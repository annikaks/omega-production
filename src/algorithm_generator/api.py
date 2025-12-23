import os
import uuid
import importlib.util
import traceback
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import anthropic
from concurrent.futures import ProcessPoolExecutor
from fastapi.staticfiles import StaticFiles  # <-- Add this
from fastapi.responses import FileResponse

from generate import AlgoGen 
from evaluate import BenchmarkSuite, eval_one_benchmark_task, BenchmarkTask
from metaprompt import LOG_FILE, GENERATION_DIRECTORY_PATH

app = FastAPI(title="OMEGA API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

algo_gen = None
suite = None

@app.on_event("startup")
def startup_event():
    """This ensures data loads ONCE in the parent process only"""
    global algo_gen, suite
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    algo_gen = AlgoGen(anthropic_client=client, log_file=LOG_FILE)
    
    suite = BenchmarkSuite()

class SynthesisRequest(BaseModel):
    description: str

def eval_single_ds(args):
    """
    Worker function. Note: We do NOT refer to the global 'suite' here.
    Everything the worker needs is passed in via 'args'.
    """
    dataset_name, model_content, class_name, X_train, X_test, y_train, y_test = args
    
    spec = importlib.util.spec_from_loader("temp_mod", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(model_content, module.__dict__)
    
    Cls = getattr(module, class_name)
    model_instance = Cls()
    
    from evaluate import BenchmarkTask, eval_one_benchmark_task
    task = BenchmarkTask(model_instance, class_name, dataset_name, X_train, X_test, y_train, y_test)
    m_name, d_name, cell, err, stats = eval_one_benchmark_task(task)
    return m_name, d_name, cell, stats

@app.post("/generate")
async def handle_synthesis(req: SynthesisRequest):
    try:
        # generate
        generated_files_result = algo_gen.parallel_genML([req.description])
        if not generated_files_result or generated_files_result[0] is None:
            raise Exception("Synthesis engine failed to generate a model.")
            
        fname, cname, _ = generated_files_result[0]
        fpath = os.path.join(GENERATION_DIRECTORY_PATH, fname)
        with open(fpath, "r") as f:
            code_string = f.read()

        # evaluate
        tasks = []
        for name, data in suite.datasets.items():
            X_train, X_test, y_train, y_test = data
            tasks.append((name, code_string, cname, X_train, X_test, y_train, y_test))
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            results_list = list(executor.map(eval_single_ds, tasks))

        suite.results = {cname: {}}
        suite.runtime_stats = {cname: {}}

        for m_name, d_name, cell, stats in results_list:
            suite.results[m_name][d_name] = cell
            suite.runtime_stats[m_name][d_name] = stats

        # tex generation
        latex_path = os.path.join(GENERATION_DIRECTORY_PATH, f"{cname}.tex")
        suite.save_latex_table_multirow(filepath=latex_path)
        
        with open(latex_path, "r") as f:
            latex_table = f.read()
            
        return {
            "id": str(uuid.uuid4()),
            "script": code_string,
            "metrics": {d_name: suite.results[cname][d_name].get("Accuracy", 0) for d_name in suite.dataset_names},
            "latex": latex_table
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)