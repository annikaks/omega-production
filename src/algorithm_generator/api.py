import os
import uuid
import json
import time
import importlib.util
import traceback
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

import anthropic

from generate import AlgoGen
from evaluate import BenchmarkSuite, eval_one_benchmark_task, BenchmarkTask
from metaprompt import LOG_FILE, GENERATION_DIRECTORY_PATH
from describe import ModelAnalyzer 
from storage.display_benchmarks import SKLEARN_BENCHMARKS

load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
if not URL or not KEY:
    raise RuntimeError("Supabase credentials not set in .env")
supabase: Client = create_client(URL, KEY)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"
BOUNDS_PATH = STORAGE_DIR / "bounds.json"
STORAGE_DIR.mkdir(exist_ok=True)

def _read_json(path: Path, default: Any):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception: pass
    return default

def _write_json_atomic(path: Path, data: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

algo_gen = None
suite = None
analyzer = None 
app = FastAPI(title="OMEGA Arena")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
def startup_event():
    global algo_gen, suite, analyzer
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    algo_gen = AlgoGen(anthropic_client=client, log_file=LOG_FILE)
    suite = BenchmarkSuite()
    analyzer = ModelAnalyzer(anthropic_client=client) 

@app.get("/config")
def get_config():
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY") 
    }

class SynthesisRequest(BaseModel):
    description: str
    user_id: str
    creator_name: str

def eval_single_ds(args):
    dataset_name, model_content, class_name, X_train, X_test, y_train, y_test = args
    spec = importlib.util.spec_from_loader("temp_mod", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(model_content, module.__dict__)
    Cls = getattr(module, class_name)
    model_instance = Cls()
    task = BenchmarkTask(model=model_instance, model_name=class_name, dataset_name=dataset_name,
                         X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    m_name, d_name, cell, err, stats = eval_one_benchmark_task(task)
    return m_name, d_name, cell, stats

def update_bounds_and_calculate_score(new_metrics: Dict[str, float]):
    bounds = _read_json(BOUNDS_PATH, {})
    clean_metrics = {k: (v if isinstance(v, (int, float)) else 0.0) for k, v in new_metrics.items()}
    bounds_changed = False
    for ds, val in new_metrics.items():
        if ds not in bounds:
            bounds[ds] = {"min": val, "max": val}
            bounds_changed = True
        else:
            if val < bounds[ds]["min"]: bounds[ds]["min"] = val; bounds_changed = True
            if val > bounds[ds]["max"]: bounds[ds]["max"] = val; bounds_changed = True
    
    if bounds_changed: 
        _write_json_atomic(BOUNDS_PATH, bounds)
    
    rel_scores = []
    for ds, val in new_metrics.items():
        mn, mx = bounds[ds]["min"], bounds[ds]["max"]
        denom = mx - mn
        rel_scores.append((val - mn) / denom if denom > 0 else 1.0)
    
    return sum(rel_scores) / len(rel_scores) if rel_scores else 0

@app.get("/")
async def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/generate")
async def handle_synthesis(req: SynthesisRequest):
    try:
        start_time = time.time()
        gen_result = algo_gen.parallel_genML([req.description])
        fname, cname, strategy = gen_result[0]
        with open(os.path.join(GENERATION_DIRECTORY_PATH, fname), "r") as f:
            code_string = f.read()

        tasks = [(n, code_string, cname, d[0], d[1], d[2], d[3]) for n, d in suite.datasets.items()]
        with ProcessPoolExecutor(max_workers=4) as executor:
            results_list = list(executor.map(eval_single_ds, tasks))

        metrics_out = {d_n: float(c.get("Accuracy", 0.0)) for _, d_n, c, _ in results_list}
        
        total_min_max = update_bounds_and_calculate_score(metrics_out)
        eval_time = time.time() - start_time

        # 4. Save to Database
        db_payload = {
            "user_id": req.user_id,
            "creator_name": req.creator_name,
            "user_prompt": req.description,
            "strategy_label": strategy or "Parallel Synthesis",
            "class_name": cname,
            "file_name": fname,
            "algorithm_code": code_string,
            "code_hash": hashlib.sha256(code_string.strip().encode()).hexdigest(),
            "eval_time_seconds": eval_time,
            "aggregate_acc": sum(metrics_out.values()) / len(metrics_out),
            "min_max_score": total_min_max,
            # Map metrics to columns
            "iris_acc": metrics_out.get("Iris"),
            "wine_acc": metrics_out.get("Wine"),
            "breast_cancer_acc": metrics_out.get("Breast Cancer"),
            "digits_acc": metrics_out.get("Digits"),
            "balance_scale_acc": metrics_out.get("Balance Scale"),
            "blood_transfusion_acc": metrics_out.get("Blood Transfusion"),
            "haberman_acc": metrics_out.get("Haberman"),
            "seeds_acc": metrics_out.get("Seeds"),
            "teaching_assistant_acc": metrics_out.get("Teaching Assistant"),
            "zoo_acc": metrics_out.get("Zoo"),
            "planning_relax_acc": metrics_out.get("Planning Relax"),
            "ionosphere_acc": metrics_out.get("Ionosphere"),
            "sonar_acc": metrics_out.get("Sonar"),
            "glass_acc": metrics_out.get("Glass"),
            "vehicle_acc": metrics_out.get("Vehicle"),
            "liver_disorders_acc": metrics_out.get("Liver Disorders"),
            "heart_statlog_acc": metrics_out.get("Heart Statlog"),
            "pima_diabetes_acc": metrics_out.get("Pima Indians Diabetes"),
            "australian_acc": metrics_out.get("Australian"),
            "monks_1_acc": metrics_out.get("Monks-1")
        }

        db_res = supabase.table("algorithms").insert(db_payload).execute()
        new_id = db_res.data[0]['id']

        return {"id": new_id, "name": cname, "metrics": metrics_out, "display_acc": db_payload["min_max_score"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
def get_leaderboard():
    res = supabase.table("algorithms") \
        .select("id, class_name, aggregate_acc, min_max_score, creator_name, user_prompt") \
        .order("min_max_score", desc=True) \
        .execute()
    
    user_models = []
    for row in res.data:
        user_models.append({
            "id": row['id'],
            "name": row['class_name'],
            "display_acc": row['aggregate_acc'],
            "total_score": row['min_max_score'],
            "creator_name": row.get('creator_name'),
            "user_prompt": row.get('user_prompt'),
            "is_baseline": row.get('user_prompt') == 'benchmark'
        })

    
    return {"ranked_list": user_models}

@app.get("/summarize/{model_id}")
async def get_summary(model_id: str):
    try:
        res = supabase.table("algorithms").select("summary, file_name").eq("id", model_id).single().execute()
        if res.data.get("summary"):
            return {"summary": res.data["summary"]}

        summary = analyzer.describe_single(GENERATION_DIRECTORY_PATH, res.data["file_name"])
        if " : " in summary: summary = summary.split(" : ", 1)[1]
        
        supabase.table("algorithms").update({"summary": summary}).eq("id", model_id).execute()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset-stats")
def get_dataset_stats():
    return {"stats": dict(sorted(_read_json(BOUNDS_PATH, {}).items()))}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)