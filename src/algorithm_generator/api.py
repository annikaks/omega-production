import os
import uuid
import json
import time
import importlib.util
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import anthropic

# Assuming these are your local files
from generate import AlgoGen
from evaluate import BenchmarkSuite, eval_one_benchmark_task, BenchmarkTask
from metaprompt import LOG_FILE, GENERATION_DIRECTORY_PATH
from describe import ModelAnalyzer 
from storage.display_benchmarks import SKLEARN_BENCHMARKS

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"
BOUNDS_PATH = STORAGE_DIR / "bounds.json"
MODELS_DATA_PATH = STORAGE_DIR / "all_models.json"
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

_LOCK_PATH = STORAGE_DIR / ".lock"
def _acquire_lock(timeout_s: float = 3.0):
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > timeout_s: raise RuntimeError("Lock timeout")
            time.sleep(0.05)
def _release_lock():
    _LOCK_PATH.unlink(missing_ok=True)

algo_gen = None
suite = None
analyzer = None 
app = FastAPI(title="OMEGA Arena")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
def startup_event():
    global algo_gen, suite, analyzer
    if not MODELS_DATA_PATH.exists():
        _write_json_atomic(MODELS_DATA_PATH, {"models": []})

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    algo_gen = AlgoGen(anthropic_client=client, log_file=LOG_FILE)
    suite = BenchmarkSuite()
    analyzer = ModelAnalyzer(anthropic_client=client) 


class SynthesisRequest(BaseModel):
    description: str

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

def update_bounds_and_rankings(new_metrics: Dict[str, float]):
    bounds = _read_json(BOUNDS_PATH, {})
    bounds_changed = False
    for ds, val in new_metrics.items():
        if ds not in bounds:
            bounds[ds] = {"min": val, "max": val}
            bounds_changed = True
        else:
            if val < bounds[ds]["min"]: bounds[ds]["min"] = val; bounds_changed = True
            if val > bounds[ds]["max"]: bounds[ds]["max"] = val; bounds_changed = True
    if bounds_changed: _write_json_atomic(BOUNDS_PATH, bounds)
    
    data = _read_json(MODELS_DATA_PATH, {"models": []})
    for m in data["models"]:
        rel_scores = []
        for ds, val in m["metrics"].items():
            mn, mx = bounds[ds]["min"], bounds[ds]["max"]
            denom = mx - mn
            rel_scores.append((val - mn) / denom if denom > 0 else 1.0)
        m["total_score"] = sum(rel_scores) / len(rel_scores) if rel_scores else 0
    _write_json_atomic(MODELS_DATA_PATH, data)

@app.get("/")
async def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/generate")
async def handle_synthesis(req: SynthesisRequest):
    try:
        gen_result = algo_gen.parallel_genML([req.description])
        fname, cname, _ = gen_result[0]
        with open(os.path.join(GENERATION_DIRECTORY_PATH, fname), "r") as f:
            code_string = f.read()

        tasks = [(n, code_string, cname, d[0], d[1], d[2], d[3]) for n, d in suite.datasets.items()]
        with ProcessPoolExecutor(max_workers=4) as executor:
            results_list = list(executor.map(eval_single_ds, tasks))

        metrics_out = {d_n: float(c.get("Accuracy", 0.0)) for _, d_n, c, _ in results_list}
        new_id = str(uuid.uuid4())
        model_entry = {"id": new_id, "name": cname, "filename": fname, "description": req.description, "metrics": metrics_out, "timestamp": time.time()
        }

        try:
            _acquire_lock()
            data = _read_json(MODELS_DATA_PATH, {"models": []})
            data["models"].append(model_entry)
            _write_json_atomic(MODELS_DATA_PATH, data)
            update_bounds_and_rankings(metrics_out)
        finally: _release_lock()
        return {**model_entry, "script": code_string}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/leaderboard")
def get_leaderboard():
    data = _read_json(MODELS_DATA_PATH, {"models": []})
    all_entries = data.get("models", []) + SKLEARN_BENCHMARKS
    
    # Recalculate Aggregates for ALL (Real Mean Accuracy)
    for m in all_entries:
        vals = list(m["metrics"].values())
        m["display_acc"] = sum(vals) / len(vals) if vals else 0
        # We still use the total_score (Z-score/Min-Max) for the actual sorting rank
        # If baseline doesn't have a score yet, use display_acc as a fallback for sorting
        if "total_score" not in m:
            m["total_score"] = m["display_acc"]

    all_entries.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    return {"top_10": [
        {
            "id": m["id"], 
            "name": m["name"], 
            "display_acc": m["display_acc"], 
            "is_baseline": m.get("is_baseline", False),
            "summary": m.get("summary", "Scikit-Learn Standard Implementation") if m.get("is_baseline") else m.get("summary")
        } for m in all_entries
    ]}

@app.get("/summarize/{model_id}")
async def get_summary(model_id: str):
    global analyzer 
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
        
    try:
        _acquire_lock()
        data = _read_json(MODELS_DATA_PATH, {"models": []})
        m = next((x for x in data["models"] if x["id"] == model_id), None)
        if not m: raise HTTPException(status_code=404, detail="Model not found")
        
        if "summary" in m and m["summary"]: 
            return {"summary": m["summary"]}

        filename = m.get("filename")
        if not filename:
            filename = f"{m['name']}.py"

        summary = analyzer.describe_single(GENERATION_DIRECTORY_PATH, filename)
        
        if " : " in summary:
            summary = summary.split(" : ", 1)[1]
        
        m["summary"] = summary
        _write_json_atomic(MODELS_DATA_PATH, data)
        return {"summary": summary}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally: _release_lock()

@app.get("/dataset-stats")
def get_dataset_stats():
    return {"stats": dict(sorted(_read_json(BOUNDS_PATH, {}).items()))}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)