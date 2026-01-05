import os
import uuid
import json
import time
import difflib
import importlib.util
import traceback
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException, Response
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
from scoring import fetch_bounds_from_supabase, recompute_min_max_scores
from sandbox_queue import SandboxQueueManager

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
if not URL or not KEY:
    raise RuntimeError("Supabase credentials not set in .env")
supabase: Client = create_client(URL, KEY)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

algo_gen = None
suite = None
analyzer = None 
queue_manager = None
app = FastAPI(title="OMEGA")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
def startup_event():
    global algo_gen, suite, analyzer, queue_manager
    logger.info("startup_event called")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    algo_gen = AlgoGen(anthropic_client=client, log_file=LOG_FILE)
    suite = BenchmarkSuite()
    analyzer = ModelAnalyzer(anthropic_client=client) 

    use_e2b = os.getenv("USE_E2B_SANDBOX", "").lower() in ("1", "true", "yes", "on")
    if use_e2b:
        pool_size = int(os.getenv("E2B_POOL_SIZE", "15"))
        queue_limit = int(os.getenv("E2B_QUEUE_LIMIT", "150"))
        job_timeout_s = int(os.getenv("E2B_JOB_TIMEOUT_S", "1800"))
        worker_count = int(os.getenv("E2B_WORKERS", str(pool_size)))
        queue_manager = SandboxQueueManager(
            supabase,
            pool_size=pool_size,
            queue_limit=queue_limit,
            job_timeout_s=job_timeout_s,
            worker_count=worker_count,
        )
        queue_manager.start()
        logger.info(
            "sandbox queue manager started (pool_size=%s, queue_limit=%s, workers=%s, instance=%s)",
            pool_size,
            queue_limit,
            worker_count,
            queue_manager.instance_id,
        )

@app.get("/config")
def get_config():
    logger.info("get_config called")
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY") 
    }

class SynthesisRequest(BaseModel):
    description: str
    user_id: str
    creator_name: str

def _fetch_existing_class_names() -> list[str]:
    res = supabase.table("algorithms").select("class_name").execute()
    names = []
    for row in res.data or []:
        name = row.get("class_name")
        if isinstance(name, str):
            names.append(name)
    return names


def _find_similar_names(target: str, candidates: list[str], limit: int = 12, threshold: float = 0.8) -> list[str]:
    if not target:
        return []
    target_lower = target.lower()
    scored = []
    for name in candidates:
        if not isinstance(name, str):
            continue
        ratio = difflib.SequenceMatcher(None, target_lower, name.lower()).ratio()
        if ratio >= threshold:
            scored.append((ratio, name))
    scored.sort(reverse=True)
    return [name for _ratio, name in scored[:limit]]


def eval_single_ds(args):
    logger.debug("eval_single_ds called")
    dataset_name, model_content, class_name, X_train, X_test, y_train, y_test = args
    try:
        spec = importlib.util.spec_from_loader("temp_mod", loader=None)
        module = importlib.util.module_from_spec(spec)
        exec(model_content, module.__dict__)
        Cls = getattr(module, class_name)
        model_instance = Cls()
        task = BenchmarkTask(model=model_instance, model_name=class_name, dataset_name=dataset_name,
                             X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        m_name, d_name, cell, err, stats = eval_one_benchmark_task(task)
        return m_name, d_name, cell, stats
    except Exception:
        return class_name, dataset_name, {"Accuracy": 0.0}, {}

@app.get("/")
async def read_index():
    logger.info("read_index called")
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.post("/generate")
async def handle_synthesis(req: SynthesisRequest):
    try:
        logger.info("handle_synthesis called")
        start_time = time.time()
        use_e2b = os.getenv("USE_E2B_SANDBOX", "").lower() in ("1", "true", "yes", "on")
        if use_e2b:
            logger.info("handle_synthesis queueing E2B sandbox job")
            if queue_manager is None:
                raise RuntimeError("E2B queue manager not initialized")
            queue_res = queue_manager.enqueue_job(
                description=req.description,
                user_id=req.user_id,
                creator_name=req.creator_name,
            )
            if queue_res.get("status") == "rejected":
                raise HTTPException(
                    status_code=429,
                    detail="Queue is full, try again later.",
                )
            return {
                "status": "queued",
                "job_id": queue_res.get("job_id"),
                "position": queue_res.get("position"),
            }
        else:
            logger.info("handle_synthesis using local ProcessPoolExecutor for evaluation")
            existing_names = _fetch_existing_class_names()
            fname = cname = strategy = None
            forbidden = []
            for attempt in range(2):
                gen_result = algo_gen.parallel_genML([req.description], forbidden_names=forbidden)
                fname, cname, strategy = gen_result[0]
                similar = _find_similar_names(cname or "", existing_names)
                if cname and cname not in existing_names and not similar:
                    break
                forbidden = list(dict.fromkeys(similar + ([cname] if cname else [])))
            if cname in existing_names or _find_similar_names(cname or "", existing_names):
                raise HTTPException(
                    status_code=409,
                    detail="Generated class name already exists or is too similar. Please try again.",
                )
            file_path = os.path.join(GENERATION_DIRECTORY_PATH, fname)
            with open(file_path, "r") as f:
                code_string = f.read()
            try:
                import_string = f"from {IMPORT_STRUCTURE_PREFIX}{fname.split('.py')[0]} import *"
                init_file_path = os.path.join(GENERATION_DIRECTORY_PATH, "__init__.py")
                algo_gen.remove_import_from_init(init_file_path, import_string)
                os.remove(file_path)
            except Exception as exc:
                logger.warning("Failed to clean up generated file %s: %s", file_path, exc)
            tasks = [(n, code_string, cname, d[0], d[1], d[2], d[3]) for n, d in suite.datasets.items()]
            with ProcessPoolExecutor(max_workers=4) as executor:
                results_list = list(executor.map(eval_single_ds, tasks))
            metrics_out = {d_n: float(c.get("Accuracy", 0.0)) for _, d_n, c, _ in results_list}
            logger.info("handle_synthesis local evaluation completed in %.2fs", time.time() - start_time)
        
        eval_time = time.time() - start_time

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
            "aggregate_acc": sum(metrics_out.values()) / len(metrics_out) if metrics_out else 0,
            "min_max_score": 0.0,
            "iris_acc": metrics_out.get("Iris", 0),
            "wine_acc": metrics_out.get("Wine", 0),
            "breast_cancer_acc": metrics_out.get("Breast Cancer", 0),
            "digits_acc": metrics_out.get("Digits", 0),
            "balance_scale_acc": metrics_out.get("Balance Scale", 0),
            "blood_transfusion_acc": metrics_out.get("Blood Transfusion", 0),
            "haberman_acc": metrics_out.get("Haberman", 0),
            "seeds_acc": metrics_out.get("Seeds", 0),
            "teaching_assistant_acc": metrics_out.get("Teaching Assistant", 0),
            "zoo_acc": metrics_out.get("Zoo", 0),
            "planning_relax_acc": metrics_out.get("Planning Relax", 0),
            "ionosphere_acc": metrics_out.get("Ionosphere", 0),
            "sonar_acc": metrics_out.get("Sonar", 0),
            "glass_acc": metrics_out.get("Glass", 0),
            "vehicle_acc": metrics_out.get("Vehicle", 0),
            "liver_disorders_acc": metrics_out.get("Liver Disorders", 0),
            "heart_statlog_acc": metrics_out.get("Heart Statlog", 0),
            "pima_diabetes_acc": metrics_out.get("Pima Indians Diabetes", 0),
            "australian_acc": metrics_out.get("Australian", 0),
            "monks_1_acc": metrics_out.get("Monks-1", 0)
        }

        db_res = supabase.table("algorithms").insert(db_payload).execute()
        new_id = db_res.data[0]['id']
        recompute_min_max_scores(supabase)
        updated = supabase.table("algorithms").select("min_max_score").eq("id", new_id).single().execute()
        display_acc = updated.data.get("min_max_score") if updated.data else 0.0

        return {"id": new_id, "name": cname, "metrics": metrics_out, "display_acc": display_acc}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue-status/{job_id}")
def get_queue_status(job_id: str):
    logger.info("get_queue_status called for job_id=%s", job_id)
    if queue_manager is None:
        raise HTTPException(status_code=400, detail="Queue manager not enabled")
    return queue_manager.get_job_status(job_id)

@app.get("/leaderboard")
def get_leaderboard():
    logger.info("get_leaderboard called")
    res = supabase.table("algorithms") \
        .select("id, class_name, aggregate_acc, min_max_score, creator_name, user_prompt") \
        .order("min_max_score", desc=True) \
        .execute()
    
    user_models = []
    for row in res.data:
        user_models.append({
            "id": row['id'],
            "name": row['class_name'],
            "raw_acc": row['aggregate_acc'], 
            "display_acc": row['min_max_score'], 
            "creator_name": row.get('creator_name'),
            "user_prompt": row.get('user_prompt'),
            "is_baseline": row.get('user_prompt') == 'benchmark'
        })
    
    return {"ranked_list": user_models}

@app.get("/leaderboard-history")
def get_leaderboard_history():
    logger.info("get_leaderboard_history called")
    res = (
        supabase.table("algorithms")
        .select("id, class_name, aggregate_acc, min_max_score, creator_name, user_prompt, created_at")
        .execute()
    )
    rows = res.data or []
    parsed = []
    for row in rows:
        created_at = row.get("created_at")
        created_dt = None
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_dt = None
        parsed.append({**row, "_created_at_dt": created_dt})

    now = datetime.now(timezone.utc)
    days = []
    for offset in range(7):
        day = (now - timedelta(days=offset)).date()
        day_end = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
        candidates = [
            row for row in parsed
            if row["_created_at_dt"] is not None and row["_created_at_dt"] < day_end
        ]
        candidates.sort(key=lambda r: (r.get("min_max_score") or 0.0), reverse=True)
        top = candidates[:10]
        ranked_list = []
        for row in top:
            ranked_list.append({
                "id": row["id"],
                "name": row["class_name"],
                "raw_acc": row.get("aggregate_acc"),
                "display_acc": row.get("min_max_score"),
                "creator_name": row.get("creator_name"),
                "user_prompt": row.get("user_prompt"),
                "created_at": row.get("created_at"),
            })
        days.append({"date": day.isoformat(), "ranked_list": ranked_list})

    return {"days": days}

@app.get("/dataset-stats")
def get_dataset_stats():
    logger.info("get_dataset_stats called")
    data = fetch_bounds_from_supabase(supabase)
    return {"stats": dict(sorted(data.items()))}

@app.get("/summarize/{model_id}")
async def get_summary(model_id: str):
    try:
        logger.info("get_summary called for model_id=%s", model_id)
        res = (
            supabase.table("algorithms")
            .select("summary, file_name, algorithm_code")
            .eq("id", model_id)
            .single()
            .execute()
        )
        if res.data.get("summary"):
            return {"summary": res.data["summary"]}

        if res.data.get("algorithm_code"):
            summary = analyzer.describe_code(res.data["algorithm_code"])
        else:
            summary = analyzer.describe_single(GENERATION_DIRECTORY_PATH, res.data["file_name"])
        if "Error" in summary:
            return {"summary": "Error"}
        
        supabase.table("algorithms").update({"summary": summary}).eq("id", model_id).execute()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/algorithm-code/{model_id}")
def get_algorithm_code(model_id: str):
    logger.info("get_algorithm_code called for model_id=%s", model_id)
    res = (
        supabase.table("algorithms")
        .select("class_name, file_name, algorithm_code")
        .eq("id", model_id)
        .single()
        .execute()
    )
    if not res.data or not res.data.get("algorithm_code"):
        raise HTTPException(status_code=404, detail="Algorithm code not found")
    class_name = res.data.get("class_name") or "model"
    filename = res.data.get("file_name") or f"{class_name}.py"
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return Response(res.data["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
