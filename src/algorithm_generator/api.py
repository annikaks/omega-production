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
from fastapi import FastAPI, HTTPException, Response, Header
from postgrest.exceptions import APIError
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
app = FastAPI(title="OMEGA", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

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

@app.get("/docs")
async def read_docs():
    logger.info("read_docs called")
    return FileResponse(str(STATIC_DIR / "apidocs.html"))

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
        .select("id, class_name, min_max_score, creator_name, user_prompt") \
        .order("min_max_score", desc=True) \
        .execute()
    
    user_models = []
    for row in res.data:
        user_models.append({
            "id": row['id'],
            "name": row['class_name'],
            "display_acc": row['min_max_score'], 
            "creator_name": row.get('creator_name'),
            "is_baseline": row.get('user_prompt') == 'benchmark',
            "source": "algorithms",
        })
    
    return {"ranked_list": user_models[:500]}

@app.get("/leaderboard-generations")
def get_leaderboard_generations():
    logger.info("get_leaderboard_generations called")

    def _parse_created_at(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    algo_rows = (
        supabase.table("algorithms")
        .select("id, class_name, min_max_score, creator_name, summary, created_at, user_prompt")
        .order("created_at", desc=True)
        .execute()
    )
    try:
        gen_rows = (
            supabase.table("model_generations")
            .select("id, class_name, min_max_score, generator_llm, summary, created_at")
            .order("created_at", desc=True)
            .execute()
        )
    except APIError:
        logger.warning("model_generations.created_at missing; falling back to min_max_score ordering")
        gen_rows = (
            supabase.table("model_generations")
            .select("id, class_name, min_max_score, generator_llm, summary")
            .order("min_max_score", desc=True)
            .execute()
        )

    merged: Dict[str, Dict[str, Any]] = {}
    for row in algo_rows.data or []:
        class_name = row.get("class_name")
        if not class_name:
            continue
        merged[class_name] = {
            "id": row.get("id"),
            "name": class_name,
            "display_acc": _coerce_float(row.get("min_max_score")),
            "creator_name": row.get("creator_name"),
            "summary": row.get("summary"),
            "is_baseline": row.get("user_prompt") == "benchmark",
            "source": "algorithms",
            "_created_at": _parse_created_at(row.get("created_at")),
        }

    for row in gen_rows.data or []:
        class_name = row.get("class_name")
        if not class_name:
            continue
        created_at = _parse_created_at(row.get("created_at")) if isinstance(row, dict) else None
        candidate = {
            "id": row.get("id") or class_name,
            "name": class_name,
            "display_acc": _coerce_float(row.get("min_max_score")),
            "creator_name": row.get("generator_llm"),
            "summary": row.get("summary"),
            "is_baseline": False,
            "source": "model_generations",
            "_created_at": created_at,
        }
        existing = merged.get(class_name)
        if not existing:
            merged[class_name] = candidate
            continue
        existing_ts = existing.get("_created_at")
        if existing_ts is None and created_at is None:
            continue
        if existing_ts is None or (created_at and created_at > existing_ts):
            merged[class_name] = candidate

    models = list(merged.values())
    models.sort(key=lambda item: (item.get("display_acc") if item.get("display_acc") is not None else -1.0), reverse=True)
    models = models[:500]
    for item in models:
        item.pop("_created_at", None)
    return {"ranked_list": models}

@app.get("/leaderboard-history")
def get_leaderboard_history():
    logger.info("get_leaderboard_history called")
    now = datetime.now(timezone.utc)
    days = []
    for offset in range(7):
        day = (now - timedelta(days=offset)).date()
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        day_end = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
        res = (
            supabase.table("algorithms")
            .select("id, class_name, min_max_score, creator_name, created_at")
            .lt("created_at", day_end.isoformat())
            .order("min_max_score", desc=True)
            .limit(10)
            .execute()
        )
        ranked_list = []
        for row in res.data or []:
            ranked_list.append({
                "id": row["id"],
                "name": row["class_name"],
                "display_acc": row.get("min_max_score"),
                "creator_name": row.get("creator_name"),
                "source": "algorithms",
            })
        days.append({"date": day.isoformat(), "ranked_list": ranked_list})

    return {"days": days}

@app.get("/dataset-stats")
def get_dataset_stats():
    logger.info("get_dataset_stats called")
    data = fetch_bounds_from_supabase(supabase)
    return {"stats": dict(sorted(data.items()))}

@app.get("/summarize/{model_id}")
async def get_summary(model_id: str, source: Optional[str] = None):
    try:
        logger.info("get_summary called for model_id=%s", model_id)
        if source == "model_generations":
            gen_res = (
                supabase.table("model_generations")
                .select("summary, algorithm_code")
                .eq("id", model_id)
                .limit(1)
                .execute()
            )
            if not gen_res.data:
                raise HTTPException(status_code=404, detail="Summary not found")
            row = gen_res.data[0]
            if row.get("summary"):
                return {"summary": row["summary"]}
            if row.get("algorithm_code"):
                summary = analyzer.describe_code(row["algorithm_code"])
                if "Error" in summary:
                    return {"summary": "Error"}
                supabase.table("model_generations").update({"summary": summary}).eq("id", model_id).execute()
                return {"summary": summary}
            raise HTTPException(status_code=404, detail="Summary not found")

        if source == "algorithms":
            try:
                uuid.UUID(model_id)
            except ValueError:
                raise HTTPException(status_code=404, detail="Summary not found")
            res = (
                supabase.table("algorithms")
                .select("summary, file_name, algorithm_code")
                .eq("id", model_id)
                .limit(1)
                .execute()
            )
            if not res.data:
                raise HTTPException(status_code=404, detail="Summary not found")
            res_data = res.data[0]
            if res_data.get("summary"):
                return {"summary": res_data["summary"]}
            if res_data.get("algorithm_code"):
                summary = analyzer.describe_code(res_data["algorithm_code"])
            else:
                summary = analyzer.describe_single(GENERATION_DIRECTORY_PATH, res_data["file_name"])
            if "Error" in summary:
                return {"summary": "Error"}
            supabase.table("algorithms").update({"summary": summary}).eq("id", model_id).execute()
            return {"summary": summary}

        res_data = None
        try:
            uuid.UUID(model_id)
            res = (
                supabase.table("algorithms")
                .select("summary, file_name, algorithm_code")
                .eq("id", model_id)
                .limit(1)
                .execute()
            )
            res_data = res.data[0] if res.data else None
        except ValueError:
            res_data = None
        except APIError as exc:
            logger.warning("summary lookup in algorithms failed: %s", exc)
            res_data = None
        if res_data:
            if res_data.get("summary"):
                return {"summary": res_data["summary"]}
            if res_data.get("algorithm_code"):
                summary = analyzer.describe_code(res_data["algorithm_code"])
            else:
                summary = analyzer.describe_single(GENERATION_DIRECTORY_PATH, res_data["file_name"])
            if "Error" in summary:
                return {"summary": "Error"}
            supabase.table("algorithms").update({"summary": summary}).eq("id", model_id).execute()
            return {"summary": summary}

        gen_res = (
            supabase.table("model_generations")
            .select("summary, algorithm_code, created_at")
            .eq("class_name", model_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if gen_res.data:
            row = gen_res.data[0]
            if row.get("summary"):
                return {"summary": row["summary"]}
            if row.get("algorithm_code"):
                summary = analyzer.describe_code(row["algorithm_code"])
                if "Error" in summary:
                    return {"summary": "Error"}
                supabase.table("model_generations").update({"summary": summary}).eq("class_name", model_id).execute()
                return {"summary": summary}
        raise HTTPException(status_code=404, detail="Summary not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/algorithm-code/by-class/{class_name}")
def get_algorithm_code_by_class(class_name: str):
    logger.info("get_algorithm_code_by_class called for class_name=%s", class_name)
    res = (
        supabase.table("algorithms")
        .select("class_name, file_name, algorithm_code")
        .eq("class_name", class_name)
        .limit(1)
        .execute()
    )
    if res.data:
        row = res.data[0]
        if row.get("algorithm_code"):
            filename = row.get("file_name") or f"{class_name}.py"
            headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
            return Response(row["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)
    alt = (
        supabase.table("model_generations")
        .select("class_name, algorithm_code")
        .eq("class_name", class_name)
        .limit(1)
        .execute()
    )
    if not alt.data:
        raise HTTPException(status_code=404, detail="Algorithm code not found")
    row = alt.data[0]
    if not row.get("algorithm_code"):
        raise HTTPException(status_code=404, detail="Algorithm code not found")
    filename = f"{class_name}.py"
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return Response(row["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)

@app.get("/algorithm-code/by-id/{model_id}")
def get_algorithm_code_by_id(model_id: str, source: Optional[str] = None):
    logger.info("get_algorithm_code_by_id called for model_id=%s source=%s", model_id, source)
    if source == "model_generations":
        res = (
            supabase.table("model_generations")
            .select("id, class_name, algorithm_code")
            .eq("id", model_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Algorithm code not found")
        row = res.data[0]
        if not row.get("algorithm_code"):
            raise HTTPException(status_code=404, detail="Algorithm code not found")
        filename = f"{row.get('class_name') or 'model'}.py"
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
        return Response(row["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)

    if source == "algorithms":
        res = (
            supabase.table("algorithms")
            .select("id, class_name, file_name, algorithm_code")
            .eq("id", model_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            raise HTTPException(status_code=404, detail="Algorithm code not found")
        row = res.data[0]
        if not row.get("algorithm_code"):
            raise HTTPException(status_code=404, detail="Algorithm code not found")
        filename = row.get("file_name") or f"{row.get('class_name') or 'model'}.py"
        headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
        return Response(row["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)

    res = (
        supabase.table("algorithms")
        .select("id, class_name, file_name, algorithm_code")
        .eq("id", model_id)
        .limit(1)
        .execute()
    )
    if res.data:
        row = res.data[0]
        if row.get("algorithm_code"):
            filename = row.get("file_name") or f"{row.get('class_name') or 'model'}.py"
            headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
            return Response(row["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)
    alt = (
        supabase.table("model_generations")
        .select("id, class_name, algorithm_code")
        .eq("id", model_id)
        .limit(1)
        .execute()
    )
    if not alt.data:
        raise HTTPException(status_code=404, detail="Algorithm code not found")
    row = alt.data[0]
    if not row.get("algorithm_code"):
        raise HTTPException(status_code=404, detail="Algorithm code not found")
    filename = f"{row.get('class_name') or 'model'}.py"
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return Response(row["algorithm_code"], media_type="text/plain; charset=utf-8", headers=headers)

@app.get("/my_info")
def get_my_info(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    token = parts[1]
    try:
        user_res = supabase.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = getattr(user_res, "user", None)
    if user is None and isinstance(user_res, dict):
        user = user_res.get("user")
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    user_metadata = getattr(user, "user_metadata", None) or (user.get("user_metadata") if isinstance(user, dict) else {})
    email = getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else None)
    display_name = None
    if isinstance(user_metadata, dict):
        display_name = user_metadata.get("display_name") or user_metadata.get("full_name")
    if not display_name and email:
        display_name = email.split("@")[0]
    return {"user_id": user_id, "creator_name": display_name}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
