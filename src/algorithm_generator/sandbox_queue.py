import hashlib
import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from postgrest.exceptions import APIError
from supabase import Client

from e2b_sandbox import (
    E2BSandboxError,
    create_e2b_sandbox,
    close_e2b_sandbox,
    generate_and_eval_with_sandbox,
)
from scoring import recompute_min_max_scores


logger = logging.getLogger(__name__)


class SandboxQueueManager:
    def __init__(
        self,
        supabase: Client,
        pool_size: int = 15,
        queue_limit: int = 150,
        job_timeout_s: int = 6000,
        worker_count: int = 1,
        poll_interval_s: float = 1.0,
        retry_max_attempts: int = 3,
        retry_backoff_s: float = 0.5,
        queue_table: str = "sandbox_queue",
    ) -> None:
        self.supabase = supabase
        self.pool_size = pool_size
        self.queue_limit = queue_limit
        self.job_timeout_s = job_timeout_s
        self.worker_count = max(1, worker_count)
        self.poll_interval_s = poll_interval_s
        self.retry_max_attempts = max(1, retry_max_attempts)
        self.retry_backoff_s = max(0.0, retry_backoff_s)
        self.queue_table = queue_table
        self.pool: queue.Queue = queue.Queue(maxsize=pool_size)
        self._sandbox_slots = threading.BoundedSemaphore(pool_size)
        self._pool_lock = threading.Lock()
        self._current_pool = 0
        self._last_create_failed_at: Optional[float] = None
        self._stop_event = threading.Event()
        self._worker_threads = []

    def start(self) -> None:
        for idx in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker_loop, daemon=True, name=f"sandbox-worker-{idx}"
            )
            self._worker_threads.append(thread)
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def enqueue_job(self, description: str, user_id: str, creator_name: str) -> Dict[str, Any]:
        queued_count = self._count_by_status("queued")
        if queued_count >= self.queue_limit:
            logger.info("queue full (queued=%s limit=%s)", queued_count, self.queue_limit)
            return {
                "status": "rejected",
                "reason": "queue_full",
                "queued_limit": self.queue_limit,
            }

        job_id = str(uuid.uuid4())
        insert_payload = {
            "id": job_id,
            "description": description,
            "user_id": user_id,
            "creator_name": creator_name,
            "status": "queued",
        }
        res = self._execute_with_retry(
            lambda: self.supabase.table(self.queue_table).insert(insert_payload).execute(),
            "enqueue_job",
        )
        row = res.data[0] if res.data else insert_payload
        created_at = row.get("created_at")
        position = self._queue_position(created_at)
        logger.info("queued job_id=%s position=%s queued=%s", job_id, position, queued_count + 1)
        return {
            "status": "queued",
            "job_id": job_id,
            "position": position,
        }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        res = self._execute_with_retry(
            lambda: self.supabase.table(self.queue_table)
            .select("*")
            .eq("id", job_id)
            .single()
            .execute(),
            "get_job_status",
        )
        if not res.data:
            return {"status": "not_found", "job_id": job_id}
        row = res.data
        status = row.get("status")
        error = row.get("error")
        if error and status == "completed":
            status = "failed"
        response = {
            "status": status,
            "job_id": job_id,
            "algorithm_id": row.get("algorithm_id"),
            "algorithm_id_text": row.get("algorithm_id_text"),
            "error": error,
        }
        if status == "completed":
            algo_id = row.get("algorithm_id")
            algo_id_text = row.get("algorithm_id_text")
            algo_key = algo_id if algo_id is not None else algo_id_text
            if algo_key is not None:
                try:
                    algo_res = self._execute_with_retry(
                        lambda: self.supabase.table("algorithms")
                        .select("class_name")
                        .eq("id", algo_key)
                        .single()
                        .execute(),
                        "get_algorithm_class_name",
                    )
                    if algo_res.data and algo_res.data.get("class_name"):
                        response["class_name"] = algo_res.data["class_name"]
                except Exception:
                    pass
        if status == "queued":
            response["position"] = self._queue_position(row.get("created_at"))
        return response

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._fetch_next_job()
            except Exception as exc:
                logger.error("queue poll failed: %s", exc)
                time.sleep(self.poll_interval_s)
                continue
            if not job:
                time.sleep(self.poll_interval_s)
                continue

            if not self._sandbox_slots.acquire(timeout=self.poll_interval_s):
                continue
            started_at = datetime.utcnow().isoformat()
            if not self._claim_job(job["id"], started_at):
                self._sandbox_slots.release()
                continue
            logger.info("processing job_id=%s description=%s", job["id"], job.get("description"))
            try:
                sandbox = self._create_sandbox_with_retry()
            except Exception as exc:
                self._mark_failed(job["id"], f"Sandbox unavailable: {exc}")
                self._sandbox_slots.release()
                continue
            try:
                self._process_job(job, sandbox)
            finally:
                close_e2b_sandbox(sandbox)
                self._sandbox_slots.release()

    def _replace_sandbox(self) -> None:
        try:
            sandbox = create_e2b_sandbox()
            with self._pool_lock:
                self._current_pool += 1
            self.pool.put(sandbox)
        except Exception as exc:
            logger.error("Failed to create E2B sandbox: %s", exc)
            self._last_create_failed_at = time.time()

    def _get_or_create_sandbox(self) -> Any:
        try:
            return self.pool.get_nowait()
        except queue.Empty:
            self._replace_sandbox()
            return self.pool.get()

    def _scale_pool_for_work(self) -> None:
        queued_count = self._count_by_status("queued")
        desired = min(self.pool_size, max(1, queued_count))
        with self._pool_lock:
            current = self._current_pool
        if desired <= current:
            return
        if self._last_create_failed_at and time.time() - self._last_create_failed_at < 5:
            return
        to_create = desired - current
        for _ in range(to_create):
            self._replace_sandbox()

    def _fetch_next_job(self) -> Optional[Dict[str, Any]]:
        httpx_logger = logging.getLogger("httpx")
        prev_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        try:
            res = self._execute_with_retry(
                lambda: self.supabase.table(self.queue_table)
                .select("*")
                .eq("status", "queued")
                .order("created_at")
                .limit(1)
                .execute(),
                "fetch_next_job",
            )
        finally:
            httpx_logger.setLevel(prev_level)
        if not res.data:
            queued_count = self._count_by_status("queued")
            if queued_count:
                logger.info("queue has %s queued jobs but none fetched", queued_count)
            return None
        job = res.data[0]
        logger.info("fetched job_id=%s status=%s", job.get("id"), job.get("status"))
        return job

    def _process_job(self, job: Dict[str, Any], sandbox: Any) -> None:
        job_id = job["id"]
        description = job["description"]
        user_id = self._safe_user_id(job.get("user_id"))
        creator_name = job.get("creator_name")
        created_at = self._parse_dt(job.get("created_at"))
        if created_at and self._is_timed_out(created_at):
            self._mark_failed(job_id, "Timed out waiting in queue")
            return
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self._mark_failed(job_id, "ANTHROPIC_API_KEY not set")
            return

        try:
            t0 = time.time()
            result = generate_and_eval_with_sandbox(
                sandbox,
                description=description,
                dataset_names=self._default_dataset_names(),
                anthropic_api_key=api_key,
            )
            eval_time = time.time() - t0
            if eval_time > self.job_timeout_s:
                self._mark_failed(job_id, "Processing timeout")
                return
        except E2BSandboxError as exc:
            self._mark_failed(job_id, str(exc))
            return
        except Exception as exc:
            self._mark_failed(job_id, f"{type(exc).__name__}: {exc}")
            return

        metrics = {k: float(v) for k, v in result.get("metrics", {}).items()}
        class_name = result.get("class_name") or "GeneratedModel"
        file_name = result.get("file_name") or f"{class_name}.py"
        code_string = result.get("code_string") or ""
        strategy = result.get("strategy") or "E2B Synthesis"

        db_payload = {
            "user_id": user_id,
            "creator_name": creator_name,
            "user_prompt": description,
            "strategy_label": strategy,
            "class_name": class_name,
            "file_name": file_name,
            "algorithm_code": code_string,
            "code_hash": hashlib.sha256(code_string.strip().encode()).hexdigest(),
            "eval_time_seconds": eval_time,
            "aggregate_acc": sum(metrics.values()) / len(metrics) if metrics else 0,
            "min_max_score": 0.0,
            "iris_acc": metrics.get("Iris", 0),
            "wine_acc": metrics.get("Wine", 0),
            "breast_cancer_acc": metrics.get("Breast Cancer", 0),
            "digits_acc": metrics.get("Digits", 0),
            "balance_scale_acc": metrics.get("Balance Scale", 0),
            "blood_transfusion_acc": metrics.get("Blood Transfusion", 0),
            "haberman_acc": metrics.get("Haberman", 0),
            "seeds_acc": metrics.get("Seeds", 0),
            "teaching_assistant_acc": metrics.get("Teaching Assistant", 0),
            "zoo_acc": metrics.get("Zoo", 0),
            "planning_relax_acc": metrics.get("Planning Relax", 0),
            "ionosphere_acc": metrics.get("Ionosphere", 0),
            "sonar_acc": metrics.get("Sonar", 0),
            "glass_acc": metrics.get("Glass", 0),
            "vehicle_acc": metrics.get("Vehicle", 0),
            "liver_disorders_acc": metrics.get("Liver Disorders", 0),
            "heart_statlog_acc": metrics.get("Heart Statlog", 0),
            "pima_diabetes_acc": metrics.get("Pima Indians Diabetes", 0),
            "australian_acc": metrics.get("Australian", 0),
            "monks_1_acc": metrics.get("Monks-1", 0),
        }
        try:
            db_res = self.supabase.table("algorithms").insert(db_payload).execute()
        except APIError as exc:
            if exc.args and isinstance(exc.args[0], dict):
                code = exc.args[0].get("code")
                details = exc.args[0].get("details", "")
                if code == "23505" and "code_hash" in details:
                    self._mark_failed(job_id, "Duplicate model code. Please try again.")
                    return
            self._mark_failed(job_id, "Database error while saving results.")
            return
        algorithm_id = db_res.data[0]["id"] if db_res.data else None
        try:
            recompute_min_max_scores(self.supabase)
        except Exception as exc:
            logger.error("Failed to recompute min_max_score bounds: %s", exc)
        update_payload = {
            "status": "completed",
            "finished_at": datetime.utcnow().isoformat(),
        }
        if isinstance(algorithm_id, int):
            update_payload["algorithm_id"] = algorithm_id
        elif algorithm_id is not None:
            update_payload["algorithm_id_text"] = str(algorithm_id)
        self._update_job(
            job_id,
            update_payload,
        )

    def _update_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        try:
            self._execute_with_retry(
                lambda: self.supabase.table(self.queue_table).update(payload).eq("id", job_id).execute(),
                "update_job",
            )
        except Exception as exc:
            logger.error("Failed to update job_id=%s: %s", job_id, exc)

    def _mark_failed(self, job_id: str, error: str) -> None:
        self._update_job(
            job_id,
            {"status": "failed", "error": error, "finished_at": datetime.utcnow().isoformat()},
        )

    def _count_by_status(self, status: str) -> int:
        res = self._execute_with_retry(
            lambda: self.supabase.table(self.queue_table)
            .select("id", count="exact")
            .eq("status", status)
            .execute(),
            f"count_by_status:{status}",
        )
        return res.count or 0

    def _queue_position(self, created_at: Optional[str]) -> Optional[int]:
        if not created_at:
            return None
        res = self._execute_with_retry(
            lambda: self.supabase.table(self.queue_table)
            .select("id", count="exact")
            .eq("status", "queued")
            .lt("created_at", created_at)
            .execute(),
            "queue_position",
        )
        return (res.count or 0) + 1

    def _default_dataset_names(self) -> list:
        return [
            "Iris",
            "Wine",
            "Breast Cancer",
            "Digits",
            "Balance Scale",
            "Blood Transfusion",
            "Haberman",
            "Seeds",
            "Teaching Assistant",
            "Zoo",
            "Planning Relax",
            "Ionosphere",
            "Sonar",
            "Glass",
            "Vehicle",
            "Liver Disorders",
            "Heart Statlog",
            "Pima Indians Diabetes",
            "Australian",
            "Monks-1",
        ]

    def _parse_dt(self, raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _is_timed_out(self, created_at: datetime) -> bool:
        return (datetime.utcnow() - created_at.replace(tzinfo=None)).total_seconds() > self.job_timeout_s

    def _claim_job(self, job_id: str, started_at: str) -> bool:
        try:
            res = self._execute_with_retry(
                lambda: self.supabase.table(self.queue_table)
                .update({"status": "processing", "started_at": started_at})
                .eq("id", job_id)
                .eq("status", "queued")
                .execute(),
                "claim_job",
            )
            return bool(res.data)
        except Exception as exc:
            logger.error("Failed to claim job_id=%s: %s", job_id, exc)
            return False

    def _execute_with_retry(self, action, context: str):
        delay = self.retry_backoff_s
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return action()
            except httpx.HTTPError as exc:
                if attempt >= self.retry_max_attempts:
                    logger.error("retry exhausted for %s: %s", context, exc)
                    raise
                logger.debug(
                    "retryable http error for %s (attempt %s/%s): %s",
                    context,
                    attempt,
                    self.retry_max_attempts,
                    exc,
                )
                if delay:
                    time.sleep(delay)
                    delay *= 2
        return action()

    def _create_sandbox_with_retry(self) -> Any:
        delay = self.retry_backoff_s
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return create_e2b_sandbox()
            except Exception as exc:
                if attempt >= self.retry_max_attempts:
                    raise
                logger.debug(
                    "retryable sandbox create error (attempt %s/%s): %s",
                    attempt,
                    self.retry_max_attempts,
                    exc,
                )
                if delay:
                    time.sleep(delay)
                    delay *= 2

    def _safe_user_id(self, user_id: Optional[str]) -> Optional[str]:
        if not user_id:
            return None
        try:
            uuid.UUID(user_id)
            return user_id
        except ValueError:
            return None
