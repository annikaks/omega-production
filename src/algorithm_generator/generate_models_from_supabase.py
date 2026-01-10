import argparse
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client

from e2b_sandbox import (
    E2BSandboxError,
    close_e2b_sandbox,
    create_e2b_sandbox,
    eval_with_sandbox,
)
from llm_clients import build_llm_clients, LLMClient
from scoring import DATASET_COLUMNS, recompute_min_max_scores_for_table


PROMPT_TABLE = "prompt_ideas"
MODEL_TABLE = "model_generations"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_supabase() -> Client:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set")
    return create_client(url, key)


def _build_model_prompt(model_family: str, idea: str) -> str:
    return f"""
Design a {model_family} classifier inspired by this idea: {idea}

1. Provide a succinct pythonic class name between <class_name></class_name> tags.
2. Provide a succinct pythonic filename (ending in .py) between <file_name></file_name> tags.
3. Provide the complete implementation in a single markdown python code block.

The class must inherit from sklearn.base.BaseEstimator.
The class must implement fit(self, X_train, y_train) and predict(self, X_test).
Only return the tags and the code block.
""".strip()


def _extract_code_snippet(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _default_dataset_names() -> List[str]:
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


def _evaluate_code_with_e2b(
    sandbox,
    code: str,
    class_name: str,
    dataset_names: List[str],
) -> Tuple[Dict[str, float], float]:
    start = time.time()
    metrics = eval_with_sandbox(
        sandbox=sandbox,
        code_string=code,
        class_name=class_name,
        dataset_names=dataset_names,
    )
    elapsed = time.time() - start
    return metrics, elapsed


def _should_recreate_sandbox(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "sandbox" in msg and ("not found" in msg or "timeout" in msg)


def _resolve_clients(requested: List[str], available: dict) -> List[LLMClient]:
    clients: List[LLMClient] = []
    for name in requested:
        client = available.get(name)
        if client is None:
            raise RuntimeError(f"LLM client '{name}' not configured; set API key env vars")
        clients.append(client)
    return clients


def _prompt_done(supabase: Client, prompt_id: str, llm_name: str) -> bool:
    res = (
        supabase.table(MODEL_TABLE)
        .select("id")
        .eq("prompt_id", prompt_id)
        .eq("generator_llm", llm_name)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


def _build_payload(
    prompt_row: Dict[str, object],
    llm_name: str,
    class_name: str,
    file_name: str,
    code: str,
    metrics: Dict[str, float],
    eval_time: float,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "prompt_id": prompt_row["id"],
        "prompt_text": prompt_row.get("prompt_text"),
        "prompt_source_llm": prompt_row.get("source_llm"),
        "generator_llm": llm_name,
        "class_name": class_name,
        "file_name": file_name,
        "algorithm_code": code,
        "code_hash": _code_hash(code),
        "eval_time_seconds": eval_time,
        "aggregate_acc": sum(metrics.values()) / len(metrics) if metrics else 0.0,
        "min_max_score": 0.0,
        "summary": "",
    }
    for dataset, column in DATASET_COLUMNS:
        payload[column] = float(metrics.get(dataset, 0.0))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate models from Supabase prompts.")
    parser.add_argument("--llms", default="gemini,claude,openai,groq", help="comma-separated LLMs")
    parser.add_argument("--shard-count", type=int, default=1, help="total number of shards")
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based shard index")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")

    supabase = _get_supabase()
    available_clients = build_llm_clients()
    llm_names = [name.strip() for name in args.llms.split(",") if name.strip()]
    clients = _resolve_clients(llm_names, available_clients)

    dataset_names = _default_dataset_names()
    prompt_res = supabase.table(PROMPT_TABLE).select("*").execute()
    prompt_rows = prompt_res.data or []
    logger.info("loaded prompts | count=%s", len(prompt_rows))

    sandbox = create_e2b_sandbox()
    inserted = 0
    try:
        for idx, row in enumerate(prompt_rows):
            if args.shard_count > 1 and (idx % args.shard_count) != args.shard_index:
                continue
            model_family = row.get("model_family") or "classifier"
            prompt_text = row.get("prompt_text") or ""
            prompt_id = row.get("id")
            if not prompt_id:
                continue
            for client in clients:
                try:
                    if _prompt_done(supabase, prompt_id, client.name):
                        logger.info(
                            "skipping existing generation | prompt_id=%s llm=%s",
                            prompt_id,
                            client.name,
                        )
                        continue
                    prompt = _build_model_prompt(model_family, prompt_text)
                    logger.info("generating model | prompt_id=%s llm=%s", prompt_id, client.name)
                    response = client.generate(prompt)
                    class_name = _extract_tag(response, "class_name")
                    file_name = _extract_tag(response, "file_name")
                    code = _extract_code_snippet(response)
                    if not class_name or not code:
                        logger.warning("missing code/class | prompt_id=%s llm=%s", prompt_id, client.name)
                        continue
                    logger.info(
                        "evaluating in e2b | prompt_id=%s llm=%s class=%s",
                        prompt_id,
                        client.name,
                        class_name,
                    )
                try:
                    metrics, eval_time = _evaluate_code_with_e2b(
                        sandbox,
                        code,
                        class_name,
                        dataset_names,
                    )
                except E2BSandboxError as exc:
                    if _should_recreate_sandbox(exc):
                        logger.warning(
                            "e2b sandbox expired; recreating | prompt_id=%s llm=%s error=%s",
                            prompt_id,
                            client.name,
                            exc,
                        )
                        close_e2b_sandbox(sandbox)
                        sandbox = create_e2b_sandbox()
                        try:
                            metrics, eval_time = _evaluate_code_with_e2b(
                                sandbox,
                                code,
                                class_name,
                                dataset_names,
                            )
                        except E2BSandboxError as retry_exc:
                            logger.error(
                                "e2b evaluation failed after recreate | prompt_id=%s llm=%s error=%s",
                                prompt_id,
                                client.name,
                                retry_exc,
                            )
                            continue
                    else:
                        logger.error(
                            "e2b evaluation failed | prompt_id=%s llm=%s error=%s",
                            prompt_id,
                            client.name,
                            exc,
                        )
                        continue
                    payload = _build_payload(row, client.name, class_name, file_name, code, metrics, eval_time)
                    supabase.table(MODEL_TABLE).insert(payload).execute()
                    inserted += 1
                    logger.info("saved model | prompt_id=%s llm=%s class=%s", prompt_id, client.name, class_name)
                except Exception as exc:
                    logger.error(
                        "generation failed | prompt_id=%s llm=%s error=%s",
                        prompt_id,
                        client.name,
                        exc,
                    )
                    continue
    finally:
        close_e2b_sandbox(sandbox)

    if inserted:
        recompute_min_max_scores_for_table(supabase, MODEL_TABLE)
        logger.info("recomputed min-max scores | table=%s", MODEL_TABLE)


if __name__ == "__main__":
    main()
