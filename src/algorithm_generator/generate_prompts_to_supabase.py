import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from supabase import create_client, Client

from llm_clients import build_llm_clients, LLMClient
import metaprompt


PROMPT_TABLE = "prompt_ideas"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_supabase() -> Client:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set")
    return create_client(url, key)


def _build_idea_prompt(model_name: str, num_ideas: int) -> str:
    principles = "\n".join(metaprompt.RESEARCH_PRINCIPLES)
    return f"""Principles of Machine Learning:
{principles}

Inspired by these principles, come up with {num_ideas} very creative ideas for how to vary the {model_name} classifier for better performance.

IMPORTANT: Format your response as exactly one idea per line, with each line starting with "IDEA: " followed by the idea description.
Example format:
IDEA: Use adaptive compression based on local density
IDEA: Implement multi-level abstraction layers
IDEA: Apply entropy-guided feature selection

Do not include any other text, just the {num_ideas} IDEA lines."""


def _extract_ideas(text: str) -> List[str]:
    ideas: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("IDEA: "):
            idea = line[6:].strip()
            if idea:
                ideas.append(idea)
    return ideas


def _prompt_hash(model_name: str, idea: str, llm_name: str) -> str:
    raw = f"{model_name}|{llm_name}|{idea}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _insert_prompts(
    supabase: Client,
    model_name: str,
    llm_name: str,
    ideas: Iterable[str],
    raw_response: str,
) -> None:
    payloads = []
    for idea in ideas:
        payloads.append(
            {
                "model_family": model_name,
                "prompt_text": idea,
                "source_llm": llm_name,
                "prompt_hash": _prompt_hash(model_name, idea, llm_name),
                "raw_response": raw_response,
            }
        )
    if payloads:
        supabase.table(PROMPT_TABLE).upsert(payloads, on_conflict="prompt_hash").execute()


def _resolve_clients(requested: List[str], available: dict) -> List[LLMClient]:
    clients: List[LLMClient] = []
    for name in requested:
        client = available.get(name)
        if client is None:
            raise RuntimeError(f"LLM client '{name}' not configured; set API key env vars")
        clients.append(client)
    return clients


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt ideas and save to Supabase.")
    parser.add_argument("--llms", default="gemini,claude,openai,groq", help="comma-separated LLMs")
    default_ideas = getattr(metaprompt, "NUM_IDEAS", 10)
    parser.add_argument("--num-ideas", type=int, default=default_ideas, help="ideas per model")
    args = parser.parse_args()

    supabase = _get_supabase()
    available_clients = build_llm_clients()
    llm_names = [name.strip() for name in args.llms.split(",") if name.strip()]
    clients = _resolve_clients(llm_names, available_clients)

    for model_name in metaprompt.MODELS:
        for client in clients:
            prompt = _build_idea_prompt(model_name, args.num_ideas)
            logger.info("generating ideas | model=%s llm=%s", model_name, client.name)
            raw = client.generate(prompt)
            ideas = _extract_ideas(raw)[: args.num_ideas]
            if not ideas:
                logger.warning("no ideas returned | model=%s llm=%s", model_name, client.name)
                continue
            _insert_prompts(supabase, model_name, client.name, ideas, raw)
            logger.info("saved prompts | model=%s llm=%s count=%s", model_name, client.name, len(ideas))


if __name__ == "__main__":
    main()
