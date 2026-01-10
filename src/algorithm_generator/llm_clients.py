import os
import time
import logging
from dataclasses import dataclass
from typing import Dict

import anthropic

logger = logging.getLogger(__name__)

@dataclass
class LLMClient:
    name: str

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class ClaudeClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(name="claude")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=0,
            system="You are a world-class research engineer.",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        return message.content[0].text


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(name="openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed (pip install openai)") from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0,
        )
        return response.output_text


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(name="gemini")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package not installed (pip install google-genai)"
            ) from exc
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return getattr(response, "text", "") or ""


class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(name="groq")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("groq package not installed (pip install groq)") from exc
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt >= max_retries:
                    logger.error("groq failed after %s attempts: %s", attempt, exc)
                    raise
                wait_s = 1.5 * attempt
                logger.warning(
                    "groq request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    wait_s,
                )
                time.sleep(wait_s)


def build_llm_clients() -> Dict[str, LLMClient]:
    clients: Dict[str, LLMClient] = {}

    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if claude_key:
        claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
        clients["claude"] = ClaudeClient(api_key=claude_key, model=claude_model)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        clients["openai"] = OpenAIClient(api_key=openai_key, model=openai_model)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        clients["gemini"] = GeminiClient(api_key=gemini_key, model=gemini_model)

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        groq_model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        clients["groq"] = GroqClient(api_key=groq_key, model=groq_model)

    return clients
