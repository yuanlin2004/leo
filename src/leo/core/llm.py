from __future__ import annotations

import os
import random
import time

import openai
from openai import OpenAI

try:
    from langsmith.wrappers import wrap_openai
except ImportError:
    wrap_openai = None

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"

RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return 500 <= getattr(exc, "status_code", 0) < 600
    return False


class LLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("LEO_LLM_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("LEO_LLM_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("LEO_LLM_API_KEY", DEFAULT_API_KEY)
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        if wrap_openai is not None and os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
            client = wrap_openai(client)
        self.client = client

    def chat(
        self,
        messages: list[dict],
        enable_thinking: bool = True,
        tools: list[dict] | None = None,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        }
        if tools:
            kwargs["tools"] = tools
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message
            except Exception as e:
                if attempt == RETRY_MAX_ATTEMPTS - 1 or not _is_retryable(e):
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.random()
                print(f"\n(llm retry {attempt + 1}/{RETRY_MAX_ATTEMPTS - 1} after {type(e).__name__}: sleeping {delay:.1f}s)")
                time.sleep(delay)
