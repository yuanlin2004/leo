from __future__ import annotations

import os
import random
import time
from types import SimpleNamespace
from typing import Callable

import openai
from openai import OpenAI

try:
    from langsmith.wrappers import wrap_openai
except ImportError:
    wrap_openai = None

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MAX_TOKENS = 262144

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
        self.max_tokens = int(os.environ.get("LEO_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        self.last_total_tokens = 0
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        if wrap_openai is not None and os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
            client = wrap_openai(client)
        self.client = client

    def chat(
        self,
        messages: list[dict],
        enable_thinking: bool = True,
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                return self._stream_once(kwargs, on_text, on_reasoning)
            except Exception as e:
                if attempt == RETRY_MAX_ATTEMPTS - 1 or not _is_retryable(e):
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.random()
                print(f"\n(llm retry {attempt + 1}/{RETRY_MAX_ATTEMPTS - 1} after {type(e).__name__}: sleeping {delay:.1f}s)")
                time.sleep(delay)

    def _stream_once(self, kwargs, on_text, on_reasoning):
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.usage is not None:
                self.last_total_tokens = chunk.usage.total_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                content_parts.append(text)
                if on_text is not None:
                    on_text(text)
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning is not None:
                    on_reasoning(reasoning)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] += fn.name
                    if fn.arguments:
                        slot["arguments"] += fn.arguments

        content = "".join(content_parts) or None
        reasoning_content = "".join(reasoning_parts) or None
        msg_tool_calls = None
        if tool_calls:
            msg_tool_calls = [
                SimpleNamespace(
                    id=slot["id"],
                    type="function",
                    function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
                )
                for _, slot in sorted(tool_calls.items())
            ]
        return SimpleNamespace(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=msg_tool_calls,
        )
