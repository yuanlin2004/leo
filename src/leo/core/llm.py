from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"


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
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[dict], enable_thinking: bool = True) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        return response.choices[0].message.content
