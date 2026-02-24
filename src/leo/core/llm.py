import os
from typing import Any, Callable, Literal, Optional

from .env import load_project_env

# Keep this one simple for now. Customized for one provider and one model is fine. 

SUPPORTED_PROVIDERS = Literal[
    "openrouter",
    "ollama",
]

BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}

class LeoLLMException(Exception):
    pass

class LeoLLMClient:
    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise LeoLLMException(
                "Missing dependency: openai. Install it with `pip install openai`."
            ) from exc

        self._model = model 
        self._provider = provider.lower() 
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._kwargs = kwargs

        load_project_env()

        if self._provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = BASE_URLS["openrouter"]
        elif self._provider == "ollama":
            api_key = None  # Ollma does not need an API key 
            base_url = BASE_URLS["ollama"] 
        else:
            raise LeoLLMException(f"Unsupported LLM provider: {self._provider}")

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=self._timeout)
    
    def invoke(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[..., Any] | None = None,
        max_iterations: int = 1,
        **kwargs,
    ) -> str:
        # Keep backward compatibility for callers that expect invoke() to return
        # plain text, while tool orchestration now lives in agents.
        if tool_executor is not None or max_iterations != 1:
            raise LeoLLMException(
                "invoke() no longer orchestrates tool calls. "
                "Use an agent to execute tools."
            )
        assistant_message = self.complete(messages=messages, tools=tools, **kwargs)
        return assistant_message.content or ""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Any:
        try:
            request = dict(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["temperature", "max_tokens"]
                },
            )
            if tools:
                request["tools"] = tools

            response = self._client.chat.completions.create(**request)
            return response.choices[0].message
        except Exception as e:
            if isinstance(e, LeoLLMException):
                raise
            raise LeoLLMException(f"LLM chat completion failed: {str(e)}")
