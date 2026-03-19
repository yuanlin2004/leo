import os
import time
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
        timeout: Optional[float] = None,
        max_retries: int = 1,
        **kwargs
    ):
        try:
            import openai as openai_module
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
        self._max_retries = max(0, int(max_retries))
        self._kwargs = kwargs
        self._retryable_exception_types = tuple(
            exception_type
            for exception_type in (
                getattr(openai_module, "APITimeoutError", None),
                getattr(openai_module, "APIConnectionError", None),
                getattr(openai_module, "RateLimitError", None),
                getattr(openai_module, "InternalServerError", None),
            )
            if isinstance(exception_type, type)
        )

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

        last_error: Exception | None = None
        for attempt_number in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
                return response.choices[0].message
            except Exception as exc:
                if isinstance(exc, LeoLLMException):
                    raise
                last_error = exc
                if attempt_number >= self._max_retries or not self._is_retryable_error(exc):
                    break
                time.sleep(min(4.0, float(2**attempt_number)))

        assert last_error is not None
        raise LeoLLMException(f"LLM chat completion failed: {str(last_error)}")

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, TimeoutError):
            return True
        if self._retryable_exception_types and isinstance(
            error, self._retryable_exception_types
        ):
            return True
        message = str(error).lower()
        return any(
            token in message
            for token in (
                "timeout",
                "timed out",
                "connection reset",
                "temporarily unavailable",
                "rate limit",
                "server error",
                "overloaded",
            )
        )
