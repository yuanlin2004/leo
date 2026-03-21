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
    "ollama": "http://192.168.29.160:11434/v1",
}


def _normalize_base_url(value: str) -> str:
    base_url = value.strip()
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"
    return base_url


def _resolve_base_url(provider: str) -> str:
    if provider == "ollama":
        return _normalize_base_url(os.getenv("OLLAMA_BASE_URL", BASE_URLS["ollama"]))
    return BASE_URLS[provider]


def _resolve_api_key(provider: str) -> str | None:
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY")
    if provider == "ollama":
        return os.getenv("OLLAMA_API_KEY", "ollama")
    return None


def _sanitize_schema_for_ollama(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "type" and isinstance(item, list):
                allowed_types = [entry for entry in item if entry != "null"]
                sanitized[key] = allowed_types[0] if allowed_types else "string"
                continue
            if key == "anyOf" and isinstance(item, list):
                non_null_options = [
                    entry
                    for entry in item
                    if not (isinstance(entry, dict) and entry.get("type") == "null")
                ]
                if len(non_null_options) == 1:
                    replacement = _sanitize_schema_for_ollama(non_null_options[0])
                    if isinstance(replacement, dict):
                        for nested_key, nested_value in replacement.items():
                            sanitized[nested_key] = nested_value
                        continue
                sanitized[key] = [
                    _sanitize_schema_for_ollama(entry) for entry in non_null_options
                ]
                continue
            sanitized[key] = _sanitize_schema_for_ollama(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_schema_for_ollama(item) for item in value]
    return value


def _sanitize_tools_for_provider(
    provider: str,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if tools is None or provider != "ollama":
        return tools
    return [_sanitize_schema_for_ollama(tool) for tool in tools]

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
            api_key = _resolve_api_key("openrouter")
            base_url = _resolve_base_url("openrouter")
        elif self._provider == "ollama":
            api_key = _resolve_api_key("ollama")
            base_url = _resolve_base_url("ollama")
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
        request_kwargs = dict(kwargs)
        if self._provider == "ollama":
            request_kwargs.pop("response_format", None)
        request = dict(
            model=self._model,
            messages=messages,
            temperature=request_kwargs.get("temperature", self._temperature),
            max_tokens=request_kwargs.get("max_tokens", self._max_tokens),
            **{
                k: v
                for k, v in request_kwargs.items()
                if k not in ["temperature", "max_tokens"]
            },
        )
        sanitized_tools = _sanitize_tools_for_provider(self._provider, tools)
        if sanitized_tools:
            request["tools"] = sanitized_tools

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
