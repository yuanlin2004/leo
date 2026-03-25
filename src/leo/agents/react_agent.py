import datetime
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.logging_utils import CONCISE_LEVEL, TRACE_LEVEL
from ..core.llm import LeoLLMClient, LeoLLMException
from ..knowledge import KnowledgeBase
from ..tools.registry import ToolsRegistry
from .agent import Agent
from .session import AgentSession

LOGGER = logging.getLogger("leo.agents.react_agent")

# Capture a reference to the real datetime.now before any external library
# (e.g. freezegun used by AppWorld) can patch datetime.datetime.
# NOTE: time.perf_counter() cannot be bypassed the same way — freezegun patches
# it such that even a module-level captured reference returns frozen time.
# Instead, use _real_datetime_now().timestamp() for elapsed-time measurements;
# datetime.datetime.now is patched at the class level (datetime.datetime →
# FakeDatetime), so a captured reference to the original method still resolves
# through the original (unfrozen) implementation.
_real_datetime_now = datetime.datetime.now


def _real_now_secs() -> float:
    """Return real wall-clock seconds (float), bypassing freezegun."""
    return _real_datetime_now().timestamp()


@dataclass(frozen=True)
class ContextConfig:
    """Controls how message history is compacted before each LLM call."""

    dedup: bool = True
    """Drop older (assistant tool_call + tool result) pairs when a later identical
    call (same tool name and arguments) returned the same content."""

    drop_errors: bool = True
    """Drop older (assistant tool_call + tool result) pairs whose result looks like
    an error, when a later identical call succeeded."""

    truncate_chars: int = 0
    """Truncate old tool results to this many characters. 0 = disabled."""

REACT_AGENT_SYSTEM_PROMPT_BASE = """
You are a ReAct-style assistant.
You operate in iterative steps:
1) Think briefly about what is needed.
2) If external facts are needed, call one or more tools.
3) Read the observation and decide the next step.
4) When done, call the `final_answer` tool with the complete user-facing answer in its
   `answer` field. Do not put any draft, recap, status line, or other text outside that tool call.

Rules:
- Prefer tool use for uncertain or time-sensitive facts.
- Do not call the same tool with the same arguments repeatedly unless new evidence justifies it.
- Keep intermediate reasoning short and practical.
- Final answer must be clear and user-facing. For writing tasks, include the full deliverable in `final_answer.answer`.
- If a skill may help, call list_available_skills first and activate_skill before using any tool or bundled resource from that skill.
- If an activated skill mentions companion guides, scripts, or reference files, load them with get_skill_resource instead of guessing.
- If a skill workflow depends on binaries, MCP servers, auth, or environment variables, inspect get_skill_requirements before proceeding.
- If an activated skill exposes runnable commands, inspect them with list_skill_commands and execute them with run_skill_command.
- Return every assistant turn as a single JSON object with the fields `thought`, `content`, `code`, and `tool_calls`.
- Always include `thought`. Keep it short, practical, and safe to expose in logs.
- Use `content` for a short status/update, `code` for any relevant code snippet, and `tool_calls` for every tool invocation.
- When you are done, emit exactly one `tool_calls` entry for `final_answer` with the complete user-facing answer in `arguments.answer`.
- If a field is not needed, use `null` for `content` or `code`, and use `[]` for `tool_calls`.
- Example:
  {"thought":"Need to inspect the API.","content":"Listing available endpoints.","code":null,"tool_calls":[{"name":"list_appworld_apis","arguments":{"app_name":"spotify"}}]}
- Do not output markdown fences or any text outside that JSON object.
"""


class ReActStructuredToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReActStructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1)
    content: str | None = None
    code: str | None = None
    tool_calls: list[ReActStructuredToolCall] = Field(default_factory=list)


@dataclass
class _StructuredFunctionCall:
    name: str
    arguments: str


@dataclass
class _StructuredToolCall:
    id: str
    function: _StructuredFunctionCall
    type: str = "function"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class ReActAgent(Agent):
    """
    Agent that follows a ReAct loop (reason -> act with tools -> observe -> respond).
    """

    _MAX_REPEAT_ACTIONS = 3
    _MAX_STRUCTURED_RESPONSE_ATTEMPTS = 5
    _MAX_LOG_PREVIEW_CHARS = 200
    _FINAL_ANSWER_TOOL_NAME = "final_answer"
    _STRUCTURED_RESPONSE_FORMAT_NAME = "react_agent_turn"
    _STRUCTURED_RESPONSE_FALLBACK_THOUGHT = "Continuing with the task."
    _FINAL_ANSWER_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": _FINAL_ANSWER_TOOL_NAME,
            "description": "Return the complete final answer to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": ["string", "number", "null"],
                        "description": (
                            "The complete, user-facing final answer. "
                            "For drafting tasks, include the full draft body here. "
                            "Use null for state-mutation tasks that do not require a textual answer."
                        ),
                    }
                },
                "required": ["answer"],
            },
        },
    }

    def __init__(
        self,
        name: str,
        llm: LeoLLMClient,
        tools_registry: ToolsRegistry | None = None,
        extra_system_prompt: str | None = None,
        context_config: ContextConfig | None = None,
        knowledge: KnowledgeBase | None = None,
        knowledge_top_k: int = 15,
    ):
        self.tools_registry = tools_registry or ToolsRegistry()
        self._context_config = context_config or ContextConfig()
        self._knowledge = knowledge
        self._knowledge_top_k = knowledge_top_k

        system_prompt = REACT_AGENT_SYSTEM_PROMPT_BASE
        if extra_system_prompt:
            system_prompt += extra_system_prompt

        super().__init__(name, llm, system_prompt)

    def __str__(self) -> str:
        return f"ReActAgent(name={self.name})"

    @staticmethod
    def _extract_final_answer(content: str) -> str | None:
        text = (content or "").strip()
        if not text:
            return None

        marker = "final answer:"
        lower = text.lower()
        if marker in lower:
            idx = lower.rfind(marker)
            return text[idx + len(marker) :].strip() or None
        return None

    @classmethod
    def _build_tool_schemas(cls, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [*tool_schemas, cls._FINAL_ANSWER_TOOL_SCHEMA]

    @classmethod
    def _extract_final_answer_from_tool_call(cls, tool_call: Any) -> str | None:
        if tool_call.function.name != cls._FINAL_ANSWER_TOOL_NAME:
            return None
        parsed_args = cls._parse_tool_args(tool_call.function.arguments)
        answer = parsed_args.get("answer")
        if answer is None:
            return None
        if isinstance(answer, (int, float)):
            text = str(answer)
        elif isinstance(answer, str):
            text = answer.strip()
        else:
            raise ValueError("final_answer.answer must be a string, number, or null.")
        if not text:
            return None
        return text

    @staticmethod
    def _parse_tool_args(raw_args: str | None) -> dict[str, Any]:
        if not raw_args:
            return {}
        parsed = json.loads(raw_args)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed

    @staticmethod
    def _build_action_key(tool_name: str, tool_args: dict[str, Any]) -> str:
        try:
            canonical_args = json.dumps(
                tool_args,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except TypeError:
            canonical_args = repr(tool_args)
        return f"{tool_name}:{canonical_args}"

    def _format_tool_result(self, result: Any) -> str:
        return self._summarize_tool_result(result)

    @staticmethod
    def _summarize_tool_result(result: Any) -> str:
        if isinstance(result, dict):
            # Structured execution output produced by the AppWorld adapter:
            # re-join the parts cleanly so the LLM sees combined output with
            # no internal sentinel noise.
            if "__stdout__" in result or "__return_value__" in result:
                parts = [result.get("__stdout__", ""), result.get("__return_value__", "")]
                return "\n".join(p for p in parts if p)
            nested_result = result.get("result")
            code = result.get("code")
            if nested_result is not None and isinstance(code, str):
                return (
                    nested_result
                    if isinstance(nested_result, str)
                    else json.dumps(nested_result)
                )
        return result if isinstance(result, str) else json.dumps(result)

    @staticmethod
    def _preview_text(text: str, max_chars: int = _MAX_LOG_PREVIEW_CHARS) -> str:
        normalized = " ".join((text or "").split())
        return normalized
        ## do not cut off the preview for now, as it may contain important info about tool results or final answers. if needed, we can re-enable truncation later with the above code.
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars]}..."

    @staticmethod
    def _summarize_args(tool_args: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key, value in tool_args.items():
            if isinstance(value, str):
                # Never truncate code — it needs to be readable in logs.
                summary[key] = value if key == "code" or len(value) <= 80 else f"{value[:77]}..."
            elif isinstance(value, (int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = f"<list len={len(value)}>"
            elif isinstance(value, dict):
                summary[key] = f"<dict keys={len(value)}>"
            else:
                summary[key] = f"<{type(value).__name__}>"
        return summary

    @staticmethod
    def _summarize_tool_names(tool_calls: list[Any]) -> str:
        names = [tool_call.function.name for tool_call in tool_calls]
        if not names:
            return "-"
        return ", ".join(names)

    @staticmethod
    def _extract_auto_final_answer(result: Any) -> tuple[bool, str | None]:
        if not isinstance(result, dict):
            return False, None
        if "_auto_final_answer" not in result:
            return False, None
        answer = result.get("_auto_final_answer")
        if answer is None:
            return True, None
        if not isinstance(answer, str):
            raise ValueError("_auto_final_answer must be a string or null.")
        text = answer.strip()
        return True, (text or None)

    def _build_knowledge_query(self, conversation: list[dict[str, Any]]) -> str:
        """Build a retrieval query from the task description and last turn context."""
        parts: list[str] = []
        # First user message — the task description.
        for msg in conversation:
            if msg.get("role") == "user":
                parts.append(msg.get("content") or "")
                break
        # Last assistant message — thought and tool call names/args as context.
        for msg in reversed(conversation):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content") or ""
            try:
                parsed = json.loads(content)
                thought = parsed.get("thought") or ""
                if thought:
                    parts.append(thought)
            except (json.JSONDecodeError, AttributeError):
                pass
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append(fn.get("name", ""))
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    parts.extend(str(v) for v in args.values() if isinstance(v, str))
                except (json.JSONDecodeError, TypeError):
                    pass
            break
        return " ".join(parts)

    def _build_knowledge_message(self, conversation: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not self._knowledge:
            return None
        query = self._build_knowledge_query(conversation)
        matches = self._knowledge.retrieve(query, top_k=self._knowledge_top_k)
        if not matches:
            return None
        examples = "\n\n".join(f'"{action}":\n{code}' for action, code in matches)
        return {"role": "system", "content": f"Relevant code examples:\n\n{examples}"}

    def _build_model_messages(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        runtime_messages = self.tools_registry.get_runtime_context_messages()
        tool_schema_message = {
            "role": "system",
            "content": self._build_tool_schema_prompt(),
        }
        knowledge_message = self._build_knowledge_message(conversation)
        extra = [knowledge_message] if knowledge_message else []
        if not runtime_messages:
            if not conversation:
                return [{"role": "system", "content": self.system_prompt}, tool_schema_message, *extra]
            return [
                conversation[0],
                tool_schema_message,
                *extra,
                *self._compact_history(conversation[1:]),
            ]
        system_message = (
            conversation[0]
            if conversation
            else {"role": "system", "content": self.system_prompt}
        )
        remainder = conversation[1:] if conversation else []
        return [
            system_message,
            tool_schema_message,
            *runtime_messages,
            *extra,
            *self._compact_history(remainder),
        ]

    @staticmethod
    def _is_error_result(content: str) -> bool:
        s = content.strip()
        if s.startswith(("[Loop detected]", "[Error", "Error:")):
            return True
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and "error" in parsed:
                    return True
            except json.JSONDecodeError:
                pass
        return False

    def _compact_history(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compact message history before each LLM call.

        Depending on ContextConfig:
        - dedup: drop older (assistant tool_call + tool result) pairs when a
          later identical call returned the same content.
        - drop_errors: drop older error results superseded by a later successful
          call with the same tool and arguments.
        - truncate_chars > 0: truncate remaining old tool results to that limit.
        """
        cfg = self._context_config
        if not cfg.dedup and not cfg.drop_errors and cfg.truncate_chars <= 0:
            return messages

        # Build call_id → (tool_name, args_json) from assistant messages.
        call_info: dict[str, tuple[str, str]] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = tc.get("id")
                    if cid:
                        fn = tc.get("function") or {}
                        call_info[cid] = (fn.get("name", ""), fn.get("arguments", ""))

        # Build call_id → result content from tool messages.
        call_results: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    call_results[cid] = msg.get("content") or ""

        # Group call_ids by (tool_name, args_json), preserving order of appearance.
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid and cid in call_info:
                    groups[call_info[cid]].append(cid)

        # Determine which call_ids to drop entirely.
        drop_ids: set[str] = set()

        if cfg.dedup:
            for cids in groups.values():
                for i, cid in enumerate(cids[:-1]):
                    later_contents = [call_results.get(c, "") for c in cids[i + 1 :]]
                    if call_results.get(cid, "") in later_contents:
                        drop_ids.add(cid)

        if cfg.drop_errors:
            for cids in groups.values():
                for i, cid in enumerate(cids[:-1]):
                    if cid in drop_ids:
                        continue
                    if self._is_error_result(call_results.get(cid, "")):
                        later_succeeded = any(
                            not self._is_error_result(call_results.get(c, ""))
                            for c in cids[i + 1 :]
                        )
                        if later_succeeded:
                            drop_ids.add(cid)

        # Find the index of the last surviving tool message for truncation exemption.
        last_tool_idx = -1
        if cfg.truncate_chars > 0:
            for i, msg in enumerate(messages):
                if msg.get("role") == "tool" and msg.get("tool_call_id") not in drop_ids:
                    last_tool_idx = i

        # Rebuild message list.
        result: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "tool":
                cid = msg.get("tool_call_id")
                if cid in drop_ids:
                    continue
                if cfg.truncate_chars > 0 and i != last_tool_idx:
                    content = msg.get("content") or ""
                    max_chars = cfg.truncate_chars
                    if isinstance(content, str) and len(content) > max_chars:
                        half = max_chars // 2
                        content = (
                            content[:half]
                            + f"\n...[{len(content) - max_chars} chars truncated from history]...\n"
                            + content[-half:]
                        )
                        msg = {**msg, "content": content}
            elif role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    kept = [tc for tc in tool_calls if tc.get("id") not in drop_ids]
                    if len(kept) < len(tool_calls):
                        if not kept and not (msg.get("content") or "").strip():
                            continue
                        msg = {**msg, "tool_calls": kept if kept else None}
                        if not kept:
                            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            result.append(msg)
        return result

    @classmethod
    def _build_structured_response_format(cls) -> dict[str, Any]:
        schema = ReActStructuredResponse.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "json_schema",
            "json_schema": {
                "name": cls._STRUCTURED_RESPONSE_FORMAT_NAME,
                "strict": True,
                "schema": schema,
            },
        }

    def _build_tool_schema_prompt(self) -> str:
        tool_schemas = self._build_tool_schemas(self.tools_registry.get_tool_schemas())
        # Strip the {"type": "function", "function": {...}} wrapper so the model
        # sees plain tool definitions instead of OpenAI native-function-calling
        # format, which can cause fine-tuned models to generate native tool_calls
        # that conflict with our structured JSON response format.
        simplified = [
            s["function"] if isinstance(s, dict) and "function" in s else s
            for s in tool_schemas
        ]
        return (
            "Tool schemas are provided below. Embed tool invocations ONLY inside "
            "the `tool_calls` field of your JSON response. "
            "Do NOT use native function/tool calling.\n"
            "Each `tool_calls` item must contain `name` and an object `arguments`.\n"
            f"{json.dumps(simplified, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @classmethod
    def _extract_json_payload(cls, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _coerce_tool_call_arguments(raw_arguments: Any) -> dict[str, Any]:
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            stripped = raw_arguments.strip()
            if not stripped:
                return {}
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("Tool arguments must decode to a JSON object.")

    @classmethod
    def _coerce_tool_call_item(cls, raw_tool_call: Any) -> dict[str, Any]:
        if not isinstance(raw_tool_call, dict):
            raise ValueError("Tool call entries must be objects.")
        if isinstance(raw_tool_call.get("name"), str):
            if "arguments" not in raw_tool_call:
                # Flat argument format: model emitted args directly in the tool call
                # object instead of nesting them under "arguments".
                # e.g. {"name": "execute_appworld_code", "code": "..."}
                flat_args = {k: v for k, v in raw_tool_call.items() if k != "name"}
                arguments = flat_args if flat_args else {}
            else:
                arguments = cls._coerce_tool_call_arguments(raw_tool_call.get("arguments"))
            return {
                "name": raw_tool_call["name"],
                "arguments": arguments,
            }
        function_payload = raw_tool_call.get("function")
        if not isinstance(function_payload, dict):
            raise ValueError("Tool call entries must contain `name` or `function.name`.")
        name = function_payload.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call function name must be a non-empty string.")
        return {
            "name": name,
            "arguments": cls._coerce_tool_call_arguments(
                function_payload.get("arguments")
            ),
        }

    @classmethod
    def _normalize_structured_payload(
        cls,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_tool_calls = payload.get("tool_calls", [])
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            raise ValueError("`tool_calls` must be a list.")
        thought = payload.get("thought")
        if not isinstance(thought, str) or not thought.strip():
            thought = cls._STRUCTURED_RESPONSE_FALLBACK_THOUGHT
        content = payload.get("content")
        code = payload.get("code")
        return {
            "thought": thought.strip(),
            "content": None if content is None else str(content),
            "code": None if code is None else str(code),
            "tool_calls": [
                cls._coerce_tool_call_item(item) for item in raw_tool_calls
            ],
        }

    @classmethod
    def _build_structured_response_from_native_tool_calls(
        cls,
        assistant_message: Any,
        raw_content: str | None,
    ) -> ReActStructuredResponse | None:
        native_tool_calls = getattr(assistant_message, "tool_calls", None) or []
        if not native_tool_calls:
            return None
        payload = {
            "thought": cls._STRUCTURED_RESPONSE_FALLBACK_THOUGHT,
            "content": cls._strip_code_fences(raw_content or "") or None,
            "code": None,
            "tool_calls": [
                tool_call.model_dump() if hasattr(tool_call, "model_dump") else tool_call
                for tool_call in native_tool_calls
            ],
        }
        try:
            return ReActStructuredResponse.model_validate(
                cls._normalize_structured_payload(payload)
            )
        except ValidationError as exc:
            raise ValueError(
                f"Native tool call response validation failed: {exc}"
            ) from exc

    @classmethod
    def _parse_structured_response(
        cls,
        raw_content: str | None,
        assistant_message: Any | None = None,
    ) -> ReActStructuredResponse:
        text = cls._strip_code_fences(raw_content or "")
        payload = cls._extract_json_payload(text)
        if payload is not None:
            try:
                return ReActStructuredResponse.model_validate(
                    cls._normalize_structured_payload(payload)
                )
            except ValidationError as exc:
                raise ValueError(f"Structured response validation failed: {exc}") from exc
        if assistant_message is not None:
            native_tool_response = cls._build_structured_response_from_native_tool_calls(
                assistant_message,
                raw_content,
            )
            if native_tool_response is not None:
                return native_tool_response
        if not text:
            raise ValueError("Structured response was empty.")
        raise ValueError("Structured response was not valid JSON: no JSON object found.")

    @classmethod
    def _render_structured_response(cls, response: ReActStructuredResponse) -> str:
        return json.dumps(response.model_dump(exclude_none=False), indent=2, sort_keys=True)

    @staticmethod
    def _extract_raw_from_llm_error(error_message: str) -> str:
        """Extract raw model output from an Ollama 'error parsing tool call' 500 message.

        Ollama returns errors of the form:
            error parsing tool call: raw='<model output>', err=<parse error>

        Single quotes inside the raw content are backslash-escaped by Ollama.
        """
        m = re.search(r"raw='((?:[^'\\]|\\.)*)'", error_message, re.DOTALL)
        if not m:
            return ""
        return m.group(1).replace("\\'", "'")

    @staticmethod
    def _build_structured_retry_message(error: Exception, is_final: bool = False) -> str:
        if is_final:
            return (
                "Your previous response did not match the required JSON schema. "
                "You must act now: either call a tool via `tool_calls` or emit `final_answer` with your best answer. "
                "Do NOT output a response with all null/empty action fields. "
                f"Validation error: {error}"
            )
        return (
            "Your previous response did not match the required JSON schema. "
            "Return exactly one JSON object with keys `thought`, `content`, `code`, and `tool_calls`. "
            "Use null for unused `content` or `code`, and [] for no tool calls. "
            "Do not return an all-null placeholder object; make one concrete tool call or emit final_answer. "
            f"Validation error: {error}"
        )

    @staticmethod
    def _is_blank_text(value: str | None) -> bool:
        return value is None or not value.strip()

    @classmethod
    def _is_effectively_empty_structured_response(
        cls,
        response: ReActStructuredResponse,
    ) -> bool:
        return (
            not response.tool_calls
            and cls._is_blank_text(response.content)
            and cls._is_blank_text(response.code)
        )

    _STRUCTURED_SCHEMA_KEYS = frozenset({"thought", "content", "code", "tool_calls"})

    def _try_recover_bare_tool_arg_response(
        self,
        raw_content: str,
    ) -> ReActStructuredResponse | None:
        """Attempt to recover a valid structured response from a bare tool-argument dict.

        Some OSS models (e.g. Qwen-20B via Ollama) output the raw arguments of a
        tool call as a flat JSON object — e.g. ``{"app_name": "venmo", "max_results": 50}``
        — instead of wrapping them in the required ``{thought, content, code, tool_calls}``
        structure.  This method detects that pattern and reconstructs a proper
        ``ReActStructuredResponse`` by matching the argument keys against every
        registered tool schema.

        Returns the recovered response, or ``None`` if no unambiguous match is found.
        """
        text = self._strip_code_fences(raw_content or "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or not payload:
            return None
        # If the dict contains any structured-schema key it is NOT a bare tool arg.
        if payload.keys() & self._STRUCTURED_SCHEMA_KEYS:
            return None

        arg_keys = set(payload.keys())

        # Build a map of tool_name -> set of all parameter names (required + optional).
        all_schemas = list(self.tools_registry.get_tool_schemas())
        all_schemas.append(self._FINAL_ANSWER_TOOL_SCHEMA)
        matches: list[str] = []
        for schema in all_schemas:
            fn = schema.get("function", {})
            tool_name = fn.get("name", "")
            params = fn.get("parameters", {})
            properties = params.get("properties", {})
            required = set(params.get("required", []))
            allowed_keys = set(properties.keys())
            # The arg keys must all be valid params, and all required params must be present.
            if arg_keys <= allowed_keys and required <= arg_keys:
                matches.append(tool_name)

        if len(matches) != 1:
            # Ambiguous or no match — cannot recover safely.
            LOGGER.debug(
                "Bare tool-arg recovery: %d candidate tool(s) for keys %s; skipping.",
                len(matches),
                sorted(arg_keys),
            )
            return None

        tool_name = matches[0]
        LOGGER.warning(
            "Bare tool-arg recovery: inferred tool=%r from bare argument keys %s.",
            tool_name,
            sorted(arg_keys),
        )
        try:
            return ReActStructuredResponse.model_validate(
                self._normalize_structured_payload(
                    {
                        "thought": f"Calling {tool_name}.",
                        "content": None,
                        "code": None,
                        "tool_calls": [{"name": tool_name, "arguments": payload}],
                    }
                )
            )
        except (ValidationError, ValueError) as exc:
            LOGGER.debug("Bare tool-arg recovery: constructed response failed validation: %s", exc)
            return None

    @classmethod
    def _build_structured_tool_calls(
        cls,
        response: ReActStructuredResponse,
        turn_number: int,
    ) -> list[_StructuredToolCall]:
        tool_calls: list[_StructuredToolCall] = []
        for index, tool_call in enumerate(response.tool_calls, start=1):
            tool_calls.append(
                _StructuredToolCall(
                    id=f"structured-call-{turn_number}-{index}",
                    function=_StructuredFunctionCall(
                        name=tool_call.name,
                        arguments=json.dumps(
                            tool_call.arguments,
                            sort_keys=True,
                            ensure_ascii=True,
                        ),
                    ),
                )
            )
        return tool_calls

    def _complete_structured_turn(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> tuple[Any, str]:
        # Tools are described in the system prompt; do not pass them as native
        # API tools — that activates provider-side tool-call parsing which
        # conflicts with our structured JSON output format.
        request_kwargs: dict[str, Any] = {"response_format": self._build_structured_response_format()}
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        try:
            assistant_message = self.llm.complete(
                messages=messages,
                tools=None,
                **request_kwargs,
            )
            return assistant_message, assistant_message.content or ""
        except LeoLLMException as exc:
            error_str = str(exc)
            if "error parsing tool call" in error_str:
                # Ollama failed to parse the model output as a native tool call
                # and embeds the raw model response in the error message.  Only
                # use it when it actually looks like a Leo structured response
                # (contains "thought"); otherwise it is Ollama's partial
                # argument extraction, which is not usable as a response.
                raw = self._extract_raw_from_llm_error(error_str)
                if raw and '"thought"' in raw:
                    LOGGER.warning(
                        "Recovering model output from Ollama tool-call interception. error=%s",
                        exc,
                    )
                    return None, raw
                LOGGER.warning(
                    "Ollama tool-call parse error; raw fragment is not a full response "
                    "(no 'thought' key) — retrying without response_format. error=%s",
                    exc,
                )
            else:
                LOGGER.warning(
                    "Structured response_format request failed; retrying without response_format. error=%s",
                    exc,
                )
        assistant_message = self.llm.complete(messages=messages, tools=None)
        return assistant_message, assistant_message.content or ""

    @staticmethod
    def _render_concise_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            if not value:
                return "-"
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                try:
                    parsed = json.loads(stripped)
                    return json.dumps(parsed, indent=2, sort_keys=True)
                except json.JSONDecodeError:
                    pass
            return value
        try:
            return json.dumps(value, indent=2, sort_keys=True)
        except TypeError:
            return repr(value)

    @classmethod
    def _render_concise_tool_args(cls, value: Any) -> str:
        if isinstance(value, dict):
            rendered_parts: list[str] = []
            non_code_items: dict[str, Any] = {}
            for key, item in value.items():
                if key == "code" and isinstance(item, str):
                    rendered_parts.append(f"code:\n{cls._indent_code_block(item)}")
                else:
                    non_code_items[key] = item
            if non_code_items:
                rendered_parts.insert(0, cls._render_concise_value(non_code_items))
            if rendered_parts:
                return "\n".join(rendered_parts)
        return cls._render_concise_value(value)

    @staticmethod
    def _indent_code_block(code: str) -> str:
        lines = code.splitlines() or [code]
        return "\n".join(f"        {line}" for line in lines)

    @classmethod
    def _extract_role_prompts(
        cls,
        messages: list[dict[str, Any]],
        role_name: str,
    ) -> str:
        contents: list[str] = []
        for item in messages:
            if str(item.get("role") or "").strip() != role_name:
                continue
            contents.append(cls._render_concise_value(item.get("content")))
        if not contents:
            return "-"
        return "\n\n".join(contents)

    @classmethod
    def _extract_current_user_prompt(cls, messages: list[dict[str, Any]]) -> str:
        for item in reversed(messages):
            if str(item.get("role") or "").strip() != "user":
                continue
            return cls._render_concise_value(item.get("content"))
        return "-"

    @staticmethod
    def _count_role_messages(messages: list[dict[str, Any]], role_name: str) -> int:
        return sum(
            1
            for item in messages
            if str(item.get("role") or "").strip() == role_name
        )

    @classmethod
    def _extract_new_role_prompts(
        cls,
        messages: list[dict[str, Any]],
        role_name: str,
        already_seen_count: int,
    ) -> tuple[str, int]:
        role_messages = [
            cls._render_concise_value(item.get("content"))
            for item in messages
            if str(item.get("role") or "").strip() == role_name
        ]
        total_count = len(role_messages)
        if total_count <= already_seen_count:
            return "-", total_count
        new_messages = role_messages[already_seen_count:]
        if not new_messages:
            return "-", total_count
        return "\n\n".join(new_messages), total_count

    _TURN_WIDTH = 72

    @classmethod
    def _fmt_banner(cls, label: str, char: str = "═") -> str:
        """Return a full-width banner:  ══════ LABEL ══════"""
        inner = f" {label} "
        pad = max(0, cls._TURN_WIDTH - len(inner))
        left = pad // 2
        right = pad - left
        return char * left + inner + char * right

    def _log_concise_initial_prompts(self, messages: list[dict[str, Any]]) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        LOGGER.log(CONCISE_LEVEL, self._fmt_banner("SYSTEM PROMPT", "─"))
        LOGGER.log(
            CONCISE_LEVEL,
            "%s",
            self._extract_role_prompts(messages, "system"),
        )
        assistant_prompt = self._extract_role_prompts(messages, "assistant")
        if assistant_prompt != "-":
            LOGGER.log(CONCISE_LEVEL, self._fmt_banner("INITIAL ASSISTANT", "─"))
            LOGGER.log(CONCISE_LEVEL, "%s", assistant_prompt)
        LOGGER.log(CONCISE_LEVEL, self._fmt_banner("INITIAL USER PROMPT", "─"))
        LOGGER.log(
            CONCISE_LEVEL,
            "%s",
            self._extract_role_prompts(messages, "user"),
        )

    def _log_concise_turn_start(
        self,
        turn_number: int,
        messages: list[dict[str, Any]],
        seen_user_prompt_count: int,
    ) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        ts = _real_datetime_now().strftime("%H:%M:%S")
        LOGGER.log(CONCISE_LEVEL, "")
        LOGGER.log(CONCISE_LEVEL, self._fmt_banner(f"TURN {turn_number}  {ts}"))
        user_prompt, _ = self._extract_new_role_prompts(
            messages,
            "user",
            seen_user_prompt_count,
        )
        if user_prompt != "-":
            LOGGER.log(CONCISE_LEVEL, "[USER]\n%s", user_prompt)

    def _log_concise_llm_response(self, content: str | None) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        # Parse the structured response JSON and format each field clearly.
        try:
            parsed = json.loads(content or "")
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if isinstance(parsed, dict):
            thought = (parsed.get("thought") or "").strip()
            text_content = (parsed.get("content") or "").strip()
            code = (parsed.get("code") or "").strip()
            tool_calls: list[Any] = parsed.get("tool_calls") or []

            if thought:
                LOGGER.log(CONCISE_LEVEL, "[THOUGHT] %s", thought)
            if text_content:
                LOGGER.log(CONCISE_LEVEL, "[CONTENT] %s", text_content)
            if code:
                LOGGER.log(CONCISE_LEVEL, "[CODE]\n%s", self._indent_code_block(code))
            if tool_calls:
                names = ", ".join(
                    tc.get("name", "?") if isinstance(tc, dict) else "?"
                    for tc in tool_calls
                )
                LOGGER.log(CONCISE_LEVEL, "[CALLS] %s", names)
        else:
            LOGGER.log(CONCISE_LEVEL, "[LLM]\n%s", self._render_concise_value(content))

    def _log_concise_tool_call(
        self, tool_name: str, tool_args: dict[str, Any], attempt: int = 1
    ) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        LOGGER.log(CONCISE_LEVEL, "[CALL] %s attempt=%d", tool_name, attempt)
        LOGGER.log(CONCISE_LEVEL, "[ARGS]\n%s", self._render_concise_tool_args(tool_args))

    def _log_concise_tool_result(self, result: Any) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        if isinstance(result, dict) and (
            "__stdout__" in result or "__return_value__" in result
        ):
            stdout = result.get("__stdout__", "")
            return_val = result.get("__return_value__", "")
            if stdout:
                LOGGER.log(CONCISE_LEVEL, "[STDOUT]\n%s", self._render_concise_value(stdout))
            if return_val:
                LOGGER.log(CONCISE_LEVEL, "[RETURN]\n%s", self._render_concise_value(return_val))
            elif not stdout:
                LOGGER.log(CONCISE_LEVEL, "[OUTPUT] (empty)")
        else:
            LOGGER.log(CONCISE_LEVEL, "[OUTPUT]\n%s", self._render_concise_value(result))

    def _run_loop(
        self,
        conversation: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        user_input = ""
        if conversation:
            user_input = str(conversation[-1].get("content") or "")
        self.tools_registry.activate_relevant_skills_for_input(user_input)
        action_counts: dict[str, int] = {}
        seen_user_prompt_count = 0
        LOGGER.info(
            "Run start: agent=%s max_iterations=%d user_input=%s",
            self.name,
            max_iterations,
            self._preview_text(user_input),
        )

        for iteration in range(max_iterations):
            turn_number = iteration + 1
            model_messages = self._build_model_messages(conversation)
            if turn_number == 1:
                self._log_concise_initial_prompts(model_messages)
                seen_user_prompt_count = self._count_role_messages(
                    model_messages,
                    "user",
                )
            self._log_concise_turn_start(
                turn_number,
                model_messages,
                seen_user_prompt_count,
            )
            _, seen_user_prompt_count = self._extract_new_role_prompts(
                model_messages,
                "user",
                seen_user_prompt_count,
            )
            # At TRACE level log only the tail of the message history — the
            # new messages since the previous turn (not the full history which
            # Log a compact context summary (not the full history — each
            # piece is already captured by [THOUGHT]/[→]/[←] in prior turns).
            if LOGGER.isEnabledFor(TRACE_LEVEL):
                role_counts: dict[str, int] = {}
                for m in model_messages:
                    r = str(m.get("role") or "").strip()
                    role_counts[r] = role_counts.get(r, 0) + 1
                summary = ", ".join(f"{r}×{n}" for r, n in sorted(role_counts.items()))
                LOGGER.log(
                    TRACE_LEVEL,
                    "[request turn %d context] %d messages (%s)",
                    turn_number,
                    len(model_messages),
                    summary,
                )
            attempt_messages = list(model_messages)
            assistant_message = None
            raw_response_content = ""
            llm_elapsed_ms = 0.0
            structured_response: ReActStructuredResponse | None = None
            last_error: Exception | None = None
            for attempt_number in range(1, self._MAX_STRUCTURED_RESPONSE_ATTEMPTS + 1):
                if attempt_number > 1:
                    LOGGER.warning(
                        "Turn %d attempt %d/%d: retrying model after invalid structured response.",
                        turn_number,
                        attempt_number,
                        self._MAX_STRUCTURED_RESPONSE_ATTEMPTS,
                    )
                # Use slightly elevated temperature on retries to break out of
                # deterministic loops (e.g. temperature=0 always produces the same
                # empty response). Apply a minimum temperature floor on all attempts
                # for providers/models known to produce thought-only or bare-argument
                # empty responses at temperature=0.
                llm_provider = getattr(self.llm, "provider", None)
                if attempt_number > 1 or llm_provider in ("openrouter", "ollama"):
                    retry_temperature = 0.3
                else:
                    retry_temperature = None
                llm_start = _real_now_secs()
                try:
                    assistant_message, raw_response_content = self._complete_structured_turn(
                        attempt_messages, temperature=retry_temperature
                    )
                except LeoLLMException as llm_exc:
                    # Ollama (and some other providers) return HTTP 500 when the
                    # model outputs something that can't be parsed as a native tool
                    # call (e.g. raw Python, a comment, or bare JSON).  Treat this
                    # as a retryable format failure rather than a fatal crash.
                    raw_response_content = self._extract_raw_from_llm_error(str(llm_exc))
                    assistant_message = None
                    last_error = ValueError(f"LLM provider error (bad model output): {llm_exc}")
                    LOGGER.warning(
                        "Turn %d attempt %d: LLM provider error treated as retryable format failure: %s",
                        turn_number,
                        attempt_number,
                        llm_exc,
                    )
                    if attempt_number == self._MAX_STRUCTURED_RESPONSE_ATTEMPTS:
                        break
                    is_final_attempt = (
                        attempt_number == self._MAX_STRUCTURED_RESPONSE_ATTEMPTS - 1
                    )
                    attempt_messages.append(
                        {
                            "role": "assistant",
                            "content": raw_response_content,
                        }
                    )
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": self._build_structured_retry_message(
                                last_error, is_final=is_final_attempt
                            ),
                        }
                    )
                    continue
                llm_elapsed_ms = (_real_now_secs() - llm_start) * 1000
                try:
                    candidate_response = self._parse_structured_response(
                        raw_response_content,
                        assistant_message=assistant_message,
                    )
                    if self._is_effectively_empty_structured_response(candidate_response):
                        # Before giving up, try to recover a bare tool-argument dict
                        # that the model emitted without the required wrapper schema.
                        recovered = self._try_recover_bare_tool_arg_response(raw_response_content)
                        if recovered is not None:
                            candidate_response = recovered
                        else:
                            raise ValueError(
                                "Structured response was empty: content, code, and tool_calls were all empty."
                            )
                except ValueError as exc:
                    last_error = exc
                    LOGGER.warning(
                        "Turn %d attempt %d: structured response validation failed: %s",
                        turn_number,
                        attempt_number,
                        exc,
                    )
                    if attempt_number == self._MAX_STRUCTURED_RESPONSE_ATTEMPTS:
                        break
                    is_final_attempt = (
                        attempt_number == self._MAX_STRUCTURED_RESPONSE_ATTEMPTS - 1
                    )
                    attempt_messages.append(
                        {
                            "role": "assistant",
                            "content": raw_response_content,
                        }
                    )
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": self._build_structured_retry_message(
                                exc, is_final=is_final_attempt
                            ),
                        }
                    )
                    continue
                structured_response = candidate_response
                break
            if structured_response is None:
                assert last_error is not None
                if raw_response_content.strip():
                    LOGGER.warning(
                        "Turn %d: dropping invalid structured response from persistent conversation after exhausting retries.",
                        turn_number,
                    )
                LOGGER.error(
                    "Turn %d: exhausted %d structured response attempts; carrying a retry instruction into the next turn. last_error=%s",
                    turn_number,
                    self._MAX_STRUCTURED_RESPONSE_ATTEMPTS,
                    last_error,
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": self._build_structured_retry_message(last_error),
                    }
                )
                continue
            tool_calls = self._build_structured_tool_calls(structured_response, turn_number)
            assistant_content = structured_response.content or ""
            rendered_response = self._render_structured_response(structured_response)
            LOGGER.info(
                "Turn %d: model responded latency_ms=%.1f tool_calls=%d",
                turn_number,
                llm_elapsed_ms,
                len(tool_calls),
            )
            self._log_concise_llm_response(rendered_response)
            # At TRACE level log only the non-redundant parts of the raw model
            # message: the model's internal reasoning (if any) and the raw text
            # content only when it differs meaningfully from the structured
            # response already printed by _log_concise_llm_response.
            if LOGGER.isEnabledFor(TRACE_LEVEL):
                reasoning = (
                    getattr(assistant_message, "reasoning", None)
                    if assistant_message is not None
                    else None
                )
                if reasoning:
                    LOGGER.log(
                        TRACE_LEVEL,
                        "[reasoning turn %d]\n%s",
                        turn_number,
                        reasoning,
                    )
                raw_content_str = (raw_response_content or "").strip()
                rendered_str = (rendered_response or "").strip()
                if raw_content_str and raw_content_str != rendered_str:
                    LOGGER.log(
                        TRACE_LEVEL,
                        "[raw model output turn %d]\n%s",
                        turn_number,
                        raw_content_str,
                    )

            if (
                len(tool_calls) == 1
                and tool_calls[0].function.name == self._FINAL_ANSWER_TOOL_NAME
            ):
                conversation.append(
                    {
                        "role": "assistant",
                        "content": rendered_response,
                        "tool_calls": [tool_calls[0].model_dump()],
                    }
                )
                final_answer = self._extract_final_answer_from_tool_call(tool_calls[0])
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0].id,
                        "content": "" if final_answer is None else final_answer,
                    }
                )
                final_call_args = self._parse_tool_args(tool_calls[0].function.arguments)
                self._log_concise_tool_call(
                    self._FINAL_ANSWER_TOOL_NAME,
                    final_call_args,
                )
                self._log_concise_tool_result("" if final_answer is None else final_answer)
                LOGGER.info("Returning final answer after %d turns.", turn_number)
                return final_answer

            if not tool_calls:
                conversation.append({"role": "assistant", "content": rendered_response})
                final_answer = self._extract_final_answer(assistant_content)
                if final_answer:
                    LOGGER.info("Returning final answer after %d turns.", turn_number)
                    return final_answer
                if not assistant_content.strip():
                    reminder = (
                        "Your previous response was empty. Continue the task by either "
                        "calling an appropriate tool or using the final_answer tool."
                    )
                    conversation.append({"role": "user", "content": reminder})
                    LOGGER.warning(
                        "Turn %d: empty assistant response without tool calls; requesting a retry.",
                        turn_number,
                    )
                    continue
                reminder = (
                    "Your previous response did not make progress toward completion. "
                    "Continue the task by either calling an appropriate tool or using the "
                    "final_answer tool when the work is actually complete."
                )
                conversation.append({"role": "user", "content": reminder})
                LOGGER.warning(
                    "Turn %d: non-final assistant content without tool calls; requesting continuation. preview=%s",
                    turn_number,
                    self._preview_text(assistant_content or (structured_response.code or "")),
                )
                continue

            conversation.append(
                {
                    "role": "assistant",
                    "content": rendered_response,
                    "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                }
            )

            # Execute all tool calls returned in this assistant turn.
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                call_start = _real_now_secs()
                attempt = 0
                try:
                    tool_args = self._parse_tool_args(tool_call.function.arguments)
                except Exception as exc:
                    tool_args = {}
                    result = f"Tool argument parsing failed for {tool_name}: {exc}"
                    LOGGER.error(
                        "Tool argument parsing failed: tool=%s error=%s",
                        tool_name,
                        exc,
                    )
                else:
                    action_key = self._build_action_key(tool_name, tool_args)
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1
                    attempt = action_counts[action_key]
                    LOGGER.log(
                        TRACE_LEVEL,
                        "Turn %d: executing tool=%s args=%s attempt=%d",
                        turn_number,
                        tool_name,
                        json.dumps(self._summarize_args(tool_args), sort_keys=True),
                        attempt,
                    )
                    if action_counts[action_key] > self._MAX_REPEAT_ACTIONS:
                        LOGGER.warning(
                            "Skipping repeated tool action: %s args=%s",
                            tool_name,
                            json.dumps(self._summarize_args(tool_args), sort_keys=True),
                        )
                        result = (
                            f"[Loop detected] `{tool_name}` with the same arguments has already "
                            f"been called {self._MAX_REPEAT_ACTIONS} time(s). This call was SKIPPED.\n"
                            "You are stuck. Take a completely different action:\n"
                            "- Do NOT call the same tool with the same arguments again.\n"
                            "- Review what you already learned from previous tool results.\n"
                            "- If you have an API list, call describe_appworld_api for the specific API you need.\n"
                            "- If you know what API to use, write and execute code with execute_appworld_code.\n"
                            "- If you still need information, use a DIFFERENT tool or different arguments."
                        )
                    else:
                        try:
                            result = self.tools_registry.execute(tool_name, **tool_args)
                            if tool_name == "activate_skill":
                                skill_name = tool_args.get("skill_name")
                                if isinstance(skill_name, str) and skill_name:
                                    LOGGER.info("Activated skill: %s", skill_name)
                        except Exception as exc:
                            result = f"Tool execution failed for {tool_name}: {exc}"
                            LOGGER.error(
                                "Tool execution failed: tool=%s error=%s",
                                tool_name,
                                exc,
                            )
                call_elapsed_ms = (_real_now_secs() - call_start) * 1000
                formatted_result = self._format_tool_result(result)
                LOGGER.log(
                    TRACE_LEVEL,
                    "Turn %d: tool completed name=%s latency_ms=%.1f result=%s",
                    turn_number,
                    tool_name,
                    call_elapsed_ms,
                    self._preview_text(formatted_result),
                )
                self._log_concise_tool_call(tool_name, tool_args, attempt=attempt)
                self._log_concise_tool_result(result)

                auto_finalized, auto_final_answer = self._extract_auto_final_answer(result)
                if auto_finalized:
                    LOGGER.info(
                        "Returning automatic final answer after %d turns. preview=%s",
                        turn_number,
                        self._preview_text("" if auto_final_answer is None else auto_final_answer),
                    )
                    return auto_final_answer

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": formatted_result,
                    }
                )

        LOGGER.error("Max iterations reached without a final response.")
        raise LeoLLMException("Max iterations reached without a final response.")

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        self.tools_registry.reset_run_state(preserve_environment=True)
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self._run_loop(conversation, max_iterations)

    def create_session(self) -> AgentSession:
        return AgentSession(
            system_prompt=self.system_prompt,
            run_loop=self._run_loop,
            reset_callback=self.tools_registry.reset_session_state,
        )
