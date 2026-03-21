import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.logging_utils import CONCISE_LEVEL, TRACE_LEVEL
from ..core.llm import LeoLLMClient, LeoLLMException
from ..tools.registry import ToolsRegistry
from .agent import Agent
from .session import AgentSession

LOGGER = logging.getLogger("leo.agents.react_agent")

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

    _MAX_REPEAT_ACTIONS = 2
    _MAX_STRUCTURED_RESPONSE_ATTEMPTS = 3
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
    ):
        self.tools_registry = tools_registry or ToolsRegistry()

        system_prompt = REACT_AGENT_SYSTEM_PROMPT_BASE
        system_prompt += "The following tools are available to you:"
        for tool_name, tool_desc in self.tools_registry.get_all_tools().items():
            system_prompt += f"\n- {tool_name}: {tool_desc}"
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
                summary[key] = value if len(value) <= 80 else f"{value[:77]}..."
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

    def _build_model_messages(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        runtime_messages = self.tools_registry.get_runtime_context_messages()
        tool_schema_message = {
            "role": "system",
            "content": self._build_tool_schema_prompt(),
        }
        if not runtime_messages:
            if not conversation:
                return [
                    {"role": "system", "content": self.system_prompt},
                    tool_schema_message,
                ]
            return [
                conversation[0],
                tool_schema_message,
                *conversation[1:],
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
            *remainder,
        ]

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
        return (
            "Tool schemas are provided below. Use them to populate `tool_calls` exactly. "
            "Each `tool_calls` item must contain `name` and an object `arguments`.\n"
            f"{json.dumps(tool_schemas, indent=2, sort_keys=True)}"
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
            return {
                "name": raw_tool_call["name"],
                "arguments": cls._coerce_tool_call_arguments(
                    raw_tool_call.get("arguments")
                ),
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
    def _build_structured_retry_message(error: Exception) -> str:
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
    ) -> tuple[Any, str]:
        tool_schemas = self._build_tool_schemas(self.tools_registry.get_tool_schemas())
        request_kwargs = {"response_format": self._build_structured_response_format()}
        try:
            assistant_message = self.llm.complete(
                messages=messages,
                tools=tool_schemas,
                **request_kwargs,
            )
            return assistant_message, assistant_message.content or ""
        except LeoLLMException as exc:
            LOGGER.warning(
                "Structured response_format request failed; retrying without response_format. error=%s",
                exc,
            )
        assistant_message = self.llm.complete(messages=messages, tools=tool_schemas)
        return assistant_message, assistant_message.content or ""

    @staticmethod
    def _render_concise_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value if value else "-"
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

    def _log_concise_initial_prompts(self, messages: list[dict[str, Any]]) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        LOGGER.log(
            CONCISE_LEVEL,
            "Initial System Prompt:\n%s",
            self._extract_role_prompts(messages, "system"),
        )
        LOGGER.log(
            CONCISE_LEVEL,
            "Initial Assistant Prompt:\n%s",
            self._extract_role_prompts(messages, "assistant"),
        )
        LOGGER.log(
            CONCISE_LEVEL,
            "Initial User Prompt:\n%s",
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
        if turn_number > 1:
            LOGGER.log(CONCISE_LEVEL, "===============")
            LOGGER.log(CONCISE_LEVEL, "")
        LOGGER.log(CONCISE_LEVEL, "Turn %d", turn_number)
        user_prompt, _ = self._extract_new_role_prompts(
            messages,
            "user",
            seen_user_prompt_count,
        )
        if user_prompt != "-":
            LOGGER.log(CONCISE_LEVEL, "User Prompt:\n%s", user_prompt)

    def _log_concise_llm_response(self, content: str | None) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        LOGGER.log(CONCISE_LEVEL, "LLM:\n%s", self._render_concise_value(content))

    def _log_concise_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        LOGGER.log(CONCISE_LEVEL, "Tool Call:\n%s", tool_name)
        LOGGER.log(CONCISE_LEVEL, "Arguments:\n%s", self._render_concise_tool_args(tool_args))

    def _log_concise_tool_result(self, result: Any) -> None:
        if not LOGGER.isEnabledFor(CONCISE_LEVEL):
            return
        LOGGER.log(CONCISE_LEVEL, "Result:\n%s", self._render_concise_value(result))

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
            LOGGER.info("Turn %d: calling model", turn_number)
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
            LOGGER.log(
                TRACE_LEVEL,
                "[request turn %d messages]\n%s",
                turn_number,
                json.dumps(model_messages, indent=2, default=str),
            )
            attempt_messages = list(model_messages)
            assistant_message = None
            raw_response_content = ""
            llm_elapsed_ms = 0.0
            structured_response: ReActStructuredResponse | None = None
            last_error: Exception | None = None
            for attempt_number in range(1, self._MAX_STRUCTURED_RESPONSE_ATTEMPTS + 1):
                llm_start = time.perf_counter()
                assistant_message, raw_response_content = self._complete_structured_turn(
                    attempt_messages
                )
                llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
                try:
                    candidate_response = self._parse_structured_response(
                        raw_response_content,
                        assistant_message=assistant_message,
                    )
                    if self._is_effectively_empty_structured_response(candidate_response):
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
                    attempt_messages.append(
                        {
                            "role": "assistant",
                            "content": raw_response_content,
                        }
                    )
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": self._build_structured_retry_message(exc),
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
                "Turn %d: model responded latency_ms=%.1f tool_calls=%d content=%s",
                turn_number,
                llm_elapsed_ms,
                len(tool_calls),
                self._preview_text(rendered_response),
            )
            LOGGER.debug(
                "Turn %d: llm_latency_ms=%.1f tool_calls=%d",
                turn_number,
                llm_elapsed_ms,
                len(tool_calls),
            )
            LOGGER.debug(
                "[assistant turn %d] %s",
                turn_number,
                self._preview_text(rendered_response),
            )
            self._log_concise_llm_response(rendered_response)
            if tool_calls:
                LOGGER.info(
                    "Turn %d: tool plan=%s",
                    turn_number,
                    self._summarize_tool_names(tool_calls),
                )
                LOGGER.debug(
                    "[assistant turn %d tool calls] %s",
                    turn_number,
                    ", ".join(tool_call.function.name for tool_call in tool_calls),
                )
            response_payload = (
                assistant_message.model_dump()
                if hasattr(assistant_message, "model_dump")
                else {
                    "content": raw_response_content,
                    "tool_calls": [],
                }
            )
            response_payload["structured_response"] = structured_response.model_dump(
                exclude_none=False
            )
            LOGGER.log(
                TRACE_LEVEL,
                "[assistant turn %d full response]\n%s",
                turn_number,
                json.dumps(response_payload, indent=2, default=str),
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
                LOGGER.info(
                    "Turn %d: final answer tool received preview=%s",
                    turn_number,
                    self._preview_text("" if final_answer is None else final_answer),
                )
                LOGGER.info("Returning final answer after %d turns.", turn_number)
                return final_answer

            if not tool_calls:
                conversation.append({"role": "assistant", "content": rendered_response})
                final_answer = self._extract_final_answer(assistant_content)
                if final_answer:
                    LOGGER.info(
                        "Turn %d: final answer detected from text preview=%s",
                        turn_number,
                        self._preview_text(final_answer),
                    )
                    LOGGER.info("Returning final answer after %d turns.", turn_number)
                    LOGGER.debug(
                        "Final answer preview: %s", self._preview_text(final_answer)
                    )
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
                call_start = time.perf_counter()
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
                    LOGGER.log(
                        TRACE_LEVEL,
                        "[tool input] id=%s name=%s args=%s",
                        tool_call.id,
                        tool_name,
                        json.dumps(tool_args, indent=2, default=str, sort_keys=True),
                    )
                    self._log_concise_tool_call(tool_name, tool_args)
                    action_key = self._build_action_key(tool_name, tool_args)
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1
                    LOGGER.info(
                        "Turn %d: executing tool=%s args=%s attempt=%d",
                        turn_number,
                        tool_name,
                        self._summarize_args(tool_args),
                        action_counts[action_key],
                    )
                    if action_counts[action_key] > self._MAX_REPEAT_ACTIONS:
                        LOGGER.warning(
                            "Skipping repeated tool action: %s args=%s",
                            tool_name,
                            tool_args,
                        )
                        result = (
                            "Skipped repeated tool action to avoid loops. "
                            "Use a different query/arguments or provide Final Answer."
                        )
                    else:
                        try:
                            LOGGER.debug(
                                "Executing tool '%s' with args=%s",
                                tool_name,
                                self._summarize_args(tool_args),
                            )
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
                call_elapsed_ms = (time.perf_counter() - call_start) * 1000
                formatted_result = self._format_tool_result(result)
                LOGGER.info(
                    "Turn %d: tool completed id=%s name=%s latency_ms=%.1f result=%s",
                    turn_number,
                    tool_call.id,
                    tool_name,
                    call_elapsed_ms,
                    self._preview_text(formatted_result),
                )
                LOGGER.debug(
                    "Tool completed: id=%s name=%s latency_ms=%.1f",
                    tool_call.id,
                    tool_name,
                    call_elapsed_ms,
                )
                LOGGER.log(
                    TRACE_LEVEL,
                    "[tool result] id=%s name=%s content=%s",
                    tool_call.id,
                    tool_name,
                    formatted_result,
                )
                self._log_concise_tool_result(formatted_result)

                auto_finalized, auto_final_answer = self._extract_auto_final_answer(result)
                if auto_finalized:
                    LOGGER.info(
                        "Turn %d: tool=%s requested automatic final answer preview=%s",
                        turn_number,
                        tool_name,
                        self._preview_text("" if auto_final_answer is None else auto_final_answer),
                    )
                    LOGGER.info("Returning automatic final answer after %d turns.", turn_number)
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
