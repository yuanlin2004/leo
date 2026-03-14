import json
import logging
import time
from typing import Any

from ..core.logging_utils import TRACE_LEVEL
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
"""


class ReActAgent(Agent):
    """
    Agent that follows a ReAct loop (reason -> act with tools -> observe -> respond).
    """

    _MAX_REPEAT_ACTIONS = 2
    _MAX_TOOL_OUTPUT_CHARS = 4000
    _MAX_LOG_PREVIEW_CHARS = 200
    _FINAL_ANSWER_TOOL_NAME = "final_answer"
    _FINAL_ANSWER_TOOL_SCHEMA = {
        "type": "function",
        "function": {
            "name": _FINAL_ANSWER_TOOL_NAME,
            "description": "Return the complete final answer to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": ["string", "null"],
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
        if not isinstance(answer, str):
            raise ValueError("final_answer.answer must be a string.")
        text = answer.strip()
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
        tool_text = result if isinstance(result, str) else json.dumps(result)
        if len(tool_text) <= self._MAX_TOOL_OUTPUT_CHARS:
            return tool_text
        return (
            tool_text[: self._MAX_TOOL_OUTPUT_CHARS]
            + "\n...[truncated to keep context window manageable]"
        )

    @staticmethod
    def _preview_text(text: str, max_chars: int = _MAX_LOG_PREVIEW_CHARS) -> str:
        normalized = " ".join((text or "").split())
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
        if not runtime_messages:
            return conversation
        system_message = (
            conversation[0]
            if conversation
            else {"role": "system", "content": self.system_prompt}
        )
        remainder = conversation[1:] if conversation else []
        return [
            system_message,
            *runtime_messages,
            *remainder,
        ]

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
        LOGGER.info(
            "Run start: agent=%s max_iterations=%d user_input=%s",
            self.name,
            max_iterations,
            self._preview_text(user_input),
        )

        for iteration in range(max_iterations):
            turn_number = iteration + 1
            LOGGER.info("Turn %d: calling model", turn_number)
            tools = self._build_tool_schemas(self.tools_registry.get_tool_schemas())
            model_messages = self._build_model_messages(conversation)
            LOGGER.log(
                TRACE_LEVEL,
                "[request turn %d messages]\n%s",
                turn_number,
                json.dumps(model_messages, indent=2, default=str),
            )
            llm_start = time.perf_counter()
            assistant_message = self.llm.complete(messages=model_messages, tools=tools)
            llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
            tool_calls = assistant_message.tool_calls or []
            assistant_content = assistant_message.content or ""
            LOGGER.info(
                "Turn %d: model responded latency_ms=%.1f tool_calls=%d content=%s",
                turn_number,
                llm_elapsed_ms,
                len(tool_calls),
                self._preview_text(assistant_content),
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
                self._preview_text(assistant_content),
            )
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
                    "content": assistant_content,
                    "tool_calls": [
                        tool_call.model_dump()
                        if hasattr(tool_call, "model_dump")
                        else {
                            "id": getattr(tool_call, "id", None),
                            "name": getattr(
                                getattr(tool_call, "function", None), "name", None
                            ),
                            "arguments": getattr(
                                getattr(tool_call, "function", None), "arguments", None
                            ),
                        }
                        for tool_call in tool_calls
                    ],
                }
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
                        "content": assistant_content,
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
                LOGGER.info(
                    "Turn %d: final answer tool received preview=%s",
                    turn_number,
                    self._preview_text("" if final_answer is None else final_answer),
                )
                LOGGER.info("Returning final answer after %d turns.", turn_number)
                return final_answer

            if not tool_calls:
                conversation.append({"role": "assistant", "content": assistant_content})
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
                LOGGER.info(
                    "Turn %d: returning assistant content preview=%s",
                    turn_number,
                    self._preview_text(assistant_content),
                )
                LOGGER.info("Returning assistant content after %d turns.", turn_number)
                LOGGER.debug(
                    "Assistant content preview: %s",
                    self._preview_text(assistant_content),
                )
                return assistant_content

            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
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
