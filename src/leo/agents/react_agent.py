import json
import logging
import time
from typing import Any

from ..core.logging_utils import TRACE_LEVEL
from ..core.llm import LeoLLMClient, LeoLLMException
from ..tools.registry import ToolsRegistry
from .agent import Agent

LOGGER = logging.getLogger("leo.agents.react_agent")

REACT_AGENT_SYSTEM_PROMPT_BASE = """
You are a ReAct-style assistant.
You operate in iterative steps:
1) Think briefly about what is needed.
2) If external facts are needed, call one or more tools.
3) Read the observation and decide the next step.
4) When done, respond with: Final Answer: <answer>

Rules:
- Prefer tool use for uncertain or time-sensitive facts.
- Do not call the same tool with the same arguments repeatedly unless new evidence justifies it.
- Keep intermediate reasoning short and practical.
- Final answer must be clear, user-facing, and concise.
- If you suspect a skill may help, call list_available_skills first to discover options, then call get_skill_details before using any skill action.
"""


class ReActAgent(Agent):
    """
    Agent that follows a ReAct loop (reason -> act with tools -> observe -> respond).
    """

    _MAX_REPEAT_ACTIONS = 2
    _MAX_TOOL_OUTPUT_CHARS = 4000
    _MAX_LOG_PREVIEW_CHARS = 200

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

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        action_counts: dict[str, int] = {}
        LOGGER.info(
            "Run start: agent=%s max_iterations=%d user_input=%s",
            self.name,
            max_iterations,
            self._preview_text(user_input),
        )

        for iteration in range(max_iterations):
            tools = self.tools_registry.get_tool_schemas()
            LOGGER.log(
                TRACE_LEVEL,
                "[request turn %d messages]\n%s",
                iteration + 1,
                json.dumps(conversation, indent=2, default=str),
            )
            llm_start = time.perf_counter()
            assistant_message = self.llm.complete(messages=conversation, tools=tools)
            llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
            tool_calls = assistant_message.tool_calls or []
            assistant_content = assistant_message.content or ""
            LOGGER.debug(
                "Turn %d: llm_latency_ms=%.1f tool_calls=%d",
                iteration + 1,
                llm_elapsed_ms,
                len(tool_calls),
            )
            LOGGER.debug(
                "[assistant turn %d] %s",
                iteration + 1,
                self._preview_text(assistant_content),
            )
            if tool_calls:
                LOGGER.debug(
                    "[assistant turn %d tool calls] %s",
                    iteration + 1,
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
                iteration + 1,
                json.dumps(response_payload, indent=2, default=str),
            )

            if not tool_calls:
                final_answer = self._extract_final_answer(assistant_content)
                if final_answer:
                    LOGGER.info("Returning final answer after %d turns.", iteration + 1)
                    LOGGER.debug(
                        "Final answer preview: %s", self._preview_text(final_answer)
                    )
                    return final_answer
                LOGGER.info("Returning assistant content after %d turns.", iteration + 1)
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
                            if tool_name == "get_skill_details":
                                skill_name = tool_args.get("skill_name")
                                if isinstance(skill_name, str) and skill_name:
                                    LOGGER.info("Loaded skill: %s", skill_name)
                        except Exception as exc:
                            result = f"Tool execution failed for {tool_name}: {exc}"
                            LOGGER.error(
                                "Tool execution failed: tool=%s error=%s",
                                tool_name,
                                exc,
                            )
                call_elapsed_ms = (time.perf_counter() - call_start) * 1000
                formatted_result = self._format_tool_result(result)
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

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": formatted_result,
                    }
                )

        LOGGER.error("Max iterations reached without a final response.")
        raise LeoLLMException("Max iterations reached without a final response.")
