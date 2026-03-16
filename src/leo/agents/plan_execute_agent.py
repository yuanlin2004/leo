from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..core.llm import LeoLLMException
from ..core.logging_utils import TRACE_LEVEL
from ..tools.registry import ToolsRegistry
from .react_agent import ReActAgent
from .session import AgentSession

LOGGER = logging.getLogger("leo.agents.plan_execute_agent")

PLAN_EXECUTE_PLANNER_PROMPT = """
You are the planning phase for a tool-using agent.
Produce a short execution plan for the next user task before tool execution begins.

Rules:
- Be concise.
- Prefer the fewest tool calls needed to reach a correct answer.
- If the task is an environment task, front-load only the API discovery needed to write one coherent execution snippet.
- If the task exposes an environment-specific execution tool, keep the computation inside that environment tool instead of copying partial observations into a generic local execution tool.
- Do not suggest repeated schema lookups once the relevant APIs are known.
- If authentication is required, include credential lookup and login before the main execution step.
- End with an explicit finish condition.
- Do not answer the user's task directly.
- Do not call tools.

Return plain text with:
Plan:
1. ...
2. ...
Finish when: ...
"""


class PlanExecuteAgent(ReActAgent):
    """
    Agent that plans first, then executes with replanning when progress stalls.
    """

    _MAX_TURNS_PER_PLAN = 4
    _REPLAN_MARKERS = (
        "Execution failed.",
        "Tool execution failed",
        "Tool argument parsing failed",
        "Skipped repeated tool action",
    )

    def __init__(
        self,
        name: str,
        llm: Any,
        tools_registry: ToolsRegistry | None = None,
        extra_system_prompt: str | None = None,
    ):
        self._ephemeral_plan_prompt: str | None = None
        super().__init__(
            name=name,
            llm=llm,
            tools_registry=tools_registry,
            extra_system_prompt=extra_system_prompt,
        )

    def __str__(self) -> str:
        return f"PlanExecuteAgent(name={self.name})"

    def _build_model_messages(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages = super()._build_model_messages(conversation)
        if not self._ephemeral_plan_prompt:
            return messages
        if not messages:
            return [{"role": "system", "content": self._ephemeral_plan_prompt}]
        return [
            messages[0],
            {"role": "system", "content": self._ephemeral_plan_prompt},
            *messages[1:],
        ]

    def _build_plan_messages(
        self,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tool_lines = []
        for tool_name, tool_desc in self.tools_registry.get_all_tools().items():
            tool_lines.append(f"- {tool_name}: {tool_desc}")
        planner_system = (
            PLAN_EXECUTE_PLANNER_PROMPT + "\nAvailable tools:\n" + "\n".join(tool_lines)
        )
        recent_messages: list[dict[str, Any]] = []
        for message in conversation[1:]:
            role = message.get("role")
            if role not in {"user", "assistant", "tool"}:
                continue
            content = message.get("content")
            if content is None:
                content = ""
            recent_messages.append({"role": role, "content": str(content)})
        return [
            {"role": "system", "content": planner_system},
            *self.tools_registry.get_runtime_context_messages(),
            *recent_messages[-8:],
        ]

    def _create_execution_plan(
        self,
        conversation: list[dict[str, Any]],
    ) -> str | None:
        plan_messages = self._build_plan_messages(conversation)
        LOGGER.info("Planning step: calling model")
        plan_message = self.llm.complete(messages=plan_messages, tools=None)
        plan_text = (plan_message.content or "").strip()
        LOGGER.info("Planning step complete: %s", self._preview_text(plan_text))
        return plan_text or None

    def _apply_execution_plan(self, conversation: list[dict[str, Any]]) -> None:
        try:
            plan_text = self._create_execution_plan(conversation)
        except Exception as exc:
            raise LeoLLMException(f"Planning step failed: {exc}") from exc

        if not plan_text:
            self._ephemeral_plan_prompt = None
            return
        self._ephemeral_plan_prompt = (
            "Execution plan for this run. Follow it unless new tool evidence requires a correction.\n"
            "Prefer a single coherent execution step once the required APIs are known.\n\n"
            f"{plan_text}"
        )

    def _should_replan(
        self,
        *,
        tool_results: list[str],
        turns_since_plan: int,
    ) -> bool:
        if any(
            any(marker in result for marker in self._REPLAN_MARKERS)
            for result in tool_results
        ):
            return True
        return turns_since_plan >= self._MAX_TURNS_PER_PLAN

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
        turns_since_plan = 0
        self._apply_execution_plan(conversation)

        LOGGER.info(
            "Run start: agent=%s max_iterations=%d user_input=%s",
            self.name,
            max_iterations,
            self._preview_text(user_input),
        )

        try:
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
                    return assistant_content

                conversation.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                    }
                )

                tool_results: list[str] = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    call_start = time.perf_counter()
                    try:
                        tool_args = self._parse_tool_args(tool_call.function.arguments)
                    except Exception as exc:
                        tool_args = {}
                        result = f"Tool argument parsing failed for {tool_name}: {exc}"
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
                                result = self.tools_registry.execute(tool_name, **tool_args)
                            except Exception as exc:
                                result = f"Tool execution failed for {tool_name}: {exc}"
                    call_elapsed_ms = (time.perf_counter() - call_start) * 1000
                    formatted_result = self._format_tool_result(result)
                    tool_results.append(formatted_result)
                    LOGGER.info(
                        "Turn %d: tool completed id=%s name=%s latency_ms=%.1f result=%s",
                        turn_number,
                        tool_call.id,
                        tool_name,
                        call_elapsed_ms,
                        self._preview_text(formatted_result),
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

                turns_since_plan += 1
                if iteration < max_iterations - 1 and self._should_replan(
                    tool_results=tool_results,
                    turns_since_plan=turns_since_plan,
                ):
                    LOGGER.info(
                        "Turn %d: replanning after new tool evidence.",
                        turn_number,
                    )
                    self._apply_execution_plan(conversation)
                    turns_since_plan = 0

            LOGGER.error("Max iterations reached without a final response.")
            raise LeoLLMException("Max iterations reached without a final response.")
        finally:
            self._ephemeral_plan_prompt = None

    def create_session(self) -> AgentSession:
        return AgentSession(
            system_prompt=self.system_prompt,
            run_loop=self._run_loop,
            reset_callback=self.tools_registry.reset_session_state,
        )
