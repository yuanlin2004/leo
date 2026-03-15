from __future__ import annotations

import logging
from typing import Any

from ..core.llm import LeoLLMException
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
    Agent that plans first, then executes with the ReAct loop.
    """

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
        planner_system = PLAN_EXECUTE_PLANNER_PROMPT + "\nAvailable tools:\n" + "\n".join(tool_lines)
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
            *recent_messages[-6:],
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

    def _run_loop(
        self,
        conversation: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        try:
            plan_text = self._create_execution_plan(conversation)
        except Exception as exc:
            raise LeoLLMException(f"Planning step failed: {exc}") from exc

        if plan_text:
            self._ephemeral_plan_prompt = (
                "Execution plan for this run. Follow it unless new tool evidence requires a correction.\n"
                "Prefer a single coherent execution step once the required APIs are known.\n\n"
                f"{plan_text}"
            )
        try:
            return super()._run_loop(conversation, max_iterations)
        finally:
            self._ephemeral_plan_prompt = None

    def create_session(self) -> AgentSession:
        return AgentSession(
            system_prompt=self.system_prompt,
            run_loop=self._run_loop,
            reset_callback=self.tools_registry.reset_session_state,
        )
