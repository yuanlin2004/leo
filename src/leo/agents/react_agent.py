import json
from typing import Any

from ..core.llm import LeoLLMClient, LeoLLMException
from ..tools.registry import ToolsRegistry
from .agent import Agent

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

    def run(self, user_input: str, max_iterations: int = 10) -> str:
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        action_counts: dict[str, int] = {}

        for _iteration in range(max_iterations):
            tools = self.tools_registry.get_tool_schemas()
            assistant_message = self.llm.complete(messages=conversation, tools=tools)
            tool_calls = assistant_message.tool_calls or []
            assistant_content = assistant_message.content or ""

            if not tool_calls:
                final_answer = self._extract_final_answer(assistant_content)
                if final_answer:
                    return final_answer
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
                try:
                    tool_args = self._parse_tool_args(tool_call.function.arguments)
                except Exception as exc:
                    tool_args = {}
                    result = f"Tool argument parsing failed for {tool_name}: {exc}"
                else:
                    action_key = self._build_action_key(tool_name, tool_args)
                    action_counts[action_key] = action_counts.get(action_key, 0) + 1
                    if action_counts[action_key] > self._MAX_REPEAT_ACTIONS:
                        result = (
                            "Skipped repeated tool action to avoid loops. "
                            "Use a different query/arguments or provide Final Answer."
                        )
                    else:
                        try:
                            result = self.tools_registry.execute(tool_name, **tool_args)
                        except Exception as exc:
                            result = f"Tool execution failed for {tool_name}: {exc}"

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._format_tool_result(result),
                    }
                )

        raise LeoLLMException("Max iterations reached without a final response.")
