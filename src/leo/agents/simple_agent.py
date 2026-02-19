import json
from typing import Any

from ..core.llm import LeoLLMClient, LeoLLMException
from ..tools.registry import ToolsRegistry
from .agent import Agent

SIMPLE_AGENT_SYSTEM_PROMPT_BASE = """
You are a helpful assistant that can answer questions and perform tasks using available tools. 
Always try to use tools if they can help you answer the question or complete the task more effectively. 
If you don't know the answer, use the tools to find it out. 
Always provide a final answer to the user after using tools, even if you had to use them multiple times. 
Be concise and clear in your responses.
If you suspect a skill may help, call list_available_skills first to discover options, then call get_skill_details before using any skill action.
"""

class SimpleAgent(Agent):
    """
    A simple agent that takes in a command and responds with an answer. 
    It uses available tools if needed.  

    """
    def __init__(
        self,
        name: str,
        llm: LeoLLMClient,
        tools_registry: ToolsRegistry | None = None,
        extra_system_prompt: str | None = None,
    ):
        self.tools_registry = tools_registry or ToolsRegistry()

        # construct the system prompt by combining the base prompt with any extra instructions
        system_prompt = SIMPLE_AGENT_SYSTEM_PROMPT_BASE

        system_prompt += "The following tools are available to you:"
        for tool_name, tool_desc in self.tools_registry.get_all_tools().items():
            system_prompt += f"\n- {tool_name}: {tool_desc}"

        if extra_system_prompt:
            system_prompt += extra_system_prompt

        super().__init__(name, llm, system_prompt)

    def __str__(self) -> str:
        return f"SimpleAgent(name={self.name})"

    @staticmethod
    def _parse_tool_args(raw_args: str | None) -> dict[str, Any]:
        if not raw_args:
            return {}
        parsed = json.loads(raw_args)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed

    def _format_tool_result(self, result: Any) -> str:
        tool_text = result if isinstance(result, str) else json.dumps(result)
        if len(tool_text) <= self._MAX_TOOL_OUTPUT_CHARS:
            return tool_text
        return (
            tool_text[: self._MAX_TOOL_OUTPUT_CHARS]
            + "\n...[truncated to keep context window manageable]"
        )


    def run(self, user_input: str, max_iterations: int = 10) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        conversation = list(messages)

        for _ in range(max_iterations):
            tools = self.tools_registry.get_tool_schemas()
            assistant_message = self.llm.complete(messages=conversation, tools=tools)
            tool_calls = assistant_message.tool_calls or []

            if not tool_calls:
                return assistant_message.content or ""

            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = self._parse_tool_args(tool_call.function.arguments)
                except Exception as exc:
                    tool_args = {}
                    result = f"Tool argument parsing failed for {tool_name}: {exc}"
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
    _MAX_TOOL_OUTPUT_CHARS = 4000
