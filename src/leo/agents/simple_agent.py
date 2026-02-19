import json
from typing import Any

from ..core.llm import LeoLLMClient
from ..tools.registry import ToolsRegistry, ToolsRegistryError
from .agent import Agent

SIMPLE_AGENT_SYSTEM_PROMPT_BASE = """
You are a helpful assistant that can answer questions and perform tasks using available tools. 
Always try to use tools if they can help you answer the question or complete the task more effectively. 
If you don't know the answer, use the tools to find it out. 
Always provide a final answer to the user after using tools, even if you had to use them multiple times. 
Be concise and clear in your responses.
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
        self.tools_registry = tools_registry or ToolsRegistry.default_registry()

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


    def run(self, user_input: str, max_iterations: int = 10) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        current_iteration = 0
        while current_iteration < max_iterations:
            current_iteration += 1
            response = self.llm.invoke(
                messages=messages,
                tools=self.tools_registry.get_tool_schemas(),
            )

            tool_calls = response.tool_calls or []
            if not tool_calls:
                return response.content or ""

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
            }
            messages.append(assistant_message)

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                try:
                    result = self.tools_registry.execute(tool_name, **tool_args)
                except (ToolsRegistryError, TypeError, ValueError) as exc:
                    result = f"Tool execution failed for {tool_name}: {exc}"

                tool_content = result if isinstance(result, str) else json.dumps(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    }
                )
        return "Max iterations reached without a final response."
