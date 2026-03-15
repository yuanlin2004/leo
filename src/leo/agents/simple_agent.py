import json
from typing import Any

from ..core.llm import LeoLLMClient, LeoLLMException
from ..tools.registry import ToolsRegistry
from .agent import Agent
from .session import AgentSession

SIMPLE_AGENT_SYSTEM_PROMPT_BASE = """
You are a helpful assistant that can answer questions and perform tasks using available tools. 
Always try to use tools if they can help you answer the question or complete the task more effectively. 
If you don't know the answer, use the tools to find it out. 
Always provide a final answer to the user after using tools, even if you had to use them multiple times. 
Be concise and clear in your responses.
If a skill may help, call list_available_skills first, then activate_skill before using any tool or bundled resource from that skill.
If an activated skill points to companion guides, scripts, or reference files, load them with get_skill_resource instead of guessing.
If a skill depends on external binaries, MCP servers, auth, or env vars, inspect get_skill_requirements.
If a skill exposes runnable commands, inspect them with list_skill_commands and execute them with run_skill_command.
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

    @staticmethod
    def _parse_tool_args(raw_args: str | None) -> dict[str, Any]:
        if not raw_args:
            return {}
        parsed = json.loads(raw_args)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed

    def _format_tool_result(self, result: Any) -> str:
        tool_text = self._summarize_tool_result(result)
        if len(tool_text) <= self._MAX_TOOL_OUTPUT_CHARS:
            return tool_text
        return (
            tool_text[: self._MAX_TOOL_OUTPUT_CHARS]
            + "\n...[truncated to keep context window manageable]"
        )

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

    def _run_loop(
        self,
        conversation: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        user_input = str(conversation[-1].get("content") or "") if conversation else ""
        self.tools_registry.activate_relevant_skills_for_input(user_input)
        for _ in range(max_iterations):
            tools = self.tools_registry.get_tool_schemas()
            assistant_message = self.llm.complete(
                messages=self._build_model_messages(conversation),
                tools=tools,
            )
            tool_calls = assistant_message.tool_calls or []
            assistant_content = assistant_message.content or ""

            if not tool_calls:
                conversation.append({"role": "assistant", "content": assistant_content})
                return assistant_content

            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
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

    _MAX_TOOL_OUTPUT_CHARS = 4000
