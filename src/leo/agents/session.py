from __future__ import annotations

import copy
from typing import Any, Callable


class AgentSession:
    def __init__(
        self,
        *,
        system_prompt: str,
        run_loop: Callable[[list[dict[str, Any]], int], str],
        reset_callback: Callable[[], None] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._run_loop = run_loop
        self._reset_callback = reset_callback
        self._conversation: list[dict[str, Any]] = []
        self.reset()

    def send(self, user_input: str, max_iterations: int = 10) -> str:
        self._conversation.append({"role": "user", "content": user_input})
        return self._run_loop(self._conversation, max_iterations)

    def reset(self) -> None:
        if self._reset_callback is not None:
            self._reset_callback()
        self._conversation = [{"role": "system", "content": self._system_prompt}]

    @staticmethod
    def _validate_message(message: Any, index: int) -> None:
        if not isinstance(message, dict):
            raise ValueError(f"Invalid message at index {index}: expected object.")

        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError(f"Invalid message at index {index}: missing string role.")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"Invalid message at index {index}: unsupported role '{role}'."
            )
        if role == "tool" and not isinstance(message.get("tool_call_id"), str):
            raise ValueError(
                f"Invalid message at index {index}: tool role requires tool_call_id."
            )
        if "content" in message and message["content"] is not None and not isinstance(
            message["content"], str
        ):
            raise ValueError(
                f"Invalid message at index {index}: content must be string or null."
            )

    def export_conversation(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._conversation)

    def load_conversation(self, conversation: list[dict[str, Any]]) -> None:
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("conversation must be a non-empty list of messages.")
        for idx, message in enumerate(conversation):
            self._validate_message(message, idx)
        first_role = conversation[0].get("role")
        if first_role != "system":
            raise ValueError("conversation must start with a system message.")
        self._conversation = copy.deepcopy(conversation)

    @property
    def conversation(self) -> list[dict[str, Any]]:
        return self._conversation
