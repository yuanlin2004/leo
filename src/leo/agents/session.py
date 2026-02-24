from __future__ import annotations

from typing import Any, Callable


class AgentSession:
    def __init__(
        self,
        *,
        system_prompt: str,
        run_loop: Callable[[list[dict[str, Any]], int], str],
    ) -> None:
        self._system_prompt = system_prompt
        self._run_loop = run_loop
        self._conversation: list[dict[str, Any]] = []
        self.reset()

    def send(self, user_input: str, max_iterations: int = 10) -> str:
        self._conversation.append({"role": "user", "content": user_input})
        return self._run_loop(self._conversation, max_iterations)

    def reset(self) -> None:
        self._conversation = [{"role": "system", "content": self._system_prompt}]

    @property
    def conversation(self) -> list[dict[str, Any]]:
        return self._conversation
