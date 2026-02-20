from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow running directly from repo root without setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from leo import LeoLLMClient
from leo.agents import ReActAgent
from leo.tools.registry import ToolsRegistry

TRACE_LEVEL = 5


def _configure_logging(level_name: str) -> None:
    if not hasattr(logging, "TRACE"):
        setattr(logging, "TRACE", TRACE_LEVEL)
        logging.addLevelName(TRACE_LEVEL, "TRACE")

    if not hasattr(logging.Logger, "trace"):
        def trace(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
            if self.isEnabledFor(TRACE_LEVEL):
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace  # type: ignore[attr-defined]

    normalized = (level_name or "INFO").strip().upper()
    level_map = {
        "TRACE": TRACE_LEVEL,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    resolved_level = level_map.get(normalized, logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(message)s",
        force=True,
    )

    class _LeoOnlyFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.name == "leo" or record.name.startswith("leo.")

    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_LeoOnlyFilter())
    logging.getLogger("leo").setLevel(resolved_level)


class TracingLLM:
    """
    Thin wrapper that prints each assistant turn and requested tool calls
    to make the ReAct loop visible when running the sample.
    """

    def __init__(self, inner: LeoLLMClient, logger: logging.Logger) -> None:
        self._inner = inner
        self._logger = logger
        self._turn = 0
        self._color_enabled = sys.stdout.isatty() and not os.getenv("NO_COLOR")
        self._input_color = "\033[94m"
        self._response_color = "\033[92m"
        self._reset = "\033[0m"

    def _styled(self, text: str, *, is_input: bool) -> str:
        if not self._color_enabled:
            return text
        color = self._input_color if is_input else self._response_color
        return f"{color}{text}{self._reset}"

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self._turn += 1
        self._logger.trace(
            self._styled(f"\n[request turn {self._turn} messages]", is_input=True)
        )
        self._logger.trace(
            self._styled(json.dumps(messages, indent=2, default=str), is_input=True)
        )
        response = self._inner.complete(messages=messages, tools=tools)
        content = (response.content or "").strip()
        tool_calls = response.tool_calls or []

        self._logger.debug(
            self._styled(f"\n[assistant turn {self._turn}]", is_input=False)
        )
        if content:
            self._logger.debug(self._styled(content, is_input=False))
        if tool_calls:
            names = ", ".join(tool_call.function.name for tool_call in tool_calls)
            self._logger.debug(self._styled(f"[tool calls] {names}", is_input=False))
        self._logger.trace(
            self._styled(
                f"\n[assistant turn {self._turn} full response]",
                is_input=False,
            )
        )
        if hasattr(response, "model_dump"):
            self._logger.trace(
                self._styled(
                    json.dumps(response.model_dump(), indent=2, default=str),
                    is_input=False,
                )
            )
        else:
            serialized_calls = [
                {
                    "id": getattr(tool_call, "id", None),
                    "name": getattr(getattr(tool_call, "function", None), "name", None),
                    "arguments": getattr(
                        getattr(tool_call, "function", None), "arguments", None
                    ),
                }
                for tool_call in tool_calls
            ]
            self._logger.trace(
                self._styled(
                    json.dumps(
                        {"content": response.content, "tool_calls": serialized_calls},
                        indent=2,
                        default=str,
                    ),
                    is_input=False,
                )
            )
        return response


def _default_task() -> str:
    return (
        "Find the latest Python release information and one recent headline "
        "about OpenAI. Compare recency and summarize key points."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end ReAct sample using lazy-loaded web_search skill "
            "with a multi-step tool flow."
        )
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("LEO_PROVIDER", "openrouter"),
        help="LLM provider (openrouter or ollama).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LEO_MODEL", "openai/gpt-4o-mini"),
        help="Model ID for the selected provider.",
    )
    parser.add_argument(
        "--skills-root",
        default=str(Path(".agents/skills").resolve()),
        help="Path to skills root directory.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum ReAct loop iterations.",
    )
    parser.add_argument(
        "--task",
        default=_default_task(),
        help="User task prompt.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LEO_LOG_LEVEL", "INFO"),
        help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    logger = logging.getLogger("leo.samples.react_web_search_e2e")

    if not (os.getenv("TAVILY_API_KEY") or os.getenv("TAVILYKEY")):
        raise SystemExit(
            "Missing Tavily key. Set TAVILY_API_KEY (or TAVILYKEY) before running."
        )

    registry = ToolsRegistry(skills_root=args.skills_root)
    skill_names = {item["name"] for item in registry.list_available_skills()}
    if "web_search" not in skill_names:
        raise SystemExit(
            f"web_search skill not found under {args.skills_root}. "
            "Expected .agents/skills/web_search/SKILL.md."
        )

    llm = LeoLLMClient(model=args.model, provider=args.provider, temperature=0.2)
    tracing_llm = TracingLLM(llm, logger)
    agent = ReActAgent(
        name="react-web-search",
        llm=tracing_llm,
        tools_registry=registry,
        extra_system_prompt="""
For this task, explicitly use a multi-step ReAct flow:
1) Call list_available_skills to discover skills first.
2) Call get_skill_details with skill_name="web_search".
3) Call web_search at least two times with different targeted queries.
4) Then return Final Answer.
""",
    )

    logger.info("== ReAct E2E Sample ==")
    logger.info("provider=%s model=%s", args.provider, args.model)
    logger.info("skills_root=%s", args.skills_root)
    logger.info("task=%s", args.task)

    final_answer = agent.run(args.task, max_iterations=args.max_iterations)

    logger.info("\n[final answer]")
    logger.info("%s", final_answer)


if __name__ == "__main__":
    main()
