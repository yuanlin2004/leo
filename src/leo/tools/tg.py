from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ToolCallGist:
    name: str
    tool_input: str | None = None
    result: str | None = None


@dataclass
class TurnGist:
    turn_number: int
    agent_messages: list[str] = field(default_factory=list)
    llm_message: str | None = None
    tool_calls: list[ToolCallGist] = field(default_factory=list)


_MODEL_RESPONDED_RE = re.compile(
    r"^Turn (?P<turn>\d+): model responded .* content=(?P<content>.*)$"
)
_TOOL_PLAN_RE = re.compile(r"^Turn (?P<turn>\d+): tool plan=(?P<plan>.*)$")
_EXECUTING_TOOL_RE = re.compile(
    r"^Turn (?P<turn>\d+): executing tool=(?P<tool>[^\s]+) args=(?P<args>\{.*\}) attempt=\d+$"
)
_TRACE_TOOL_INPUT_RE = re.compile(
    r"^\[tool input\] id=(?P<id>[^\s]+) name=(?P<tool>[^\s]+) args=(?P<args>.*)$"
)
_TOOL_COMPLETED_RE = re.compile(
    r"^Turn (?P<turn>\d+): tool completed id=.* name=(?P<tool>[^\s]+) .* result=(?P<result>.*)$"
)
_FINAL_ANSWER_TEXT_RE = re.compile(
    r"^Turn (?P<turn>\d+): final answer detected from text preview=(?P<preview>.*)$"
)
_FINAL_ANSWER_TOOL_RE = re.compile(
    r"^Turn (?P<turn>\d+): final answer tool received preview=(?P<preview>.*)$"
)
_RETURNING_ASSISTANT_RE = re.compile(
    r"^Turn (?P<turn>\d+): returning assistant content preview=(?P<preview>.*)$"
)
_AUTO_FINAL_RE = re.compile(
    r"^Turn (?P<turn>\d+): tool=(?P<tool>[^\s]+) requested automatic final answer preview=(?P<preview>.*)$"
)
_EMPTY_RETRY_RE = re.compile(
    r"^Turn (?P<turn>\d+): empty assistant response without tool calls; requesting a retry\.$"
)


def parse_trace_gist(text: str) -> list[TurnGist]:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        parsed = _parse_structured_trace(text)
        if parsed:
            return parsed

    turns: dict[int, TurnGist] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _MODEL_RESPONDED_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            content = match.group("content").strip()
            turn.llm_message = content or None
            continue

        match = _TOOL_PLAN_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            plan = match.group("plan").strip()
            if plan and plan != "-":
                turn.agent_messages.append(f"Plans tool use: {plan}")
            continue

        match = _EXECUTING_TOOL_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            tool_name = match.group("tool")
            tool_input = _normalize_plaintext_tool_input(
                tool_name,
                match.group("args").strip(),
            )
            turn.tool_calls.append(ToolCallGist(name=tool_name, tool_input=tool_input))
            continue

        match = _TRACE_TOOL_INPUT_RE.match(line)
        if match:
            tool_name = match.group("tool")
            tool_input = _normalize_plaintext_tool_input(
                tool_name,
                match.group("args").strip(),
            )
            _attach_tool_input_to_latest_turn(turns, tool_name, tool_input)
            continue

        match = _TOOL_COMPLETED_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            tool_name = match.group("tool")
            result = match.group("result").strip() or None
            _attach_tool_result(turn, tool_name, result)
            continue

        match = _FINAL_ANSWER_TEXT_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            preview = match.group("preview").strip()
            turn.agent_messages.append(f"Final answer detected: {preview}")
            continue

        match = _FINAL_ANSWER_TOOL_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            preview = match.group("preview").strip()
            turn.agent_messages.append(f"Final answer tool received: {preview}")
            continue

        match = _RETURNING_ASSISTANT_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            preview = match.group("preview").strip()
            turn.agent_messages.append(f"Returns assistant content: {preview}")
            continue

        match = _AUTO_FINAL_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            tool_name = match.group("tool")
            preview = match.group("preview").strip()
            turn.agent_messages.append(
                f"Automatic final answer requested by {tool_name}: {preview}"
            )
            continue

        match = _EMPTY_RETRY_RE.match(line)
        if match:
            turn = _get_turn(turns, int(match.group("turn")))
            turn.agent_messages.append("Empty response; retry requested.")
            continue

    return [turns[key] for key in sorted(turns)]


def render_trace_gist(turns: Sequence[TurnGist]) -> str:
    blocks: list[str] = []
    for turn in turns:
        block = [
            f"Turn {turn.turn_number}",
            f"Agent: {_render_agent_messages(turn.agent_messages)}",
            f"LLM: {_render_value(turn.llm_message)}",
            f"Tool: {_render_tool_names(turn.tool_calls)}",
            f"Input: {_render_tool_inputs(turn.tool_calls)}",
            f"Result: {_render_tool_results(turn.tool_calls)}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def trace_gist_from_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    text = resolved.read_text(encoding="utf-8", errors="replace")
    return render_trace_gist(parse_trace_gist(text))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m leo.tools.tg",
        description="Render a compact, human-readable gist of Leo run logs.",
    )
    parser.add_argument("log_file", help="Path to a Leo plain-text run log.")
    args = parser.parse_args(argv)
    print(trace_gist_from_path(args.log_file))
    return 0


def _get_turn(turns: dict[int, TurnGist], turn_number: int) -> TurnGist:
    turn = turns.get(turn_number)
    if turn is None:
        turn = TurnGist(turn_number=turn_number)
        turns[turn_number] = turn
    return turn


def _attach_tool_result(turn: TurnGist, tool_name: str, result: str | None) -> None:
    normalized_result = _normalize_plaintext_tool_result(tool_name, result)
    for tool_call in reversed(turn.tool_calls):
        if tool_call.name == tool_name and tool_call.result is None:
            tool_call.result = normalized_result
            return
    turn.tool_calls.append(ToolCallGist(name=tool_name, result=normalized_result))


def _attach_tool_input_to_latest_turn(
    turns: dict[int, TurnGist],
    tool_name: str,
    tool_input: str | None,
) -> None:
    for turn_number in sorted(turns.keys(), reverse=True):
        turn = turns[turn_number]
        for tool_call in reversed(turn.tool_calls):
            if tool_call.name == tool_name and tool_call.tool_input is None:
                tool_call.tool_input = tool_input
                return
    if turns:
        latest_turn = turns[max(turns)]
        latest_turn.tool_calls.append(ToolCallGist(name=tool_name, tool_input=tool_input))


def _render_agent_messages(messages: Sequence[str]) -> str:
    if not messages:
        return "-"
    return " | ".join(message for message in messages if message) or "-"


def _render_value(value: str | None) -> str:
    if value is None:
        return "-"
    return value


def _render_tool_names(tool_calls: Sequence[ToolCallGist]) -> str:
    if not tool_calls:
        return "-"
    return " | ".join(tool_call.name for tool_call in tool_calls)


def _render_tool_results(tool_calls: Sequence[ToolCallGist]) -> str:
    if not tool_calls:
        return "-"
    parts = []
    for tool_call in tool_calls:
        result = _render_value(tool_call.result)
        parts.append(f"{tool_call.name}: {result}")
    return " | ".join(parts)


def _render_tool_inputs(tool_calls: Sequence[ToolCallGist]) -> str:
    if not tool_calls:
        return "-"
    parts = []
    for tool_call in tool_calls:
        tool_input = _render_value(tool_call.tool_input)
        parts.append(f"{tool_call.name}: {tool_input}")
    return " | ".join(parts)


def _parse_structured_trace(text: str) -> list[TurnGist]:
    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict) or "event_type" not in payload:
            return []
        events.append(payload)

    if not events:
        return []

    turns: dict[int, TurnGist] = {}
    current_turn = 0

    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "model_request":
            current_turn += 1
            _get_turn(turns, current_turn)
            continue

        if current_turn < 1:
            continue

        turn = _get_turn(turns, current_turn)

        if event_type == "model_response":
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                turn.llm_message = content.strip()
            tool_calls = payload.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                tool_names: list[str] = []
                for item in tool_calls:
                    if not isinstance(item, dict):
                        continue
                    function_payload = item.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    tool_name = str(function_payload.get("name") or "").strip()
                    if tool_name:
                        tool_names.append(tool_name)
                if tool_names:
                    turn.agent_messages.append(
                        f"Plans tool use: {', '.join(tool_names)}"
                    )
            continue

        if event_type == "tool_call":
            tool_name = str(payload.get("tool_name") or "").strip()
            if tool_name:
                tool_input = _normalize_structured_tool_input(
                    tool_name,
                    payload.get("tool_args"),
                )
                turn.tool_calls.append(
                    ToolCallGist(name=tool_name, tool_input=tool_input)
                )
            continue

        if event_type == "tool_result":
            tool_name = str(payload.get("tool_name") or "").strip()
            result = payload.get("result")
            rendered = _structured_result_preview(tool_name, result)
            if tool_name:
                _attach_tool_result(turn, tool_name, rendered)
            continue

        if event_type == "final_answer":
            answer = payload.get("answer")
            if answer is None:
                turn.agent_messages.append("Final answer recorded: null")
            else:
                turn.agent_messages.append(
                    f"Final answer recorded: {_structured_result_preview('', answer)}"
                )

    return [turns[key] for key in sorted(turns)]


def _structured_result_preview(tool_name: str, value: Any) -> str:
    if tool_name == "execute_appworld_code" and isinstance(value, dict):
        nested_result = value.get("result")
        if isinstance(nested_result, str):
            text = " ".join(nested_result.split())
            return text or "-"
        if nested_result is not None:
            return _structured_result_preview("", nested_result)
    if isinstance(value, str):
        text = " ".join(value.split())
        return text or "-"
    if value is None:
        return "null"
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = repr(value)
    text = " ".join(text.split())
    if len(text) <= 240:
        return text
    return f"{text[:237]}..."


def _normalize_plaintext_tool_result(tool_name: str, result: str | None) -> str | None:
    if tool_name != "execute_appworld_code" or result is None:
        return result
    stripped = result.strip()
    if not stripped.startswith("{"):
        return result
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict):
        return result
    nested_result = payload.get("result")
    if isinstance(nested_result, str):
        normalized = " ".join(nested_result.split())
        return normalized or "-"
    if nested_result is None:
        return result
    return _structured_result_preview("", nested_result)


def _normalize_plaintext_tool_input(tool_name: str, tool_input: str | None) -> str | None:
    if tool_input is None:
        return None
    try:
        payload = json.loads(tool_input.replace("'", '"'))
    except json.JSONDecodeError:
        return tool_input
    return _normalize_structured_tool_input(tool_name, payload)


def _normalize_structured_tool_input(tool_name: str, tool_input: Any) -> str | None:
    if tool_name == "execute_appworld_code" and isinstance(tool_input, dict):
        code = tool_input.get("code")
        if isinstance(code, str):
            return code.strip() or "-"
    if isinstance(tool_input, str):
        text = " ".join(tool_input.split())
        return text or "-"
    if tool_input is None:
        return None
    try:
        text = json.dumps(tool_input, sort_keys=True)
    except TypeError:
        text = repr(tool_input)
    text = " ".join(text.split())
    if len(text) <= 240:
        return text
    return f"{text[:237]}..."


if __name__ == "__main__":
    raise SystemExit(main())
