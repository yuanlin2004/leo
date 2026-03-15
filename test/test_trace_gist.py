from __future__ import annotations

from pathlib import Path

from leo.tools.tg import parse_trace_gist, render_trace_gist, trace_gist_from_path


def test_parse_trace_gist_extracts_turn_summaries() -> None:
    turns = parse_trace_gist(
        "\n".join(
            [
                "Run start: agent=leo-react max_iterations=4 user_input=do thing",
                "Turn 1: calling model",
                "Turn 1: model responded latency_ms=0.0 tool_calls=1 content=Thought: need data",
                "Turn 1: tool plan=lookup",
                "Turn 1: executing tool=lookup args={'query': 'a'} attempt=1",
                "Turn 1: tool completed id=call-1 name=lookup latency_ms=0.0 result=obs:a",
                "Turn 2: calling model",
                "Turn 2: model responded latency_ms=0.0 tool_calls=0 content=Final Answer: done",
                "Turn 2: final answer detected from text preview=done",
                "Returning final answer after 2 turns.",
            ]
        )
    )

    assert [turn.turn_number for turn in turns] == [1, 2]
    assert turns[0].agent_messages == ["Plans tool use: lookup"]
    assert turns[0].llm_message == "Thought: need data"
    assert len(turns[0].tool_calls) == 1
    assert turns[0].tool_calls[0].name == "lookup"
    assert turns[0].tool_calls[0].tool_input == '{"query": "a"}'
    assert turns[0].tool_calls[0].result == "obs:a"
    assert turns[1].agent_messages == ["Final answer detected: done"]
    assert turns[1].llm_message == "Final Answer: done"


def test_render_trace_gist_formats_human_readable_blocks() -> None:
    rendered = render_trace_gist(
        parse_trace_gist(
            "\n".join(
                [
                    "Turn 1: model responded latency_ms=0.0 tool_calls=2 content=",
                    "Turn 1: tool plan=lookup, other",
                    "Turn 1: executing tool=lookup args={'query': 'a'} attempt=1",
                    "Turn 1: tool completed id=call-1 name=lookup latency_ms=0.0 result=obs:a",
                    "Turn 1: executing tool=other args={} attempt=1",
                    "Turn 1: tool completed id=call-2 name=other latency_ms=0.0 result=unused",
                ]
            )
        )
    )

    assert "Turn 1" in rendered
    assert "Agent: Plans tool use: lookup, other" in rendered
    assert "LLM: -" in rendered
    assert "Tool: lookup | other" in rendered
    assert 'Input: lookup: {"query": "a"} | other: {}' in rendered
    assert "Result: lookup: obs:a | other: unused" in rendered


def test_render_trace_gist_extracts_execute_appworld_code_result_only() -> None:
    rendered = render_trace_gist(
        parse_trace_gist(
            "\n".join(
                [
                    "Turn 1: model responded latency_ms=0.0 tool_calls=1 content=",
                    "Turn 1: tool plan=execute_appworld_code",
                    "Turn 1: executing tool=execute_appworld_code args={'code': 'print(1)'} attempt=1",
                    'Turn 1: tool completed id=call-1 name=execute_appworld_code latency_ms=0.0 result={"task_id":"t1","code":"print(1)","result":"1\\n"}',
                ]
            )
        )
    )

    assert "Tool: execute_appworld_code" in rendered
    assert "Input: execute_appworld_code: print(1)" in rendered
    assert 'Result: execute_appworld_code: 1' in rendered
    assert '"code":"print(1)"' not in rendered


def test_trace_gist_from_path_reads_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Turn 1: model responded latency_ms=0.0 tool_calls=0 content=hello",
                "Turn 1: returning assistant content preview=hello",
            ]
        ),
        encoding="utf-8",
    )

    rendered = trace_gist_from_path(log_path)

    assert "Turn 1" in rendered
    assert "Agent: Returns assistant content: hello" in rendered
    assert "LLM: hello" in rendered
    assert "Input: -" in rendered


def test_trace_gist_from_path_reads_structured_trace_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-03-15T00:00:00+00:00","event_type":"model_request","payload":{"messages":[],"tool_names":["lookup","final_answer"]}}',
                '{"timestamp":"2026-03-15T00:00:01+00:00","event_type":"model_response","payload":{"content":"Thought: need data","tool_calls":[{"function":{"name":"lookup","arguments":"{\\"query\\": \\"a\\"}"}}]}}',
                '{"timestamp":"2026-03-15T00:00:02+00:00","event_type":"tool_call","payload":{"tool_name":"lookup","tool_args":{"query":"a"}}}',
                '{"timestamp":"2026-03-15T00:00:03+00:00","event_type":"tool_result","payload":{"tool_name":"lookup","result":"obs:a"}}',
                '{"timestamp":"2026-03-15T00:00:04+00:00","event_type":"final_answer","payload":{"answer":"done"}}',
            ]
        ),
        encoding="utf-8",
    )

    rendered = trace_gist_from_path(trace_path)

    assert "Turn 1" in rendered
    assert "Agent: Plans tool use: lookup | Final answer recorded: done" in rendered
    assert "LLM: Thought: need data" in rendered
    assert "Tool: lookup" in rendered
    assert 'Input: lookup: {"query": "a"}' in rendered
    assert "Result: lookup: obs:a" in rendered


def test_trace_gist_structured_trace_extracts_execute_appworld_code_result_only(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-03-15T00:00:00+00:00","event_type":"model_request","payload":{"messages":[],"tool_names":["execute_appworld_code"]}}',
                '{"timestamp":"2026-03-15T00:00:01+00:00","event_type":"model_response","payload":{"content":"","tool_calls":[{"function":{"name":"execute_appworld_code","arguments":"{\\"code\\": \\"print(1)\\"}"}}]}}',
                '{"timestamp":"2026-03-15T00:00:02+00:00","event_type":"tool_call","payload":{"tool_name":"execute_appworld_code","tool_args":{"code":"print(1)"}}}',
                '{"timestamp":"2026-03-15T00:00:03+00:00","event_type":"tool_result","payload":{"tool_name":"execute_appworld_code","result":{"task_id":"t1","code":"print(1)","result":"1\\n"}}}',
            ]
        ),
        encoding="utf-8",
    )

    rendered = trace_gist_from_path(trace_path)

    assert "Tool: execute_appworld_code" in rendered
    assert "Input: execute_appworld_code: print(1)" in rendered
    assert "Result: execute_appworld_code: 1" in rendered
    assert '"code":"print(1)"' not in rendered
