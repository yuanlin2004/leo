from __future__ import annotations

from types import SimpleNamespace

import pytest

from leo.core.lessons.reflector import (
    CreateOp,
    ReflectorError,
    SkipOp,
    UpdateOp,
    build_reflection_messages,
    parse_ops,
    reflect,
)
from leo.core.lessons.schema import Lesson, Scope, Trigger


# -- parse_ops -------------------------------------------------------------


def test_parse_ops_empty_text():
    assert parse_ops("") == []


def test_parse_ops_no_json_raises():
    with pytest.raises(ReflectorError, match="no JSON envelope"):
        parse_ops("just some prose, no json here")


def test_parse_ops_unbalanced_braces_treated_as_no_envelope():
    # The balanced-brace extractor never converges, so we fall through to
    # "no JSON envelope found" rather than "not valid JSON". Either error
    # is acceptable to the caller — both surface as ReflectorError.
    with pytest.raises(ReflectorError):
        parse_ops('{"ops": [malformed')


def test_parse_ops_invalid_json_inside_braces():
    with pytest.raises(ReflectorError, match="not valid JSON"):
        parse_ops('{"ops": [1,]}')  # trailing comma, balanced braces


def test_parse_ops_top_level_array_not_recognized():
    # The extractor only looks for {...}; a bare array is rejected as
    # "no JSON envelope found". The reflector prompt asks for an object.
    with pytest.raises(ReflectorError, match="no JSON envelope"):
        parse_ops('["not", "an", "object"]')


def test_parse_ops_object_without_ops_key_raises():
    with pytest.raises(ReflectorError, match="object with an 'ops' field"):
        parse_ops('{"foo": "bar"}')


def test_parse_ops_ops_must_be_list():
    with pytest.raises(ReflectorError, match="'ops' must be a list"):
        parse_ops('{"ops": "nope"}')


def test_parse_ops_strips_code_fence():
    text = (
        "Here's my output:\n"
        "```json\n"
        '{"ops": [{"op": "skip", "reason": "uneventful"}]}\n'
        "```"
    )
    ops = parse_ops(text)
    assert len(ops) == 1
    assert isinstance(ops[0], SkipOp)
    assert ops[0].reason == "uneventful"


def test_parse_ops_finds_object_amid_prose():
    text = (
        "I think this is what we learned:\n"
        '{"ops": [{"op": "skip", "reason": "x"}]}\n'
        "Hope that helps!"
    )
    ops = parse_ops(text)
    assert len(ops) == 1


def test_parse_ops_handles_empty_ops_list():
    assert parse_ops('{"ops": []}') == []


def test_parse_ops_create():
    text = (
        '{"ops": [{"op": "create", "lesson": '
        '{"title": "T", "category": "fact", "trigger": {"type": "always"}, '
        '"rule": "r", "why": "w", "how_to_apply": "h"}}]}'
    )
    ops = parse_ops(text)
    assert len(ops) == 1
    assert isinstance(ops[0], CreateOp)
    assert ops[0].lesson["title"] == "T"


def test_parse_ops_create_missing_lesson_raises():
    with pytest.raises(ReflectorError, match="missing 'lesson' object"):
        parse_ops('{"ops": [{"op": "create"}]}')


def test_parse_ops_update():
    text = (
        '{"ops": [{"op": "update", "id": "abc", '
        '"fields": {"why": "new reason"}}]}'
    )
    ops = parse_ops(text)
    assert isinstance(ops[0], UpdateOp)
    assert ops[0].id == "abc"
    assert ops[0].fields == {"why": "new reason"}


def test_parse_ops_update_missing_id_raises():
    with pytest.raises(ReflectorError, match="missing string 'id'"):
        parse_ops('{"ops": [{"op": "update", "fields": {"x": 1}}]}')


def test_parse_ops_update_missing_fields_raises():
    with pytest.raises(ReflectorError, match="missing 'fields' object"):
        parse_ops('{"ops": [{"op": "update", "id": "x"}]}')


def test_parse_ops_unknown_op_raises():
    with pytest.raises(ReflectorError, match="unknown 'op'"):
        parse_ops('{"ops": [{"op": "bogus"}]}')


def test_parse_ops_mixed():
    text = """{"ops": [
        {"op": "create", "lesson": {"title": "A", "category": "fact",
         "trigger": {"type": "always"}, "rule": "r", "why": "w",
         "how_to_apply": "h"}},
        {"op": "update", "id": "old", "fields": {"why": "new"}},
        {"op": "skip", "reason": "nothing"}
    ]}"""
    ops = parse_ops(text)
    assert [type(o).__name__ for o in ops] == [
        "CreateOp", "UpdateOp", "SkipOp"
    ]


# -- build_reflection_messages --------------------------------------------


def _existing_lesson(lid="x", title="T", category="fact"):
    return Lesson(
        id=lid,
        title=title,
        category=category,
        trigger=Trigger(type="on_prompt", keywords=["foo"]),
        scope=Scope(),
        rule="r",
        why="w",
        how_to_apply="h",
        created="2026-04-25",
        updated="2026-04-25",
    )


def test_build_reflection_messages_includes_existing_lessons():
    msgs = build_reflection_messages(
        [{"role": "user", "content": "hi"}],
        [_existing_lesson(lid="ex1", title="Existing")],
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "ex1" in msgs[1]["content"]
    assert "Existing" in msgs[1]["content"]


def test_build_reflection_messages_serializes_tool_calls():
    trace = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
                }
            ],
        }
    ]
    msgs = build_reflection_messages(trace, [])
    body = msgs[1]["content"]
    assert "bash" in body
    assert "ls" in body


def test_build_reflection_messages_says_none_when_empty():
    msgs = build_reflection_messages([], [])
    assert "(none)" in msgs[1]["content"]


# -- reflect (with FakeLLM) -----------------------------------------------


class FakeLLM:
    def __init__(self, scripted_content: str):
        self.scripted_content = scripted_content
        self.last_messages = None
        self.model = "fake"

    def chat(self, messages, enable_thinking=True, tools=None,
             on_text=None, on_reasoning=None):
        self.last_messages = messages
        return SimpleNamespace(
            content=self.scripted_content,
            reasoning_content=None,
            tool_calls=None,
        )


def test_reflect_returns_parsed_ops():
    llm = FakeLLM('{"ops": [{"op": "skip", "reason": "x"}]}')
    result = reflect(llm, [], [])
    assert len(result.ops) == 1
    assert isinstance(result.ops[0], SkipOp)


def test_reflect_propagates_parser_error():
    llm = FakeLLM("not json at all")
    with pytest.raises(ReflectorError):
        reflect(llm, [], [])


def test_reflect_passes_disabled_thinking_and_no_tools():
    """The reflector should not stream thinking tokens or expose tools."""
    llm = FakeLLM('{"ops": []}')
    captured = {}

    orig_chat = llm.chat

    def chat(messages, enable_thinking=True, tools=None, **kw):
        captured["enable_thinking"] = enable_thinking
        captured["tools"] = tools
        return orig_chat(messages, enable_thinking=enable_thinking,
                         tools=tools, **kw)

    llm.chat = chat
    reflect(llm, [], [])
    assert captured["enable_thinking"] is False
    assert captured["tools"] is None
