from __future__ import annotations

import pytest

from leo.core.lessons.schema import (
    HOW_MAX,
    RULE_MAX,
    SchemaError,
    WHY_MAX,
    parse_lesson,
)


def test_parse_valid_minimal(write):
    path = write("preference", "terse")
    lesson = parse_lesson(path)
    assert lesson.id == "terse"
    assert lesson.category == "preference"
    assert lesson.trigger.type == "always"
    assert lesson.trigger.keywords == []
    assert lesson.trigger.tool is None
    assert lesson.scope.is_empty()
    assert lesson.rule == "A rule."
    assert lesson.why == "A reason."
    assert lesson.how_to_apply == "At some point."
    assert lesson.path == path


def test_parse_valid_full(write):
    trigger = (
        "trigger:\n"
        "  type: on_tool_call\n"
        "  tool: bash\n"
        "  keywords: [curl]"
    )
    scope = (
        "scope:\n"
        "  project: leo\n"
        "  model: [claude-opus-4-7, claude-sonnet-4-6]"
    )
    path = write("gotcha", "vllm-batched", trigger=trigger, scope=scope)
    lesson = parse_lesson(path)
    assert lesson.trigger.type == "on_tool_call"
    assert lesson.trigger.tool == "bash"
    assert lesson.trigger.keywords == ["curl"]
    assert lesson.scope.project == ["leo"]
    assert lesson.scope.model == ["claude-opus-4-7", "claude-sonnet-4-6"]


def test_missing_frontmatter_open(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("just text, no frontmatter\n")
    with pytest.raises(SchemaError, match="missing YAML frontmatter"):
        parse_lesson(p)


def test_unterminated_frontmatter(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("---\nid: foo\n")
    with pytest.raises(SchemaError, match="unterminated"):
        parse_lesson(p)


def test_invalid_yaml(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text("---\nid: [unclosed\n---\nbody\n")
    with pytest.raises(SchemaError, match="invalid YAML"):
        parse_lesson(p)


def test_missing_required_field_id(write, tmp_path):
    path = write("preference", "x")
    text = path.read_text().replace("id: x\n", "")
    path.write_text(text)
    with pytest.raises(SchemaError, match="missing required field 'id'"):
        parse_lesson(path)


def test_bad_category_value(write):
    path = write("preference", "x")
    text = path.read_text().replace("category: preference", "category: bogus")
    path.write_text(text)
    with pytest.raises(SchemaError, match="category must be one of"):
        parse_lesson(path)


@pytest.mark.parametrize(
    "trigger,err",
    [
        # always must not have keywords or tool
        (
            "trigger:\n  type: always\n  keywords: [foo]",
            "always must not have keywords",
        ),
        (
            "trigger:\n  type: always\n  tool: bash",
            "always must not have keywords",
        ),
        # on_prompt requires non-empty keywords
        (
            "trigger:\n  type: on_prompt",
            "on_prompt requires non-empty keywords",
        ),
        (
            "trigger:\n  type: on_prompt\n  keywords: []",
            "on_prompt requires non-empty keywords",
        ),
        # on_monologue requires non-empty keywords
        (
            "trigger:\n  type: on_monologue\n  keywords: []",
            "on_monologue requires non-empty keywords",
        ),
        # on_tool_call needs tool or keywords
        (
            "trigger:\n  type: on_tool_call",
            "requires either tool or keywords",
        ),
        # tool only valid for on_tool_call
        (
            "trigger:\n  type: on_prompt\n  keywords: [foo]\n  tool: bash",
            "trigger.tool only valid for on_tool_call",
        ),
        # bad type value
        (
            "trigger:\n  type: never",
            "trigger.type must be one of",
        ),
        # null keywords
        (
            "trigger:\n  type: on_prompt\n  keywords:",
            "trigger.keywords must not be null",
        ),
    ],
)
def test_trigger_validation(write, trigger, err):
    path = write("fact", "x", trigger=trigger)
    with pytest.raises(SchemaError, match=err):
        parse_lesson(path)


def test_trigger_must_be_mapping(write):
    path = write("fact", "x", trigger="trigger: always")
    with pytest.raises(SchemaError, match="trigger must be a mapping"):
        parse_lesson(path)


def test_scope_null_value_is_error(write):
    scope = "scope:\n  project:"
    path = write("fact", "x", scope=scope)
    with pytest.raises(SchemaError, match="scope.project is null"):
        parse_lesson(path)


def test_scope_unknown_key(write):
    scope = "scope:\n  bogus: [x]"
    path = write("fact", "x", scope=scope)
    with pytest.raises(SchemaError, match="unknown scope keys"):
        parse_lesson(path)


def test_scope_scalar_sugar(write):
    scope = "scope:\n  skill: git"
    path = write("fact", "x", scope=scope)
    lesson = parse_lesson(path)
    assert lesson.scope.skill == ["git"]


def test_scope_empty_list_wildcard(write):
    scope = "scope:\n  skill: []"
    path = write("fact", "x", scope=scope)
    lesson = parse_lesson(path)
    assert lesson.scope.skill == []


def test_scope_value_wrong_type(write):
    scope = "scope:\n  skill: 42"
    path = write("fact", "x", scope=scope)
    with pytest.raises(SchemaError, match="must be a string or list of strings"):
        parse_lesson(path)


def test_scope_list_with_non_string(write):
    scope = "scope:\n  skill: [git, 7]"
    path = write("fact", "x", scope=scope)
    with pytest.raises(SchemaError, match="must be a string or list of strings"):
        parse_lesson(path)


def test_body_missing_rule(write):
    body = "## Why\na\n\n## How to apply\nb\n"
    path = write("fact", "x", body=body)
    with pytest.raises(SchemaError, match="missing '## Rule' section"):
        parse_lesson(path)


def test_body_missing_why(write):
    body = "## Rule\na\n\n## How to apply\nb\n"
    path = write("fact", "x", body=body)
    with pytest.raises(SchemaError, match="missing '## Why' section"):
        parse_lesson(path)


def test_body_missing_how(write):
    body = "## Rule\na\n\n## Why\nb\n"
    path = write("fact", "x", body=body)
    with pytest.raises(SchemaError, match="missing '## How to apply' section"):
        parse_lesson(path)


def test_rule_too_long(write):
    long_rule = "x" * (RULE_MAX + 1)
    body = f"## Rule\n{long_rule}\n\n## Why\nw\n\n## How to apply\nh\n"
    path = write("fact", "x", body=body)
    with pytest.raises(SchemaError, match=f"rule exceeds {RULE_MAX}"):
        parse_lesson(path)


def test_why_too_long(write):
    long_why = "y" * (WHY_MAX + 1)
    body = f"## Rule\nr\n\n## Why\n{long_why}\n\n## How to apply\nh\n"
    path = write("fact", "x", body=body)
    with pytest.raises(SchemaError, match=f"why exceeds {WHY_MAX}"):
        parse_lesson(path)


def test_how_too_long(write):
    long_how = "h" * (HOW_MAX + 1)
    body = f"## Rule\nr\n\n## Why\nw\n\n## How to apply\n{long_how}\n"
    path = write("fact", "x", body=body)
    with pytest.raises(SchemaError, match=f"how-to-apply exceeds {HOW_MAX}"):
        parse_lesson(path)


def test_iso_date_object_accepted(write):
    # YAML parses bare ISO dates as date objects; we should normalize to strings.
    path = write("fact", "x", created="2026-04-25", updated="2026-05-01")
    lesson = parse_lesson(path)
    assert lesson.created == "2026-04-25"
    assert lesson.updated == "2026-05-01"
