from __future__ import annotations

import pytest

from leo.cli.leo import _parse_exit_command


@pytest.mark.parametrize(
    "user_input,expected",
    [
        ("/exit",        (True, False)),
        ("/quit",        (True, False)),
        ("/exit noref",  (True, True)),
        ("/quit noref",  (True, True)),
        ("/exit  noref", (True, True)),  # extra space — split() collapses
    ],
)
def test_exit_recognized(user_input, expected):
    assert _parse_exit_command(user_input) == expected


@pytest.mark.parametrize(
    "user_input",
    [
        "",
        "hello",
        "/help",
        "/reset",
        "/exit nope",       # unknown arg
        "/quit foo bar",    # extra args
        "/exit noref now",  # extra word
    ],
)
def test_exit_not_recognized(user_input):
    assert _parse_exit_command(user_input) is None
