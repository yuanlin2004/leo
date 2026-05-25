from __future__ import annotations

import pytest

from leo.cli.leo import _cmd_session
from leo.core.setup import WORKSPACE_MARKER, WORKSPACE_SUBDIRS


def _make_workspace(path):
    leo_dir = path / WORKSPACE_MARKER
    leo_dir.mkdir()
    for sub in WORKSPACE_SUBDIRS:
        (leo_dir / sub).mkdir()
    return path


@pytest.mark.parametrize(
    "argv",
    [
        ["list", "--workspace", "{ws}"],   # flag after verb
        ["--workspace", "{ws}", "list"],   # flag before verb
    ],
)
def test_session_list_accepts_workspace_in_either_position(tmp_path, argv, capsys):
    ws = _make_workspace(tmp_path)
    argv = [a.format(ws=str(ws)) for a in argv]
    assert _cmd_session(argv) == 0
    assert "(no sessions)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "missing-id", "--workspace", "{ws}"],
        ["--workspace", "{ws}", "show", "missing-id"],
        ["rm", "missing-id", "--workspace", "{ws}"],
        ["--workspace", "{ws}", "rm", "missing-id"],
    ],
)
def test_session_show_rm_accept_workspace_in_either_position(tmp_path, argv, capsys):
    # We only assert that argparse accepts the args and the command reaches
    # the FileNotFoundError branch (return code 1). The point of the test
    # is parser acceptance, not the missing-session error message itself.
    ws = _make_workspace(tmp_path)
    argv = [a.format(ws=str(ws)) for a in argv]
    assert _cmd_session(argv) == 1
