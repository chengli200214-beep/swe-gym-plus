from __future__ import annotations

import pytest

from codeagentbench.sandbox.workspace import workspace_digest


def test_workspace_digest_does_not_follow_symlink_outside_workspace(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("first private value", encoding="utf-8")
    link = workspace / "probe.txt"
    try:
        link.symlink_to(private)
    except OSError:
        pytest.skip("file symlinks unavailable")

    before = workspace_digest(workspace)
    private.write_text("second private value", encoding="utf-8")
    assert workspace_digest(workspace) == before
    assert link.is_symlink()
