from types import SimpleNamespace

import pytest

from scripts.autodl_dev_gate import outcome_flags, visible_test_command


def test_passing_patch_does_not_imply_clean_agent_finish():
    patch = "diff --git a/x b/x\n"
    result = SimpleNamespace(diff=patch, status="failed")
    evaluation = SimpleNamespace(passed=True)
    flags = outcome_flags(result, evaluation)
    assert flags["patch_verified"] is True
    assert flags["run_completed"] is False
    assert flags["autonomous_success"] is False
    assert len(flags["final_patch_sha256"]) == 64


def test_clean_agent_finish_still_needs_passing_patch():
    result = SimpleNamespace(diff="diff", status="completed")
    assert outcome_flags(result, SimpleNamespace(passed=False))["autonomous_success"] is False
    result = SimpleNamespace(diff="", status="completed")
    assert outcome_flags(result, SimpleNamespace(passed=True))["patch_verified"] is False


def test_visible_test_command_requires_existing_non_symlink_directory(tmp_path):
    (tmp_path / "tests" / "test_rds").mkdir(parents=True)
    assert visible_test_command(tmp_path, "tests/test_rds") == "python -m pytest -q -x tests/test_rds"
    for bad in ("../tests/test_rds", "/tests/test_rds", "tests/../test_rds", "tests/test_missing"):
        with pytest.raises(ValueError):
            visible_test_command(tmp_path, bad)
