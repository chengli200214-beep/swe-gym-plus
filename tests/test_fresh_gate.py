import pytest

from scripts.freeze_fresh_gate import mentioned_ids, select


def row(number, **changes):
    return {"instance_id": f"getmoto__moto-{number}", "repo": "getmoto/moto",
            "base_commit": str(number), "problem_statement": f"issue {number}",
            "FAIL_TO_PASS": ["tests/test_x.py::test_x"], "PASS_TO_PASS": [],
            **changes}


def test_exposure_includes_all_partitions_and_historical_run_ids():
    value = {"tasks": [row(1)], "eval": ["getmoto__moto-2"],
             "prior": {"task_id": "getmoto__moto-3"}, "issue": "not an id"}
    assert mentioned_ids(value) == {"getmoto__moto-1", "getmoto__moto-2", "getmoto__moto-3"}


def test_selection_excludes_known_and_near_duplicate_groups_without_gold_access():
    rows = [row(n) for n in range(1, 12)]
    rows[2]["problem_statement"] = "issue   1"
    rows[3]["base_commit"] = "1"
    known = {"getmoto__moto-1", "getmoto__moto-2"}
    selected, profile = select(rows, known)
    ids = {r["instance_id"] for r in selected}
    assert len(ids) == 3 and not ids & {f"getmoto__moto-{n}" for n in range(1, 5)}
    selected_ids = [r["instance_id"] for r in selected]
    assert profile["rows"] == 11 and profile["eligible_fresh"] == 7
    assert select(list(reversed(rows)), known)[0] == selected
    for r in rows:
        r["patch"] = "DO NOT SELECT BASED ON THIS"
        r["model_result"] = "passed"
    assert [r["instance_id"] for r in select(rows, known)[0]] == selected_ids


def test_invalid_grain_missing_data_and_insufficient_fresh_tasks_fail_closed():
    with pytest.raises(ValueError, match="duplicates"):
        select([row(1), row(1)], set())
    with pytest.raises(ValueError, match="missing"):
        select([row(1, problem_statement=None)], set())
    with pytest.raises(ValueError, match="three fresh"):
        select([row(1), row(2)], set())
