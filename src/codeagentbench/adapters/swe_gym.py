"""SWE-Gym/SWE-Bench field mapping kept separate from the runtime."""

from __future__ import annotations

from typing import Any

from codeagentbench.models import EvalSpec, TaskRecord


def normalize_swe_gym_row(row: dict[str, Any], *, split: str = "unspecified") -> TaskRecord:
    """Map common SWE-Gym and SWE-Bench keys into the internal contract."""

    data = dict(row)
    data.setdefault("split", split)
    data.setdefault("instance_id", data.get("instance_id", data.get("id", "")))
    data.setdefault("repo", data.get("repo", data.get("repository", "")))
    data.setdefault("base_commit", data.get("base_commit", data.get("base_sha", "")))
    data.setdefault("issue", data.get("problem_statement", data.get("issue", "")))
    if not data.get("test_command"):
        spec = EvalSpec.from_dict(data)
        test_names = list(dict.fromkeys((*spec.fail_to_pass, *spec.pass_to_pass)))
        if test_names:
            data["test_command"] = "python -m pytest -q " + " ".join(test_names)
    data.setdefault("metadata", {
        key: data[key]
        for key in ("hints_text", "created_at", "version")
        if data.get(key) is not None
    })
    data["eval_spec"] = EvalSpec.from_dict(data).to_dict()
    return TaskRecord.from_dict(data)
