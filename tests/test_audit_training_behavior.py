"""Synthetic checks for aggregate behavior auditing; not training evidence."""
import json

from scripts.audit_training_behavior import action_info, audit, patch_flags


def row(index, action, history=()):
    return {"task_id": "task-1", "run_id": "run-1", "action_index": index,
            "messages": [{"role": "system", "content": "rules"},
                         {"role": "user", "content": "issue"}, *history,
                         {"role": "assistant", "content": json.dumps(action)}]}


def test_audit_separates_action_and_run_grain_without_certifying_shell_execution():
    edit = {"command": "python -c 'Path(\"src.py\").write_text(\"x\")'", "done": False}
    test = {"command": "python -m pytest tests/test_src.py", "done": False}
    receipt = [{"role": "assistant", "content": json.dumps(test)},
               {"role": "user", "content": 'Tool result:\n{"exit_code":0,"timed_out":false}'}]
    actions = [row(0, edit), row(1, test), row(2, {"done": True}, receipt)]
    patch = "diff --git a/src.py b/src.py\n@@ -1 +1 @@\n-old\n+new\n"
    trajectories = [{"task_id": "task-1", "run_id": "run-1", "agent_status": "completed",
                     "evaluation_verdict": "passed", "patch": patch, "events_path": "missing"}]
    result = audit(actions, trajectories, raw_events_exist=lambda _: False)
    assert result["action_rows"] == 3 and result["distinct_tasks"] == 1
    assert result["behavior_sequence_heuristic"]["runs_with_edit_then_test_heuristic"] == 1
    assert result["runs_with_passing_test_receipt_visible_in_done_context"] == 1
    assert result["raw_event_paths_resolved"] == 0
    assert "heuristics" in result["caveats"][0]


def test_patch_proxy_and_typed_actions():
    assert action_info('{"edit":{"path":"src.py","before":"x","after":"y"}}')[0] == "edit"
    assert patch_flags("diff --git a/src.py b/src.py\n@@ -1 +1 @@\n-# old\n+# TODO new\n")["comment_or_blank_only"]
    assert patch_flags("diff --git a/tests/test_x.py b/tests/test_x.py\n@@ -1 +1 @@\n-x\n+y\n")["test_file_modified"]
