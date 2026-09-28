"""Aggregate behavioral coverage of an action-level SFT export.

This profiles already exported JSONL data. It never executes model commands,
prints source text, or upgrades heuristic shell labels to verified receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from codeagentbench.adapters.action import parse_action


EDIT = re.compile(r"\b(?:write_text|apply_patch)\b|\bsed\s+-i\b|\bpatch\s|\bopen\([^\n]*['\"]w", re.I)
TEST = re.compile(r"\b(?:pytest|unittest|tox)\b", re.I)
DIFF = re.compile(r"\bgit\s+diff\b", re.I)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_info(text: str) -> tuple[str, set[str]]:
    try:
        action = parse_action(text)
    except ValueError:
        return "invalid", set()
    if action.done:
        return "done", set()
    for kind in ("search", "read", "edit"):
        if getattr(action, kind) is not None:
            return kind, set()
    tags = set()
    if EDIT.search(action.command):
        tags.add("likely_shell_edit")
    if TEST.search(action.command):
        tags.add("likely_shell_test")
    if DIFF.search(action.command):
        tags.add("shell_diff")
    return "command", tags


def visible_passing_test_in_done_context(messages: list[dict]) -> bool:
    """Only count a visible action/receipt pair, not a claimed test result."""
    prior_test = False
    for message in messages[2:-1]:
        content = message.get("content", "")
        if message.get("role") == "assistant":
            prior_test = "likely_shell_test" in action_info(content)[1]
        elif message.get("role") == "user" and content.startswith("Tool result:\n"):
            try:
                receipt = json.loads(content.split("\n", 1)[1])
            except (ValueError, TypeError):
                continue
            if prior_test and receipt.get("exit_code") == 0 and receipt.get("timed_out") is False:
                return True
            prior_test = False
    return False


def patch_flags(diff: str) -> dict[str, bool | int]:
    paths, changed, noncomment, todo = set(), 0, 0, False
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.add(parts[3].removeprefix("b/"))
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed += 1
            body = line[1:].strip()
            noncomment += bool(body and not body.startswith("#"))
            if line.startswith("+") and re.search(r"\bTODO\b", body, re.I):
                todo = True
    return {
        "nonempty": bool(diff.strip()),
        "file_count": len(paths),
        "test_file_modified": any(path.startswith("tests/") or "/tests/" in path for path in paths),
        "comment_or_blank_only": bool(changed) and not noncomment,
        "todo_added": todo,
    }


def audit(actions: list[dict], trajectories: list[dict], *, raw_events_exist) -> dict:
    """Return only aggregate checks and public task IDs; no prompts or patches."""
    by_run: dict[tuple[str, str], list[tuple[int, str, set[str]]]] = defaultdict(list)
    target_kinds, command_tags, contexts = Counter(), Counter(), Counter()
    duplicate_keys = set()
    identities = set()
    done_test_receipt = set()
    for row in actions:
        task, run, index = row["task_id"], row["run_id"], row["action_index"]
        identity = (task, run, index)
        if identity in identities:
            duplicate_keys.add(identity)
        identities.add(identity)
        kind, tags = action_info(row["messages"][-1]["content"])
        target_kinds[kind] += 1
        command_tags.update(tags)
        by_run[(task, run)].append((index, kind, tags))
        history = row["messages"][:-1]
        contexts["warning_rows"] += any(m["content"].startswith("Harness warning:") for m in history)
        contexts["protocol_result_rows"] += any(m["content"].startswith("Protocol result:\n") for m in history)
        contexts["typed_read_receipt_rows"] += any(
            m["content"].startswith("Tool result:\n") and '"source_read"' in m["content"]
            for m in history)
        if kind == "done" and visible_passing_test_in_done_context(row["messages"]):
            done_test_receipt.add((task, run))

    sequence_counts = Counter()
    for steps in by_run.values():
        ordered = sorted(steps)
        edits = [index for index, kind, tags in ordered if kind == "edit" or "likely_shell_edit" in tags]
        tests = [index for index, _, tags in ordered if "likely_shell_test" in tags]
        sequence_counts["runs_with_edit_target_or_shell_heuristic"] += bool(edits)
        sequence_counts["runs_with_test_shell_heuristic"] += bool(tests)
        sequence_counts["runs_with_edit_then_test_heuristic"] += any(test > edit for edit in edits for test in tests)
        sequence_counts["runs_with_done"] += any(kind == "done" for _, kind, _ in ordered)

    outcomes, patch_counts = Counter(), Counter()
    patches = {}
    for row in trajectories:
        identity = (row["task_id"], row["run_id"])
        outcomes[(row["agent_status"], row["evaluation_verdict"])] += 1
        flags = patch_flags(row["patch"])
        patches[identity] = flags
        patch_counts["runs_with_nonempty_patch"] += flags["nonempty"]
        patch_counts["runs_modifying_tests"] += flags["test_file_modified"]
        patch_counts["comment_or_blank_only_patches"] += flags["comment_or_blank_only"]
        patch_counts["patches_adding_todo"] += flags["todo_added"]
    event_paths = [row.get("events_path", "") for row in trajectories]
    return {
        "grain": "one supervised assistant action; run/task counts are separate",
        "action_rows": len(actions), "trajectory_rows": len(trajectories),
        "distinct_tasks": len({row["task_id"] for row in trajectories}),
        "distinct_action_runs": len(by_run),
        "duplicate_action_identities": len(duplicate_keys),
        "action_runs_without_trajectory": len(set(by_run) - set(patches)),
        "trajectories_without_actions": len(set(patches) - set(by_run)),
        "target_kinds": dict(target_kinds), "shell_command_tags_heuristic": dict(command_tags),
        "context_coverage": dict(contexts), "behavior_sequence_heuristic": dict(sequence_counts),
        "runs_with_passing_test_receipt_visible_in_done_context": len(done_test_receipt),
        "declared_outcomes": {f"{status}/{verdict}": count for (status, verdict), count in outcomes.items()},
        "patch_structure": dict(patch_counts),
        "raw_event_paths_resolved": sum(bool(path) and raw_events_exist(path) for path in event_paths),
        "raw_event_paths_declared": sum(bool(path) for path in event_paths),
        "caveats": [
            "Shell edit/test tags and edit-then-test sequences are heuristics, not execution receipts.",
            "A missing test receipt in the compact done context does not prove the run never tested.",
            "Declared independent pass cannot be re-certified where original event files are missing.",
            "Comment-only/TODO patch flags are syntactic proxies, not semantic correctness judgments.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior audit; choose a new output path")
    report = audit(load_jsonl(args.actions), load_jsonl(args.trajectories),
                   raw_events_exist=lambda path: Path(path).is_file())
    report["action_file_sha256"] = hashlib.sha256(args.actions.read_bytes()).hexdigest()
    report["trajectory_file_sha256"] = hashlib.sha256(args.trajectories.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
