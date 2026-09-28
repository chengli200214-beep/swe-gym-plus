"""Necessary deployment-alignment checks; never an authenticity certificate.

Original event receipts, blind splits and tokenizer spans still need independent
audits. Coverage can only reject unsuitable data, not prove that data is genuine
or sufficient. Do not manufacture examples to satisfy this gate (ADR 0012).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from codeagentbench.adapters.action import parse_action
from codeagentbench.harness.context_history import POLICIES


REQUIRED_TARGETS = frozenset({"search", "read", "edit", "done"})


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _kind(action) -> str:
    if action.done:
        return "done"
    for name in ("edit", "read", "search"):
        if getattr(action, name) is not None:
            return name
    return "command"


def _changed_after_rejection(messages, target) -> bool:
    """Count a changed executable action after an explicit non-execution result.

    This examines serialized context only; matching its bytes to original events
    remains the export audit's responsibility. A done claim is not recovery.
    """
    if not target.executable:
        return False
    indices = [i for i, m in enumerate(messages[:-1])
               if m["role"] == "user" and m["content"].startswith("Protocol result:\n")]
    if not indices:
        return False
    index = indices[-1]
    result = json.loads(messages[index]["content"].split("\n", 1)[1])
    if not isinstance(result, dict) or result.get("executed") is not False:
        return False
    # Only the immediately following action is evidence of recovery.
    if any(m["role"] == "assistant" for m in messages[index + 1:-1]):
        return False
    if index < 1 or messages[index - 1]["role"] != "assistant":
        return False
    try:
        prior = parse_action(messages[index - 1]["content"])
    except ValueError:
        return True  # Rejected malformed action followed by a valid one.
    before, after = prior.to_dict(), target.to_dict()
    before.pop("message", None)
    after.pop("message", None)
    return before != after


def audit_readiness(records, *, expected_system_prompt: str, expected_prompt_policy: str) -> dict:
    """Return bounded diagnostics without exposing source text or model outputs."""
    counts, errors, tasks, policies, systems = Counter(), Counter(), set(), set(), set()
    runs, finished, identities = set(), set(), set()
    recovery = 0
    if not isinstance(expected_prompt_policy, str) or expected_prompt_policy not in POLICIES:
        errors["deployment_policy_missing_or_unknown"] += 1
    if not records:
        errors["empty_dataset"] += 1
    for row in records:
        if not isinstance(row, dict):
            errors["invalid_record"] += 1
            continue
        task, run, index = row.get("task_id"), row.get("run_id"), row.get("action_index")
        if (not isinstance(task, str) or not task or not isinstance(run, str) or not run
                or type(index) is not int or index < 0):
            errors["invalid_source_identity"] += 1
        else:
            identity = (task, run, index)
            if identity in identities:
                errors["duplicate_action_identity"] += 1
            identities.add(identity)
            tasks.add(task)
            runs.add((task, run))
        policy = row.get("prompt_policy")
        if isinstance(policy, str):
            policies.add(policy)
        if policy != expected_prompt_policy:
            errors["context_policy_mismatch"] += 1
        if row.get("assistant_only_loss") is not True or row.get("next_action_only_loss") is not True:
            errors["next_action_masking_not_declared"] += 1
        if row.get("source_evaluation_verdict") != "passed":
            errors["source_not_declared_passed"] += 1
        messages = row.get("messages")
        if (not isinstance(messages, list) or len(messages) < 3
                or any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
                       or not isinstance(m.get("content"), str) for m in messages)
                or [messages[0]["role"], messages[1]["role"], messages[-1]["role"]]
                != ["system", "user", "assistant"]):
            errors["invalid_next_action_messages"] += 1
            continue
        systems.add(prompt_hash(messages[0]["content"]))
        if messages[0]["content"] != expected_system_prompt:
            errors["system_prompt_mismatch"] += 1
        try:
            target = parse_action(messages[-1]["content"])
        except ValueError:
            errors["invalid_target_action"] += 1
            continue
        counts[_kind(target)] += 1
        if target.done and isinstance(task, str) and isinstance(run, str):
            finished.add((task, run))
        try:
            recovery += _changed_after_rejection(messages, target)
        except (ValueError, TypeError):
            errors["invalid_rejection_result"] += 1
    for missing in sorted(REQUIRED_TARGETS - counts.keys()):
        errors[f"missing_{missing}_targets"] += 1
    if not recovery:
        errors["missing_changed_action_after_rejection"] += 1
    if runs - finished:
        errors["source_runs_without_done_target"] += len(runs - finished)
    return {"ready": not errors, "records": len(records), "distinct_tasks": len(tasks),
            "target_kinds": dict(counts), "recovery_examples": recovery,
            "expected_prompt_sha256": prompt_hash(expected_system_prompt),
            "observed_prompt_sha256": sorted(systems), "observed_policies": sorted(policies),
            "errors": dict(errors),
            "scope": "alignment and minimum coverage only; not receipt/split/tokenizer or quality certification"}
