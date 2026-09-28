"""Rebuild next-action examples from qualified exports and real event receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

from codeagentbench.runtime import AgentRuntime, UNVERIFIED_FINISH_WARNING_PREFIX
from codeagentbench.adapters.action import parse_action
from codeagentbench.training.action_context import POLICY, encode_next_action
from codeagentbench.harness.context_history import POLICIES
from codeagentbench.harness.source_evidence import grounded_command, observe_source
from codeagentbench.harness.tool_observation import tool_observation
from codeagentbench.adapters.repository_inventory import inventory_command, inventory_packet
from codeagentbench.tasks.manifest import load_manifest
from scripts.accelerate_campaign import seed_ok


def _observed_finish_rejection(previous_context, next_event, prompt):
    """Accept only the actual next model input as evidence of a rejected done.

    Older runs did not journal a separate completion-rejection event. Their
    recorded next input still contains the runner's exact warning. We do not
    infer a rejected finish merely because another model action exists.
    """
    allowed = json.loads(prompt).get("allowed_test_command")
    recorded = next_event.get("context_messages")
    if not isinstance(allowed, str) or not allowed or not isinstance(recorded, list) or not recorded:
        return False
    warning = recorded[-1]
    return (isinstance(warning, dict) and warning.get("role") == "user"
            and isinstance(warning.get("content"), str)
            and warning["content"].startswith(UNVERIFIED_FINISH_WARNING_PREFIX)
            and warning["content"].endswith("Exact command: " + allowed)
            and warning not in previous_context)


def examples(row, events, *, policy=POLICY):
    if row.get("agent_status") != "completed" or row.get("evaluation_verdict") != "passed":
        raise ValueError("require normally completed, independently passed source")
    prompts = [e["content"] for e in events if e.get("type") == "prompt"]
    if prompts != [row["messages"][0]["content"]]:
        raise ValueError("source prompt and authoritative events disagree")
    inventory = json.loads(prompts[0]).get("repository_inventory")
    inventories = [e for e in events if e.get("type") == "harness_tool" and e.get("name") == "repository_inventory"]
    if inventory is not None:
        if len(inventories) != 1:
            raise ValueError("initial inventory lacks one real harness receipt")
        observed = inventories[0]
        if (observed["intent"]["command"] != inventory_command()
                or observed["receipt"]["command"] != observed["intent"]["command"]
                or inventory_packet(observed["receipt"]) != inventory):
            raise ValueError("initial inventory disagrees with the real observation")
    elif inventories:
        raise ValueError("real inventory is absent from the initial model context")
    output, pending, observations = [], None, []
    model_texts = [e["content"] for e in events if e.get("type") == "model"]
    if model_texts != [m["content"] for m in row["messages"] if m["role"] == "assistant"]:
        raise ValueError("source model turns and events disagree")
    for event in events:
        if event.get("type") == "model":
            if pending is not None:
                if (pending[0] is None or not pending[0].done
                        or not _observed_finish_rejection(pending[1], event, prompts[0])):
                    raise ValueError("model action has no real tool receipt or verified finish rejection")
                pending = None  # The earlier done was rejected; never supervise it.
            if event.get("context_contract") != "runtime-once-v1":
                raise ValueError("source lacks the runtime-once context contract")
            if event.get("prompt_policy") != policy:
                raise ValueError("source inference policy differs from export policy")
            recorded = event.get("context_messages")
            if (not isinstance(recorded, list) or len(recorded) < 2
                    or any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
                           or not isinstance(m.get("content"), str) for m in recorded)
                    or recorded[0] != {"role": "system", "content": AgentRuntime._system_prompt()}
                    or recorded[1] != {"role": "user", "content": prompts[0]}):
                raise ValueError("recorded inference context disagrees with deployed prompt or original task")
            try:
                action = parse_action(event["content"])
            except ValueError:
                action = None  # A real protocol_rejection must follow; never supervise malformed output.
            pending = (action, [dict(m) for m in recorded])
        elif event.get("type") == "tool":
            if pending is None or pending[0] is None or not pending[0].executable:
                raise ValueError("tool receipt has no executable model action")
            action, context = pending
            if event["intent"]["command"] != grounded_command(action, observations):
                raise ValueError("tool receipt does not match real model action")
            recorded_action = event["intent"].get("action_json")
            if recorded_action and json.loads(recorded_action) != action.to_dict():
                raise ValueError("journal action differs from actual model action")
            receipt = event["receipt"]
            if "command" in receipt and receipt["command"] != event["intent"]["command"]:
                raise ValueError("receipt command differs from journal intent")
            tool_observation(action, receipt)  # Reject invalid projected observations.
            observe_source(observations, action, receipt)
            output.append(_example(row, policy, context, action, len(output)))
            pending = None
        elif event.get("type") in {"protocol_rejection", "no_progress", "source_navigation_blocked"}:
            if pending is None:
                raise ValueError("rejection has no preceding model action")
            kind, action = event["type"], pending[0]
            if kind == "protocol_rejection" and action is not None:
                try:
                    grounded_command(action, observations)
                except ValueError:
                    pass
                else:
                    raise ValueError("protocol rejection has no invalid action or grounding error")
            if kind == "no_progress" and (action is None or not action.executable):
                raise ValueError("repeat block has no executable action")
            if kind == "source_navigation_blocked" and (action is None or action.read is None):
                raise ValueError("source navigation block has no read action")
            pending = None
    if pending is None or pending[0] is None or not pending[0].done:
        raise ValueError("source lacks verified termination")
    output.append(_example(row, policy, pending[1], pending[0], len(output)))
    return output


def _example(row, policy, context, action, index):
    canonical = json.dumps(action.to_dict(), ensure_ascii=False, separators=(",", ":"))
    return {"task_id": row["task_id"], "run_id": row["run_id"], "action_index": index,
            "source_evaluation_verdict": "passed", "assistant_only_loss": True,
            "next_action_only_loss": True, "prompt_policy": policy,
            "messages": context + [{"role": "assistant", "content": canonical}]}


def visible_test_commands_from_manifest(path: Path) -> dict[str, str]:
    """Opt in only to explicitly declared, bounded public-test commands.

    The operator must separately verify the path exists in the base checkout;
    this syntax/split check does not establish that provenance on its own.
    """
    commands = {}
    for task in load_manifest(path).tasks:
        command = task.metadata.get("agent_test_command")
        if not command:
            continue
        if task.split != "train" or not isinstance(command, str):
            raise ValueError("visible-test manifest must contain train tasks with string commands")
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise ValueError("invalid visible-test command") from exc
        if len(tokens) != 6 or tokens[:5] != ["python", "-m", "pytest", "-q", "-x"]:
            raise ValueError("visible-test command must be a bounded pytest path")
        parts = tokens[5].split("/")
        if (parts[0] != "tests" or len(parts) < 2
                or any(part in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                       for part in parts[1:])):
            raise ValueError("visible-test command must target a relative tests/ path")
        commands[task.instance_id] = command
    return commands


def prepare(source: Path, output: Path, split_path: Path, tokenizer, max_seq_len=8192, *,
            policy=POLICY, visible_test_manifest: Path | None = None):
    if output.exists():
        raise ValueError("preserve prior data; output already exists")
    split = json.loads(split_path.read_text())
    rows = [json.loads(s) for s in source.read_text().splitlines() if s]
    visible_test_commands = (visible_test_commands_from_manifest(visible_test_manifest)
                             if visible_test_manifest else None)
    records, receipts, seen, lengths, target_lengths = [], [], set(), [], []
    for row in rows:
        if row["task_id"] in seen or not seed_ok(row, split["train"], visible_test_commands=visible_test_commands):
            raise ValueError("source must be unique, blind, completed, passed and train-only")
        seen.add(row["task_id"])
        event_path = Path(row["events_path"])
        events = [json.loads(s) for s in event_path.read_text().splitlines() if s]
        converted = examples(row, events, policy=policy)
        for record in converted:
            encoded = encode_next_action(tokenizer, record["messages"], max_seq_len)
            lengths.append(len(encoded["input_ids"]))
            target_lengths.append(sum(x != -100 for x in encoded["labels"]))
        records.extend(converted)
        receipts.append({"task_id": row["task_id"], "events_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest(), "records": len(converted)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit = {"policy": policy, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(), "visible_test_manifest_sha256": hashlib.sha256(visible_test_manifest.read_bytes()).hexdigest() if visible_test_manifest else None, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "distinct_tasks": len(seen), "records": len(records), "done_examples": sum(json.loads(r["messages"][-1]["content"])["done"] for r in records), "total_tokens": sum(lengths), "max_tokens": max(lengths), "supervised_tokens": sum(target_lengths), "max_target_tokens": max(target_lengths), "target_tokens_above_768": sum(n > 768 for n in target_lengths), "truncated": 0, "rejected": [], "real_receipts_verified": True, "prior_assistant_loss_masked": True, "sources": receipts}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--policy", choices=sorted(POLICIES), default=POLICY)
    p.add_argument("--visible-test-manifest", type=Path,
                   help="explicit train-only manifest for public base-checkout visible tests")
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.source, args.output, args.split, tokenizer, policy=args.policy,
                             visible_test_manifest=args.visible_test_manifest)), flush=True)


if __name__ == "__main__":
    main()
