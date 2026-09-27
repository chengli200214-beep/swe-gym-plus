"""Rebuild next-action examples from qualified exports and real event receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from codeagentbench.runtime import AgentRuntime
from codeagentbench.adapters.action import parse_action
from codeagentbench.training.action_context import POLICY, compact_context, encode_next_action
from scripts.accelerate_campaign import seed_ok


def examples(row, events):
    if row.get("agent_status") != "completed" or row.get("evaluation_verdict") != "passed":
        raise ValueError("require normally completed, independently passed source")
    prompts = [e["content"] for e in events if e.get("type") == "prompt"]
    if prompts != [row["messages"][0]["content"]]:
        raise ValueError("source prompt and authoritative events disagree")
    history = [{"role": "system", "content": AgentRuntime._system_prompt()}, {"role": "user", "content": prompts[0]}]
    output, pending = [], None
    model_texts = [e["content"] for e in events if e.get("type") == "model"]
    if model_texts != [m["content"] for m in row["messages"] if m["role"] == "assistant"]:
        raise ValueError("source model turns and events disagree")
    for event in events:
        if event.get("type") == "model":
            if pending is not None:
                raise ValueError("model action has no real tool receipt")
            action = parse_action(event["content"])
            canonical = json.dumps({"command": action.command, "done": action.done, "message": action.message}, ensure_ascii=False, separators=(",", ":"))
            output.append({"task_id": row["task_id"], "run_id": row["run_id"], "action_index": len(output), "source_evaluation_verdict": "passed", "assistant_only_loss": True, "next_action_only_loss": True, "prompt_policy": POLICY, "messages": compact_context(history) + [{"role": "assistant", "content": canonical}]})
            history.append({"role": "assistant", "content": canonical})
            pending = None if action.done else action.command
        elif event.get("type") == "tool":
            if pending is None or event["intent"]["command"] != pending:
                raise ValueError("tool receipt does not match real model action")
            receipt = event["receipt"]
            history.append({"role": "user", "content": "Tool result:\n" + json.dumps({k: receipt[k] for k in ("exit_code", "stdout", "stderr", "timed_out")}, ensure_ascii=False)})
            pending = None
    if pending is not None or not output or not json.loads(output[-1]["messages"][-1]["content"])["done"]:
        raise ValueError("source lacks verified termination")
    return output


def prepare(source: Path, output: Path, split_path: Path, tokenizer, max_seq_len=8192):
    if output.exists():
        raise ValueError("preserve prior data; output already exists")
    split = json.loads(split_path.read_text())
    rows = [json.loads(s) for s in source.read_text().splitlines() if s]
    records, receipts, seen, lengths, target_lengths = [], [], set(), [], []
    for row in rows:
        if row["task_id"] in seen or not seed_ok(row, split["train"]):
            raise ValueError("source must be unique, blind, completed, passed and train-only")
        seen.add(row["task_id"])
        event_path = Path(row["events_path"])
        events = [json.loads(s) for s in event_path.read_text().splitlines() if s]
        converted = examples(row, events)
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
    audit = {"policy": POLICY, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "distinct_tasks": len(seen), "records": len(records), "done_examples": sum(json.loads(r["messages"][-1]["content"])["done"] for r in records), "total_tokens": sum(lengths), "max_tokens": max(lengths), "supervised_tokens": sum(target_lengths), "max_target_tokens": max(target_lengths), "target_tokens_above_768": sum(n > 768 for n in target_lengths), "truncated": 0, "rejected": [], "real_receipts_verified": True, "prior_assistant_loss_masked": True, "sources": receipts}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.source, args.output, args.split, tokenizer)), flush=True)


if __name__ == "__main__":
    main()
