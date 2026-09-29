"""Replay audited train pre-edit/pre-finish contexts for diagnosis, not scoring.

This never executes the predicted action. It is an in-sample action-choice
probe, not an autonomous task result or an independent benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from codeagentbench.adapters.action import AgentAction, parse_action
from codeagentbench.adapters.model import LocalHFModel


def action_kind(action: AgentAction) -> str:
    if action.done:
        return "done"
    for kind in ("edit", "read", "search"):
        if getattr(action, kind) is not None:
            return kind
    return "command"


def replay_records(records: list[dict], model: LocalHFModel) -> list[dict]:
    results = []
    for record in records:
        messages = record["messages"]
        expected_action = parse_action(messages[-1]["content"])
        response = model.complete(messages[:-1], temperature=0)
        try:
            actual_action = parse_action(response.text)
            actual = action_kind(actual_action)
            before, after = expected_action.to_dict(), actual_action.to_dict()
            before.pop("message", None)
            after.pop("message", None)
            exact = before == after
        except ValueError:
            actual, exact = "invalid", False
        results.append({
            "task_id": record["task_id"],
            "run_id": record["run_id"],
            "action_index": record["action_index"],
            "expected": action_kind(expected_action),
            "actual": actual,
            "exact_action": exact,
            "completion_tokens": response.completion_tokens,
            "generated_text": response.text,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-jsonl", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("local replay must not inherit an API credential")
    if os.getenv("CODEAGENTBENCH_LOCAL_PROMPT_POLICY") != "recent-history-v3":
        raise RuntimeError("replay requires the frozen recent-history-v3 policy")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids = set(split["train"])
    records = [json.loads(line) for line in args.aligned_jsonl.read_text(encoding="utf-8").splitlines() if line]
    selected = []
    identities = set()
    for record in records:
        if record["task_id"] not in train_ids or record["source_evaluation_verdict"] != "passed":
            raise ValueError("replay source must be independently passed train data")
        identity = (record["task_id"], record["run_id"], record["action_index"])
        if identity in identities:
            raise ValueError("duplicate source action identity")
        identities.add(identity)
        if action_kind(parse_action(record["messages"][-1]["content"])) in {"edit", "done"}:
            selected.append(record)
    if not selected or args.max_new_tokens < 1:
        raise ValueError("no edit/done targets or invalid token bound")
    model = LocalHFModel(args.adapter, base_model_path=args.base, max_new_tokens=args.max_new_tokens)
    results = replay_records(selected, model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter((row["expected"], row["actual"]) for row in results)
    print(json.dumps({"scope": "in-sample diagnostic, not autonomous repair", "records": len(results),
                      "expected_actual": {f"{a}->{b}": n for (a, b), n in sorted(counts.items())},
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
