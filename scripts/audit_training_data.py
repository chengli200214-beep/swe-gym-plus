"""Token-level admission: never silently truncate the task or target action."""
import argparse
import hashlib
import json
from pathlib import Path

from codeagentbench.train_sft import encode_example, normalise_messages, IGNORE_INDEX


def main():
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=8192)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists; preserve previous audit")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    split = json.loads(args.split.read_text())
    rows = [json.loads(s) for s in args.source.read_text().splitlines() if s.strip()]
    kept, rejected, seen, lengths, supervised = [], [], set(), [], []
    for row in rows:
        if row["task_id"] not in split["train"] or row["source_evaluation_verdict"] != "passed":
            raise ValueError("training source is outside admitted train data")
        messages = normalise_messages(row["messages"])
        fingerprint = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        reason = "duplicate" if fingerprint in seen else "too_long" if count > args.max_length else ""
        example = encode_example(tokenizer, messages, args.max_length) if not reason else None
        if not reason and (not example or not any(v != IGNORE_INDEX for v in example["labels"])):
            reason = "no_supervision"
        if reason:
            rejected.append({"task_id": row["task_id"], "action_index": row["action_index"], "reason": reason, "tokens": count})
            continue
        seen.add(fingerprint)
        kept.append(row)
        lengths.append(count)
        supervised.append(sum(v != IGNORE_INDEX for v in example["labels"]))
    if not kept:
        raise ValueError("all examples rejected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    receipt = {"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
               "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(), "records": len(kept), "distinct_tasks": len({r["task_id"] for r in kept}),
               "max_tokens": max(lengths), "total_tokens": sum(lengths), "supervised_tokens": sum(supervised), "truncated": 0,
               "done_examples": sum(json.loads(r["messages"][-1]["content"])["done"] for r in kept), "rejected": rejected}
    args.output.with_suffix(".audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
