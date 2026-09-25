"""Combine distinct, independently passed trajectory exports for SFT.

Usage: python scripts/prepare_passed_sft.py OUTPUT.jsonl EXPORT1.jsonl ...
The output is private experiment data; do not commit it to a public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def combine(output: Path, sources: list[Path]) -> list[dict]:
    if len(sources) < 5:
        raise ValueError("SFT pilot requires at least 5 distinct exports")
    rows: list[dict] = []
    seen_tasks: set[str] = set()
    seen_runs: set[str] = set()
    receipts: list[dict] = []
    for source in sources:
        content = source.read_bytes()
        lines = [line for line in content.decode("utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"expected one record in {source}, got {len(lines)}")
        row = json.loads(lines[0])
        task_id, run_id = row.get("task_id"), row.get("run_id")
        if row.get("evaluation_verdict") != "passed":
            raise ValueError(f"independent evaluation did not pass: {source}")
        if not task_id or not run_id or task_id in seen_tasks or run_id in seen_runs:
            raise ValueError(f"missing or duplicate task/run ID: {source}")
        if not row.get("assistant_only_loss") or not any(
            message.get("role") == "assistant" for message in row.get("messages", [])
        ):
            raise ValueError(f"missing assistant supervision: {source}")
        seen_tasks.add(task_id)
        seen_runs.add(run_id)
        rows.append(row)
        receipts.append({
            "source": str(source), "task_id": task_id, "run_id": run_id,
            "agent_status": row.get("agent_status"), "sha256": hashlib.sha256(content).hexdigest(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps({"records": len(rows), "sources": receipts}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    receipts = combine(args.output, args.sources)
    print(json.dumps({"output": str(args.output), "records": len(receipts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
