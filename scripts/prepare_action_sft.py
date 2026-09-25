"""Build protocol-aligned, one-action SFT examples from passed private trajectories.

Usage: PYTHONPATH=src python scripts/prepare_action_sft.py INPUT.jsonl OUTPUT.jsonl
Keep both files outside the public repository when they contain raw trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from codeagentbench.runtime import AgentRuntime, parse_action


def action_examples(row: dict[str, Any], *, system_prompt: str, context_chars: int = 800, include_done: bool = False, history: bool = False) -> list[dict[str, Any]]:
    if row.get("evaluation_verdict") != "passed":
        raise ValueError("source trajectory lacks a passed independent evaluation")
    task_id, run_id = row.get("task_id"), row.get("run_id")
    if not task_id or not run_id:
        raise ValueError("source trajectory lacks task_id or run_id")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "user":
        raise ValueError(f"{run_id}: expected an initial user task prompt")
    if context_chars < 0:
        raise ValueError("context_chars must be nonnegative")

    issue = str(messages[0].get("content", ""))
    if not issue:
        raise ValueError(f"{run_id}: empty task prompt")
    try:
        task_prompt = json.loads(issue)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{run_id}: task prompt is not auditable JSON") from exc
    if not isinstance(task_prompt, dict) or task_prompt.get("instance_id") != task_id:
        raise ValueError(f"{run_id}: task prompt does not match source task")
    if task_prompt.get("allowed_test_command"):
        raise ValueError(f"{run_id}: task prompt exposes an evaluator test command")
    if task_prompt.get("test_patch") or task_prompt.get("gold_patch"):
        raise ValueError(f"{run_id}: task prompt exposes evaluator-only patches")
    latest_tool = ""
    transcript: list[str] = []
    examples: list[dict[str, Any]] = []
    for message in messages[1:]:
        role = message.get("role")
        if role == "tool":
            latest_tool = str(message.get("content", ""))
            transcript.append("Tool result:\n" + latest_tool)
            continue
        if role != "assistant":
            continue
        action = parse_action(str(message.get("content", "")))
        # A five-task experiment should teach executable tool actions, not
        # premature completion. Final done turns can be trained separately
        # after there is a larger, fully completed set of trajectories.
        if action.done and not (include_done and row.get("agent_status") == "completed"):
            continue
        if not action.command and not action.done:
            raise ValueError(f"{run_id}: assistant turn lacks a command")
        user_prompt = issue
        if history and transcript and context_chars:
            user_prompt += "\n\nRecent interaction history (context only):\n" + "\n".join(transcript)[-context_chars:]
        elif latest_tool and context_chars:
            user_prompt += "\n\nLatest tool observation (tail):\n" + latest_tool[-context_chars:]
        canonical = json.dumps(
            {"command": action.command, "done": action.done, "message": action.message},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        examples.append({
            "task_id": task_id,
            "run_id": run_id,
            "action_index": len(examples),
            "source_evaluation_verdict": "passed",
            "assistant_only_loss": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": canonical},
            ],
        })
        transcript.append("Previous action:\n" + canonical)
    if not examples:
        raise ValueError(f"{run_id}: no executable assistant actions")
    return examples


def prepare(source: Path, output: Path, *, context_chars: int = 800, include_done: bool = False, history: bool = False) -> dict[str, Any]:
    content = source.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty source dataset")
    seen_tasks: set[str] = set()
    examples: list[dict[str, Any]] = []
    for row in rows:
        task_id = row.get("task_id")
        if task_id in seen_tasks:
            raise ValueError(f"duplicate source task: {task_id}")
        seen_tasks.add(task_id)
        examples.extend(action_examples(row, system_prompt=AgentRuntime._system_prompt(), context_chars=context_chars, include_done=include_done, history=history))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in examples), encoding="utf-8")
    receipt = {
        "source": str(source),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "distinct_tasks": len(seen_tasks),
        "action_examples": len(examples),
        "context_chars": context_chars,
        "include_done": include_done,
        "history": history,
        "done_examples": sum(json.loads(e["messages"][-1]["content"])["done"] for e in examples),
        "blind_prompt_checked": True,
    }
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-chars", type=int, default=800)
    parser.add_argument("--include-done", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, context_chars=args.context_chars, include_done=args.include_done, history=args.history), ensure_ascii=False))


if __name__ == "__main__":
    main()
