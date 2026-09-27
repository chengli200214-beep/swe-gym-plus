"""Independent, credential-free evidence verification; run after the pipeline.

No repair commands or API requests are executed. New JSON receipts are created
exclusively; existing experiment artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args()
    root, prior = args.root, args.prior
    assert not (root / "verification.json").exists(), "preserve existing verification"
    pipeline = read(root / "pipeline.receipt.json")
    assert pipeline["returncode"] == 0
    exp, old_exp = read(root / "experiment.json"), read(prior / "experiment.json")
    assert exp["action_data_mode"] == "last-action-v2"
    assert exp["evaluation_workers"] == 2
    assert exp["exports"] == old_exp["exports"]
    assert exp["config_sha256"] == old_exp["config_sha256"]
    assert exp["evaluation_limits"] == old_exp["evaluation_limits"]
    assert exp["model_path"] == old_exp["model_path"]
    for name in ("manifest", "split"):
        assert digest(args.pool / (name + ".json")) == exp[name + "_sha256"] == old_exp[name + "_sha256"]
    actual_source = {
        p.relative_to(args.repo).as_posix(): digest(p)
        for folder in ("src/codeagentbench", "scripts")
        for p in sorted((args.repo / folder).rglob("*.py"))
    }
    assert actual_source == exp["source_sha256"]
    for name, expected in old_exp["source_sha256"].items():
        assert digest(root / "source-before" / name) == expected
    setup = read(root / "setup.receipt.json")
    old_assets = {name: digest(prior / name) for name in setup["previous_artifact_sha256"]}
    assert old_assets == setup["previous_artifact_sha256"]
    assert read(root / "model-files.json")["files"] == read(prior / "model-files.json")["files"]
    for name, receipt in read(root / "model-files.json")["files"].items():
        model_file = Path(exp["model_path"]) / name
        assert model_file.stat().st_size == receipt["size"] and digest(model_file) == receipt["sha256"]
    environment_observations = lines(root / "inference-environments.jsonl")
    observed_models = set()
    for observation in environment_observations:
        for process in observation["processes"]:
            assert process["prompt_policy"] == "last-action-v2" and process["executor"] == "bwrap"
            assert process["max_new_tokens"] == "768" and not process["credential_names_present"]
            observed_models.add(process["model_path"])
    assert observed_models == {exp["model_path"], str(root / "checkpoint")}

    # Independently locate target boundaries with the actual tokenizer; do not
    # call the modified training encoder to verify its own reported totals.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp["model_path"], local_files_only=True, trust_remote_code=False)
    rows = lines(root / "data/actions.jsonl")
    audit = read(root / "data/actions.audit.json")
    assert digest(root / "data/actions.jsonl") == audit["output_sha256"]
    assert digest(root / "data/trajectories.jsonl") == audit["source_sha256"]
    split = read(args.pool / "split.json")
    task_ids = {row["task_id"] for row in rows}
    assert len(rows) == 250 and len(task_ids) == 20
    assert task_ids <= set(split["train"])
    assert task_ids.isdisjoint(split["dev"] + split["eval"])
    input_lengths, target_lengths, done, roles = [], [], 0, Counter()
    for row in rows:
        assert row["prompt_policy"] == "last-action-v2" and row["next_action_only_loss"]
        messages = row["messages"]
        roles[",".join(m["role"] for m in messages)] += 1
        assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
        assert messages[-1]["role"] == "assistant"
        before = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_ids = tokenizer(before, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        assert full_ids[:len(prefix_ids)] == prefix_ids
        assert len(prefix_ids) < len(full_ids) <= 8192
        input_lengths.append(len(full_ids))
        target_lengths.append(len(full_ids) - len(prefix_ids))
        target = json.loads(messages[-1]["content"])
        done += int(target["done"])
        if len(messages) > 3:
            assert messages[2]["role"] == "assistant" and messages[3]["role"] == "user"
            assert messages[3]["content"].startswith("Tool result:\n")
            observation = json.loads(messages[3]["content"].split("\n", 1)[1])
            assert {"exit_code", "stdout", "stderr", "timed_out"} <= observation.keys()
            assert len(observation["stdout"]) <= 4000 and len(observation["stderr"]) <= 4000
    recomputed = {
        "records": len(rows), "distinct_tasks": len(task_ids), "done_examples": done,
        "total_tokens": sum(input_lengths), "max_tokens": max(input_lengths),
        "supervised_tokens": sum(target_lengths), "max_target_tokens": max(target_lengths),
        "target_tokens_above_768": sum(n > 768 for n in target_lengths),
    }
    assert all(audit[key] == value for key, value in recomputed.items())
    assert audit["truncated"] == 0 and not audit["rejected"] and done == 20
    assert set(s["task_id"] for s in audit["sources"]) == task_ids
    trajectories = lines(root / "data/trajectories.jsonl")
    assert len(trajectories) == len(task_ids)
    for row in trajectories:
        assert row["agent_status"] == "completed" and row["evaluation_verdict"] == "passed"
        expected = next(s for s in audit["sources"] if s["task_id"] == row["task_id"])
        assert digest(Path(row["events_path"])) == expected["events_sha256"]

    training = read(root / "checkpoint/train_metrics.json")
    assert read(root / "logs/training.receipt.json")["returncode"] == 0
    assert training["global_step"] == 32 and training["epochs"] == 1 and training["seed"] == 42
    assert training["encoded_examples"] == training["train_records"] == len(rows)
    assert training["dropped_no_supervision"] == training["truncated_examples"] == 0
    import torch
    from safetensors.torch import load_file
    weights = load_file(str(root / "checkpoint/adapter_model.safetensors"), device="cpu")
    assert weights and all(bool(torch.isfinite(t).all()) for t in weights.values())
    lora_b = {k: v for k, v in weights.items() if "lora_B" in k}
    nonzero_b = sum(bool(t.ne(0).any()) for t in lora_b.values())
    assert nonzero_b == len(lora_b) > 0

    pairs = read(root / "pairs.json")
    assert [r["task_id"] for r in pairs] == split["dev"] + split["eval"]
    report = read(root / "report.json")
    verified_pairs, exclusions, aggregates = [], [], {}
    for pair in pairs:
        task = pair["task_id"]
        assert pair["partition"] == ("dev" if task in split["dev"] else "eval")
        quality = read(root / "quality" / (task + ".json"))
        assert quality["task_id"] == task
        if not quality["admitted"]:
            assert pair["status"] == "not_admitted"
            exclusions.append({"task_id": task, "partition": pair["partition"], "reason": quality["reason"],
                               "controls": [{k: c[k] for k in ("name", "exit_code", "expected_pass", "observed_pass", "reason")} for c in quality["controls"]]})
            continue
        assert pair["status"] == "evaluated"
        assert len(quality["controls"]) == 2 and all(c["expected_pass"] == c["observed_pass"] and not c["reason"] for c in quality["controls"])
        item = {"task_id": task, "partition": pair["partition"]}
        for label in ("base", "sft"):
            run_id = task + "-" + label
            run_dir = root / "paired/runs" / run_id
            config = read(run_dir / "run.json")
            assert config["task_id"] == task and config["run_id"] == run_id
            for name, value in exp["evaluation_limits"].items():
                if name != "max_new_tokens":
                    assert config["config"][name] == value
            events = lines(run_dir / "events.jsonl")
            evaluations = [e for e in events if e["type"] == "evaluation"]
            assert len(evaluations) == 1
            evaluation = evaluations[0]
            receipt = read(root / "logs" / (run_id + "-evaluate.receipt.json"))
            assert receipt["returncode"] == (0 if evaluation["verdict"] == "passed" else 1)
            if evaluation["verdict"] == "passed":
                assert evaluation["exit_code"] == 0
            summary = read(run_dir / "summary.json")
            assert pair[label]["verdict"] == evaluation["verdict"]
            assert pair[label]["agent_status"] == summary["status"]
            assert pair[label]["patch_present"] == bool(summary["diff"])
            item[label] = {"verdict": evaluation["verdict"], "agent_status": summary["status"],
                           "patch_present": bool(summary["diff"]), "reason": summary["failure_reason"],
                           "steps": summary["steps"], "model_events": sum(e["type"] == "model" for e in events),
                           "tool_events": sum(e["type"] == "tool" for e in events)}
        verified_pairs.append(item)
    for partition in ("dev", "eval"):
        items = [r for r in verified_pairs if r["partition"] == partition]
        wins = sum(r["sft"]["verdict"] == "passed" and r["base"]["verdict"] != "passed" for r in items)
        losses = sum(r["base"]["verdict"] == "passed" and r["sft"]["verdict"] != "passed" for r in items)
        n = wins + losses
        stats = {"frozen_tasks": len(split[partition]), "admitted": len(items), "not_admitted": len(split[partition]) - len(items),
                 "base_passed": sum(r["base"]["verdict"] == "passed" for r in items),
                 "sft_passed": sum(r["sft"]["verdict"] == "passed" for r in items),
                 "sft_only_passed": wins, "base_only_passed": losses,
                 "paired_exact_p": min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0}
        assert all(report["partitions"][partition][k] == v for k, v in stats.items())
        stats["diagnostics"] = {label: {
            "statuses": dict(Counter(r[label]["agent_status"] for r in items)),
            "reasons": dict(Counter(r[label]["reason"] for r in items)),
            "patches": sum(r[label]["patch_present"] for r in items),
            "steps": [r[label]["steps"] for r in items],
        } for label in ("base", "sft")}
        aggregates[partition] = stats
    verification = {"status": "verified_completed", "executed_at": datetime.now(timezone.utc).isoformat(),
                    "pipeline": pipeline, "training_data_recomputed": recomputed, "role_shapes": dict(roles),
                    "weight_tensors_finite": len(weights), "lora_b_tensors": len(lora_b), "nonzero_lora_b_tensors": nonzero_b,
                    "source_identity_matches": True, "prior_assets_unchanged": old_assets,
                    "method_sha256": digest(Path(__file__)), "inference_environment_observations": environment_observations,
                    "training": {k: v for k, v in training.items() if k != "trainer_metrics"},
                    "partitions": aggregates, "pairs": verified_pairs, "exclusions": exclusions,
                    "prior_partitions": read(prior / "report.json")["partitions"],
                    "limitations": ["Moto-only single-seed diagnostic pilot, not full SWE-Gym", "Previously inspected heldout tasks are reused, not a fresh sealed test", "Both groups use changed context; cross-round changes cannot be attributed only to SFT", "Only most recent interaction is retained", "One training target exceeds shared 768-token inference limit", "No independent external backup"]}
    save_new(root / "verification.json", verification)
    files = []
    for directory, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in {"workspace", ".repo_cache", "contracts-temp", "__pycache__"})
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_symlink():
                files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": digest(path)})
    files.sort(key=lambda item: item["path"])
    save_new(root / "final-assets.verified.json", {"status": "inventory_not_backup", "files": files, "verification_sha256": digest(root / "verification.json")})
    print("ROUND2_VERIFIED_JSON " + json.dumps({"status": verification["status"], "partitions": aggregates,
          "finite_tensors": len(weights), "nonzero_lora_b": nonzero_b, "file_count": len(files),
          "sha256": {name: digest(root / name) for name in ("report.json", "pairs.json", "verification.json", "final-assets.verified.json", "pipeline.receipt.json", "checkpoint/adapter_model.safetensors")}}), flush=True)


if __name__ == "__main__":
    main()
