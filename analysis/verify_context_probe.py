"""Independently verify the completed three-task context intervention probe."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root, repo = args.root, args.repo
    identity, report, pairs = [read(root / name) for name in ("experiment.json", "report.json", "pairs.json")]
    assert read(root.parent / (root.name + "-setup") / "pipeline.receipt.json")["returncode"] == 0
    assert [p["task_id"] for p in pairs] == identity["tasks"]
    assert len(pairs) == 3 and report["status"] == "completed"
    assert all(sha(repo / name) == value for name, value in identity["source_sha256"].items())
    assert sha(Path(identity["model_path"]) / "adapter_model.safetensors") == identity["adapter_sha256"]
    prior = Path(identity["model_path"]).parent
    assert sha(prior / "report.json") == "ea037488146e08cbaeb8f94c9c0a8f22825b6d2aa4f54ab9330c91561b1a630e"
    assert sha(prior / "verification.json") == "9b9f3663266248a898dba6a25b11a28e478fb843c597fa2d2772d971750433c4"
    verified, groups = [], {}
    for pair in pairs:
        for policy in identity["policies"]:
            run_id = pair["task_id"] + "-" + policy
            directory = root / "paired/runs" / run_id
            summary, config = read(directory / "summary.json"), read(directory / "run.json")["config"]
            assert config["temperature"] == identity["temperature"] and config["seed"] == identity["seed"]
            assert all(config[k] == v for k, v in identity["limits"].items() if k != "max_new_tokens")
            events, trace = lines(directory / "events.jsonl"), lines(root / "traces" / (run_id + ".jsonl"))
            models = [e for e in events if e["type"] == "model"]
            inputs = [e for e in trace if e["type"] == "input"]
            outputs = [e for e in trace if e["type"] == "output"]
            assert len(models) == len(inputs) == len(outputs)
            evaluations = [e for e in events if e["type"] == "evaluation"]
            assert len(evaluations) == 1
            passed = evaluations[0]["verdict"] == "passed"
            assert read(root / "logs" / (run_id + ".receipt.json"))["returncode"] == 0
            assert read(root / "logs" / (run_id + "-evaluate.receipt.json"))["returncode"] == (0 if passed else 1)
            expected = pair[policy]
            assert expected["agent_status"] == summary["status"]
            assert expected["patch_present"] == bool(summary["diff"])
            assert expected["verdict"] == evaluations[0]["verdict"]
            calls, signature, streak, delivered, changed = 0, None, 0, 0, 0
            for event in events:
                if event["type"] == "tool":
                    current = (event["intent"]["command"], event["intent"]["pre_digest"])
                    streak = streak + 1 if current == signature else 1
                    signature = current
                elif event["type"] == "model":
                    inp, out = inputs[calls], outputs[calls]
                    calls += 1
                    assert inp["call"] == out["call"] == calls
                    assert inp["policy"] == policy and inp["temperature"] == identity["temperature"]
                    assert out["prompt_tokens"] == event["prompt_tokens"]
                    assert out["completion_tokens"] == event["completion_tokens"]
                    view = inp["messages"]
                    assert hashlib.sha256(json.dumps(view, ensure_ascii=False).encode()).hexdigest() == inp["context_sha256"]
                    assert sum(m["role"] == "assistant" for m in view) <= (4 if policy == "recent-history-v3" else 1)
                    last = max((i for i, m in enumerate(view) if m["content"].startswith("Tool result:\n")), default=1)
                    current_warnings = [m["content"] for m in view[last + 1:]
                                        if m["role"] == "user" and m["content"].startswith(("Harness warning:", "Harness checkpoint:"))]
                    assert current_warnings == inp["latest_interventions"]
                    has_repeat = any(w.startswith("Harness warning: the same command") for w in current_warnings)
                    if streak == 2:
                        assert has_repeat == (policy == "recent-history-v3")
                    if has_repeat:
                        delivered += 1
                        action = out["action"]
                        changed += bool(action and (action["done"] or action["command"] != signature[0]))
            verified.append({"task": pair["task_id"], "policy": policy, "passed": passed,
                             "patch": bool(summary["diff"]), "status": summary["status"],
                             "repeat_exit": "identical command repeated" in summary["failure_reason"],
                             "reason": summary["failure_reason"], "warnings_delivered": delivered,
                             "changed_after_warning": changed, "steps": summary["steps"],
                             "model_calls": len(models),
                             "output_limit_hits": sum(e["completion_tokens"] >= identity["limits"]["max_new_tokens"] for e in models),
                             "truncated_action_repairs": sum((o.get("action") or {}).get("message") == "parsed truncated JSON command" for o in outputs),
                             "tool_nonzero_exits": sum(e["receipt"]["exit_code"] != 0 for e in events if e["type"] == "tool")})
    for policy in identity["policies"]:
        rows = [r for r in verified if r["policy"] == policy]
        counts = {"tasks": len(rows), "repeat_exits": sum(r["repeat_exit"] for r in rows),
                  "patches": sum(r["patch"] for r in rows), "passed": sum(r["passed"] for r in rows), "blocked": 0}
        assert report["groups"][policy] == counts
        groups[policy] = counts | {"warnings_delivered": sum(r["warnings_delivered"] for r in rows),
                                  "changed_after_warning": sum(r["changed_after_warning"] for r in rows),
                                  "model_calls": sum(r["model_calls"] for r in rows),
                                  "output_limit_hits": sum(r["output_limit_hits"] for r in rows),
                                  "truncated_action_repairs": sum(r["truncated_action_repairs"] for r in rows),
                                  "tool_nonzero_exits": sum(r["tool_nonzero_exits"] for r in rows),
                                  "failure_reasons": dict(Counter(r["reason"] for r in rows))}
    result = {"status": "verified_completed", "groups": groups, "runs": verified,
              "source_identity_matches": True, "prior_assets_unchanged": True,
              "report_sha256": sha(root / "report.json"), "method_sha256": sha(Path(__file__)),
              "limitations": report["limitations"]}
    with (root / "verification.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print("CONTEXT_PROBE_VERIFIED_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
