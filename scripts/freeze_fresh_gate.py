"""Freeze three unexposed development tasks before admission/model outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from codeagentbench.adapters.swe_gym import normalize_swe_gym_row
from codeagentbench.tasks.manifest import Manifest, save_manifest
from scripts.freeze_expanded_pool import REVISION

SEED = "cab-source-observation-20260927-v1:"
SOURCE_SHA256 = "60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97"
_ID = re.compile(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+\Z")


def mentioned_ids(value):
    """Conservatively union identity fields and split lists, not raw source text."""
    if isinstance(value, str):
        return {value} if _ID.fullmatch(value) else set()
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return set().union(*(mentioned_ids(v) for v in value))
    return set()


def issue_key(row):
    return row["repo"], " ".join(row["problem_statement"].split())


def select(rows, known):
    required = ("instance_id", "repo", "base_commit", "problem_statement")
    missing = {k: sum(not isinstance(r.get(k), str) or not r[k].strip() for r in rows) for k in required}
    counts = Counter(r.get("instance_id") for r in rows)
    duplicates = sum(n - 1 for n in counts.values())
    if any(missing.values()) or duplicates:
        raise ValueError(f"invalid source grain/required fields: missing={missing}, duplicates={duplicates}")
    exposed = [r for r in rows if r["instance_id"] in known]
    known_issues = {issue_key(r) for r in exposed}
    known_commits = {(r["repo"], r["base_commit"]) for r in exposed}
    eligible = []
    for row in rows:
        if (row["repo"] != "getmoto/moto" or row["instance_id"] in known
                or issue_key(row) in known_issues or (row["repo"], row["base_commit"]) in known_commits):
            continue
        fail, passed = row.get("FAIL_TO_PASS"), row.get("PASS_TO_PASS")
        if not isinstance(fail, list) or not isinstance(passed, list):
            raise ValueError("source test identities must be lists, never coerce missing labels")
        if 1 <= len(fail) <= 5 and len(passed) <= 30:
            eligible.append(row)
    eligible.sort(key=lambda r: hashlib.sha256((SEED + r["instance_id"]).encode()).hexdigest())
    selected, issues, commits = [], set(), set()
    for row in eligible:
        key, commit = issue_key(row), (row["repo"], row["base_commit"])
        if key in issues or commit in commits:
            continue
        selected.append(row)
        issues.add(key)
        commits.add(commit)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError(f"need three fresh source groups; eligible={len(eligible)}")
    profile = {"rows": len(rows), "distinct_task_ids": len(counts), "duplicate_task_ids": duplicates,
               "required_missing": missing, "known_exposure_ids": len(known),
               "known_ids_found_in_source": len(exposed), "eligible_fresh": len(eligible)}
    return selected, profile


def freeze(source: Path, prior_paths: list[Path], output: Path):
    if output.exists():
        raise ValueError("preserve the existing frozen task partition")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("dataset bytes differ from the inspected pinned upstream snapshot")
    prior = [{"label": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
              "value": json.loads(p.read_text(encoding="utf-8"))} for p in prior_paths]
    known = set().union(*(mentioned_ids(p["value"]) for p in prior))
    if not known:
        raise ValueError("must supply historical exposure/split evidence")
    import pyarrow.parquet as pq
    rows = pq.read_table(source).to_pylist()
    selected, profile = select(rows, known)
    tasks = tuple(normalize_swe_gym_row(r, split="dev") for r in selected)
    output.mkdir(parents=True, exist_ok=False)
    save_manifest(Manifest("SWE-Gym", REVISION, tasks), output / "manifest.json")
    receipt = {"executed_at": datetime.now(timezone.utc).isoformat(), "source_revision": REVISION,
               "source_sha256": source_hash, "selection_seed": SEED, "profile": profile,
               "tasks": [t.instance_id for t in tasks], "excluded_ids": sorted(known),
               "prior_sources": [{k: v for k, v in p.items() if k != "value"} for p in prior],
               "rule": "Moto; 1<=FAIL_TO_PASS count<=5; PASS_TO_PASS count<=30; exclude all known IDs, matching issue text and base snapshots; hash order; three distinct issue/base groups",
               "no_gold_content_or_model_outcome_selection": True,
               "scope": "Development pilot, not an untouched final evaluation set or full SWE-Gym estimate",
               "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--prior", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(freeze(args.source, args.prior, args.output)), flush=True)


if __name__ == "__main__":
    main()
