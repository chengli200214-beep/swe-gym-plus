"""Derive a versioned execution manifest from real isolated test collection.

    Original task IDs, patches and labels remain unchanged and inspectable.
    No gold patch is applied for collection; no model is invoked here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import tempfile
from dataclasses import replace
from pathlib import Path

from codeagentbench.adapters.file_tools import fixed_python_command, validate_source_path
from codeagentbench.models import ToolIntent
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.tasks.manifest import Manifest, load_manifest, save_manifest
from codeagentbench.tasks.quality import _apply_patch
from codeagentbench.verification.selectors import resolve_selectors

_COLLECT = r'''
import base64,json,sys,pytest
data=json.loads(base64.b64decode(sys.argv[1]))
class Capture:
    def pytest_collection_finish(self,session):
        print('CAB_SELECTOR_NODES='+json.dumps([item.nodeid for item in session.items]))
sys.exit(pytest.main(['--collect-only','-q','-s',*data['files']],plugins=[Capture()]))
'''


def derive(source: Path, cache: Path, output: Path):
    if selected_backend() != "nsjail" or output.exists():
        raise ValueError("require the existing sandbox and a new output version")
    original = load_manifest(source)
    tasks, receipts = [], []
    for task in original.tasks:
        labels = list(dict.fromkeys((*task.eval_spec.fail_to_pass, *task.eval_spec.pass_to_pass)))
        files = sorted({validate_source_path(n.split("::", 1)[0]) for n in labels})
        with tempfile.TemporaryDirectory(prefix="cab-selector-") as temp:
            workspace = WorkspaceManager(temp, cache_root=cache).create(task, "collect")
            ok, error = _apply_patch(workspace.path, task.eval_spec.test_patch)
            if not ok:
                raise ValueError("sealed test patch collection preparation failed: " + error)
            command = fixed_python_command(_COLLECT, {"files": files})
            receipt = BashExecutor(workspace.path, backend="nsjail", output_limit=200000).execute(
                ToolIntent("selector-collection", command, str(workspace.path), 120))
            records = re.findall(r"^CAB_SELECTOR_NODES=(.*)$", receipt.stdout, flags=re.MULTILINE)
            if receipt.exit_code != 0 or receipt.timed_out or len(records) != 1:
                raise ValueError("isolated collection failed: " + receipt.stdout[-3000:] + receipt.stderr[-3000:])
            collected = json.loads(records[0])
            mapping = resolve_selectors(labels, collected)
        resolved = list(dict.fromkeys(mapping.values()))
        test_command = shlex.join(["python", "-m", "pytest", "-q", *resolved])
        changed = {k: v for k, v in mapping.items() if k != v}
        identity = {"task_id": task.instance_id, "mapping": mapping,
                    "changed_labels": changed, "required_labels": len(labels),
                    "collected_nodes": collected, "no_gold_patch_used": True,
                    "original_command_sha256": hashlib.sha256(task.eval_spec.test_command.encode()).hexdigest(),
                    "resolved_command_sha256": hashlib.sha256(test_command.encode()).hexdigest(),
                    "test_patch_sha256": hashlib.sha256(task.eval_spec.test_patch.encode()).hexdigest()}
        receipts.append(identity)
        tasks.append(replace(task, eval_spec=replace(task.eval_spec, test_command=test_command),
                             metadata={**task.metadata, "evaluation_selector_mapping": changed}))
        print(json.dumps({"task_id": task.instance_id, "required": len(labels), "repaired": len(changed)}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    save_manifest(Manifest(original.dataset, original.revision, tuple(tasks)), output / "manifest.json")
    audit = {"source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
             "same_task_ids": [t.instance_id for t in tasks], "no_requirement_dropped": True,
             "scope": "selector-equivalent execution adapter, not the unmodified upstream command",
             "tasks": receipts}
    (output / "selector-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = derive(args.source, args.cache, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "tasks"}), flush=True)


if __name__ == "__main__":
    main()
