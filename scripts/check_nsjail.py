"""Bounded no-key admission, run on the actual AutoDL host before model code."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from codeagentbench.models import ToolIntent
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.nsjail import NsjailSandbox
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.tasks.manifest import load_manifest

ROOT = Path(__file__).parents[1]
NEGATIVE_CHECK = r'''
import os, socket, ctypes, errno, tempfile
assert os.getuid() >= 100000 and os.getgid() == os.getuid()
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(39, 0, 0, 0, 0) == 1
with tempfile.TemporaryFile() as temporary:
    temporary.write(b'tempfile-smoke')
assert not os.path.exists('/root/autodl-tmp/models')
assert not os.path.exists('/proc') or not os.listdir('/proc')
assert not os.path.exists('/sys') or not os.listdir('/sys')
assert not os.path.exists('/dev/nvidia0')
assert all(not any(marker in key.upper() for marker in ('DEEPSEEK', 'PASSWORD', 'TOKEN')) for key in os.environ)
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    try: socket.socket(family)
    except PermissionError: pass
    else: raise AssertionError('network socket permitted')
for operation in (lambda: os.setuid(0), lambda: os.rename('.git', 'renamed-git'),
                  lambda: open('.git/config', 'a'), lambda: os.mkfifo('fifo'), lambda: os.setsid()):
    try: operation()
    except PermissionError: pass
    else: raise AssertionError('privileged/metadata operation permitted')
os.symlink('/root/autodl-tmp/models', 'outside-link')
assert not os.path.exists('outside-link')
os.unlink('outside-link')
open('sandbox-smoke.py','w').write('VALUE = 1\n')
assert open('sandbox-smoke.py').read() == 'VALUE = 1\n'
print('NEGATIVE_ADMISSION_PASSED', os.getuid())
'''


def _live_uid_processes(uid: int) -> list[int]:
    result = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            text = path.read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if f"Uid:\t{uid}\t" in text and "State:\tZ" not in text:
            result.append(int(path.parent.name))
    return result


def check_admission(artifact_root: Path) -> dict:
    assert os.getenv("CODEAGENTBENCH_EXECUTOR") == "nsjail"
    task = load_manifest(ROOT / "data/manifests/demo.json").tasks[0]
    workspace = WorkspaceManager(artifact_root).create(task, "negative-admission")
    executor = BashExecutor(workspace.path, backend="nsjail", output_limit=20000)
    receipt = executor.execute(ToolIntent("smoke", "set -e\npython - <<'PY'\n" + NEGATIVE_CHECK + "\nPY\nsed -i 's/VALUE = 1/VALUE = 2/' sandbox-smoke.py\npython -c \"assert 'VALUE = 2' in open('sandbox-smoke.py').read()\"\ngit status --porcelain", str(workspace.path), 30))
    assert receipt.exit_code == 0 and "NEGATIVE_ADMISSION_PASSED" in receipt.stdout and not receipt.stderr, asdict(receipt)
    output = executor.execute(ToolIntent("output", "python -c \"print('x'*2000000)\"", str(workspace.path), 10))
    assert output.status == "output_limit", asdict(output)
    timeout = executor.execute(ToolIntent("timeout", "sleep 60 & wait", str(workspace.path), 1))
    assert timeout.timed_out, asdict(timeout)
    background = executor.execute(ToolIntent("background", "sleep 60 & echo spawned", str(workspace.path), 5))
    assert background.exit_code == 0, asdict(background)
    uid = NsjailSandbox(workspace.path).uid
    time.sleep(0.2)
    assert not _live_uid_processes(uid), "guest descendants survived"
    other = WorkspaceManager(artifact_root).create(task, "other-uid")
    assert NsjailSandbox(other.path).uid != uid, "UID reused across runs"
    result = {"passed": True, "uid": uid, "checks": ["python/git", "filesystem", "network", "uid", "no_new_privs", ".git", "output_limit", "timeout", "descendant_cleanup", "unique_uid"]}
    (artifact_root / "admission.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(check_admission(Path(sys.argv[1])), indent=2))
