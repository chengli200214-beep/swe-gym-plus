"""Shared boundary for the two fixed, data-only sandbox file tools."""
from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
from pathlib import PurePosixPath


def validate_source_path(path: object) -> str:
    if isinstance(path, str) and path.endswith("/"):
        raise ValueError("path must name one source file, not a directory; list directory entries with a shell command")
    if (not isinstance(path, str) or not path or len(path) > 512
            or PurePosixPath(path).is_absolute() or "\\" in path or ":" in path
            or any(ord(c) < 32 for c in path)
            or any(p in {"", ".", ".."} or p.casefold() == ".git" for p in path.split("/"))):
        raise ValueError("path must be a relative POSIX source path, not Git metadata")
    return path


def fixed_python_command(program: str, data: dict) -> str:
    """Only trusted program bytes are executable; model fields remain JSON data."""
    encoded = base64.b64encode(program.encode()).decode()
    payload = base64.b64encode(json.dumps(data, ensure_ascii=False).encode()).decode()
    # Keep the traceback's -c source short so its actual error is not buried
    # behind kilobytes of encoded program text when observations are bounded.
    loader = "import base64,sys;exec(base64.b64decode(sys.argv.pop(1)))"
    args = ["python", "-c", loader, encoded, payload]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


# Both tools run this inside the sandbox, never on the model/inference host.
# No repository module is imported. The caller validates the relative path.
READ_SOURCE_PROGRAM = r'''
import base64, hashlib, json, os, stat, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
proposal = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
path = Path(proposal['path'])
root = Path.cwd().resolve()
for part in (path, *path.parents):
    if part.is_symlink():
        raise ValueError('file tool rejects symlink components')
if not path.resolve().is_relative_to(root):
    raise ValueError('file tool path escapes workspace')
info = path.stat()
if not stat.S_ISREG(info.st_mode) or info.st_size > 1048576 or info.st_nlink != 1:
    raise ValueError('file tool requires a regular single-link file of at most 1 MiB')
raw = path.read_bytes()
if len(raw) > 1048576:
    raise ValueError('file grew beyond 1 MiB')
text = raw.decode('utf-8')
'''.strip()
