"""Validate edit data and lower it to a fixed program run by the sandbox.

No model-provided code is evaluated by the host. This is a tool, not a repair:
the model must locate the file and choose both exact source and replacement.
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class TextEdit:
    path: str
    before: str
    after: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_edit(payload: object) -> TextEdit:
    if not isinstance(payload, dict) or set(payload) != {"path", "before", "after"}:
        raise ValueError("edit requires exactly path/before/after")
    if any(not isinstance(v, str) or len(v) > 8000 for v in payload.values()):
        raise ValueError("edit fields must be strings of at most 8000 characters")
    path, before, after = (payload[k] for k in ("path", "before", "after"))
    parts = path.split("/")
    if (not path or len(path) > 512 or PurePosixPath(path).is_absolute()
            or "\\" in path or ":" in path or any(ord(c) < 32 for c in path)
            or any(p in {"", ".", "..", ".git"} for p in parts)):
        raise ValueError("edit path must be relative POSIX source path, not Git metadata")
    if not before or before == after:
        raise ValueError("edit before must be non-empty and differ from after")
    return TextEdit(path, before, after)


# Runs inside the same restricted tool environment as bash. Atomic replacement
# protects against partial writes; syntax is checked before opening an output.
# No repository imports or test code run as part of the adapter.
_PROGRAM = r'''
import base64, json, os, stat, sys, tempfile
from pathlib import Path
proposal = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
path = Path(proposal['path'])
root = Path.cwd().resolve()
for part in (path, *path.parents):
    if part.is_symlink():
        raise ValueError('edit rejects symlink components')
if not path.resolve().is_relative_to(root):
    raise ValueError('edit path escapes workspace')
info = path.stat()
if not stat.S_ISREG(info.st_mode) or info.st_size > 1048576 or info.st_nlink != 1:
    raise ValueError('edit requires a regular single-link file of at most 1 MiB')
raw = path.read_bytes()
text = raw.decode('utf-8')
count = text.count(proposal['before'])
if count != 1:
    raise ValueError('before must match exactly once in current source; observed ' + str(count))
updated = text.replace(proposal['before'], proposal['after'], 1)
if len(updated.encode('utf-8')) > 1048576:
    raise ValueError('updated file exceeds 1 MiB')
if path.suffix == '.py':
    compile(updated, str(path), 'exec')
if path.read_bytes() != raw:
    raise ValueError('source changed during edit; no write applied')
fd, name = tempfile.mkstemp(prefix='.agent-edit-', dir=str(path.parent))
try:
    with os.fdopen(fd, 'wb') as stream:
        stream.write(updated.encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(name, stat.S_IMODE(info.st_mode))
    os.replace(name, path)
finally:
    if os.path.exists(name):
        os.unlink(name)
print('EDIT_APPLIED ' + proposal['path'])
'''.strip()


def edit_command(edit: TextEdit) -> str:
    """Encode data, never interpolate it as shell/Python source."""
    edit = validate_edit(edit.to_dict())
    program = base64.b64encode(_PROGRAM.encode()).decode()
    data = base64.b64encode(json.dumps(edit.to_dict(), ensure_ascii=False).encode()).decode()
    loader = "import base64;exec(base64.b64decode('" + program + "'))"
    args = ["python", "-c", loader, data]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
