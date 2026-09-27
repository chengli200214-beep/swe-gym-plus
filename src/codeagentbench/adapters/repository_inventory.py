"""A fixed, bounded filename observation; never reads task/source contents."""
from __future__ import annotations

import json

from codeagentbench.adapters.file_tools import fixed_python_command

VERSION = "directory-inventory-v1"
PROGRAM = r'''
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
excluded = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache'}
result = {'format': 'directory-inventory-v1', 'root_entries': [],
          'child_directories': {}, 'truncated': False}
def entries(path):
    selected = []
    with os.scandir(path) as stream:
        for entry in stream:
            if entry.name in excluded:
                continue
            if len(selected) == 512:
                result['truncated'] = True
                break
            selected.append(entry)
    return sorted(selected, key=lambda e: e.name)
def fits():
    return len(json.dumps(result, ensure_ascii=False).encode('utf-8')) <= 3600
roots = entries('.')
for entry in roots:
    result['root_entries'].append(entry.name)
    if not fits():
        result['root_entries'].pop()
        result['truncated'] = True
        break
for entry in roots:
    if entry.name not in result['root_entries'] or not entry.is_dir(follow_symlinks=False):
        continue
    result['child_directories'][entry.name] = []
    if not fits():
        del result['child_directories'][entry.name]
        result['truncated'] = True
        break
    for child in entries(entry.path):
        if not child.is_dir(follow_symlinks=False):
            continue
        children = result['child_directories'][entry.name]
        children.append(child.name)
        if not fits():
            children.pop()
            result['truncated'] = True
            break
print(json.dumps(result, ensure_ascii=False))
'''


def inventory_command() -> str:
    return fixed_python_command(PROGRAM, {})


def inventory_packet(receipt: dict) -> dict:
    if receipt.get("status") != "completed" or receipt["exit_code"] != 0 or receipt["timed_out"]:
        raise ValueError("repository inventory did not complete successfully")
    raw = receipt["stdout"]
    if len(raw.encode("utf-8")) > 3800:
        raise ValueError("repository inventory exceeded its observation limit")
    result = json.loads(raw)
    if (not isinstance(result, dict) or set(result) != {"format", "root_entries", "child_directories", "truncated"}
            or result["format"] != VERSION or type(result["truncated"]) is not bool
            or not isinstance(result["root_entries"], list) or not isinstance(result["child_directories"], dict)):
        raise ValueError("invalid repository inventory packet")
    def names(items):
        return isinstance(items, list) and all(isinstance(s, str) and s and s not in {".", ".."}
            and "/" not in s and "\x00" not in s for s in items) and len(set(items)) == len(items)
    if not names(result["root_entries"]) or any(k not in result["root_entries"] or not names(v)
            for k, v in result["child_directories"].items()):
        raise ValueError("invalid repository inventory names")
    return result
