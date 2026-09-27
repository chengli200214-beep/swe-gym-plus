"""A bounded, versioned source observation chosen by the agent, not the host."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from codeagentbench.adapters.file_tools import (
    READ_SOURCE_PROGRAM, fixed_python_command, validate_source_path,
)


@dataclass(frozen=True)
class SourceRead:
    path: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        return asdict(self)


def validate_read(payload: object) -> SourceRead:
    if not isinstance(payload, dict) or set(payload) != {"path", "start_line", "end_line"}:
        raise ValueError("read requires exactly path/start_line/end_line")
    path = validate_source_path(payload["path"])
    start, end = payload["start_line"], payload["end_line"]
    if type(start) is not int or type(end) is not int or not 1 <= start <= end < start + 80:
        raise ValueError("read needs positive integer lines covering at most 80 lines")
    return SourceRead(path, start, end)


_PROGRAM = READ_SOURCE_PROGRAM + r'''

lines = text.splitlines(keepends=True)
start = proposal['start_line']
stop = min(proposal['end_line'], len(lines))
if start > len(lines):
    raise ValueError('start_line exceeds file length ' + str(len(lines)))
result = {'path': proposal['path'], 'sha256': hashlib.sha256(raw).hexdigest(),
          'start_line': start, 'end_line': start - 1, 'text': '', 'next_line': None}
for number in range(start, stop + 1):
    candidate = dict(result, end_line=number, text=result['text'] + lines[number - 1])
    candidate['next_line'] = number + 1 if number < len(lines) else None
    if len(json.dumps(candidate, ensure_ascii=False)) > 3800:
        break
    result = candidate
if result['end_line'] < start:
    raise ValueError('single source line exceeds bounded observation; no partial line returned')
print(json.dumps(result, ensure_ascii=False))
'''


def read_command(read: SourceRead) -> str:
    return fixed_python_command(_PROGRAM, validate_read(read.to_dict()).to_dict())
