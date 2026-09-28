"""Bounded literal source search executed inside the task sandbox."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from codeagentbench.adapters.file_tools import (
    READ_SOURCE_PROGRAM,
    fixed_python_command,
    validate_source_path,
)


@dataclass(frozen=True)
class SourceSearch:
    path: str
    query: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_search(payload: object) -> SourceSearch:
    if not isinstance(payload, dict) or set(payload) != {"path", "query"}:
        raise ValueError("search requires exactly path/query")
    path, query = payload["path"], payload["query"]
    validate_source_path(path)
    if (not isinstance(query, str) or not query.strip() or len(query) > 200
            or any(ord(character) < 32 for character in query)):
        raise ValueError("search query must be 1-200 printable characters")
    return SourceSearch(path, query)


_PROGRAM = READ_SOURCE_PROGRAM + "\n" + r'''

query = proposal['query']
needle = query.casefold()
matches = []
match_count = 0
for number, line in enumerate(text.splitlines(), 1):
    index = line.casefold().find(needle)
    if index < 0:
        continue
    match_count += 1
    if len(matches) >= 12:
        continue
    start = max(0, index - 60)
    end = min(len(line), index + len(query) + 100)
    snippet = line[start:end]
    if start:
        snippet = '…' + snippet
    if end < len(line):
        snippet += '…'
    matches.append({'line': number, 'text': snippet})
result = {
    'path': proposal['path'],
    'file_sha256': hashlib.sha256(raw).hexdigest(),
    'query': query,
    'match_count': match_count,
    'matches': matches,
    'matches_truncated': match_count > len(matches),
    'note': 'Search locations only; perform a successful bounded read before editing.',
}
packet = json.dumps(result, ensure_ascii=False)
while len(packet.encode('utf-8')) > 3000 and matches:
    matches.pop()
    result['matches_truncated'] = True
    packet = json.dumps(result, ensure_ascii=False)
print(packet)
'''.strip()


def search_command(search: SourceSearch) -> str:
    """Encode literal search data; only the fixed scanner program is executed."""
    search = validate_search(search.to_dict())
    return fixed_python_command(_PROGRAM, search.to_dict())
