"""Resolve upstream test-name serialization defects, never drop a requirement."""
from __future__ import annotations


def _escaped_unicode(text: str) -> str:
    return "".join(c.encode("unicode_escape").decode("ascii") if ord(c) > 127 else c for c in text)


def resolve_selectors(required: list[str], collected: list[str]) -> dict[str, str]:
    """Each label must resolve to exactly one real collected node.

    Prefer exact IDs. Only Unicode escaping and an *incomplete* parameter ID
    prefix are eligible for repair. Do not match function names, broaden to a
    parent test, select by outcome or remove tests. See ADR 0007.
    """
    if not required or not collected or len(collected) != len(set(collected)):
        raise ValueError("require non-empty test identities and unique collected nodes")
    mapping = {}
    for label in required:
        if not isinstance(label, str) or "::" not in label:
            raise ValueError("invalid upstream test identity")
        if label in collected:
            mapping[label] = label
            continue
        variants = {label, _escaped_unicode(label)}
        matches = {n for n in collected if n in variants}
        if not matches and "[" in label and not label.endswith("]"):
            matches = {n for n in collected if n.endswith("]") and any(n.startswith(v) for v in variants)}
        if len(matches) != 1:
            raise ValueError(f"upstream test identity has {len(matches)} possible collected nodes: {label!r}")
        mapping[label] = matches.pop()
    return mapping
