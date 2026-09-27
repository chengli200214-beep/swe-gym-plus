"""Readable model observations from raw receipts, shared by live/SFT contexts.

Raw journal/events remain unchanged. Only a validated typed read is decoded;
arbitrary shell stdout can never become a trusted source-read packet here.
"""
from __future__ import annotations

from codeagentbench.adapters.action import AgentAction
from codeagentbench.harness.source_evidence import source_read_result

SOURCE_FORMAT = "decoded-source-text-v1"


def tool_observation(action: AgentAction, receipt: dict) -> dict:
    observation = {k: receipt[k] for k in ("exit_code", "stdout", "stderr", "timed_out")}
    for key in ("stdout_truncated", "stderr_truncated"):
        if receipt.get(key) is True:
            observation[key] = True
    projected = receipt.get("stdout_format") == SOURCE_FORMAT
    if projected and (action.read is None or receipt["exit_code"] != 0 or receipt["timed_out"]):
        raise ValueError("decoded source observation requires an actual successful read action")
    if action.read is None or receipt["exit_code"] != 0 or receipt["timed_out"]:
        return observation
    if projected:
        metadata = receipt.get("source_read")
        if (not isinstance(metadata, dict) or set(metadata) != {"path", "sha256", "start_line", "end_line", "next_line"}
                or metadata["path"] != action.read.path or metadata["start_line"] != action.read.start_line
                or type(metadata["end_line"]) is not int
                or not action.read.start_line <= metadata["end_line"] <= action.read.end_line
                or not isinstance(metadata["sha256"], str) or len(metadata["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in metadata["sha256"])
                or not isinstance(observation["stdout"], str)):
            raise ValueError("invalid decoded source observation metadata")
    else:
        result = source_read_result(action, receipt)
        metadata = {k: v for k, v in result.items() if k != "text"}
        observation["stdout"] = result["text"]
    return observation | {"stdout_format": SOURCE_FORMAT, "source_read": dict(metadata)}
