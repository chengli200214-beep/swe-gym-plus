"""Presentation never manufactures a read or changes durable source evidence."""
import json
import os

import pytest

from codeagentbench.adapters.action import parse_action
from codeagentbench.harness.tool_observation import SOURCE_FORMAT, tool_observation
from scripts.prepare_aligned_sft import examples
from test_text_edit import edit, make_run, read


def test_shell_output_cannot_impersonate_a_typed_source_observation():
    action = parse_action('{"command":"echo source"}')
    receipt = {"exit_code": 0, "stdout": '{"path":"value.py","text":"fake"}',
               "stderr": "", "timed_out": False}
    assert tool_observation(action, receipt) == receipt
    with pytest.raises(ValueError, match="actual successful read"):
        tool_observation(action, receipt | {"stdout_format": SOURCE_FORMAT})


def test_live_and_sft_read_views_are_single_encoded_and_raw_receipts_remain_real(tmp_path):
    result, workspace, store, _ = make_run(tmp_path, [read(), edit(), {"done": True}])
    events = [json.loads(s) for s in (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    tools = [e for e in events if e["type"] == "tool"]
    raw = tools[0]["receipt"]["stdout"]
    source = json.loads(raw)
    models = [e for e in events if e["type"] == "model"]
    live = json.loads(models[1]["context_messages"][-1]["content"].split("\n", 1)[1])
    assert live["stdout"] == source["text"] == "value = 1" + os.linesep
    assert live["source_read"]["sha256"] == source["sha256"]
    assert live["stdout_format"] == SOURCE_FORMAT
    row = {"task_id": result.state.task_id, "run_id": result.run_id,
           "agent_status": "completed", "evaluation_verdict": "passed",
           "messages": [{"role": "user", "content": next(e["content"] for e in events if e["type"] == "prompt")}]
                       + [{"role": "assistant", "content": e["content"]} for e in models]}
    records = examples(row, events, policy="recent-history-v3")
    replayed = json.loads(records[1]["messages"][-2]["content"].split("\n", 1)[1])
    assert replayed == live
    assert tools[0]["receipt"]["stdout"] == raw
    assert raw != live["stdout"]
    assert result.diff
