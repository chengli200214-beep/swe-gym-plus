"""Keep framework observation receipts separate from model-generated actions."""
from __future__ import annotations

from dataclasses import asdict
import json

from codeagentbench.adapters.repository_inventory import inventory_command, inventory_packet
from codeagentbench.harness.budget import BudgetExceeded
from codeagentbench.models import ToolIntent


def capture_inventory(workspace, executor, journal, *, run_id, timeout, consumed=False, failure_injector=None):
    action_id = run_id + "-inventory"
    records = [r for r in journal.records() if r.get("action_id") == action_id]
    if records:
        intents = [r for r in records if r["type"] == "intent"]
        receipts = [r for r in records if r["type"] == "receipt"]
        if len(intents) != 1 or len(receipts) != 1 or intents[0]["command"] != inventory_command():
            raise RuntimeError("inventory action has an unknown or mismatched outcome; do not replay")
        if (receipts[0]["command"] != intents[0]["command"] or intents[0]["cwd"] != str(workspace.path)
                or intents[0]["side_effect"] is not False):
            raise RuntimeError("inventory receipt does not match its fixed observation intent")
        return intents[0], receipts[0], True
    if consumed:
        raise RuntimeError("consumed inventory lacks its real receipt")
    intent = ToolIntent(action_id, inventory_command(), str(workspace.path), min(10, timeout),
                        False, action_id, workspace.digest)
    journal.record_intent(intent)
    if failure_injector:
        failure_injector("after_inventory_intent")
    receipt = executor.execute(intent, intent_already_recorded=True)
    if failure_injector:
        failure_injector("after_inventory_receipt")
    return asdict(intent), asdict(receipt), False


def attach_inventory(state, messages, workspace, executor, journal, ledger, store, *, failure_injector=None):
    """Recover known observations once; unknown actions remain fail-closed."""
    if not state.inventory_consumed and (not ledger.can_spend(tool_calls=1) or ledger.snapshot.remaining_seconds <= 0):
        raise BudgetExceeded("insufficient budget before repository inventory")
    intent, receipt, recovered = capture_inventory(workspace, executor, journal,
        run_id=state.run_id, timeout=ledger.snapshot.remaining_seconds, consumed=state.inventory_consumed,
        failure_injector=failure_injector)
    if not state.inventory_consumed:
        if receipt["post_digest"] != workspace.digest:
            raise RuntimeError("inventory recovery receipt/workspace mismatch")
        ledger.consume(seconds=receipt["duration_seconds"], tool_calls=1)
        state.inventory_consumed = True
    event_path = store.run_dir(state.run_id) / "events.jsonl"
    events = [json.loads(s) for s in event_path.read_text(encoding="utf-8").splitlines()] if event_path.exists() else []
    if not any(e.get("type") == "harness_tool" and e.get("receipt", {}).get("action_id") == receipt["action_id"] for e in events):
        store.append_event(state.run_id, {"type": "harness_tool", "name": "repository_inventory",
            "intent": intent, "receipt": receipt, "recovered": recovered})
    if receipt["status"] == "cancelled":
        state.status, state.failure_reason = "cancelled", "cancellation requested during repository inventory"
        return False
    packet = inventory_packet(receipt)
    prompt = json.loads(messages[1]["content"])
    previous = prompt.get("repository_inventory")
    if previous is not None and previous != packet:
        raise RuntimeError("checkpoint inventory differs from its real receipt")
    prompt["repository_inventory"] = packet
    messages[1]["content"] = json.dumps(prompt, ensure_ascii=False)
    return True
