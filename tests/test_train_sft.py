"""Tests for the QLoRA SFT entry point (runbook section 9).

Real training cannot run on a CPU-only or sub-12GB machine, so these tests cover
the parts that must be correct *before* a GPU run is attempted: path resolution,
assistant-only supervision, tool output staying in context, truncation, padding
and the refusal to fall back to CPU training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagentbench import train_sft

IGNORE = train_sft.IGNORE_INDEX


class StubTokenizer:
    """Whitespace tokenizer with role markers, enough to locate supervised spans."""

    pad_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self._id(token) for token in text.split()]}

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False) -> str:
        parts: list[str] = []
        for message in messages:
            parts.append(f"<|{message['role']}|>")
            parts.append(message["content"])
        return " ".join(parts)

    def decode(self, ids) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return " ".join(inverse.get(int(i), "?") for i in ids)


def _conversation() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "fix bug"},
        {"role": "assistant", "content": "run tests"},
        {"role": "tool", "content": "all green"},
        {"role": "assistant", "content": "done"},
    ]


def test_build_config_resolves_relative_paths_against_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "sft.yaml"
    config_file.write_text(
        "train_file: ../data/sft/train.jsonl\noutput_dir: ../checkpoints/out\n", encoding="utf-8"
    )

    built = train_sft.build_config(config_file)

    assert built.train_file == (tmp_path / "data" / "sft" / "train.jsonl").resolve()
    assert built.output_dir == (tmp_path / "checkpoints" / "out").resolve()


def test_build_config_requires_train_file(tmp_path: Path) -> None:
    config_file = tmp_path / "sft.yaml"
    config_file.write_text("epochs: 1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="train_file"):
        train_sft.build_config(config_file)


def test_tool_role_is_mapped_to_a_template_safe_role() -> None:
    normalised = train_sft.normalise_messages(_conversation())

    assert [message["role"] for message in normalised] == ["user", "assistant", "user", "assistant"]
    assert normalised[2]["content"].startswith("TOOL OUTPUT:")
    assert "all green" in normalised[2]["content"]


def test_assistant_only_loss_supervises_only_assistant_spans() -> None:
    tokenizer = StubTokenizer()

    example = train_sft.encode_example(tokenizer, _conversation(), max_seq_len=512)

    assert example is not None
    supervised = [token for token, label in zip(example["input_ids"], example["labels"]) if label != IGNORE]
    supervised_text = tokenizer.decode(supervised)

    # Assistant turns are the supervision targets.
    assert "run tests" in supervised_text
    assert "done" in supervised_text
    # Tool output and the user turn stay in context but are never supervised.
    assert "all green" not in supervised_text
    assert "fix bug" not in supervised_text

    context_text = tokenizer.decode(example["input_ids"])
    assert "all green" in context_text, "tool output must remain in the context"


def test_encode_example_returns_none_when_no_assistant_turn() -> None:
    tokenizer = StubTokenizer()

    example = train_sft.encode_example(tokenizer, [{"role": "user", "content": "only a question"}], max_seq_len=512)

    assert example is None


def test_encode_example_truncates_to_max_sequence_length() -> None:
    tokenizer = StubTokenizer()
    messages = [
        {"role": "assistant", "content": " ".join(f"a{i}" for i in range(60))},
        {"role": "user", "content": " ".join(f"w{i}" for i in range(60))},
    ]

    example = train_sft.encode_example(tokenizer, messages, max_seq_len=32)

    assert example is not None
    assert len(example["input_ids"]) == 32
    assert len(example["labels"]) == 32
    assert any(label != IGNORE for label in example["labels"])


def test_long_opening_turn_keeps_a_supervised_tail() -> None:
    """Head+tail truncation keeps final assistant actions trainable."""

    tokenizer = StubTokenizer()
    messages = [
        {"role": "user", "content": " ".join(f"w{i}" for i in range(60))},
        {"role": "assistant", "content": " ".join(f"a{i}" for i in range(60))},
    ]

    example = train_sft.encode_example(tokenizer, messages, max_seq_len=32)

    assert example is not None
    assert len(example["input_ids"]) == 32
    assert any(label != IGNORE for label in example["labels"])


def test_collator_pads_inputs_and_masks_padding_from_the_loss() -> None:
    collator = train_sft.SupervisedCollator(StubTokenizer.pad_token_id)

    batch = collator([
        {"input_ids": [5, 6, 7], "labels": [IGNORE, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ])

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE, 6, 7], [8, IGNORE, IGNORE]]


def test_training_refuses_to_fall_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="cuda|CUDA"):
        train_sft._require_gpu()


def test_load_records_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not found"):
        train_sft.load_records(tmp_path / "absent.jsonl")


def test_load_records_rejects_records_without_messages(tmp_path: Path) -> None:
    data = tmp_path / "train.jsonl"
    data.write_text('{"messages": [], "assistant_only_loss": true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="messages"):
        train_sft.load_records(data)


# --------------------------------------------------------------------------- #
# section 8: exported records must retain their provenance
# --------------------------------------------------------------------------- #
def _write_run(run_dir: Path, *, verdict: str = "passed", diff: str = "diff --git a/x b/x\n") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({"run_id": "r-1", "task_id": "getmoto__moto-1", "config": {}}), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"run_id": "r-1", "task_id": "getmoto__moto-1", "status": "failed", "diff": diff}),
        encoding="utf-8",
    )
    events = [
        {"type": "prompt", "content": "fix the issue"},
        {"type": "model", "content": '{"command": "pytest -q"}', "prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.5},
        {"type": "tool", "intent": {"command": "pytest -q"}, "receipt": {"stdout": "1 passed", "stderr": ""}},
        {"type": "evaluation", "verdict": verdict, "duration_seconds": 12.5},
    ]
    events_path = run_dir / "events.jsonl"
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return events_path


def test_export_records_section_8_provenance(tmp_path: Path) -> None:
    from codeagentbench.training.export import export_run_to_sft

    events_path = _write_run(tmp_path / "runs" / "r-1")
    output = tmp_path / "out" / "r-1.jsonl"

    assert export_run_to_sft(events_path, output) is True

    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["task_id"] == "getmoto__moto-1"
    assert record["run_id"] == "r-1"
    assert record["assistant_only_loss"] is True
    assert record["evaluation_verdict"] == "passed"
    assert record["patch"].startswith("diff --git")
    assert record["duration_seconds"] == 12.5
    assert record["total_tokens"] == 120
    assert record["cost_usd"] == 0.5
    assert record["tool_calls"] == 1
    assert record["events_path"].endswith("events.jsonl")
    # The agent's own status is kept alongside the independent verdict.
    assert record["agent_status"] == "failed"


def test_export_reports_unavailable_provenance_as_none(tmp_path: Path) -> None:
    """Missing sibling artifacts must not be invented."""

    from codeagentbench.training.export import export_run_to_sft

    events_path = tmp_path / "lonely" / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"type": "prompt", "content": "issue"},
                {"type": "model", "content": '{"command": "pytest -q"}'},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"

    assert export_run_to_sft(events_path, output, successful_only=False) is True

    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["task_id"] is None
    assert record["patch"] is None
    assert record["evaluation_verdict"] is None
    assert record["total_tokens"] == 0


def test_export_without_successful_evaluation_is_skipped_by_default(tmp_path: Path) -> None:
    from codeagentbench.training.export import export_run_to_sft

    events_path = _write_run(tmp_path / "runs" / "r-2", verdict="failed")
    output = tmp_path / "out" / "r-2.jsonl"

    assert export_run_to_sft(events_path, output) is False
    assert not output.exists()
