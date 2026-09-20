"""QLoRA SFT entry point for the CodeAgentBench SFT run (runbook section 9).

    python -m codeagentbench.train_sft --config experiments/sft-v0/configs/sft-3b-qlora.yaml
    python -m codeagentbench.train_sft --config <config> --dry-run

Constraints taken from ``docs/deepseek-server-runbook.md`` section 9:

* ``Qwen/Qwen2.5-Coder-3B-Instruct`` with 4-bit NF4 QLoRA;
* LoRA rank 16, alpha 32, dropout 0.05;
* per-device batch size 1, gradient accumulation 8, gradient checkpointing on;
* max sequence length 2048, one epoch, seed 42;
* assistant-only loss: tool output stays in context but is never a supervision
  target;
* CPU training is refused (section 2) and the documented OOM ladder is applied
  (2048/8 -> 1024/16) instead of silently degrading to CPU.

Records are the JSONL objects produced by ``codeagentbench export-sft``:
``{"messages": [{"role", "content"}, ...], "assistant_only_loss": true}``.

The heavy stack (CUDA torch, peft, bitsandbytes) is imported lazily, so
``--dry-run`` validates the configuration and the data on a machine with no GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100

DEFAULTS: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "output_dir": "checkpoints/qwen2.5-coder-3b-qlora",
    "train_file": None,
    "eval_file": None,
    "max_seq_len": 2048,
    "epochs": 1,
    "seed": 42,
    "learning_rate": 2e-4,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "load_in_4bit": True,
    "bnb_quant_type": "nf4",
    "bnb_double_quant": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "logging_steps": 1,
    "save_total_limit": 3,
    # Section 9 OOM ladder, applied in order.
    "oom_ladder": [
        {"max_seq_len": 2048, "gradient_accumulation_steps": 8},
        {"max_seq_len": 1024, "gradient_accumulation_steps": 16},
    ],
}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class SFTConfig:
    base_model: str
    output_dir: Path
    train_file: Path
    eval_file: Path | None
    max_seq_len: int
    epochs: int
    seed: int
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    load_in_4bit: bool
    bnb_quant_type: str
    bnb_double_quant: bool
    gradient_checkpointing: bool
    optim: str
    logging_steps: int
    save_total_limit: int
    oom_ladder: list[dict[str, int]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("train_sft requires pyyaml: pip install codeagentbench[dataset]") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"config must be a YAML mapping: {path}")
    return data


def build_config(config_path: Path, overrides: dict[str, Any] | None = None) -> SFTConfig:
    raw = dict(DEFAULTS)
    raw.update(_load_yaml(config_path))
    raw.update(overrides or {})
    if not raw.get("train_file"):
        raise RuntimeError("config must set `train_file` (the exported SFT JSONL)")
    base = config_path.parent
    train_file = Path(raw["train_file"])
    if not train_file.is_absolute():
        train_file = (base / train_file).resolve()
    eval_file = raw.get("eval_file")
    if eval_file:
        eval_file = Path(eval_file)
        if not eval_file.is_absolute():
            eval_file = (base / eval_file).resolve()
    output_dir = Path(raw["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    return SFTConfig(
        base_model=raw["base_model"],
        output_dir=output_dir,
        train_file=train_file,
        eval_file=eval_file,
        max_seq_len=int(raw["max_seq_len"]),
        epochs=int(raw["epochs"]),
        seed=int(raw["seed"]),
        learning_rate=float(raw["learning_rate"]),
        per_device_batch_size=int(raw["per_device_batch_size"]),
        gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        lora_rank=int(raw["lora_rank"]),
        lora_alpha=int(raw["lora_alpha"]),
        lora_dropout=float(raw["lora_dropout"]),
        target_modules=list(raw["target_modules"]),
        load_in_4bit=bool(raw["load_in_4bit"]),
        bnb_quant_type=str(raw["bnb_quant_type"]),
        bnb_double_quant=bool(raw["bnb_double_quant"]),
        gradient_checkpointing=bool(raw["gradient_checkpointing"]),
        optim=str(raw["optim"]),
        logging_steps=int(raw["logging_steps"]),
        save_total_limit=int(raw["save_total_limit"]),
        oom_ladder=[dict(rung) for rung in raw.get("oom_ladder") or []],
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"training data not found: {path}")
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
        if not isinstance(record.get("messages"), list) or not record["messages"]:
            raise RuntimeError(f"{path}:{lineno} has no `messages` list")
        records.append(record)
    if not records:
        raise RuntimeError(f"training data is empty: {path}")
    return records


def normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map non-chat roles onto template-safe roles.

    The harness records tool output with role ``tool``. Not every chat template
    accepts that role, so it is rendered as a labelled user turn. Tool text
    stays in context exactly as before; it is never supervised.
    """

    normalised: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "tool":
            normalised.append({"role": "user", "content": f"TOOL OUTPUT:\n{content}"})
        elif role in {"system", "user", "assistant"}:
            normalised.append({"role": role, "content": content})
        else:
            normalised.append({"role": "user", "content": content})
    return normalised


def encode_example(tokenizer: Any, messages: list[dict[str, str]], max_seq_len: int) -> dict[str, list[int]] | None:
    """Tokenise one conversation with assistant-only supervision.

    Only assistant spans receive a label; every other span is masked with
    ``IGNORE_INDEX`` so tool output is context but never a target.

    The sequence is truncated from the right, which keeps the issue statement and
    the earliest context. A long opening tool output can therefore push every
    assistant turn past the window; such an example is dropped (returns ``None``)
    rather than trained with no supervision. Callers report the drop count.
    """

    def render(conversation: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

    input_ids: list[int] = []
    labels: list[int] = []
    for index, message in enumerate(messages):
        # Qwen chat templates reject an empty conversation.  The first turn has
        # no prefix tokens, so avoid rendering ``messages[:0]`` altogether.
        prefix_ids = (
            []
            if index == 0
            else tokenizer(render(messages[:index]), add_special_tokens=False)["input_ids"]
        )
        current_ids = tokenizer(render(messages[:index + 1]), add_special_tokens=False)["input_ids"]
        if current_ids[: len(prefix_ids)] != prefix_ids:
            # This template is not prefix-stable for this turn, so the supervised
            # span cannot be located by diffing. Re-encode the whole prefix and
            # leave it unsupervised rather than risk training on the wrong span.
            input_ids = tokenizer(render(messages[: index + 1]), add_special_tokens=False)["input_ids"]
            labels = [IGNORE_INDEX] * len(input_ids)
            continue
        delta = current_ids[len(prefix_ids):]
        input_ids.extend(delta)
        labels.extend(delta if message["role"] == "assistant" else [IGNORE_INDEX] * len(delta))

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


class SupervisedCollator:
    """Right-pad input ids while keeping pad labels masked out of the loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(feature["input_ids"]) for feature in features)
        batch_input, batch_labels, batch_mask = [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            batch_input.append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch_labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            batch_mask.append([1] * len(feature["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def dry_run(config: SFTConfig) -> int:
    """Validate config and data without touching a GPU or the hub."""

    records = load_records(config.train_file)
    assistant_turns = 0
    for record in records:
        assistant_turns += sum(1 for m in record["messages"] if m.get("role") == "assistant")
    tool_turns = sum(
        1 for record in records for m in record["messages"] if m.get("role") == "tool"
    )
    plan = {
        "base_model": config.base_model,
        "train_file": str(config.train_file),
        "output_dir": str(config.output_dir),
        "records": len(records),
        "assistant_turns": assistant_turns,
        "tool_turns_kept_in_context": tool_turns,
        "assistant_only_loss": True,
        "max_seq_len": config.max_seq_len,
        "epochs": config.epochs,
        "per_device_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora": {"r": config.lora_rank, "alpha": config.lora_alpha, "dropout": config.lora_dropout},
        "load_in_4bit": config.load_in_4bit,
        "quant_type": config.bnb_quant_type,
        "gradient_checkpointing": config.gradient_checkpointing,
        "seed": config.seed,
        "oom_ladder": config.oom_ladder,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    print("\ndry run complete: config and data are valid; no GPU was used.", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _require_gpu() -> Any:
    """Refuse to train without CUDA (runbook section 2)."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("train_sft requires torch: pip install codeagentbench[train]") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. Runbook section 5 requires fixing the "
            "CUDA environment first and section 2 forbids substituting CPU training."
        )
    return torch


def _build_model_and_tokenizer(config: SFTConfig, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not config.load_in_4bit:
        raise RuntimeError("section 9 requires 4-bit NF4 QLoRA; `load_in_4bit` must stay true")

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("QLoRA requires bitsandbytes: pip install bitsandbytes") from exc
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_quant_type,
        bnb_4bit_use_double_quant=config.bnb_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantisation,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, lora)
    return model, tokenizer


def _attempt(config: SFTConfig, torch: Any, max_seq_len: int, grad_accum: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    import inspect

    from transformers import Trainer, TrainingArguments

    model, tokenizer = _build_model_and_tokenizer(config, torch)

    encoded = []
    dropped = 0
    for record in records:
        example = encode_example(tokenizer, normalise_messages(record["messages"]), max_seq_len)
        if example is not None:
            encoded.append(example)
        else:
            dropped += 1
    if dropped:
        print(
            f"  dropped {dropped}/{len(records)} record(s): no assistant span survived "
            f"tokenisation at max_seq_len={max_seq_len}",
            file=sys.stderr,
        )
    if not encoded:
        raise RuntimeError(
            "no training example survived tokenisation with an assistant span; "
            "check that export-sft produced assistant messages"
        )

    training_kwargs = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.3,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "data_seed": config.seed,
        "bf16": True,
        "gradient_checkpointing": config.gradient_checkpointing,
        "optim": config.optim,
        "report_to": [],
        "remove_unused_columns": False,
    }
    # Transformers 5.x removed ``warmup_ratio`` from TrainingArguments. Keep
    # the intended warmup on older releases and use the compatible zero-step
    # fallback on the current server image.
    if "warmup_ratio" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["warmup_ratio"] = 0.03
    else:
        training_kwargs["warmup_steps"] = 0
    arguments = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=encoded,
        data_collator=SupervisedCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=config.raw.get("resume_from_checkpoint"))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    (config.output_dir / "training_args.json").write_text(
        json.dumps(arguments.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = {
        "train_records": len(records),
        "encoded_examples": len(encoded),
        "dropped_no_supervision": dropped,
        "max_seq_len": max_seq_len,
        "gradient_accumulation_steps": grad_accum,
        "epochs": config.epochs,
        "seed": config.seed,
        "trainer_metrics": trainer.state.log_history,
        "global_step": trainer.state.global_step,
    }
    (config.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return metrics


def train(config: SFTConfig) -> int:
    torch = _require_gpu()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    records = load_records(config.train_file)
    ladder = config.oom_ladder or [
        {"max_seq_len": config.max_seq_len, "gradient_accumulation_steps": config.gradient_accumulation_steps}
    ]

    attempts: list[dict[str, Any]] = []
    for index, rung in enumerate(ladder):
        max_seq_len = int(rung.get("max_seq_len", config.max_seq_len))
        grad_accum = int(rung.get("gradient_accumulation_steps", config.gradient_accumulation_steps))
        print(f"[rung {index + 1}/{len(ladder)}] max_seq_len={max_seq_len} grad_accum={grad_accum}", file=sys.stderr)
        try:
            metrics = _attempt(config, torch, max_seq_len, grad_accum, records)
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({"max_seq_len": max_seq_len, "gradient_accumulation_steps": grad_accum, "result": "oom"})
            print(f"  OOM: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            if index == len(ladder) - 1:
                (config.output_dir.parent / "training_status.json").write_text(
                    json.dumps(
                        {
                            "status": "blocked",
                            "reason": "out of memory on every rung of the documented ladder",
                            "attempts": attempts,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError("OOM on every rung of the ladder; not falling back to CPU training") from exc
            continue
        attempts.append({"max_seq_len": max_seq_len, "gradient_accumulation_steps": grad_accum, "result": "ok"})
        metrics["attempts"] = attempts
        metrics["lora"] = {"r": config.lora_rank, "alpha": config.lora_alpha, "dropout": config.lora_dropout}
        metrics["base_model"] = config.base_model
        (config.output_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "completed", "output_dir": str(config.output_dir), "attempts": attempts}, indent=2))
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m codeagentbench.train_sft")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="validate config and data without a GPU")
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--train-file", type=Path)
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.max_seq_len:
        overrides["max_seq_len"] = args.max_seq_len
        overrides["oom_ladder"] = [{"max_seq_len": args.max_seq_len, "gradient_accumulation_steps": 8}]
    # CLI paths are relative to the working directory; only values written inside
    # the YAML file are resolved against the config's own directory.
    if args.output_dir:
        overrides["output_dir"] = str(args.output_dir.resolve())
    if args.train_file:
        overrides["train_file"] = str(args.train_file.resolve())
    if args.resume_from_checkpoint:
        overrides["resume_from_checkpoint"] = args.resume_from_checkpoint

    try:
        config = build_config(args.config.resolve(), overrides)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        try:
            return dry_run(config)
        except Exception as exc:
            print(f"dry run failed: {exc}", file=sys.stderr)
            return 2
    try:
        return train(config)
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
