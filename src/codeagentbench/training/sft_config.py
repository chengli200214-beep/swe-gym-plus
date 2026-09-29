"""Validated SFT configuration and path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


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
    "done_sampling_weight": 1.0,
    "typed_action_sampling_weight": 1.0,
    "sampling_mode": "weighted_with_replacement",
    "oom_ladder": [
        {"max_seq_len": 2048, "gradient_accumulation_steps": 8},
        {"max_seq_len": 1024, "gradient_accumulation_steps": 16},
    ],
}


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
    done_sampling_weight: float = 1.0
    typed_action_sampling_weight: float = 1.0
    sampling_mode: str = "weighted_with_replacement"
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
    weight = float(raw["done_sampling_weight"])
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("done_sampling_weight must be finite and at least 1")
    typed_weight = float(raw["typed_action_sampling_weight"])
    if not math.isfinite(typed_weight) or typed_weight < 1:
        raise ValueError("typed_action_sampling_weight must be finite and at least 1")
    sampling_mode = str(raw["sampling_mode"])
    if sampling_mode not in {"weighted_with_replacement", "coverage_plus_weighted"}:
        raise ValueError("unsupported sampling_mode")
    if sampling_mode == "coverage_plus_weighted" and weight == typed_weight == 1:
        raise ValueError("coverage_plus_weighted requires a non-default action weight")
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
        done_sampling_weight=weight,
        typed_action_sampling_weight=typed_weight,
        sampling_mode=sampling_mode,
        oom_ladder=[dict(rung) for rung in raw.get("oom_ladder") or []],
        raw=raw,
    )
