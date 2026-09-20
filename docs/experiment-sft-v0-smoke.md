# SFT-v0 Smoke Experiment

Date: 2026-09-20  
Repository commit: `c0f2c24`  
Status: engineering smoke run, not a benchmark result

## Purpose

Validate the end-to-end path from DeepSeek-powered coding-agent rollouts to
assistant-only QLoRA training on a rented GPU server:

```text
SWE-Gym task -> Agent tool trajectory -> independent evaluation
-> export-sft JSONL -> Qwen2.5-Coder-3B QLoRA checkpoint
```

## Rollout data

Two trajectories were exported after independent formal evaluation passed:

| task | evaluation | notes |
|---|---|---|
| `getmoto__moto-7365` | passed (`fail_to_pass=true`, `pass_to_pass=true`) | Decimal arithmetic patch |
| `getmoto__moto-6920` | passed | Lambda layer metadata patch |

The agent status and evaluation verdict are stored separately. One successful
candidate reached the step budget before emitting the final protocol marker;
the patch was still accepted only because the fresh evaluator passed it.

## Training configuration

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`
- 4-bit NF4 QLoRA with double quantization
- LoRA rank 16, alpha 32, dropout 0.05
- Maximum sequence length 2048
- Per-device batch size 1, gradient accumulation 8
- One epoch, seed 42, assistant-only loss
- Records: 2; encoded examples: 2; dropped examples: 0

## Result

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately 96GB VRAM
- PyTorch: `2.8.0+cu128`
- Training loss: `2.413710355758667`
- Global step: 1
- Training runtime: 3.547 seconds after model download
- Checkpoint: `experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora`

The checkpoint and raw rollout artifacts remain on the server under
`/root/autodl-tmp/swe-gym-plus/experiments/sft-v0/`. They are intentionally not
committed to the public repository.

## Interpretation and next step

This run proves that the data contract, independent evaluator, QLoRA loader,
assistant-only masking, GPU training, and checkpoint export work together. Two
records are too few for a capability claim. The next experiment should collect
at least 20--50 independently passing trajectories, reserve a held-out smoke
split, and compare base versus SFT under the same evaluator before attempting
GRPO.
