# Local Base/SFT Rollout

The harness can run a local Transformers model when an external API is
unavailable. The backend supports both a base causal-LM checkpoint and a PEFT
LoRA adapter checkpoint.

Set the cache location on the GPU server and keep downloads offline after the
model has been cached:

```bash
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_DISABLE_XET=1
```

Base-model smoke run:

```bash
python -m codeagentbench run data/manifests/swegym-smoke.json \
  getmoto__moto-7365 \
  --repo-root experiments/sft-v0/rollouts \
  --run-id local-base-7365 \
  --model-backend local \
  --model-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --max-steps 8 --max-tool-calls 12 --max-new-tokens 512
```

SFT-adapter smoke run:

```bash
python -m codeagentbench run data/manifests/swegym-smoke.json \
  getmoto__moto-7365 \
  --repo-root experiments/sft-v0/rollouts \
  --run-id local-sft-7365 \
  --model-backend local \
  --model-path experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora \
  --base-model-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --max-steps 8 --max-tool-calls 12 --max-new-tokens 512
```

The evaluator remains independent of the model backend. Compare the
`evaluation.verdict`, `fail_to_pass`, `pass_to_pass`, patch, token count, and
runtime from the two run artifacts. The first SFT checkpoint was trained on
only two records, so it is a pipeline smoke artifact rather than a capability
claim.
