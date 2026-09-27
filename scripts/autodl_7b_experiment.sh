#!/usr/bin/env bash
# One bounded, credential-free autonomous run after a verified public download.
set -euo pipefail
cd /root/autodl-tmp/swe-gym-plus
unset DEEPSEEK_API_KEY
export CODEAGENTBENCH_EXECUTOR=nsjail
export CODEAGENTBENCH_LOCAL_PROMPT_POLICY=recent-history-v3
export MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=8
export MODELSCOPE_DOWNLOAD_MAX_RETRIES=3
export HF_HOME=/root/autodl-tmp/cache/huggingface
python_bin=/root/autodl-tmp/agent-env/bin/python
artifact_root=/root/autodl-tmp/experiments/autodl-one-task-20260927
manifest=/root/autodl-tmp/server-handoff-20260927/experiments/accelerated-20260926/pool/manifest.json
model=/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct
timeout 1200 "$python_bin" -u -c 'from modelscope import snapshot_download; from modelscope_hub.constants import DOWNLOAD_PARALLELS; assert DOWNLOAD_PARALLELS == 8; print(snapshot_download("Qwen/Qwen2.5-Coder-7B-Instruct", local_dir="/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct"))' > "$artifact_root/download-7b-parallel-resume.log" 2>&1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
timeout 900 "$python_bin" -u -m codeagentbench run "$manifest" getmoto__moto-6212 \
    --repo-root "$artifact_root" --model-backend local --model-path "$model" \
    --max-new-tokens 2048 --temperature 0 --max-steps 12 --max-tool-calls 12 \
    --max-tokens 150000 --max-seconds 600 --max-cost-usd 0 \
    --run-id base7b-6212-greedy --skip-evaluation > "$artifact_root/base7b-greedy.log" 2>&1
timeout 180 "$python_bin" -m codeagentbench evaluate-run "$manifest" base7b-6212-greedy \
    --repo-root "$artifact_root" --timeout 120 > "$artifact_root/base7b-eval.log" 2>&1
