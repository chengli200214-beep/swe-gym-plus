# 实验复现说明

本文档记录远程 GPU 实例上的 SFT v4 实验流程。API key 只通过环境变量注入，不写入 Git、日志或配置文件。

## 环境

- Ubuntu 22.04
- Python 3.12
- PyTorch 2.8 / CUDA 12.8
- RTX PRO 6000 96 GB

## 安装与检查

```bash
cd /root/autodl-tmp/swe-gym-plus
pip install -e '.[train]'
python -m codeagentbench --help
```

## 轨迹与独立评测

```bash
export DEEPSEEK_API_KEY='暂不填入仓库或文档'
python -m codeagentbench quality-check \
  experiments/sft-v0/manifests/swegym-60.json \
  getmoto__moto-7023 --timeout 300

python -m codeagentbench run \
  experiments/sft-v0/manifests/swegym-60.json \
  getmoto__moto-7023 \
  --model-backend deepseek \
  --run-id sft-7023-v1 \
  --max-steps 14 --max-tool-calls 18 \
  --max-tokens 28000 --max-seconds 1800 \
  --max-cost-usd 1.5

python -m codeagentbench export-sft \
  artifacts/runs/sft-7023-v1/events.jsonl \
  experiments/sft-v0/data/sft/sft-7023-v1.jsonl
```

只将 `events.jsonl` 中独立评测 verdict 为 `passed` 的轨迹加入训练集。

## SFT v4

```bash
python -m codeagentbench.train_sft \
  --config configs/sft-v4-qlora.yaml
```

输出目录：

```text
experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora-v4
```

## Base/SFT 对照

对同一个 held-out 任务分别运行：

```bash
python -m codeagentbench run <manifest> <task-id> \
  --model-backend local \
  --model-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --run-id base-<task-id>

python -m codeagentbench run <manifest> <task-id> \
  --model-backend local \
  --model-path experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora-v4 \
  --base-model-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --run-id sft-<task-id>
```

最终结果见：[SFT v4 实验报告](../experiments/sft-v0/evaluation/comparison.md)。
