# CodeAgentBench / SWE-Gym-plus 服务器执行任务

## 1. 目标

你现在是本项目的执行代理，不是方案顾问。请在租用的 Linux GPU 服务器上，基于现有 `swe-gym-plus` 项目完成：

1. 检查服务器和 CUDA 环境；
2. 使用 DeepSeek API 生成 Coding Agent 轨迹；
3. 将成功轨迹转换为 SFT 数据；
4. 使用 QLoRA 训练 `Qwen/Qwen2.5-Coder-3B-Instruct`；
5. 对训练前后的模型进行独立评测；
6. 保存代码修改、数据、日志、Adapter 和评测结果；
7. 打包成可下载、可恢复的实验包。

必须实际执行，不能只给方案或伪造结果。

## 2. 硬性限制

- 不做 GRPO。
- 不做全参数微调。
- 3B QLoRA 成功前，不训练 7B 或 14B。
- 不伪造训练结果、评测结果或数据量。
- 不把 API Key 写入代码、日志、配置、Git 或压缩包。
- 不把训练任务用于最终评测。
- 不使用 CPU 训练替代 GPU 训练。
- 如果 GPU、代码、数据或 API Key 缺失，停止并报告原因。
- 不删除已有实验文件。
- 每一步都必须把结果写入磁盘。

## 3. 服务器要求

优先使用 Ubuntu 22.04/24.04、NVIDIA GPU、32GB 以上系统内存和至少 100GB 可用磁盘。

推荐显存 24GB 以上；12GB 只运行 3B QLoRA。可优先选择 RTX 4090 24GB、RTX 5090 32GB、L40S 48GB 或 A10 24GB。

统一目录：

```text
~/codeagentbench-server-run/
~/codeagentbench-server-run/experiments/sft-v0/
```

## 4. API Key 安全

使用服务器 Secret、环境变量或安全输入方式注入：

```bash
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
```

不要把 Key 写入 `.env`、实验配置、日志、Shell history 或最终报告。打包前检查并脱敏。

## 5. 获取和检查项目

如果服务器已有项目，直接进入项目目录；否则使用用户提供的项目压缩包或 Git 仓库，不要根据对话重新编写一个类似项目。

```bash
pwd
git rev-parse HEAD
git status --short
mkdir -p ~/codeagentbench-server-run/experiments/sft-v0
mkdir -p ~/codeagentbench-server-run/logs

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

必须按服务器驱动安装 CUDA 版 PyTorch，不能安装 `torch+cpu`：

```bash
pip install -e ".[api,dataset,dev,train]"
pip install accelerate bitsandbytes safetensors sentencepiece
nvidia-smi
```

检查 CUDA：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, p.name, round(p.total_memory / 1024**3, 2), "GB")
PY
```

如果 `torch.cuda.is_available()` 为 `False`，先修复环境，不得继续训练。

## 6. 运行测试和记录环境

```bash
pytest -q
python -m codeagentbench validate-manifest data/manifests/swegym-smoke.json
git rev-parse HEAD > experiments/sft-v0/git_commit.txt
pip freeze > experiments/sft-v0/pip_freeze.txt
```

将 Python、PyTorch、Transformers、PEFT、TRL、bitsandbytes、CUDA、GPU 和测试结果保存到：

```text
experiments/sft-v0/environment.txt
experiments/sft-v0/preflight.json
```

## 7. 数据集和任务划分

优先使用：

```text
data/manifests/swegym-smoke.json
```

记录 SWE-Gym 数据版本：

```text
bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb
```

按照稳定排序划分任务：前 60% 为训练任务，后 40% 为评测任务，评测任务至少 3 个。创建：

```text
experiments/sft-v0/task_split.json
```

内容至少包含：

```json
{
  "dataset_revision": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
  "train_task_ids": [],
  "eval_task_ids": [],
  "split_seed": 42
}
```

如果 manifest 不存在，不要伪造任务，直接报告缺失文件。

## 8. 生成 DeepSeek 轨迹

使用现有 DeepSeek Adapter 和 Agent Harness，不要手写模拟数据。

如果存在，可以复用已有种子：

```text
artifacts-api-real7/sft/real-moto-7365-v8.jsonl
```

但单条轨迹不能作为正式训练结果，必须继续生成数据。

目标：

```text
有效成功轨迹：至少 10 条
理想有效成功轨迹：20 条以上
最大原始尝试：60 次
最大 API 总预算：30 美元
```

每次 rollout 使用不同的 `run_id)，推荐参数：

```text
max_steps=8
max_tool_calls=16
max_tokens=16000
max_seconds=900
max_cost_usd=3
temperature=0.0
```

每次运行必须保存 `run.json`、`checkpoint.json`、`actions.jsonl`、`events.jsonl`、`summary.json`、评测结果和 workspace diff，目录为：

```text
experiments/sft-v0/rollouts/
```

每完成一次运行立即写盘。只将独立评测通过的轨迹转换为主要 SFT 数据：

```bash
python -m codeagentbench export-sft \
  <events.jsonl> \
  experiments/sft-v0/data/sft/<run_id>.jsonl
```

失败轨迹单独保存到：

```text
experiments/sft-v0/data/failed/
```

最终合并为：

```text
experiments/sft-v0/data/sft/train.jsonl
```

每条训练记录必须保留 task_id、run_id、messages、assistant_only_loss、evaluation verdict、原始 events 路径、patch、运行时间、token 用量和 API 成本。

如果有效轨迹少于 10 条，仍可进行 smoke training，但必须标记 `smoke_only=true`，不得声称数据充足。

## 9. 实现并运行 QLoRA SFT

如果项目目前只有 SFT 导出接口、没有真正训练入口，请补充实现：

```bash
python -m codeagentbench.train_sft \
  --config experiments/sft-v0/configs/sft-3b-qlora.yaml
```

训练要求：

- 模型：`Qwen/Qwen2.5-Coder-3B-Instruct`；
- 4-bit NF4；
- QLoRA；
- LoRA rank 16，alpha 32，dropout 0.05；
- per-device batch size 1；
- gradient accumulation 8；
- gradient checkpointing 开启；
- max sequence length 2048；
- epoch 1；
- seed 42；
- 支持 checkpoint 恢复；
- 只对 assistant 输出计算 loss；
- tool 输出保留在上下文中，但不作为 assistant 监督目标。

配置保存到：

```text
experiments/sft-v0/configs/sft-3b-qlora.yaml
```

输出保存到：

```text
experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora/
```

至少保存 `adapter_config.json`、`adapter_model.safetensors`、tokenizer 文件、`trainer_state.json`、`training_args.json` 和 `train_metrics.json`。

如果 OOM，依次将 max sequence length 从 2048 降到 1024、将 gradient accumulation 调到 16，并确认 4-bit 和 gradient checkpointing 已开启。不得改成 CPU 训练。

## 10. 训练前后独立评测

在同一批 `eval_task_ids` 上分别评测：

1. 未训练的 `Qwen/Qwen2.5-Coder-3B-Instruct)；
2. 加载 `experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora/` 的模型。

必须使用独立 workspace 和独立 Evaluator，不得使用训练期间产生的 patch。

保存：

```text
experiments/sft-v0/evaluation/base/
experiments/sft-v0/evaluation/sft/
experiments/sft-v0/evaluation/comparison.json
experiments/sft-v0/evaluation/comparison.md
```

至少记录 base 和 SFT 的任务数、通过数、通过率、训练记录数和 `smoke_only`。如果评测任务少于 3 个，标记 `evaluation_quality=smoke_only`，不能写成正式 benchmark 结果。

## 11. 保存代码修改和实验说明

```bash
git diff > experiments/sft-v0/git_diff.patch
pip freeze > experiments/sft-v0/pip_freeze.txt
```

创建 `experiments/sft-v0/README.md`，说明服务器配置、GPU、数据版本、任务划分、样本数、训练参数、训练状态、OOM 情况、base/SFT 评测结果、API 调用次数、API 成本、已知问题和下一步建议。

## 12. 清单、校验和打包

创建 `experiments/sft-v0/MANIFEST.json`，列出重要文件及 SHA256：

```bash
find experiments/sft-v0 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > experiments/sft-v0/SHA256SUMS
```

不得打包 `.env`、`DEEPSEEK_API_KEY`、Shell history、SSH 私钥或云凭据。

```bash
tar \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.cache' \
  --exclude='hf-cache' \
  --exclude='.venv' \
  -czf codeagentbench-sft-v0.tar.gz experiments/sft-v0

sha256sum codeagentbench-sft-v0.tar.gz > codeagentbench-sft-v0.tar.gz.sha256
```

如果压缩包超过上传限制，按 200MB 分片，并保存每个分片的 SHA256：

```bash
split -b 200M codeagentbench-sft-v0.tar.gz codeagentbench-sft-v0.tar.gz.part-
```

保留服务器上的原始目录，不要只保留压缩包。

## 13. 最终报告

执行结束后报告：

```text
status: completed / smoke_only / blocked
server_gpu:
gpu_memory:
cuda:
torch:
git_commit:

rollout_attempts:
successful_trajectories:
training_records:
api_total_cost_usd:

base_passed:
base_total:
base_pass_rate:

sft_passed:
sft_total:
sft_pass_rate:

training_status:
checkpoint_path:
adapter_path:
archive_path:
archive_sha256:

known_issues:
next_step:
```

如果训练、评测或打包没有真正完成，必须标记为 `blocked` 或 `smoke_only` 并说明原因。

完成 SFT 和评测后停止，不要继续执行 GRPO。
