# SWE-Gym Plus：SFT v3 实验报告

日期：2026-09-20  
项目：[swe-gym-plus](https://github.com/chengli200214-beep/swe-gym-plus)

## 1. 实验目标

在 SWE-Gym/Moto 代码修复任务上，使用 DeepSeek 生成可验证的 coding-agent 轨迹，训练 Qwen2.5-Coder-3B-Instruct 的 QLoRA 适配器，并比较 Base 模型与 SFT 模型的独立评测结果。

本轮执行到“实验报告”阶段为止，未进入 GRPO、README 包装或额外数据扩充。

## 2. 数据构造

- 训练轨迹：5 条通过独立形式测试的轨迹
- 原有成功轨迹：`moto-7365`、`moto-6920`、`moto-4950`、`moto-5072`
- 本轮新增成功轨迹：`moto-6998`
- 本轮新增 DeepSeek rollout：8 次（包含 2 次重试）
- 本轮新增通过数：1/8
- 失败样本未加入 SFT 数据集
- 数据文件：`experiments/sft-v0/data/sft/train-v3.jsonl`

独立评测使用新工作区运行 frozen test command；模型输出的“声称已修改”不作为成功依据。

## 3. SFT 配置

- Base：`Qwen/Qwen2.5-Coder-3B-Instruct`
- 方法：QLoRA，4-bit NF4，LoRA rank 16，alpha 32，dropout 0.05
- Epochs：3
- Learning rate：`1e-4`
- Max sequence length：2048
- Gradient accumulation：1
- Train records：5
- 训练结果：completed
- Train loss：约 `0.8822`
- Checkpoint：`experiments/sft-v0/checkpoints/qwen2.5-coder-3b-qlora-v3`
- 配置：[configs/sft-v3-qlora.yaml](../../../configs/sft-v3-qlora.yaml)

## 4. Base 与 SFT 对照

两组使用相同任务、相同运行预算和相同独立评测流程。

| 任务 | Base | SFT v3 | 主要失败原因 |
|---|---|---|---|
| `moto-5085` | failed | failed | 重复探索/重复命令导致无进展 |
| `moto-5968` | failed | failed | Base 出现非 JSON 回复；SFT 达到最大步数 |
| `moto-5725` | failed | failed | Base 出现非 JSON 回复；SFT 重复失败的 `sed` |
| 合计 | 0/3 passed | 0/3 passed | 本轮小样本未观察到 fail-to-pass |

这不是 SFT 无效的充分证明：训练集只有 5 条轨迹，且本地 3B 模型在工具调用格式、精确定位代码和安全编辑命令上仍存在明显瓶颈。当前结果应作为工程基线和失败分析，而不是对模型能力的最终结论。

## 5. 对照暴露的问题与修复

已在代码中完成并通过完整测试：

1. 运行层现在可以从“解释文字 + JSON fenced block”中提取动作。
2. 兼容模型错误生成的 `\\'` shell 单引号转义，不再直接因格式错误终止。
3. 可以解析模型退化为 `**Edit Command:**` fenced shell 命令的回复。
4. shell 命令失败或同一命令/工作区/结果重复时，向模型注入明确的替代策略提示。
5. 保留重复无进展保护，避免 agent 无限重试。

验证结果：本地完整 `pytest -q` 通过。

对应提交：

- `9f3a779` — recover mixed action output and failed command loops
- `2896135` — parse labeled shell commands from model replies

## 6. 结论与限制

当前项目已经具备可展示的闭环：任务导入 → agent rollout → 轨迹保存 → 独立评测 → 失败分类 → SFT → Base/SFT 对照 → 实验报告。

本轮不应把 `0/3` 对照结果包装成提升结果。下一阶段若继续，应优先使用修复后的 runtime 重新跑 3 个 held-out 任务，并扩大通过轨迹数量，再考虑 GRPO。
