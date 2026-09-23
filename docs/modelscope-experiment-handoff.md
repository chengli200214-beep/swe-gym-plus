# ModelScope 云端实验接续记录（2026-09-23）

本记录只写入已核实的状态和复现入口，不包含 API key 或原始轨迹。项目早期的 SFT v4 报告见 [`comparison.md`](../experiments/sft-v0/evaluation/comparison.md)；报告中的 6 条训练轨迹和 3B LoRA checkpoint **不在当前公开仓库或本次 ModelScope 工作区中**，不能仅凭报告宣称已在此实例复现训练。

## 本次环境

- `moto-7365` 复验使用 `a70e188`；新的 `moto-6920` 运行使用 `6c4da71`（DeepSeek 单次输出上限与非思考模式默认值已加入）。
- ModelScope Code Workspace：`/mnt/workspace/swe-gym-plus-current` 是独立 worktree，本轮已切换到 detached HEAD `6c4da71`；原目录 `/mnt/workspace/swe-gym-plus` 保持原状，并存放仓库缓存和本次运行数据。
- 数据版本：`data/manifests/swegym-smoke.json`，SWE-Gym revision `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`，共 10 个 smoke 任务。
- 云端 PyTorch `2.12.0+git6bbd260`，ROCm `7.2.53211`，`torch.cuda.is_available()` 为 `True`。`bitsandbytes` 尚未安装；不能据此认定 QLoRA 训练环境已就绪。
- 新 DeepSeek API key 的连通性已由最小请求验证；`moto-6920` 使用 `deepseek-flash`，默认关闭思考模式、单次最多 4,096 输出 token。API key 只注入当前云端终端的环境变量，**未写入 Git、配置或本文档**。新终端/重启实例后需重新注入。
- 为任务测试补装了云端 Python 依赖 `freezegun==1.5.5`、`docker==7.2.0` 和 `pytz==2026.3.post1`；新实例必须重新安装或使用可复现环境。

## 已完成的任务与证据

| 任务/运行 ID | 结果 | 证据或原因 |
|---|---|---|
| `getmoto__moto-5876` 质量检查 | admitted | 未修复版 1 failed / 2 passed；官方补丁版 3 passed。 |
| `ms-5876-20260923-a` | 独立评测 failed | 22,000 token 上限前只完成 6 次工具调用，未产生补丁。 |
| `ms-5876-20260923-b` | 独立评测 failed | 60,000 token 上限前仍未产生补丁，工作区无代码改动。 |
| `getmoto__moto-7365` 质量检查 | admitted | 未修复版失败、官方补丁版通过；当前适合用于新一轮轨迹复现。 |
| `ms-7365-20260923-a` | 独立评测 failed | 达到 14 步上限，未产生补丁。不能因历史报告中的同任务成功记录而将本次失败轨迹导入训练。 |
| `ms-7365-20260923-b` | 独立评测 failed | 去除平台横幅后生成了 DynamoDB Decimal 补丁，但遗漏 `Decimal` 导入；形式测试报 `NameError`。 |
| `ms-7365-20260923-c` | **独立评测 passed，Agent 状态 failed** | Agent 到第 17 步触及预算，`summary.json` 记录 `maximum agent steps reached`；留下的补丁改用已导入的 `decimal.Decimal`，隔离评测 `exit_code=0`、无失败测试、`verdict=passed`。不能写成 Agent 正常完成。 |
| `getmoto__moto-6920` 质量检查 | admitted | 初次因缺少 `freezegun`、`docker` 在收集阶段未准入；补齐依赖后，未修复版 1 failed / 7 passed，官方补丁版 8 passed。 |
| `ms-6920-20260923-a` | **独立评测 passed，Agent 状态 completed** | 新 key + `deepseek-flash`、16 步/20 工具调用/45,000 总 token/900 秒限制；Agent 完成并生成 Lambda Layer 补丁，隔离评测 8 passed、`verdict=passed`。 |
| `getmoto__moto-5085` 质量检查 | admitted | 初次因缺少 `pytz` 未准入；补齐后未修复版 1 failed / 77 passed，官方补丁版 78 passed。 |
| `ms-5085-20260923-a` | 独立评测 failed | `deepseek-flash` 在 20 步/20 次工具调用后未产出补丁，`summary.json` 为 `maximum agent steps reached`，隔离评测仍是 1 failed / 77 passed；51,110 token。轨迹显示反复搜索相关源码但没有编辑，不导入 SFT。 |
| `getmoto__moto-5386` 质量检查 | admitted | 未修复版 2 failed / 83 passed，官方补丁版 85 passed。 |
| `ms-5386-20260923-a` | 独立评测 failed，Agent 状态 blocked | `deepseek-flash` 在 16 次工具调用后累计 49,567 token，下一步超过 50,000 token 上限；没有补丁，隔离评测仍是 2 failed / 83 passed。不导入 SFT。 |

原始运行数据保存在 ModelScope 的 `/mnt/workspace/swe-gym-plus/artifacts/runs/<run-id>/`。真实 SWE-Gym 通过样本目前来自 **2 个不同任务**：此前的 `/mnt/workspace/swe-gym-plus/artifacts/exports/ms-7365-20260923-c.jsonl`（1 条、约 66 KB；18 次模型动作、18 次工具反馈；同时保留 `evaluation_verdict=passed` 和 `agent_status=failed`；SHA-256 `df08cd1652e2297ef0b501167162c0e2a9f9bfae92a7467f1a414f33c9912136`），以及新的 `/mnt/workspace/swe-gym-plus/artifacts/exports/ms-6920-20260923-a.jsonl`（1 条、20 条消息、9 次工具调用、33,206 token、19,246 字节；`evaluation_verdict=passed`、`agent_status=completed`）。其他失败轨迹不应导入 SFT 训练集。实例不应被删除；停止/重启前应确认工作区仍可访问，并将需要跨平台保存的结果另行备份。

## 云端训练链路验证

用 `configs/sft-bootstrap-amd.yaml`、上述真实通过样本和云端已有的 Qwen2.5-Coder-0.5B-Instruct 做了单步 BF16 LoRA 烟雾测试；`load_in_4bit=false`，不是 QLoRA。训练前 dry-run 验证了 1 条记录、18 个 assistant 回合及 18 个工具反馈；真实 tokenizer 在 512 token 截断后仍保留 123 个 assistant 监督 token。GPU 训练完成，`global_step=1`、`encoded_examples=1`，adapter 文件约 35 MB，保存在 `/mnt/workspace/swe-gym-plus/experiments/sft-bootstrap/checkpoints/ms-7365-20260923-c-smoke/`，对应 `train_metrics.json` 可核验。单样本单步训练仅证明链路可运行，不能用于宣称模型质量提升。

从另一台电脑登录同一 ModelScope 工作区后，先检查上述样本与 checkpoint 文件是否仍在 `/mnt/workspace`，再进入 `/mnt/workspace/swe-gym-plus-current` 继续。云端运行目录和样本**尚未上传到公开 GitHub**；本仓库只保存状态与路径。若需要在 ModelScope 之外训练，必须先备份样本并检查其中是否包含敏感信息，不能直接把原始轨迹公开。

## 继续实验的门槛

1. 为新的训练任务重复“质量检查 → rollout → 独立评测 → 仅导出通过轨迹”，并将训练/held-out 任务隔离。不要用失败轨迹、演示数据或 held-out 任务充数；目前有 2 条不同任务的真实通过样本，尚不足以做有意义的 SFT 效果对照。`getmoto__moto-4950` 与 `getmoto__moto-7456` 暂留为 held-out 候选，收集训练轨迹时不要使用。
2. AMD 云实例上 BF16 LoRA 的 0.5B 单步 smoke 已通过。正式 3B SFT 仍需足够的真实通过样本、固定配置及训练/评测隔离；若走原定 4-bit QLoRA，须先验证 `bitsandbytes` 对当前 ROCm 的兼容性。BF16 LoRA 是不同配置，不能混称 QLoRA。
3. 用固定任务和相同预算进行 Base/SFT 独立评测。当前历史报告为 0/3 对 0/3，不可宣称 SFT 带来提升。

已发现的预算缺陷：DeepSeek 响应包含 token 用量，但当前适配器从不存在的 `cost_usd` 字段读取费用，运行中会将 USD 费用记为 0。因此 `--max-cost-usd` **不是有效的 DeepSeek 硬限制**；在修复计费前，只能用步数、总 token 数、单次输出 token 和运行时间限制单次实验。`ms-6920-20260923-a` 导出记录中的 `cost_usd=0.0` 并不代表实际免费。

本次还发现 ModelScope 的登录 Bash 会在每次 Agent 工具输出前注入冗长的平台欢迎横幅。运行器已改为在原生 Bash 使用非登录 `bash -c`，并加入回归测试；前三次失败运行均使用修复前代码，不能把它们归因于模型本身而忽略环境干扰。

`ms-5085-20260923-a` 和 `ms-5386-20260923-a` 暴露了 Agent 在源码搜索上消耗全部动作/token 的问题。进一步审查发现，六次无补丁动作后的 Harness 提醒写入消息后可能立即被上下文压缩清除；本地已修复为压缩后追加提醒，并覆盖后续无补丁动作，增加回归测试。该修复尚未在上述两次已结束的运行中生效，不能把它们算作修复后的对照结果。

`ms-7365-20260923-b` 已验证横幅消失，且产生了候选补丁，但也暴露了模型在 hidden tests 不在工作区时未检查新增符号导入的问题。系统提示因此加入通用的“检查新增名字定义/导入、无可见测试时做最小 smoke”要求；`ms-7365-20260923-c` 在新提示下独立评测通过，但其 Agent 仍因预算上限结束。后续应分别统计 Agent 完成率与最终补丁通过率。
