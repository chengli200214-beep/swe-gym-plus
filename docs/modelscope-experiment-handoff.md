# ModelScope 云端实验接续记录（2026-09-23）

本记录只写入已核实的状态和复现入口，不包含 API key 或原始轨迹。项目早期的 SFT v4 报告见 [`comparison.md`](../experiments/sft-v0/evaluation/comparison.md)；报告中的 6 条训练轨迹和 3B LoRA checkpoint **不在当前公开仓库或本次 ModelScope 工作区中**，不能仅凭报告宣称已在此实例复现训练。

## 本次环境

- GitHub `main`：`11cf4bcea3c75878e9303ad942982c11c0be6cef`。
- ModelScope Code Workspace：`/mnt/workspace/swe-gym-plus-current` 是从上述提交创建的独立 worktree；原目录 `/mnt/workspace/swe-gym-plus` 保持原状，并存放仓库缓存和本次运行数据。
- 数据版本：`data/manifests/swegym-smoke.json`，SWE-Gym revision `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`，共 10 个 smoke 任务。
- 云端 PyTorch `2.12.0+git6bbd260`，ROCm `7.2.53211`，`torch.cuda.is_available()` 为 `True`。`bitsandbytes` 尚未安装；不能据此认定 QLoRA 训练环境已就绪。
- DeepSeek API 连通性已由最小请求验证（HTTP 200）。API key 只注入当前云端终端的环境变量，**未写入 Git、配置或本文档**。新终端/重启实例后需重新注入。

## 已完成的任务与证据

| 任务/运行 ID | 结果 | 证据或原因 |
|---|---|---|
| `getmoto__moto-5876` 质量检查 | admitted | 未修复版 1 failed / 2 passed；官方补丁版 3 passed。 |
| `ms-5876-20260923-a` | 独立评测 failed | 22,000 token 上限前只完成 6 次工具调用，未产生补丁。 |
| `ms-5876-20260923-b` | 独立评测 failed | 60,000 token 上限前仍未产生补丁，工作区无代码改动。 |
| `getmoto__moto-7365` 质量检查 | admitted | 未修复版失败、官方补丁版通过；当前适合用于新一轮轨迹复现。 |
| `ms-7365-20260923-a` | 独立评测 failed | 达到 14 步上限，未产生补丁。不能因历史报告中的同任务成功记录而将本次失败轨迹导入训练。 |
| `ms-7365-20260923-b` | 独立评测 failed | 去除平台横幅后生成了 DynamoDB Decimal 补丁，但遗漏 `Decimal` 导入；形式测试报 `NameError`。 |

原始运行数据保存在 ModelScope 的 `/mnt/workspace/swe-gym-plus/artifacts/runs/<run-id>/`。失败轨迹不应导入 SFT 训练集。实例不应被删除；停止/重启前应确认工作区仍可访问，并将需要跨平台保存的结果另行备份。

## 继续实验的门槛

1. 对 `moto-7365` 在修复过的 Harness 上重试 DeepSeek rollout；只以 `events.jsonl` 最后一条独立 `evaluation.verdict == "passed"` 为成功依据。通过后才运行 `python -m codeagentbench export-sft <events.jsonl> <output.jsonl>`。
2. 为新的训练任务重复“质量检查 → rollout → 独立评测 → 仅导出通过轨迹”，并将训练/held-out 任务隔离。不要用失败轨迹、演示数据或 held-out 任务充数。
3. 在这台 AMD 实例上训练前，先验证 `bitsandbytes`/PEFT/TRL 对当前 ROCm 版本和本项目 QLoRA 配置的兼容性；通过一次最小训练 smoke 后再启动正式 SFT。
4. 用固定任务和相同预算进行 Base/SFT 独立评测。当前历史报告为 0/3 对 0/3，不可宣称 SFT 带来提升。

已发现的预算缺陷：DeepSeek 响应包含 token 用量，但当前适配器从不存在的 `cost_usd` 字段读取费用，运行中会将 USD 费用记为 0。因此 `--max-cost-usd` **不是有效的 DeepSeek 硬限制**；在修复计费前，只能用步数、总 token 数和运行时间限制单次实验。不要把记录的 0 美元理解为实际免费。

本次还发现 ModelScope 的登录 Bash 会在每次 Agent 工具输出前注入冗长的平台欢迎横幅。运行器已改为在原生 Bash 使用非登录 `bash -c`，并加入回归测试；上述三次失败运行均使用修复前代码，不能把它们归因于模型本身而忽略环境干扰。后续重试需使用包含此修复的新提交。

`ms-7365-20260923-b` 已验证横幅消失，且产生了候选补丁，但也暴露了模型在 hidden tests 不在工作区时未检查新增符号导入的问题。系统提示现已加入通用的“检查新增名字定义/导入、无可见测试时做最小 smoke”要求；下一轮需要在更新后的提交上验证，不能将这次失败轨迹视为成功样本。
