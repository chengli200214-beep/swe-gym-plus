# ModelScope 云端实验接续记录（更新至 2026-09-25）

本记录只写入已核实的状态和复现入口，不包含 API key 或原始轨迹。项目早期的 SFT v4 报告见 [`comparison.md`](../experiments/sft-v0/evaluation/comparison.md)；报告中的 6 条训练轨迹和 3B LoRA checkpoint **不在当前公开仓库或本次 ModelScope 工作区中**，不能仅凭报告宣称已在此实例复现训练。

## 2026-09-25：工程验收与新增训练候选准入

GitHub `main`、本地与云端 `/mnt/workspace/swe-gym-plus-next` 已同步到 `2b31bf2`。旧工作树 `/mnt/workspace/swe-gym-plus-current` 的未提交训练配置与 `/mnt/workspace/swe-gym-plus/experiments/` 中的私有数据均未覆盖。云端新工作树的 Harness 回归测试通过；现有 46 条私有动作样本通过任务级分组检查，没有混入开发或测试任务。新增共享预算耗尽前阻止多候选/评测启动的保护，但仍不能把未计价的 DeepSeek `max_cost_usd` 称为真实消费硬限制。

`getmoto__moto-6641` 经模型无关控制检查准入：未修复版本 1 failed / 6 passed；官方补丁版本 7 passed。完整私有报告为 `/mnt/workspace/swe-gym-plus/experiments/quality/moto-6641-20260925.json`。它被分配到训练候选池，**没有**产生通过独立评测的 Agent 轨迹；成功轨迹仍只有原来的 5 个不同任务。现有 bash 执行器也未提供工作目录外的系统级隔离，因此新增真实 rollout 之前仍须处理隔离与费用控制。

## 2026-09-25：动作级 SFT 与固定留出集复测（最新）

上一轮的每任务一条长轨迹训练在 2048 token 上限下严重截断；训练记录的原始长度为 4,197–15,712 token。为验证工具协议学习，使用 [`scripts/prepare_action_sft.py`](../scripts/prepare_action_sft.py) 将同一批 **5 个不同任务、独立评测通过** 的轨迹转为 **46 条命令动作级监督样本**，统一为 Harness 所需 JSON 动作格式。46 条是相关的动作样本，**不是 46 个独立任务**；完成动作不进入这次训练。源数据 SHA-256 为 `8eac5ccdc54a6ed3486145bc859e6b3ae2cc8a4ea6ee2f97e1c9597305e7842d`。私有数据及收据保存在 `/mnt/workspace/swe-gym-plus/experiments/sft-bootstrap/data/train-actions-5passed.{jsonl,receipt.json}`，不上传公开仓库。

云端采用 [`configs/sft-action-5passed-amd.yaml`](../configs/sft-action-5passed-amd.yaml) 训练 Qwen2.5-Coder-0.5B-Instruct 的 BF16 LoRA（非 QLoRA）；1 epoch、`max_seq_len=3072`、46 条编码样本、0 条无监督丢弃、`global_step=46`、最终 `train_loss≈1.334`。真实 tokenizer 审计显示这 46 条样本最长 3,024 token，故本次未因 3,072 上限截断。适配器为 `/mnt/workspace/swe-gym-plus/experiments/sft-bootstrap/checkpoints/qwen2.5-coder-0.5b-actions-5passed-lora/adapter_model.safetensors`（35,231,704 字节），同目录有 `train_metrics.json`。本轮不需要 DeepSeek API；训练在 ModelScope AMD 实例 GPU 上完成。

与下节完全相同的两个留出任务、预算和独立评测流程得到：

| 留出任务 | 新运行 ID | Agent / 工具调用 / token / 补丁 | 独立评测 |
|---|---|---|---|
| `getmoto__moto-5876` | `compare-5876-action-sft-20260925` | blocked / 5 / 12,615 / 0 字节；下一步超过 16,000 token 预算 | failed，1 failed / 2 passed |
| `getmoto__moto-5085` | `compare-5085-action-sft-20260925` | failed / 6 / 12,788 / 0 字节；重复搜索同一源码位置 | failed，1 failed / 77 passed |

新模型至少学会了发起工具调用（旧 Base 和每任务一条轨迹的 SFT 均为 0 次），但**任务解决率仍为 0/2，没有修复能力提升的证据**。两次 Agent 运行均无代码补丁，隔离评测仍保持未修复失败。运行数据位于 `/mnt/workspace/swe-gym-plus/experiments/base-sft-5passed/runs/compare-{5876,5085}-action-sft-20260925/`，评测目录位于同一实验根下的 `evaluations/`。下一步应增加不同任务的真实通过轨迹，并专门抑制重复测试/搜索，再用固定留出集复测；不要通过反复训练这 5 个任务或调留出集来声称泛化。

云端工作树本次在 `5b12cef` 基础上通过编辑器同步了与 GitHub `4467ccf` 相同的 3,072-token 配置；云端连接 GitHub 曾超时，未成功切换到新提交。恢复连接后先检查 `git status`，再同步提交，**不要覆盖 `/mnt/workspace/swe-gym-plus/experiments/` 中的私有实验数据**。公开仓库保存代码、配置和本记录，不含原始轨迹与模型权重。

## 2026-09-25：五样本 SFT 的首次有效留出对照

在已保存的五样本 Qwen2.5-Coder-0.5B-Instruct LoRA 上，使用云端本地基座模型与 LoRA 适配器分别运行两个不在这五条训练数据中的 getmoto 任务。运行代码为 `b512dc9`；共同预算为 8 步、12 次工具调用、16,000 总 token、600 秒、单次最多 512 新 token。均使用同一 SWE-Gym manifest、相同独立评测流程和已补齐的测试依赖，无需 DeepSeek API。原始运行目录位于 `/mnt/workspace/swe-gym-plus/experiments/base-sft-5passed/`，仍只保存在 ModelScope 工作区，**没有公开上传轨迹或权重**。

| 留出任务 | 质量控制：未修复 / 官方补丁 | Base 运行 ID 与独立评测 | SFT 运行 ID 与独立评测 |
|---|---|---|---|
| `getmoto__moto-5876` | 1 failed / 2 passed；3 passed，准入 | `compare-5876-base-valid-20260925`：1 failed / 2 passed | `compare-5876-sft-20260925`：1 failed / 2 passed |
| `getmoto__moto-5085` | 1 failed / 77 passed；78 passed，准入 | `compare-5085-base-valid-20260925`：1 failed / 77 passed | `compare-5085-sft-20260925`：1 failed / 77 passed |

四次有效运行的 Agent 状态虽然都是 `completed`，却都在第 1 步结束、**0 次工具调用、0 字节补丁**，独立评测均为 `failed`。Base 与 SFT 的任务解决数同为 **0/2**；这只说明本次极小样本训练未在这两个任务上体现改善，不能推断一般化解决率。轨迹中的模型回复有直接宣称完成、却未执行工具的情况，因此 `completed` 绝不等于解决任务。

新实例最初缺少 `sure`、`python-jose[cryptography]`、`responses`、`xmltodict`、`freezegun`、`pytz` 等 Moto 测试依赖；依赖缺失时的 `compare-5876-base-20260925` 收集失败，不计入对照。并行创建 `5085` 工作区时还触发仓库快照缓存的 `FileExistsError`；报错的 `compare-5085-base-20260925` 同样不计入对照，串行重跑成功。仓库代码已修复同任务并发快照的竞争，加入回归测试；在本地完整测试套件通过后才提交。云端 worktree 在下次拉取新提交前仍是旧代码。

下一轮应先检查训练数据中的工具动作监督格式和本地模型首步输出，再扩大不同任务的真实通过轨迹，并在固定留出集上复测；不要把五样本训练或 `0/2` 包装成 SFT 提升。若换用更大的模型或重新训练，需单独记录模型、数据、预算和显卡配置，不能与本轮混为同一次实验。

## 最新进展：5 条通过轨迹与云端 SFT

本轮训练使用的代码版本为 `e921a56`；实验结束后云端工作树已同步到包含本记录的最新 `main`。从 10 个冻结的 getmoto 训练候选中继续做未修复/官方补丁控制测试、Agent rollout 和独立评测；未准入或最终评测失败的运行不进入训练集。新增通过任务为 `getmoto__moto-7023`（`ms-7023-20260923-a`，Agent completed，9/9 passed）、`getmoto__moto-6208`（`ms-6208-20260923-a`，Agent blocked，5/5 passed）、`getmoto__moto-7061`（`ms-7061-20260923-a`，Agent blocked，8/8 passed）。加上已有的 `7365`、`6920`，共 **5 个不同任务的独立评测通过轨迹**，但其中 **3 个 Agent 运行状态为 failed/blocked**；这里只把最终补丁评测通过与 Agent 正常完成分开记录。`6190` 控制测试收集失败，`6410`、`5699`、`6509`、`5949` 的本轮 Agent 补丁未通过；`7168` 控制测试准入但未用于训练。`4950`、`7456` 仍保留为 held-out 候选。

5 个私有导出位于 `/mnt/workspace/swe-gym-plus/artifacts/exports/ms-{7365-20260923-c,6920-20260923-a,7023-20260923-a,6208-20260923-a,7061-20260923-a}.jsonl`。`scripts/prepare_passed_sft.py` 核对独立评测通过、任务/运行 ID 唯一及 assistant 监督后，生成 `/mnt/workspace/swe-gym-plus/experiments/sft-bootstrap/data/train-5-passed.jsonl`（5 行，SHA-256 `8eac5ccdc54a6ed3486145bc859e6b3ae2cc8a4ea6ee2f97e1c9597305e7842d`）及同目录的 `train-5-passed.receipt.json`（每个导出的来源与校验和）。原始轨迹、合并数据和模型权重未公开上传 GitHub；换电脑需登录同一 ModelScope 工作区确认 `/mnt/workspace` 数据仍在。

2026-09-25 追加审计：上述 5 个私有导出各自的首条 Agent 用户提示均包含非空 `allowed_test_command`，来源是旧版代码误将评测专用 `test_command` 暴露给 Agent。它们的独立测试通过记录有效，但不能作为无评测命令泄漏的干净轨迹或盲测证据；已训练的 0.5B 适配器属于这一历史小样本链路验证，后续正式训练需使用修复后的新轨迹，旧文件不删改。修复代码提交 `b56e7b6` 已在云端工作树通过 4 项无密钥关键回归测试。

使用 [`configs/sft-5passed-amd.yaml`](../configs/sft-5passed-amd.yaml) 先 dry-run，再在云端 AMD GPU 上进行 **Qwen2.5-Coder-0.5B-Instruct BF16 LoRA**（非 QLoRA）：5 条训练记录、5 条编码样本、0 条因无监督被丢弃，`max_seq_len=2048`、1 epoch、`global_step=5`、最终记录的 `train_loss≈1.664`。训练产物：`/mnt/workspace/swe-gym-plus/experiments/sft-bootstrap/checkpoints/qwen2.5-coder-0.5b-5passed-lora/`，其中 `adapter_model.safetensors` 约 34 MB、`train_metrics.json` 可复查。轨迹远长于 2048 token 时会截断，故该训练只是五样本链路验证；**尚未在固定 held-out 任务上完成 Base/SFT 对照，不能宣称解决率或模型质量提升**。这一步训练直接使用云端已有的本地基座模型，不需要调用 DeepSeek API；DeepSeek key 仅用于先前的轨迹生成，未写入仓库。

## 本次环境

- `moto-7365` 复验使用 `a70e188`；`moto-6920`、`moto-5085`、`moto-5386` 使用 `6c4da71`；随后依次切换到 `e763208`（无补丁提醒）、`83ed644`（JSON 响应格式）和 `2c22e5c`（空响应限次重试）。
- ModelScope Code Workspace：`/mnt/workspace/swe-gym-plus-current` 是独立 worktree，当前 detached HEAD `2c22e5c`；原目录 `/mnt/workspace/swe-gym-plus` 保持原状，并存放仓库缓存和本次运行数据。
- 数据版本：`data/manifests/swegym-smoke.json`，SWE-Gym revision `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`，共 10 个 smoke 任务。
- 云端 PyTorch `2.12.0+git6bbd260`，ROCm `7.2.53211`，`torch.cuda.is_available()` 为 `True`。`bitsandbytes` 尚未安装；不能据此认定 QLoRA 训练环境已就绪。
- 新 DeepSeek API key 的连通性已由最小请求验证；`moto-6920` 使用 `deepseek-flash`，默认关闭思考模式、单次最多 4,096 输出 token。API key **未写入 Git、配置或本文档**；当前只在云端运行终端的环境变量中可用。云端终端和平台日志不可视为绝对安全，用后应轮换；新终端/重启实例后需重新注入。
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
| `getmoto__moto-5212` 质量检查 | admitted | 未修复版 3 failed / 46 passed，官方补丁版 49 passed。 |
| `ms-5212-20260923-a` | 独立评测 failed | `e763208` + `deepseek-v4-pro` 的首次响应是 DSML 工具调用文本，不是 Harness 要求的 JSON；`summary.json` 为 `model response is not valid JSON`，仅用 761 token、无工具调用或补丁。该次是协议适配失败，不代表任务无法修复。 |
| `ms-5212-20260923-b` | 独立评测 failed，Agent 状态 completed | `83ed644` + `deepseek-v4-pro` 的 JSON 模式在真实小请求及本次 rollout 中可用；Agent 14 步、13 次工具调用、46,546 token，产生 `moto/ec2/models/vpcs.py` 补丁并正常结束，但隔离评测 5 failed / 44 passed（未修复版为 3 failed / 46 passed），`verdict=failed`。不导出 SFT。 |
| `ms-5876-20260923-c` | 独立评测 failed | `83ed644` + `deepseek-v4-pro` 首次响应为空，运行器报 `model response is not valid JSON`；1,909 token、无工具调用或补丁，隔离评测仍为 1 failed / 2 passed。说明 JSON 模式不保证每次返回非空动作。 |
| `ms-5876-20260923-d` | 独立评测 failed，Agent 状态 blocked | `2c22e5c` 加入空响应最多一次重试后，模型执行到第 15 步、14 次工具调用、56,309 token，生成约 999 字节 Cognito 补丁；下一步超过 60,000 token 上限。隔离评测 2 failed / 1 passed，`verdict=failed`；不导出 SFT。 |

原始运行数据保存在 ModelScope 的 `/mnt/workspace/swe-gym-plus/artifacts/runs/<run-id>/`。前一阶段先有 **2 个不同任务** 的真实通过样本：`/mnt/workspace/swe-gym-plus/artifacts/exports/ms-7365-20260923-c.jsonl`（1 条、约 66 KB；18 次模型动作、18 次工具反馈；同时保留 `evaluation_verdict=passed` 和 `agent_status=failed`；SHA-256 `df08cd1652e2297ef0b501167162c0e2a9f9bfae92a7467f1a414f33c9912136`），以及 `/mnt/workspace/swe-gym-plus/artifacts/exports/ms-6920-20260923-a.jsonl`（1 条、20 条消息、9 次工具调用、33,206 token、19,246 字节；`evaluation_verdict=passed`、`agent_status=completed`）。最新已增至上文所述的 5 条。其他失败轨迹不应导入 SFT 训练集。实例不应被删除；停止/重启前应确认工作区仍可访问，并将需要跨平台保存的结果另行备份。

## 云端训练链路验证

用 `configs/sft-bootstrap-amd.yaml`、上述真实通过样本和云端已有的 Qwen2.5-Coder-0.5B-Instruct 做了单步 BF16 LoRA 烟雾测试；`load_in_4bit=false`，不是 QLoRA。训练前 dry-run 验证了 1 条记录、18 个 assistant 回合及 18 个工具反馈；真实 tokenizer 在 512 token 截断后仍保留 123 个 assistant 监督 token。GPU 训练完成，`global_step=1`、`encoded_examples=1`，adapter 文件约 35 MB，保存在 `/mnt/workspace/swe-gym-plus/experiments/sft-bootstrap/checkpoints/ms-7365-20260923-c-smoke/`，对应 `train_metrics.json` 可核验。单样本单步训练仅证明链路可运行，不能用于宣称模型质量提升。

从另一台电脑登录同一 ModelScope 工作区后，先检查上述样本与 checkpoint 文件是否仍在 `/mnt/workspace`，再进入 `/mnt/workspace/swe-gym-plus-current` 继续。云端运行目录和样本**尚未上传到公开 GitHub**；本仓库只保存状态与路径。若需要在 ModelScope 之外训练，必须先备份样本并检查其中是否包含敏感信息，不能直接把原始轨迹公开。

## 继续实验的门槛

1. 为新的训练任务重复“质量检查 → rollout → 独立评测 → 仅导出通过轨迹”，并将训练/held-out 任务隔离。不要用失败轨迹、演示数据或 held-out 任务充数；现有 5 条不同任务的真实通过样本只足以验证小样本训练链路，不足以推断泛化效果。`getmoto__moto-4950` 已出现在旧版 3B SFT v4 训练报告中，不能作为该模型系的盲测任务；`getmoto__moto-7456` 仅为未使用的测试候选，正式盲测池仍需扩充并冻结。任务级分组见 [`swegym-moto-v1.json`](../data/splits/swegym-moto-v1.json)。
2. AMD 云实例上 BF16 LoRA 的 0.5B 单步 smoke 与 5 条样本、5 步训练均已通过。正式 3B SFT 仍需更多真实通过样本、固定配置及训练/评测隔离；若走原定 4-bit QLoRA，须先验证 `bitsandbytes` 对当前 ROCm 的兼容性。BF16 LoRA 是不同配置，不能混称 QLoRA。
3. 用固定任务和相同预算进行 Base/SFT 独立评测。当前历史报告为 0/3 对 0/3，不可宣称 SFT 带来提升。

已发现的预算缺陷：DeepSeek 响应包含 token 用量，但当前适配器从不存在的 `cost_usd` 字段读取费用，运行中会将 USD 费用记为 0。因此 `--max-cost-usd` **不是有效的 DeepSeek 硬限制**；在修复计费前，只能用步数、总 token 数、单次输出 token 和运行时间限制单次实验。`ms-6920-20260923-a` 导出记录中的 `cost_usd=0.0` 并不代表实际免费。

本次还发现 ModelScope 的登录 Bash 会在每次 Agent 工具输出前注入冗长的平台欢迎横幅。运行器已改为在原生 Bash 使用非登录 `bash -c`，并加入回归测试；前三次失败运行均使用修复前代码，不能把它们归因于模型本身而忽略环境干扰。

`ms-5085-20260923-a` 和 `ms-5386-20260923-a` 暴露了 Agent 在源码搜索上消耗全部动作/token 的问题。进一步审查发现，六次无补丁动作后的 Harness 提醒写入消息后可能立即被上下文压缩清除；`e763208` 已修复为压缩后追加提醒，并覆盖后续无补丁动作，增加回归测试。该修复未在上述两次已结束的运行中生效，不能把它们算作修复后的对照结果。

`ms-5212-20260923-a` 发现 Pro 模型可能返回非 JSON 工具文本。适配器按 DeepSeek 官方 JSON Output 接口添加 `response_format={"type":"json_object"}`，完成本地回归测试与真实 API 小请求验证；`ms-5212-20260923-b` 证明协议可工作，但补丁未通过隔离评测。`ms-5876-20260923-c` 又观察到一次空响应；`2c22e5c` 加入最多一次重试并累计两次请求的 token 用量，本地全套 49 个测试通过，云端适配器测试通过，`ms-5876-20260923-d` 确实进入多步工具循环，但最终补丁仍未通过。该重试不是解决率提升的证据。

`ms-7365-20260923-b` 已验证横幅消失，且产生了候选补丁，但也暴露了模型在 hidden tests 不在工作区时未检查新增符号导入的问题。系统提示因此加入通用的“检查新增名字定义/导入、无可见测试时做最小 smoke”要求；`ms-7365-20260923-c` 在新提示下独立评测通过，但其 Agent 仍因预算上限结束。后续应分别统计 Agent 完成率与最终补丁通过率。
