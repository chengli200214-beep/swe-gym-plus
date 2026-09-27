# SWE-Gym-Plus 第二轮：训练与执行输入对齐

日期：2026-09-27。当前状态：GPU 训练、完整配对评测和独立验收均已完成，`verification.json` 为 `verified_completed`。第二轮 Base 通过 1/12，SFT 通过 0/12，未证明修复能力提升。

## 本轮范围

使用现有 ModelScope AMD 实例，保留第一轮结果，不追加付费 DeepSeek 采集、不租新实例、不使用本机 GPU、不推送 GitHub。原始轨迹与权重仍仅在云端；资产哈希清单不是独立备份。

原实验根目录：`/mnt/workspace/swe-gym-plus/experiments/formal-accelerated-20260926`。

新实验根目录：`/mnt/workspace/swe-gym-plus/experiments/aligned-round2-20260927`。

云端源码目录：`/mnt/workspace/swe-gym-plus-completion`。

## 诊断依据与边界

第一轮 250 条训练样本的角色形状均为 `system,user,assistant`；真实执行输入包含先前助手动作、工具回执和运行层反馈，存在训练/执行输入差异。第一轮 SFT 在准入的 12 个 eval 任务上重复退出 10 次、达到步数上限 2 次、生成补丁 0 个。多次重复退出发生在首次上下文压缩之前，因此不能只归因于压缩，也不能把输入差异认定为已证实的唯一原因。

第二轮是对此假设的诊断实验，不是保证提升的修复。第一轮留出题已被用于诊断，第二轮相同题目属于迭代评测，不再是未接触的 sealed test。

## 已冻结的改动

- 新增共享 `last-action-v2` 上下文：原始系统提示、原始题面、最近一次真实助手动作及对应工具回执。真实错误或超时才产生确定性的失败提醒。
- 工具 stdout/stderr 各保留前 4,000 个字符，并标明工具输出裁剪；不裁剪训练目标或悄悄丢弃过长样本。
- Base 和 SFT 在本轮使用同一上下文规则；该规则会丢失更早的探索历史，是明确的设计取舍。
- 根据源轨迹及其权威事件重建动作样本，逐条匹配实际命令与回执，要求最终结束动作。
- 仅下一条助手动作及结束标记计算损失；历史助手动作只作为输入，不再计算损失。
- 默认 `native` 行为不变；第二轮通过独立 CLI 模式启用，不改写第一轮数据、权重或结果。
- 运行层原有隔离、费用/动作/时间限制及重复命令失败关闭保持不变。

运行期间不要修改 `src/codeagentbench` 或 `scripts`，否则实验源码身份会不一致。

## 已完成核验

部署前核对第一轮实际 56 个 Python 源文件哈希，并保存 `source-before/`；新版本保存为 `source-after/`，部署补丁和收据分别为 `code.patch`、`setup.receipt.json`。

第一轮不可变资产：

| 资产 | SHA256 |
| --- | --- |
| report.json | 02ad2c43198dbd95dc0dbee1d4867707cdd0c5df4655349374e11bb61762e110 |
| verification.json | 98c9225c44123f58f6dde9de613430ee321ac0c00c4197d72e2f3da0cc338700 |
| adapter_model.safetensors | ab6822e542788fcad89629501d33b2fbb13c3315fc7aa4cac1bbe64279f04705 |

实例重启后缺少 bubblewrap 及此前的 Moto 测试依赖。用户明确授权后，仅从 Ubuntu 官方 jammy 源恢复以下包及其声明依赖，未使用现有 AMD、Broadcom 或 PPA 源安装本轮软件：

- bubblewrap `0.6.1-1ubuntu0.3`
- python3-sure `2.0.0-1build1`
- python3-responses `0.18.0-1`
- python3-tz `2022.1-1ubuntu0.22.04.1`
- python3-xmltodict `0.12.0-2`

`dependencies.receipt.json` 记录完整变更清单及返回码 0。`sandbox.receipt.json` 的真实退出码为 0，探针确认工作目录为 `/workspace`、私有 `/mnt/workspace` 不可见、无凭据环境变量、上述 Python 依赖可导入。未扩大沙箱挂载或网络权限。

云端选定合同测试 JUnit：38 tests、0 failures、0 errors、0 skipped；耗时 17.081 秒。文件：`contracts.xml`、`contracts.log`。

本机补充默认模式回归 JUnit：55 tests、53 passed、1 failure、1 skipped，耗时 258.577 秒，文件 `.pytest-round2-native-regression-20260927.xml`。失败项为 Windows SQLite Worker 演示任务：在 Agent 启动前，`git init --quiet` 返回 128，未生成 run summary。只读复现显示 `_env()` 保留 `GIT_CONFIG_COUNT`、`GIT_CONFIG_VALUE_*` 却过滤掉 `GIT_CONFIG_KEY_*`，Git 报 `missing config key GIT_CONFIG_KEY_0`。这是本机服务环境过滤的已有缺陷；本轮未改 Worker，云端协调器不用该 Worker。保留待修项，不把本机回归描述为全通过，也不在活动实验期间修改冻结源码。

启动前，20 个源导出文件的名称及 SHA256 与第一轮 `experiment.json` 完全一致。

## 实际数据审计

真实 Qwen tokenizer 审计返回码 0；生成前缀稳定性检查已通过。

| 指标 | 第二轮实际值 |
| --- | --- |
| 不同训练任务 | 20 |
| 动作样本 | 250 |
| 结束动作 | 20 |
| 总输入 token | 433,848 |
| 最大样本 token | 3,856 |
| 监督 token | 28,181 |
| 最大目标 token | 1,079 |
| 超过 768-token 目标 | 1 |
| 目标截断 / 拒绝 | 0 / 0 |

总 token 与第一轮不同主要来自上下文和动作规范化，不表示新增独立任务。工具输出按明确字符规则裁剪，与“目标截断为 0”不是同一指标。

数据输出 SHA256：`4d8412ee8bd558a659762441349124c5366e6a24b5b23ec9ea1d4ede299b6350`。

源轨迹合并 SHA256：`fcb396ace5383cd5046f697dae1ad62226b579018722402cf68b05dec137eab2`。

split SHA256：`14b2e53b181ae647fc18a1bdbf340ebaa943e9f9d1cb379920de23346c16ce57`。

## 训练与评测协议

Qwen2.5-Coder-3B-Instruct，BF16 LoRA（不是 QLoRA），仍使用 `configs/sft-clean-3b-amd.yaml`：1 epoch、seed 42、lr 1e-4、batch 1、梯度累积 8、LoRA r=16/alpha=32/dropout=0.05、max_seq_len=8192。实际完成 32 个优化器步骤。

本轮重新执行 Base 和 SFT，不能复用第一轮原生上下文的 Base 结果。冻结 4 个 dev 和 20 个 eval 任务，不依据模型结果换题；先做未修复/官方补丁控制检查，未准入任务完整报告。两任务并行，每对模型顺序按冻结索引交替。

两组均保持 max_steps=12、max_tool_calls=12、max_tokens=60000、max_seconds=300、max_new_tokens=768。训练中 1 条目标超过推理输出上限属于本轮限制，不临时放宽其中一组。

协调器启动记录：`pipeline.launch.json`。控制器 PID 3716，协调器子 PID 3717；PID 只对本次实例有效。训练启动时 PID 3858 实际持有 `/dev/kfd`、`/dev/dri/renderD128` 句柄，凭据变量名检查为空。不要通过 PID 猜测下次实例状态，不要重复启动未完成日志对应的进程。

训练已完成：`logs/training.receipt.json` 返回码 0，实际耗时 166.5815645040002 秒。`checkpoint/train_metrics.json` 为 250 train_records、250 encoded_examples、0 dropped_no_supervision、0 truncated_examples、32 global_step、1 epoch、seed 42。Trainer 记录 train_runtime=151.5174 秒、train_loss=0.7325680404901505。两轮标签和上下文不同，不直接比较 loss 来推断修复能力。

`training-weights.verified.json` 已核对 504 个权重张量均为有限值，252 个 LoRA-B 张量均非零；adapter SHA256 为 `6b384c15fc7cf8d72e3ed7e338f972fb2604343c1e6aef9e28209dabfea85282`。`inference-environments.jsonl` 实际观测到 Base、SFT 均为 last-action-v2、bubblewrap、768-token 输出上限，凭据变量名为空；观测是对应进程的快照，不是对所有未来进程的独立检查。

## 完整配对评测结果

`pipeline.receipt.json` 返回码 0，实际耗时 3459.772482372 秒（约 57 分 40 秒，含训练、控制检查、全部运行及独立评测）。全部 24 个冻结任务已有最终状态，无活动任务日志。

| 分区 | 冻结任务 | 准入 | 未准入 | Base 独立通过 | SFT 独立通过 |
| --- | --- | --- | --- | --- | --- |
| dev | 4 | 0 | 4 | 无可评测任务 | 无可评测任务 |
| eval | 20 | 12 | 8 | 1/12（8.3%） | 0/12（0%） |

未准入任务不能计作模型修复失败，也不能从冻结任务覆盖率中消失。dev 无准入任务，不存在可解释的 dev 成功率。

第一轮和第二轮均为 Base 1/12、SFT 0/12。第二轮两组使用新上下文，不能把跨轮变化单独归因于 SFT；这些留出题已被查看，属于小样本、单种子、Moto 单仓库的迭代诊断，不是新的 sealed test。

唯一独立通过任务仍为 Base 的 `getmoto__moto-5406`。它的 Agent 自身因重复命令以 failed 结束，但保留补丁通过了独立测试；两类状态分别记录，不把它写成 Agent 正常完成。两组只存在 1 对 Base 独立通过而 SFT 未通过的差异，配对精确检验 p=1.0，不能证明统计显著的提升或退化。

核验后的行为诊断（只针对 12 个准入 eval 任务）：

| 行为 | Base | SFT |
| --- | --- | --- |
| Agent 状态 completed / failed | 2 / 10 | 0 / 12 |
| 生成补丁 | 5 | 0 |
| 重复命令退出 | 5 | 12 |
| 达到最大步数 | 4 | 0 |
| 无已识别的修改后通过测试 | 1 | 0 |
| 模型动作不是合法 JSON | 1 | 0 |
| failure_reason 为空 | 1 | 0 |

状态与 failure_reason 是不同维度，不能将两列相加当作独立任务数。completed 的两个 Base 任务均未通过独立评测，重复退出的一个 Base 任务反而有通过补丁。

本轮仅改变训练/执行上下文并未解决 SFT 重复问题。可以确认“该小样本实验未出现修复提升”，不能确认“仅由某个训练原因导致”或“SFT 方法普遍无效”。

## 已完成的收尾

1. [x] 核对 32 个优化器步骤、250 条样本、504 个有限权重张量、252 个非零 LoRA-B 张量。
2. [x] 完成同条件的 12 对 Base/SFT 与全部 24 个冻结任务的准入状态。
3. [x] 从实际 evaluation 事件、CLI 收据、summary、run 配置重算结果、失败分类和配对统计；保留 12 个未准入任务的控制失败原因。
4. [x] 核对第一轮不可变资产、20 个源导出、冻结 split/manifest、配置、模型实际字节哈希、本轮源码身份与实际推理环境快照。
5. [x] 保存私有 `verification.json`、`final-assets.verified.json` 和本文件结果；资产清单包含 449 个文件，不是独立外部备份。

独立方法：`analysis/verify_aligned_round2.py`，SHA256 `5c420d5fa1f09c356a0e82729c04e1cb76784689b24cd706702d398c0647eea4`。使用实际 tokenizer 独立重算前缀/监督边界和 token 统计，没有调用修改后的训练编码器来验证自身。

| 第二轮资产 | SHA256 |
| --- | --- |
| report.json | ea037488146e08cbaeb8f94c9c0a8f22825b6d2aa4f54ab9330c91561b1a630e |
| pairs.json | e120b8980d0c0cba829db1e601db65f3b42a71320945be47c7f74a0ed727eeaa |
| verification.json | 9b9f3663266248a898dba6a25b11a28e478fb843c597fa2d2772d971750433c4 |
| final-assets.verified.json | 70d662c980e49dbdc7d75fc715a8f46a91b7918dca6f6feaac59c7e6b8a1a997 |
| pipeline.receipt.json | 2b6143954f16bcb7c516c84e6042ef7d9e5aa5ca8c70dde44df552205c4f447c |
| adapter_model.safetensors | 6b384c15fc7cf8d72e3ed7e338f972fb2604343c1e6aef9e28209dabfea85282 |

## 下次继续与仍未完成的事项

本轮已停止新增实验；服务器仍运行，不自动关机。未新增 API 消耗、外部备份、GitHub commit/push 或新实例。

实例重启后先核验实验文件是否仍在，不要按旧 PID 恢复；这些 PID 的进程本轮已经结束。读取结果不需要 API 密钥：

```bash
cd /mnt/workspace/swe-gym-plus-completion
python -m json.tool /mnt/workspace/swe-gym-plus/experiments/aligned-round2-20260927/verification.json
```

不要直接重跑独立验收脚本：验证与清单采用独占创建保护，已经存在时应只读核验，不覆盖。

项目整体尚未完成：本机 Worker 的 Git 环境变量过滤缺陷仍待修复；独立外部备份已按用户要求延期；未做新的 sealed test。若继续提升训练效果，应先审查重复命令样本与退出反馈，再扩充不同任务的真实成功轨迹，并使用新的未接触评测题；不建议仅重复现有 250 条样本训练。
