# SWE-Gym-Plus 核查与提速记录（2026-09-26）

## 最快交付路线与当前状态

当前不是从零开发：Harness、队列服务、恢复机制、独立评测和页面已有云端验收收据。正式训练评测仍未完成，不能声称修复能力提升。

| 工作 | 核查结果 |
|---|---|
| 工程 | 已核验云端历史 108 项全过及恢复依赖后 104 项全过的 XML 收据；本轮 32 项提速/协议相关测试通过 |
| 训练候选 | 原冻结的 67 个训练候选全部重检，38 个准入 |
| 合格轨迹 | 最新核查为 16/20（6 种子 + 8 Flash + 2 Pro）；两批已结束，四路独立协议接续中；此数量是不同任务，不是动作样本数 |
| 留出集 | 原冻结的 4 dev + 20 eval 全部预检，12 准入；其余保留排除原因 |
| 正式训练 | 尚未启动；20 条达标后自动审计并训练 3B BF16 LoRA，1 epoch |
| 配对评测 | 训练后自动两路运行，相同预算、交替模型先后，保留全部冻结任务 |
| 外部备份 | 按用户要求暂缓；云端校验清单不等于备份 |

最短收尾只推进：`20 个不同成功任务 → 数据审计 → 3B LoRA → Base/SFT 配对评测 → 实验报告与交接`。当前实例够用，不新增服务器、大模型、GRPO 或其他功能。采集成功率决定剩余时间，不能把 GPU 空闲或账户余额等同于轨迹已经合格。

## 核查结论

ModelScope 当前实例的系统盘没有 bubblewrap；旧采集器又直接复用了失败的准入报告，导致环境恢复后也无法重新判定这些任务。旧结果文件有 50 条尝试记录，涉及 45 个任务；记录数不能当作独立任务数。

旧活动目录共有 48 份质量报告：7 份准入，41 份拒绝。拒绝原因：

| 原因 | 报告数 | 处理 |
|---|---:|---|
| 测试隔离器不可用，测试未运行 | 36 | 恢复 bubblewrap 并重新运行双控制测试 |
| 测试命令启动或测试收集错误 | 3 | 引号编码修复；保留测试收集错误并继续诊断 |
| 官方补丁版本测试失败 | 2 | 检查真实错误，不能改成通过 |

7 份导出中，6 份记录为 Agent 正常完成且独立评测通过；`getmoto__moto-7365` 的补丁通过但 Agent 状态为 failed，因此不计入本轮正常完成轨迹。6 份种子还须在当前环境通过新的任务准入控制才能复用。

## 已实施

- 从 Ubuntu 官方软件源恢复 bubblewrap、python3-sure、python3-responses、python3-tz、python3-xmltodict 及其依赖。
- 使用实际 BashExecutor/bubblewrap 验证测试依赖可导入；沙箱内无法访问 `/mnt/workspace`。
- 修复数据导入器对 pytest 参数化测试名称的 shell 引号编码，防止括号、空格等被 shell 解释。
- 新增 `scripts/accelerate_campaign.py`：重新验证原有 67 个训练候选，6 路质量验证、4 路 API rollout；每个任务最多 2 次候选运行，保留完整独立评测。
- 使用派生 manifest 修复命令编码，保留每个测试名称、任务 ID、官方补丁和原训练/开发/评测划分；原 frozen pool 文件不覆盖。
- 根据用户“现在不限制成本”的指示，新活动取消旧的两元余额窗口，余额下限配置为 0；模型适配器仍检查账户可用状态与其内置的一元余额余量。
- 正常完成且独立评测通过的不同任务累计到 20 条后停止提交新任务；不自动重发不确定的付费请求。
- 22 项相关测试通过（含来源收据、预检查一致性、并行配对顺序测试）。首次本地测试遇到默认 pytest 临时目录权限问题，改用工作区内新临时目录后通过。
- 增加 `--collect-ready`：完整通过双控制的任务可先采集，无需等待其余控制；独占采集锁防止重复付费调度。
- 正式训练收据记录实际源文件哈希（包括未提交修复），并核验提前完成的留出集控制确实使用同一 frozen manifest/split。

## 实时核查检查点（2026-09-27）

- 67 个训练候选全部重检：38 个准入，19 个测试启动/收集错误，10 个官方补丁版本测试失败；不改判，也不替换 frozen heldout。
- 6 条旧正常完成、独立评测通过且不暴露评测命令的种子通过新的任务控制，已复用。
- 4 路真实 API rollout 已启动；新成功任务已核实 7365、5321、7635，累计 9 个不同合格任务（6 个旧种子 + 3 个本轮），目标 20，尚未宣告达标。
- 固定的 4 个 dev 和 20 个 eval 任务已全部完成环境预检查：12 个准入、11 个测试启动/收集错误、1 个官方补丁版本测试失败。此前读屏把 11 和 12 对调，已以最终报告核正。收据写入正式实验目录，训练后复用这些环境控制；最终报告必须保留完整 24 个冻结任务及其排除原因。
- 当前实例 PyTorch 报告约 191.7 GiB 可见显存；BF16 矩阵运算和本地 3B tokenizer smoke 已通过。不是 GPU 型号的推断。
- 工程正式版已有历史云端 108 项回归及 PostgreSQL/恢复/页面验收证据，见 `completion-status-2026-09-25.md`；本轮不重复开发。
- 本轮 Windows 全量测试：110 passed、1 failed、2 skipped（总 113 项，新增来源测试前）。失败的服务端脚本测试被本机 `git init` 退出 128 中断，日志确认失败发生在创建测试仓库，不是修复或独立评测断言。不能声称本轮 Windows 全量通过；云端历史 Linux 验收证据另列。
- 本地与云端 `scripts/run_clean_training.py` 最终字节 SHA256 一致：`f2dafdbe50e9af7038a26eac0e69f484199061906aa6fa13d6226d9091fc4b87`。早期串行版本 SHA 为 `a42ab0d7dc035ac23d18e290d9205a6cd786feb2b6ff346e94532fe9e91ef2c3`。
- 正式评测使用 2 路独立任务并行，每题内仍顺序比较 Base/SFT，交替模型先后；同一固定预算，主协调器按冻结任务顺序写入结果。
- 达标接续进程已在云端终端 4 运行：只在采集报告为 completed 且留出集控制全部完成时，启动无密钥的数据审计、3B LoRA 和配对评测，保留日志与退出收据。
- 当前云端代码 HEAD 实测为 `a3aa038f01ec77a6d4de0961b4fa54caf71d5ba9`，新采集脚本未提交、训练脚本有修改；不声称本轮修改已经同步到 GitHub。
- 私有预训练环境清单已保存为正式目录中的 `pre-training-environment.json`，包含软件版本与 91 份质量报告的校验值。这是云端完整性清单，不是外部备份。

## 最快接续检查点：四路 Pro 已启动（2026-09-27）

- 用户明确同意将公开 SWE-Gym 题面、Moto 代码片段和沙箱工具输出发送至 `api.deepseek.com`，使用现有密钥运行四路 Pro 思考模式。首次启动被安全审核拦截，得到针对这些数据与目的地的确认后才启动。
- `scripts/fast_finish.py` 独立记录新批次，不热改原 Flash 进程。只处理已准入且两个 Flash 尝试均有 rejected 收据的训练任务；不重跑正在运行、未知结果或留出任务。
- 实际模型参数：`deepseek-v4-pro`、thinking enabled/high、4096 输出 token、每题 32 步/32 工具调用/384k token/600 秒；余额下限配置 0，仍保留一元可用余额余量。费用不再套用 Flash 估算价。
- 新批次目录：`/mnt/workspace/swe-gym-plus/experiments/thinking-20260927`。已在云端看到四个运行任务 6641、7035、5343、5725，并产生 prompt/model/tool 事件；尚未获得新批次成功收据。
- 启动检查点合并合格任务数仍为 10，原 Flash 已完成结果中 accepted 4、rejected 14，另有 6 条复用种子；这些 rejected 是任务结果，不是模型调用次数。
- 新脚本本地规范化 LF SHA256：`0a2702b2c7cc73f586931d502984b21871d6ea8ec355ac86b3346539e211f74e`。云端字节 SHA256：`d68f385b7e681ae065328826a5fed66f8db7cace0c205f8a20f5eb0094b74d58`；IDE 使用 CRLF，云端规范化 LF 校验与本地一致。云端 AST 与导入通过，本地 26 项相关测试退出码 0。
- 两批合并达到至少 20 个不同合格训练任务后，停止新增 Pro 任务并等待在途任务写完，再冻结去重的私有快照与来源哈希，启动无密钥训练和配对评测。独占 `pipeline.log` 避免两条接续进程重复训练。
- 正式训练尚未启动，`pipeline.log` 与 `report.json` 在核查时均不存在。不得把已安排的自动流水线写成已完成实验。
- 后续检查点：合并合格数增加到 11，原 Flash accepted 增至 5；Pro 第一题正常结束但独立评测未通过，不计入成功数据。Pro 其他任务继续产生工具事件，不把运行事件数或 completed 状态当成成功轨迹数。
- 最后核查：合并合格数增加到 12（6 条种子 + 6 条本轮 Flash）；Pro 已有 3 份 rejected 收据，4 路继续运行；两批均未 halted。留出预检 manifest/split 哈希核验通过，正式训练流水线和报告仍未生成。接续文档已保存到本地及云端持久工作区。

## 协议接续检查点（2026-09-27）

- Flash 最终为 14 个合格任务（6 种子 + 8 新任务），首轮 Pro 新增 2 个，合并 16/20；两批均 `insufficient_successes`，正式训练尚未启动。
- 首轮 Pro 的失败不全是代码修复能力问题：初查 21 个结果中 11 个无有效 JSON 动作。核查实际模型输出为空白、残缺工具片段或输出上限截断，不将它们重构为成功动作。
- 新脚本 `scripts/repair_protocol.py` 采用 thinking enabled/low、text 格式、16384 输出 token、四路、32 步/32 工具/512k token/600 秒。候选只能来自已准入且首轮 Pro 确定 rejected 的 train 任务；成功、heldout、在途和未知结果不进入重试池。
- 初次协议启动遗漏 CLI 必需的余额下限配置，四个子进程均在模型适配器实例化之前返回。四份日志只包含余额下限错误，0 个模型事件文件、0 个模型响应收据；原目录 `protocol-retry-20260927` 保留。
- 修复后将 API 环境加上 `DEEPSEEK_MIN_BALANCE_CNY=0`，不会把密钥或该变量带入独立评测/训练环境；新增回归测试。30 项本地相关测试退出码 0，云端 AST 与导入通过，规范化 LF SHA256 一致：`c5589bbbf87232d32c4821517cd117ff12ea4d7ee8faf8619502d69359461406`。
- 当前独立目录 `protocol-retry2-20260927` 已在终端 6 启动四路，看到各子进程的实际 teacher receipt，尚未新增合格成功收据。达标后自动快照、审计、3B LoRA 和固定留出配对评测；不降低 20 个任务门槛。
- GPU 可用、可见显存约 191.7 GiB；当前 CPU 执行沙箱任务，GPU 等待正式训练。本机不跑 GPU，不新增租用。

## 云端位置

新增高思考文本接续：低思考文本批次 `protocol-retry2-20260927` 的 13 个任务全部有明确失败收据，协调器停止新调度后等待子任务自然排空。没有强杀在途 API 请求，未把失败样本计入成功数据。独立脚本 `scripts/repair_protocol_next.py` 通过 32 项本地相关测试及云端 AST/导入；规范化 LF SHA256 `7aa597891bb1f423bf21e2a3fef5862d0a8313e755d3049eaf866b0de444a851`。当前 `protocol-hightext-20260927` 在终端 4 使用 high/text/16384、四路；identity 保存前序 13 个 outcome 哈希，实际 teacher receipt 已出现。累计仍 16/20，正式训练尚未启动。原低思考脚本保留 `c5589...` 版本，不热改旧批次；完整接续见 `FASTEST_FINISH_CHECKPOINT_20260927.md`。

- 代码：`/mnt/workspace/swe-gym-plus-completion/scripts/accelerate_campaign.py`
- 旧记录：`/mnt/workspace/swe-gym-plus/experiments/clean-expanded-safe-20260925`
- 新活动：`/mnt/workspace/swe-gym-plus/experiments/accelerated-20260926`
- 旧串行采集器 PID 1563 已暂停，原文件保留。

新活动生成 `baseline-diagnosis.json`、`identity.json`、`quality-progress.json`、`quality-summary.json`、`collection-progress.json`、`collection-report.json`；私有完整轨迹在其 `exports/` 下。

## 接续命令

当前已启动第一条无密钥命令。启动或接续前先确认没有相同活动仍在运行，避免重复启动。

```bash
cd /mnt/workspace/swe-gym-plus-completion
PYTHONPATH=src python -m scripts.accelerate_campaign \
  --source-root /mnt/workspace/swe-gym-plus/experiments/clean-expanded-safe-20260925 \
  --pool data/expanded/moto-v2 \
  --root /mnt/workspace/swe-gym-plus/experiments/accelerated-20260926 \
  --quality-only
```

质量验证已有足量完整通过报告后，核对实际 admitted 数量与未通过原因，再运行同一命令并把 `--quality-only` 换为 `--collect-ready`。密钥通过隐藏提示或进程环境输入，不写入代码、活动文件或 Git。

20 条正常完成且通过评测的独立任务轨迹达标后，使用新活动的派生 pool 执行数据审计、3B BF16 LoRA 和相同预算下的 Base/SFT 留出对照：

```bash
PYTHONPATH=src python -m scripts.run_clean_training \
  --exports /mnt/workspace/swe-gym-plus/experiments/accelerated-20260926/exports \
  --pool /mnt/workspace/swe-gym-plus/experiments/accelerated-20260926/pool \
  --root /mnt/workspace/swe-gym-plus/experiments/formal-accelerated-20260926
```

这份记录不宣称轨迹已达 20 条，也不宣称训练已经完成或修复能力已经提高；以云端正式收据为准。
