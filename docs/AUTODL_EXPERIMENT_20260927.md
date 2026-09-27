# AutoDL 同机实验：2026-09-27

本轮目标：在 RTX 5090 上完成隔离执行，并验证模型能否生成完整、有效的代码修改。
推理、代码执行与独立评测均在 AutoDL；不调用 DeepSeek API，不使用 ModelScope 算力。

## 已验证的工程进展

- 接入 NsJail no-namespace chroot 后端；没有关闭云平台安全策略，没有以宿主 root 执行模型命令。
- Agent、任务准入、独立评测统一选择 `nsjail`，未知后端失败即停止，不回退到本地 shell。
- 任务目录与模型权重、原始轨迹隔离；每个运行有独立 UID/GID 和独立 rootfs，Git 对象解除宿主缓存依赖。
- 实测通过 Python、Git、临时文件与 sed 修改；联网、提权、修改/重命名 Git 元数据被拒绝。
- 实测通过输出上限、超时、后台子进程清理和不同任务 UID 不复用。
- 保留历史运行与 `recent-history-v3` 的重复提醒/近期交互；不完整命令不再被修复后执行。
- 最终云端全量测试：190 项，189 通过，1 跳过，0 失败（含实际 NsJail 负向准入）。
- 最终 Windows 全量测试：194 项，189 通过、5 跳过，0 失败；Linux 专用检查不在 Windows 上运行。

本后端是受限云容器中的实验隔离方案，不是 VM，也不能据此宣称通过生产级多租户安全审计。
磁盘占用/文件数为周期检查，单文件大小另有 RLIMIT；不是瞬时文件系统总配额。
当前工具限制：地址空间 1024 MiB、进程数 16、CPU 时间 120 秒、打开文件数 64、
单文件 16 MiB；NsJail 自身墙钟上限为工具 timeout 向上取整后加 1 秒，
控制器同时监督进程组、输出和可写目录。实际负向准入 v7 已在这些限制下通过。
详见 `adr/0004-autodl-no-namespace-executor.md`。

## 任务准入

目标：`getmoto__moto-6212`，仓库 `getmoto/moto`。
base commit：`8c0c688c7b37f87dfcc00dbf6318d4dd959f91f5`。

- 原始代码 + 评测测试：1 失败、16 通过，错误落在默认 primary workgroup 的配置解析。
- 官方补丁 + 同一评测测试：17 通过。
- 准入通过。官方补丁/评测测试内容不提供给模型。

原冻结 split 将本题归入 eval。本轮已反复用于诊断和调整，后续不得再把它当作未使用留出测试，
也不能把本页结果直接混入原来 12 题的泛化解决率。

## 3B 已完成的诊断

所有 shell 轨迹：最大生成 2048 tokens，保留重复提醒与近期交互。
微调模型为原第二轮 Qwen2.5-Coder-3B LoRA，不重新训练或更改旧权重。

| 运行 | 条件 | 实际结果 |
| --- | --- | --- |
| `sft-6212-nsjail-2048` | 温度 0.2，自主运行 | 第 2 次响应重复扩展搜索词至生成上限；无完整 JSON、无补丁；未执行截断命令 |
| `base-6212-greedy` | 温度 0，自主运行 | 6 步，重复未命中文本的替换，重复守卫停止；无补丁 |
| `sft-6212-greedy` | 温度 0，自主运行 | 4 步，重复访问不存在的路径，重复守卫停止；无补丁 |
| `base-6212-grounded` | 温度 0，提供一次真实公共源码读取结果 | 6 步；提出配置修改方向，但生成的 shell/Python 命令未成功修改文件；独立测试仍失败 |
| `sft-6212-grounded` | 同上 | 第 2 步输出不完整 JSON；无补丁；独立测试仍失败 |
| `base-6212-proposal` | 单次结构化 before/after 提案，最多 1024 tokens | JSON 通过，但候选代码有语法问题；在写入前拒绝，无补丁 |
| `sft-6212-proposal` | 同上 | 修改前文本未匹配源码；拒绝，无补丁 |

带公共源码定位与结构化提案的运行属于编辑诊断，不是自主 Agent 基准。
控制器不代写任务修复、不供应官方补丁；实际修改仍须由模型提出并在沙箱内执行。
这些结果不证明 SFT 带来提升。失败在 NVIDIA 上仍出现，不能把此前失败单纯归因于 AMD。
CLI 的进程退出码在 `--skip-evaluation` 时不表示 Agent 成功；以 summary/回执/独立评测为准。

## 7B 接续

官方 `Qwen/Qwen2.5-Coder-7B-Instruct` 已下载完成，四个权重分片均存在，BF16 模型成功加载。
第一次并行下载触及 900 秒限时，保留分片后由官方 SDK 断点续传完成；日志保留，未从零重下。
这是新基座对照，不加载不兼容的 3B LoRA，不将 7B 的结果称作旧 SFT 提升。

`base7b-6212-greedy`：正常自主运行，温度 0，最大生成 2048 tokens，12 步。
前期读取文档和错误路径，随后找到实际初始化调用；两次编辑尝试分别因 sed 表达式错误和
Python 语法错误失败。最终无补丁，状态 failed（maximum agent steps reached）。
独立评测仍是 1 失败、16 通过。首轮不构成“完整有效修改”或自主解决成功。

接续的公共源码定位诊断单独记录，不能替代以上自主运行结果：

- `base7b-6212-grounded`：6 步，Agent 自报 completed，但没有补丁；独立测试仍 1 失败、16 通过。
- `base7b-6212-proposal`：完整 JSON 中的 before 来自 botocore 报错栈，不存在于允许编辑的 Moto 文件，拒绝写入。
- `base7b-6212-proposal-feedback`：返回真实失败回执，保留原题、源码和上一响应，最多 2 次；两次 before 都未匹配，无补丁。
- `base7b-6212-proposal-cleanissue`：去掉 `Traceback:` 后的文字，保留公开复现和源码；原题另存，不加入修复答案。模型首个提案真实写入并通过语法检查：`if name in self.work_groups:` → `if name.lower() in self.work_groups:`。独立评测仍 1 失败、16 通过，说明写入成功但没有修复错误。这个视图也删去了异常信息，不能将差异单独归因于移除库实现。
- `base7b-6212-proposal-errorview`：保留公开复现、原题异常信息、源码，去掉库实现；2 次均猜测了一段不在读取源码中的 return 语句，匹配失败，无补丁，不再追加无边界重试。

结构化诊断显式使用 native 历史，避免 bash 专用压缩器丢掉 proposal/result 消息；
不改变正常 Agent 的 recent-history-v3 或旧训练格式。每次响应与真实编辑回执单独保存，
拒绝结果不进入训练集。源码读取实际包含完整 190 行、5741 字符和 primary 初始化调用，
不能把上述错选文本归因于这次读取被截断。上述结果不支持扩大训练或盲目更换显卡。

本轮共 13 个运行/诊断记录，全部来自同一题，**不是 13 个独立任务或 13 条合格成功轨迹**。
目前只证明出现了一次完整、可执行的模型修改；正确修复和自主解决成功仍未达成。
旧第二轮 SFT 的 0/12 结论不变，本轮没有重新训练。

## 下一轮决策点

停止增加同题提示变体和盲目重复 SFT。先在少量新的开发任务上验证代码定位、准确文本编辑和
真实测试反馈三个环节；需要建立至少一个独立评测通过的基座对照，再扩数据和训练。
正常 Agent 尚未接入结构化编辑工具，格式纠正也不属于当前已完成能力；
本页的 before/after 适配和两次上限仅用于诊断，不能包装成已完成的通用 Agent 工具体系。
保留现有重复提醒与必要历史；所有无补丁或评测失败的记录继续排除在成功训练数据之外。

维护债务 `TRAIN-BOUNDARY-01`：既有 `src/codeagentbench/train_sft.py` 为 564 行，超出项目 500 行原则；
本轮未修改训练实现。下一次修改训练代码时应按真实训练职责拆分并保留回归覆盖，
不要为了本次执行器/诊断改动重写无关训练代码。本轮新增/修改代码文件均未超限。

## 私有资产位置

- 新实验：`/root/autodl-tmp/experiments/autodl-one-task-20260927`
- 原迁移资产：`/root/autodl-tmp/server-handoff-20260927`
- 基座：`/root/autodl-tmp/models`
- rootfs：`/root/autodl-tmp/nsjail-rootfs-20260927`
- UID 注册表：`/root/autodl-tmp/nsjail-state/uids.sqlite3`
- 实际负向准入：`/root/autodl-tmp/experiments/nsjail-admission-v7-20260927/admission.json`
- 训练前后旧资产不覆盖；原始轨迹和权重不上传公开 GitHub。依用户选择，尚未外部备份。

环境：RTX 5090 32GB；宿主 Python 3.12.3 / PyTorch 2.8.0+cu128 / Transformers 5.15.1；
工具 rootfs Ubuntu 22.04 / Python 3.10。不要把工具 Python 与宿主训练 Python 混为一谈。

关键校验：

- Ubuntu Base archive SHA256：`242cd8898b33ea806ef5f13b1076ed7c76f9f989d18384452f7166692438ff1a`。
- 任务 manifest SHA256：`4885fee2ea2dee7235a24fb1110bbdd72998d6278b5191aba995bcb2ec5dd301`。
- 旧第二轮 adapter SHA256：`6b384c15fc7cf8d72e3ed7e338f972fb2604343c1e6aef9e28209dabfea85282`。

已下载 7B 的四个分片（文件编号 / bytes / SHA256）：

| 编号 | bytes | SHA256 |
| --- | --- | --- |
| 00001 | 4877660776 | `0b6f069918b07c064cbba8ae4f00f529aa9bbf84b7cdfcb7fc2694a40f6aa8ef` |
| 00002 | 4932751008 | `c3d46733e7aa054ea7b063fbccd0a5a08446e7bd1814bef26936c5aa1331da62` |
| 00003 | 4330865200 | `9fe45dacee087385b3d2d6dd27a7413a8a56d95f145772facc148fa86fc73446` |
| 00004 | 1089994880 | `5aa6e5cbe642377fd441fb4e60e83cca96b2bcd9820e245b9ea06d94653f17f2` |

## 运行入口

```bash
export CODEAGENTBENCH_EXECUTOR=nsjail
export CODEAGENTBENCH_LOCAL_PROMPT_POLICY=recent-history-v3
unset DEEPSEEK_API_KEY
# 每次准入使用新的输出目录，避免覆盖既有工作区。
/root/autodl-tmp/agent-env/bin/python scripts/check_nsjail.py /root/autodl-tmp/experiments/<new-admission-dir>
```

继续单题时复用 CLI `run --model-backend local --temperature 0 --skip-evaluation`，
随后用 `evaluate-run` 在新工作区独立判分。定位诊断入口为 `scripts/autodl_grounded_probe.py`；
结构化编辑诊断入口为 `scripts/autodl_edit_proposal.py`。不得把诊断提案自动导入训练集。
该入口 `--max-attempts` 只允许 1 或 2；`--issue-view` 为 full / before-traceback /
repro-and-exception，必须连同原题与实际模型输入一起记录。默认不改变正常 Agent 的题面。

完整云端框架测试需将 agent-env 的 Python 放入 PATH；测试中的可信 demo 子进程使用 `python`。
本轮最终回归第一次启动未设置该 PATH，12 项报错；实际回执为 exit 127、python not found，
不是 NsJail 模型实验失败。该失败 XML 保留；修正启动环境后全量通过。实任务始终显式选择隔离后端。

```bash
unset DEEPSEEK_API_KEY CODEAGENTBENCH_EXECUTOR
export PATH=/root/autodl-tmp/agent-env/bin:$PATH
export CODEAGENTBENCH_TEST_NSJAIL=1
python -m pytest -q -p no:cacheprovider
```

最终报告：`/root/autodl-tmp/migration-20260927/pytest-nsjail-verified-env.xml`；
首次环境失败报告：同目录 `pytest-nsjail-verified.xml`。
冻结源码包：`/root/autodl-tmp/experiments/autodl-one-task-20260927/source-verified-20260927.tar.gz`，
校验值另记在私有接续记录，避免把归档自身的校验值写入归档造成循环。源码包不含私有轨迹/权重。
