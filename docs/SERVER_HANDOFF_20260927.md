# 换服务器接续：2026-09-27

这是当前结果与接续入口。代码、汇总报告、验证脚本和架构决策上传公开 GitHub；
原始轨迹、动作训练数据和 adapter 权重通过私有迁移包单独转移。
公开仓库不含 API 密钥、账户凭据或训练权重。

## 当前真实进度

| 实验 | 数据 / 协议 | 结果 |
| --- | --- | --- |
| 第一轮 3B BF16 LoRA | 20 个成功任务、250 条动作、1 epoch、32 步 | 12 个准入 eval：Base 1/12、SFT 0/12 |
| 第二轮训练/执行上下文对齐 | 同一批任务与动作规模、1 epoch、32 步 | Base 1/12、SFT 0/12 |
| 上下文提醒与历史保留 | 第二轮 adapter，3 题 × 2 策略 | 两组均 0 补丁、0 通过；新策略 2 次重复退出、1 次预算退出 |

两轮均保留完整冻结的 4 dev + 20 eval；其中 dev 4 个未准入，eval 8 个未准入。
失败准入任务没有根据模型表现替换。三题试验使用已查看任务，不能称为新的 sealed test。
三轮均已独立核验完成，没有证明 SFT 带来修复能力提升。

新上下文保留四轮真实动作/回执和仍有效的提醒；88 项云端回归通过。
实际模型收到 5 次重复提醒，其中 2 次更换命令，但没有生成补丁。
新策略 20 次输出中 16 次达到 768 token 上限，16 次被现有解析器标记为截断命令恢复。

用户提出“一题上出现完整、有效的修改动作”后，工作在只读检查阶段被中断。
**截断动作拒绝、有限纠正重试、模型常驻和新的单题验证尚未实现或运行。**
不要把讨论中的方案当作已经完成的改动。

详情与公开数据：

- `experiments/progress-20260927/summary.json`
- `docs/FASTEST_FINISH_CHECKPOINT_20260927.md`
- `docs/ALIGNED_ROUND2_CHECKPOINT_20260927.md`
- `docs/CONTEXT_HISTORY_PROBE_20260927.md`
- `docs/adr/0001-context-history-probe.md`

上述历史文档中的“尚未推送 GitHub”是对应实验结束时的状态；本次同步以本接续文件为准。

## GitHub 与私有资产的区别

新服务器可以先获取代码：

```bash
git clone https://github.com/chengli200214-beep/swe-gym-plus.git
cd swe-gym-plus
```

只克隆仓库不能加载已有 SFT adapter。原私有资产仍在 ModelScope：

```text
/mnt/workspace/swe-gym-plus/experiments/formal-accelerated-20260926
/mnt/workspace/swe-gym-plus/experiments/aligned-round2-20260927
/mnt/workspace/swe-gym-plus/experiments/context-history-probe-20260927
```

迁移包准备在原实例的 `transfer/` 目录；下载包和同名收据后，私下上传到新服务器，
按收据中的 archive SHA256 校验，再解压到新的空目录。具体文件及校验值见下方收尾记录。
迁移包不是已完成的独立外部备份；不要因 GitHub 已同步就删除原工作区。

迁移包包含两轮 adapter、训练数据、结果、动作事件/诊断输入、任务 manifest/split、
实际云端代码快照；排除任务 checkout、仓库缓存和完整基础模型。
基础模型为 `Qwen/Qwen2.5-Coder-3B-Instruct`，在新服务器另行准备。
没有下载或发布基础权重，不向公开仓库上传私有迁移包。

## 新服务器恢复顺序

1. 确认是 Linux GPU 环境，记录 GPU 名称、显存、驱动、PyTorch、CUDA 和依赖版本。
   NVIDIA 镜像保留其 CUDA 版 PyTorch，不复制 AMD 的 ROCm 软件环境。
2. 克隆仓库、创建 Python 环境，按 `pyproject.toml` 安装所需依赖。
   纯读取汇总报告不需要 GPU 或 DeepSeek key。
3. 私下转移迁移包并校验，解压到空目录。第二轮 adapter 的预期 SHA256 为
   `6b384c15fc7cf8d72e3ed7e338f972fb2604343c1e6aef9e28209dabfea85282`。
4. 准备同一基础模型；加载 adapter 时用 CLI `--base-model-path` 指定新服务器路径，
   不修改原 adapter 或覆盖原实验。旧配置里的 `/mnt/workspace/...` 路径需显式映射。
5. 恢复 bubblewrap 和任务测试依赖，先执行无密钥隔离探针与单题未修复/官方补丁准入。
   换环境后不能直接复用 AMD 实例的准入结论。
6. 新建实验目录，先在 `getmoto__moto-6212` 上验证完整动作、真实文件修改和非空补丁。
   独立测试通过与“有修改动作”分别报告；不要先重复训练 250 条样本。

如果目的是单独比较硬件，先保持 checkpoint、任务、提示、解析器和输出/总预算一致，
记录加载耗时、生成 token/s、截断次数与工具回执。新的协议修正应另开实验。
不同 PyTorch、驱动和推理内核也会影响结果，不能把全部变化都归因于 AMD/NVIDIA。

`scripts/probe_context_history.py` 的完整六次旧实验包含旧环境和源码身份约束，
不能在新服务器不检查就重跑；已有报告和验证文件采用独占创建，不覆盖。
保留原实验的身份信息，新的 Git commit 不会自动改变历史运行时源码版本。

## 源码与未解决问题

云端工作树 HEAD 为 `a3aa038f01ec77a6d4de0961b4fa54caf71d5ba9`，实际实验含未提交改动。
本次核对 99 个 Python/YAML 文件，97 个与本地待发布版本按 LF 内容一致；
`scripts/freeze_expanded_pool.py` 与 `tests/test_expanded_pool.py` 保留本地已有的
跨平台清单哈希修正，不回退到云端旧版。准确历史源码以私有快照和实验身份文件为准。

仍存在：截断命令被恢复执行、模型重复/超长动作、训练修复提升未证明、
Windows Worker 的 Git 配置环境过滤缺陷、独立外部备份延期、尚未做新的 sealed test。
迁移服务器不等于这些问题已解决。遵守根 `AGENTS.md`，不泄露密钥，不覆盖完成的实验。

## 本次同步收尾

提交前完整 pytest：179 项，176 passed、3 skipped、0 failures、0 errors，230.083 秒。
3 项跳过分别需要 Linux flock 或本机无法创建的文件 symlink；不涉及训练或模型能力测试。
本次测试进程先清除继承的 `GIT_CONFIG_COUNT/KEY_*/VALUE_*`；这避免已有 Windows
Worker 环境过滤缺陷影响测试，不代表该缺陷已在代码中修好。JUnit 为本机私有验证收据，
不把含本机路径的临时测试目录上传仓库。`git diff --check` 通过。

云端已生成并逐成员重新读取校验的迁移包：

```text
文件：/mnt/workspace/swe-gym-plus/transfer/server-handoff-20260927.tar.gz
收据：/mnt/workspace/swe-gym-plus/transfer/server-handoff-20260927.tar.receipt.json
大小：229945499 bytes（约 230 MB / 219.3 MiB）
成员：1045 个文件
SHA256：a0cf72f453a3ac7b33c4260c11174ee76792dafa440025a4fad8084856d27901
```

在 ModelScope 文件树打开 `swe-gym-plus/transfer`，下载上述包与收据，再通过你的
私有传输渠道上传到新服务器。包尚未离开原实例，不算独立外部备份。
新服务器验证并解压示例（假设包已传到当前目录）：

```bash
printf '%s  %s\n' \
  a0cf72f453a3ac7b33c4260c11174ee76792dafa440025a4fad8084856d27901 \
  server-handoff-20260927.tar.gz | sha256sum -c -
mkdir server-handoff-20260927
tar -xzf server-handoff-20260927.tar.gz -C server-handoff-20260927
```

基础模型没有包含在包里。`code-source/` 是实际云端源码快照；GitHub 是带公开接续
文档的当前版本，两者的文件身份差异已经在上文记录。不要覆盖原实验目录。
本次没有启动新训练或新轨迹，没有租用、关闭或释放实例。
