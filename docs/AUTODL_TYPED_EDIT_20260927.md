# 正常 Agent 精确编辑协议：实现与三题开发诊断

## 已完成的工程变更

正常 Agent 新增 `edit: {path, before, after}` 动作，与探索/测试用的 command 动作互斥。
固定适配器在 NsJail 中验证源文件、精确一次匹配和 Python 语法后原子写入。
不会为模型猜测源文件、填补截断命令、供应答案或将错匹配算作成功修改。
新动作复用既有两阶段日志与恢复；崩溃恢复不能抹掉被拒绝的编辑计数。

一次全任务格式纠正上限、两次全任务编辑拒绝上限均已实现。
原题和异常不删去；recent-history-v3 保留真实编辑/回执和协议拒绝、当前重复提醒。
每次模型调用保存实际上下文；动作样本转换器支持同一编辑动作和 recent-history-v3。
旧 last-action-v2 控制仍保留，原始训练数据和权重未覆盖。本轮没有训练。
细节与回滚见 [ADR 0005](adr/0005-typed-edit-actions.md)。

## 验证

- 云端最终完整测试：218 项，217 通过、1 跳过、0 失败/错误，18.030 秒。
  包括真实 NsJail 负向检查、编辑执行/失败纠正与 once-only 恢复。
- Windows 完整测试：221 项，214 通过、7 跳过、0 失败/错误，246.387 秒；
  随后新增的恢复后拒绝次数上限测试单独通过。不要把两次报告写成一次全量 222 项报告。
- `compileall`、空白检查通过。代码文件未超过 500 行。
- Cognito 缺失的 python3-jose 从 Ubuntu 官方源安装到独立 rootfs，旧模板未更改。
  新 rootfs 的真实隔离负向检查通过，DNS 不保留给工具进程。

初次 Windows 引号测试因测试文件 CRLF 与指定 before 的 LF 不一致失败；
测试改为明确写入 LF 字节，不放松精确匹配。旧失败报告保留。
新 rootfs 的包安装成功后，末尾验证命令因 PATH 不含 chroot 返回 127；
改为绝对工具路径后实际导入/包版本、隔离与任务双控制均已重新验证。

## 冻结开发诊断

继续使用已有冻结 split 的前三个 dev 任务，不按模型结果替换。
manifest SHA256：`2d65450d9e168bd740178183d796ec3b1a8be926d3b4e46bed31be74d75729dc`。
来源 manifest：`4885fee2ea2dee7235a24fb1110bbdd72998d6278b5191aba995bcb2ec5dd301`。

这些是当前 7B 新执行协议的开发诊断，不是新密封留出集。
5876 和 5085 在 2026-09-25 的 0.5B 实验已有曝光；因此本轮不满足计划中的
“三个全新开发任务/未使用测试集”要求，也不能声称泛化提升。后续新任务仍须另行冻结。

| 任务 | 准入 | 自主运行 | 独立评测 |
| --- | --- | --- | --- |
| getmoto__moto-5085 | 未修复失败、官方补丁通过 | 首步猜 before；收到不匹配错误后仍重复提案；2 步、无补丁 | failed |
| getmoto__moto-5212 | 未修复失败、官方补丁通过 | 首步猜不存在的路径；收到错误后继续用同一路径；2 步、无补丁 | failed |
| getmoto__moto-5876 | 首次缺 jose；独立新模板恢复后未修复失败、官方补丁通过 | 首步猜不存在的路径；收到错误后仍重复；2 步、无补丁 | failed：1 failed / 2 passed |

模型为 Qwen2.5-Coder-7B-Instruct BF16 Base；无 3B adapter、无 DeepSeek API。
共同预算：温度 0、生成上限 2048 tokens、最多 16 步/16 工具调用、180000 总 tokens、600 秒。
三题共 0/3 自主修复通过。完整 JSON 可解析并不意味着准确定位；
没有源码读取行动，before 不是从实际文件复制。主要证据指向定位/观察动作未发生，
并非 AMD/CUDA 差异、文件写入能力或工具输出被截断。

## 接续与仍未完成的工作

按计划的失败门槛停止扩充数据和训练，不再追加同样的猜测提案。
下一项是让正常 Agent 在编辑前自主获取真实仓库路径和源代码证据，
随后重新冻结未曝光的小开发集；不让控制器提供修复答案。
20 个任务的再审计、新 7B LoRA、公平未使用留出评测、AutoDL 服务验收仍未完成。

私有资产：`/root/autodl-tmp/experiments/autodl-dev-gate-20260927`。
原三题 report 中 5876 仍为首次 blocked_admission；后续结果独立保存在
`quality/getmoto__moto-5876-cognito.json` 和 run ID `base7b-edit-cognito-getmoto__moto-5876`，
没有覆盖首次失败或擅自改写原报告。
另外两个 run ID 为 `base7b-edit-getmoto__moto-5085`、`base7b-edit-getmoto__moto-5212`。
最终云端 XML：`/root/autodl-tmp/migration-20260927/typed-edit-publish.xml`。
新 rootfs：`/root/autodl-tmp/nsjail-rootfs-cognito-20260927`。
实验结束后 GPU 0 MiB、0%，没有残留 guest UID 进程；实例仍运行计费，不是已关机。
