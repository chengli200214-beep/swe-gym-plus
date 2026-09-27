# 真实源码观察与新开发任务接续

## 已实现的变更

正常 Agent 新增 `read: {path, start_line, end_line}`，由模型自主选择路径和
行范围，固定程序在现有 NsJail 内读取；每次最多 80 行，返回完整源码行、
实际路径和全文件 SHA256。单条输出不足 4000 字符，不人为补齐源码。
typed edit 必须引用真实 read 回执中出现的 before；沙箱检查文件版本、
精确一次匹配、Python 语法后原子写入。成功编辑后必须重新读取再继续改。
恢复使用已保存回执，不重复执行读取或编辑。原题、真实错误和重复提醒保留。

此门槛只证明 typed edit 的观察来源，不将任意 shell 表达式分类为只读；
更不证明模型理解了源码或补丁正确。细节见 ADR 0006。

## 工程验证

- Windows 完整回归：239 项，232 通过、7 跳过，0 失败/错误，362.513 秒。
  该次尚不包含后来新增的 6 个 selector 单元用例；其后相关 17 项单独通过。
- AutoDL 读取/编辑初版完整回归：235 项，234 通过、1 跳过，0 失败/错误，20.255 秒。
- AutoDL 最终全量回归：242 项，241 通过、1 跳过，0 失败/错误，19.547 秒。
  报告为 `source-full-tests-04.xml`，包含 selector、独立新模板和显式评测缓存接口。
- Windows 测试发现固定程序分隔和 UTF-8 输出问题，已修复，不放松 source 匹配。
- 首次云端全量测试缺少环境 bin 的 PATH，失败报告保留；修复仅作用于测试进程。

## 固定数据与开发题

使用 SWE-Gym 固定版本 `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`。
Parquet SHA256：`60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97`。
官方快照已有本机缓存；AutoDL 无法连接 Hugging Face，校验后传输了同一数据文件。
未在本机进行 GPU 训练或推理。

数据质量检查：2438 行、2438 个不同任务 ID、0 重复，instance_id/repo/
base_commit/problem_statement 缺失均为 0。排除历史/预留清单 100 个任务 ID，
以及匹配的题面和基础提交组；Moto、1–5 FAIL_TO_PASS、至多 30 PASS_TO_PASS
约束下有 57 个新候选。按固定哈希 seed `cab-source-observation-20260927-v1:`
预先选择：5164、5255、6535。没有根据官方补丁内容、准入或模型结果选题。

这是项目内新开发集，不代表模型预训练未见过公开代码，也不是最终密封测试集。
原始 pool manifest SHA256：`b8fba6fe887674a8a26b1aec98da6100cb31dd44a7250d86907631834b006833`。
第一次 gate manifest SHA256：`c937bcd325f8567e11030054991d3a40579a1521e513971ea9a87883d1935647`。

## 首次准入与环境修复（不算模型失败）

- 5164：缺少 flask-cors，测试收集失败。
- 5255：一个 Unicode 转义差异、两个上游截断的参数化测试名称，测试选择失败。
- 6535：Ubuntu cryptography 3.4.8 不提供任务使用的私有 NameOID 导入，收集失败。

首次 0/3 准入报告保留，无模型调用。依赖安装到独立新 rootfs，未覆盖旧模板。
PyPI wheels 的 SHA256 均验证，版本固定 cryptography 46.0.4、cffi 2.1.1、
pycparser 3.0、typing_extensions 4.15.0；Ubuntu flask-cors 使用官方源。
首个离线包遗漏 Python <3.11 条件依赖，已补入固定 wheel；不开放 guest 网络。

测试名称通过 base+test_patch 的真实隔离收集唯一映射，未使用 gold patch。
三题 required 数分别 2、29、5；仅 5255 的三个名称被等价映射，零测试要求删除。
原始标签/补丁/ID 保留，执行命令有独立派生版本；不称为原命令直接复现。
派生 selector manifest SHA256：`e489a25ba929bda6dc8c294c3a84567ddac3c56f332a22d16d16a6430de7afd1a`。
细节见 ADR 0007；必须重新通过双控制准入再运行模型。

派生 gate manifest SHA256：`63d1a3746f5315e67ccb53ef9d68ef8e18b6734f14b65de4c66cf03d85d8e830`。
该版本三题已全部通过双控制准入（3/3），原始版本的三份 blocked 报告未覆盖。

## 仍未完成

新开发集自主修复、20 训练任务再审计、新 7B LoRA、公平未使用留出对照、
AutoDL 服务验收尚未完成。本轮尚未开始训练，没有新修复提升结论。
私有权重、原始轨迹和完整评测资产仍在云端，不公开到 GitHub；外部备份继续暂缓。
