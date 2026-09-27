# AutoDL 服务同机验收与取消修复

日期：2026-09-27。现有 5090 实例；无 DeepSeek 调用、无公网模型/服务端口。
这是可信脚本的工程验收，**不是模型自主修复或训练提升**。

## 实际失败与修复

验收 01 的任务提交、输入边界、示例补丁独立评测及四个 artifact 端点通过。
运行取消失败：队列已标记 cancelled，但 3 秒等待内仍存在该工作区 UID 的
非僵尸 guest 进程。原 report、队列和日志均保留，不覆盖。
后续检查原验收工作区已无活跃 guest 进程；并非通过删掉失败数据获取成功。

根因是运行控制边界：Worker 通知 CLI，CLI 仅在动作间检查取消；
真正持有独立 NsJail 进程组的监督器没有收到协作取消请求。
现已将取消谓词从 CLI 分别经 Runtime/Evaluator 传到监督器，终止并等待自己
创建的进程组，保留实际回执。取消不冒充测试失败，也不启动后续模型动作。
同时覆盖退出与取消请求的竞态。设计及回滚边界见 ADR 0009。

## 同机实际验收

验收 02 五项通过，验证已确认 read 的恢复；验收 03 加强为已确认 **edit** 后故障注入。
03 原始私有报告名为 `autodl-service-acceptance-20260927-03/report.json`。
报告记录基线 HEAD `daff387` 和 9 个实际执行文件 SHA256；运行后逐文件复核一致。
因此，执行时虽尚未提交修复，实际代码版本仍可核验，不将旧 HEAD 当作修复版本。

| 验收项 | 03 实际结果 |
| --- | --- |
| 排队取消及外部输入边界 | 通过；浏览器自带 command 被拒绝 |
| 真实 CLI 执行、示例补丁独立评测 | passed；events / patch / checkpoint / summary 四端点有内容 |
| 运行中工具取消 | 0.314 秒，剩余活跃 guest 进程 0 |
| 独立评测取消 | 0.516 秒，剩余活跃 guest 进程 0；记录 blocked / formal evaluation cancelled |
| 已确认 edit 的恢复 | edit 执行一次，三条动作回执各一份；resume_count=1，独立示例评测 passed |

验收完成后，复查该目录分配的 7 个工作区 UID，活跃 guest 进程仍为 0。
仅以同一次运行计时，不将该耗时作为不同服务器性能基准。

## 回归与复现

- 云端 `service-cancellation-full-tests-03.xml`：261 项通过，0 失败/错误/跳过，33.975 秒。
  包含实际加载的 7 项服务测试和显式 live NsJail 准入，不再将缺失 FastAPI 的模块跳过
  包装为服务通过。
- 本地相关测试：21 项，17 通过、4 个 Linux 专项跳过，0 失败/错误，29.171 秒。
- 云端 `service-cancellation-full-tests-01.xml` 为失败记录：252 项、29 失败。该调用未等待
  上传完成，且可信 local fixture 的 PATH 未包含 Python；不能支持新版代码结论。
  等待上传并配置明确 PATH 后，02 为 260 项全通过；加入最终竞态用例后得到上述 03。
- 环境安装已有声明的 service extra 与 httpx，`pip check` 无破损依赖；实际 FastAPI
  0.141.1、SQLAlchemy 2.1.1、httpx 0.28.1。TestClient 有弃用警告，当前测试仍执行；
  未为消除警告新增另一套 HTTP 客户端。

云端全量命令应显式使用 agent-env 的 Python 与 PATH，设置 `PYTHONPATH=src`，
并以 `CODEAGENTBENCH_TEST_NSJAIL=1`、实际 rootfs 路径执行 `python -m pytest -q`。
同机验收入口与权限边界见 `service-acceptance.md`，使用新目录，禁止覆盖旧报告。

## PostgreSQL 临时集成验收（2026-09-27）

为补齐数据库集成证据，在 AutoDL Ubuntu 22.04 使用官方软件源安装 PostgreSQL 14，
并创建仅通过本机 Unix socket 访问的专用空测试库。以 peer 身份认证运行测试；
没有设置 API key、数据库密码或 TCP 公网监听。

第一次全套调用误把 `CODEAGENTBENCH_EXECUTOR=nsjail` 全局施加到所有测试，
而不是只启用显式 live-sandbox 准入；测试反复向系统 `/tmp` 复制 rootfs，造成
overlay 暂时耗尽。该次 JUnit 失败记录保留，不用来推断产品缺陷。确认 pytest 已退出后，
只删除本次调用创建的临时目录，系统盘恢复到 16% 使用率；此前的 pytest 记录和实验文件
均未清理。

修正调用后，不强制全局 executor，只设置 `CODEAGENTBENCH_TEST_NSJAIL=1` 运行
显式 live-sandbox 测试；使用独立数据盘 `--basetemp`、agent-env Python、真实 rootfs
和专用 `TEST_POSTGRES_URL`。最终私有 JUnit 为
`experiments/data-quality-audit-20260927/postgres-full-tests-corrected.xml`：
**278 项通过，0 失败、0 错误、0 跳过**。PostgreSQL 参数化的 6 个服务测试通过，
覆盖并发任务领取、过期租约隔离、脚本补丁与独立评测、运行中取消、Worker 执行器策略
和取消竞态。测试后确认 `jobs` 表已由 fixture 删除；专用测试库与临时角色已删除，
PostgreSQL cluster 已停止，5432 端口无响应。

此结果是 PostgreSQL 14 上的临时集成测试，不等同于生产部署、连接池压力、迁移升级、
认证授权或多租户隔离验收。复现需新建 disposable DB，不能指向现有队列库。

## 未覆盖的交付要求

已有 TestClient + SQLite + Worker/NsJail 与 PostgreSQL 临时集成验收；仍不是浏览器交互、
生产认证授权、多用户隔离或任意 SIGKILL/宿主故障下的恢复证明。
页面响应和 artifact API 可用不代表前端加载、空状态、错误、重试及可访问性已完成。
后续必须逐项验收，不能因此宣布完整项目完成。私有轨迹和权重未上传公开 GitHub，
外部备份仍按用户要求暂缓。
