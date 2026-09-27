# SWE-Gym-Plus 最快完成接续记录

核查日期：2026-09-27。目标仍是完成到第五步，不新增 GRPO、简历包装或服务器租用。

下面的长记录包含各批次的历史快照；当前状态以“当前证据”和最新正式收据为准，不能把旧的 16/20、17/20 或训练未启动描述当作最新状态。

## 本轮最终结论：已完成到第五步并停止新增实验

已经完成“20 个合格任务采集 → 动作数据审计 → 3B BF16 LoRA → 冻结留出配对评测 → 收据、资产清单和接续记录”这一轮闭环。正式流程返回码 0，记录耗时 2073.8574001660018 秒（约 34 分 34 秒）；不包含此前全部采集时间和 Base 预执行时间。

- 训练：20 个不同合格任务、250 条动作样本、20 条结束动作；1 epoch、32 个优化器步骤，真实 LoRA 参数更新已核验。
- 评测：原冻结 4 dev + 20 eval 全部保留。4 dev 全部未准入；20 eval 中 12 个准入、8 个未准入。12 组配对产生 24 份模型运行收据和 24 次真实独立判分事件，未换题、未重跑挑选结果。
- 结果：Base 补丁独立通过 **1/12（8.33%）**，SFT **0/12**。Base 唯一通过任务是 `getmoto__moto-5406`；但该 Agent 仍因步数上限以 failed 结束。两种模型的 12 个 Agent 状态均为 failed，不能称为正常完成 12 个任务。
- 结论：训练与评测系统闭环已跑通，**本轮没有证明修复能力提升**。一个不一致配对，双侧配对精确检验 p=1.0，不能把差异说成显著退化。未准入项是环境/控制检查排除，不混算为模型修复失败。
- 已保存私有 `verification.json` 和 `final-assets.verified.json`，后者索引 326 个正式训练/评测文件。原始轨迹与权重仍留在 ModelScope；用户暂缓独立外部备份，本轮代码和脱敏记录尚未推送 GitHub。

主要失败现象：Base 为 4 次原地重复命令、8 次步数上限，4 个任务产生补丁；SFT 为 10 次原地重复命令、2 次步数上限，没有产生补丁。该现象提示下一轮应先分析动作学习与运行交互，而非把“加数据或加显卡”直接当成已证明的解决办法。没有据此擅自启动下一轮训练、GRPO 或修改这轮结果。

本轮只有一个训练种子，且仅为 Moto 子集试验。没有可用的 dev 任务，不存在基于 dev 选择最佳 checkpoint 的证据；没有完整 SWE-Gym 泛化性能或自进化收益结论。

## 当前证据

| 项目 | 核查结果 |
|---|---|
| 工程版 | 已有云端 108 项全通过及服务/恢复/页面验收证据；逐条队列批次 52 项相关测试通过；可靠接续批次的 56 项选定回归测试在本机和云端通过 |
| 任务环境 | 原冻结 67 train 全部重检：38 准入、19 测试启动/收集错误、10 官方补丁测试失败 |
| 训练轨迹 | 已达到 20/20 个不同合格任务：6 条合格种子 + 8 条 Flash + 2 条首轮 Pro + 1 条初版原生工具 + 3 条可靠原生工具；正式审计保留 250 条动作样本 |
| 留出集 | 原 4 dev + 20 eval 全部保留：4 dev 未准入；20 eval 中 12 准入、8 未准入；12 组 Base/SFT 配对判分已完成 |
| 并行采集 | 可靠批次 14 个已提交任务全部收尾：3 accepted、11 rejected；达到 20 后自动停止新采集，已冻结私有合并快照 |
| 正式训练 | 无密钥 Qwen2.5-Coder-3B-Instruct BF16 LoRA 已完成，1 epoch、32 个优化器步骤、返回码 0；含加载保存约 272.19 秒；最终 Base 1/12、SFT 0/12，没有提升证据 |
| GPU 并行利用 | 两路无凭据本地 Base 推理已跑完原冻结留出集中 7 个 Base 先行任务，独立评测 0/7；不是 SFT，也不是全部 24 题对照 |
| 私有资产 | 留在 ModelScope，最终 326 文件清单及 24 次独立评测核验已落盘。用户暂缓外部备份；云端清单不是备份 |

Pro 批次已获用户对具体数据和目的地的授权。密钥仅进入隐藏提示和 API 子进程环境，不写入文件或 Git。新批次不导出模型 reasoning_content，completion token 仍计入模型返回的总消耗。

首轮 Pro 最終报告为 `insufficient_successes`：2 个任务 accepted、22 个任务 rejected。最初 21 个结果中 11 个因无有效 JSON 动作失败；这些实际返回为空白、残缺片段或输出上限截断，不能重建为成功动作。接续批次单独采用 `deepseek-v4-pro`、thinking enabled/low、text 响应、16384 输出 token，仍由原运行层检查实际动作，不修改已结束结果。

协议接续第一次启动因遗漏 CLI 必需的 `DEEPSEEK_MIN_BALANCE_CNY` 而被拦截。四份 rollout 日志均只有该启动错误，模型事件文件和 teacher receipt 均为 0，证明未发出模型调用。原目录保留。修复后使用新目录接续；30 项本地相关测试退出码 0，云端 AST/导入和规范化 LF 源码哈希校验通过。现已看到四个子进程产生实际模型响应收据，但尚未新增合格成功结果。

低思考文本批次累计 13 次尝试均 rejected。停止了协调器的新提交，并等待独立会话的子任务自然收尾；13 个已提交任务都有明确 summary/outcome，独占锁已释放，无未知付费结果。旧批次源码和结果保留，新批次独立使用 `scripts/repair_protocol_next.py`，thinking enabled/high、text、16384 输出 token，参数已在 identity 与实际 API 收据核验。32 项本地相关测试退出码 0，云端 AST/导入和规范化 LF 校验通过。新批次的 identity 保存前序 13 份 outcome 哈希，累计仍为 16/20；正式训练尚未启动。

曾询问是否改为用现有 16 条先做小规模验证；尚未收到用户确认，因此仍执行 20 个不同合格任务门槛，不自行降低标准。

最新接续核查发现两个具体瓶颈：文本协议下模型会输出自编工具结果，运行层压缩上下文后也会失去部分执行历史；原生工具协议下模型有时一次返回两个 shell 调用，而初版适配只允许一个。高思考文本批次的 21 个已提交任务均写完明确 summary/outcome，均 rejected；协调器只停止新提交，子进程自然收尾。初版原生工具批次的 10 个已提交任务也已全部写完明确结果，为 9 rejected、1 accepted；新增 `getmoto__moto-7081` 正常 completed、独立 evaluation passed，合并后累计 17 个合格任务。旧源码和结果保留，不改判失败记录。

独立接续模块 `scripts/collect_native_queue.py` 使用 DeepSeek 官方原生工具接口。它在进程内保留历史 reasoning_content，并按官方协议只回传同一 API，不打印或导出该字段。最多 8 个完整、类型合法的 shell 调用会依次交给原沙箱执行；每条都计入动作预算并要求匹配的真实 journal 回执。只有全部回执完成后才发下一次 API 请求。结束调用不得与尚未观察结果的命令混发；空白、截断、非法参数、伪造工具结果及未知传输结果仍失败关闭。不重放传输结果未知的付费请求，不恢复丢失原生会话内存的运行。

队列动作的模型事件数不等于实际 API 请求数。同一已知响应的后续队列动作不重复请求，prompt/completion token 为 0；总 API usage 记在该响应的首个动作上。独立 tool receipt 与 dispatcher 哈希用于核验各条实际动作。

逐条队列批次首组三个任务已独立判分，均 rejected，累计仍为 17/20：`getmoto__moto-5575` 正常 completed，但 CloudFront 空 invalidation 列表测试失败；`getmoto__moto-5306` 时间预算结束且 EC2 响应测试失败；`getmoto__moto-6804` 时间预算结束且伙伴事件源实现尚缺失。补丁存在或 Agent 声称完成均不算合格，程序已自动继续下一组已准入训练任务。此时尚无正式训练收据。

后续确认了另一项适配缺陷：原生会话只同步真实工具回执，没有同步运行层产生的 Harness warning/checkpoint。独立模块 `scripts/collect_native_reliable.py` 保留前一版源码不变，新增这些可信提醒的去重传递，并等一组原生调用的全部真实回执到齐后再附加提醒，避免打断 API 工具消息序列；不传入压缩摘要、伪造工具输出或 assistant 自编提醒。新测试覆盖压缩后提醒保留、重复同步不重复消息、排除非 Harness 文本及多调用中间不插入 user 消息。该批次上限为 48 动作、48 工具、1024000 总 token、1200 秒，目的在于避免高思考下过短时间预算，不降低合格标准。

逐条队列旧协调器停止新提交后已自然排空；9 个已提交任务都有明确结果，均 rejected，合并仍为 17/20。可靠接续批次持有四个已排空前序目录的独占锁，并保存其全部 outcome 哈希。云端最新选定测试 JUnit 为 56 tests、0 failures、0 errors、0 skipped，本机同组测试退出码 0。第一次本机测试因默认临时目录权限失败，改用工作区内新建的专用 pytest 临时目录后通过；没有据此修改业务代码。

为利用采集等待时间，新增 `scripts/prewarm_baselines.py`，仅预执行原定模型顺序为 Base→SFT、且此前已准入的 7 个留出任务，不以模型成败选题。两路本地模型进程已核实持有 AMD GPU 句柄且不含 DeepSeek/OpenAI API 凭据。Base 仍使用 12 步、12 工具、60000 总 token、300 秒、768 新 token 的原预算；SFT 先行的任务不提前跑 Base。原训练协调器未来启动后，在进入配对阶段前等待预执行的独占锁释放并核对完整收据、模型字节、源码、冻结清单和预算，再复用相同 run_id 的 Base 收据，防止重复推理。只有真实 Base/SFT 均有判分后才进入 pairs/report；现在 pairs.json 为 0 不能解读为没有基线进程。

预执行部署保留了原 `run_clean_training.py` 的不可变副本。未修改当前活动采集脚本、冻结任务、模型权重、样本合格门槛或评测预算。最新云端选定回归 JUnit 为 60 tests、0 failures、0 errors、0 skipped。本机新增组也无失败，Linux 文件锁测试在 Windows 上按声明跳过，云端该测试已通过。后续不要修改 src/scripts 源码，否则预执行的源码身份检查会拒绝混用。

部署时补齐了此前仅在本机的 5 份回归测试及 `preflight_controls.py`。云端测试发现旧任务适配器尚缺本机已有的 shell 参数引号修复；核对文件无其他差异后，仅同步 `shlex.join` 转义。冻结 manifest/split 和测试选择器未改动。最新云端 JUnit 为 52 tests、0 failures、0 errors、0 skipped；本机同组测试退出码 0。

### 最新正式训练进展

可靠批次新增 `getmoto__moto-5560`、`getmoto__moto-5725`、`getmoto__moto-6810`，三者均 Agent completed、独立 evaluation passed、盲提示成功导出。累计 20 个不同训练任务后自动停止采集。正式 `pipeline.log` 已启动，audit 返回码 0；`data/actions.audit.json` 为 250 records、20 distinct_tasks、20 done_examples、max_tokens=5066、total_tokens=745260、supervised_tokens=28931、truncated=0、rejected=[]。正式数据输出 SHA256 为 `923a256484e20a9713477f7e1c00b62ce155f84a35b3c8fb14973caa88ddbcc5`，split SHA256 为 `14b2e53b181ae647fc18a1bdbf340ebaa943e9f9d1cb379920de23346c16ce57`。

训练日志已加载全部模型权重并进入 0/32 优化器步骤。通过 `/proc` 仅检查进程身份、GPU 设备句柄和凭据变量名称，确认实际 `codeagentbench.train_sft` 进程持有 GPU 句柄且不含 DEEPSEEK_API_KEY/OPENAI_API_KEY；未打印环境变量值。训练尚未写出完成收据，后续不得重复启动。

后续训练已完成：`logs/training.receipt.json` 返回码 0、seconds=272.19346430899895；`checkpoint/train_metrics.json` 为 250 train_records、250 encoded_examples、dropped_no_supervision=0、truncated_examples=0、global_step=32、epochs=1、seed=42、gradient_accumulation_steps=8。LoRA r=16、alpha=32、dropout=0.05；Trainer 记录 train_runtime=255.9901 秒、train_loss=0.6879699612036347。这是 BF16 LoRA，不是 QLoRA。

权重文件 `checkpoint/adapter_model.safetensors` 为 119801528 bytes，SHA256 `ab6822e542788fcad89629501d33b2fbb13c3315fc7aa4cac1bbe64279f04705`。只读 CPU 检查确认 504 个 LoRA 张量全部有限值，252 个 LoRA-B 张量非零，实际发生参数更新。权重、tokenizer、训练参数及指标已落盘。自动配对评测已经启动；此时 pairs.json 先保存四个 dev 的 not_admitted，不可据此声称已完成全部评测。

正式训练前，已有 Base 预执行复用检查实际通过：源码、模型字节、冻结 manifest/split、预算、已结束报告和文件锁全部匹配。没有修改源码或重新推理 Base。

训练结束后生成私有 `training-assets.verified.json`：30 个已完成训练文件、20 份合格轨迹快照均记录大小和 SHA256，并再次核对当前源码与 `experiment.json` 中的 56 个源码文件哈希一致。清单 SHA256 为 `578cc3bb3c571bccb03972b1558ecb1c55d755cf96cf9cfdf8cb6375e740fba1`。该清单尚不包含仍在变化的配对日志和最终报告，也不是独立备份。首次盘点命令因没有设置项目模块路径而在 import 阶段失败，没有写文件；使用 `PYTHONPATH=src` 后成功。

Base 预执行已写完 `base-prefetch-report.json`，status=completed。7 个任务均有 rollout 与独立 evaluation 回执，独立通过 0 个。实际 Agent summary 为 failed：6082、4787、4895 因原地重复命令退出，6028、7119、6637、5899 因 12 步上限退出。CLI rollout 返回码 0 仅表示命令执行完成，不表示 Agent 正常完成或补丁通过。训练未来仍复用这些原始回执，不依据结果换题或重跑 Base。GPU 预执行进程已结束，服务器 CPU 继续执行沙箱及判分。

此前 17 任务时的只读预审在云端终端 7 完成，不启动训练、不改变 20 任务门槛、不写入正式数据目录。17 个严格合格任务经原 `action_examples` 转换为 182 条动作样本，其中 17 条结束动作；按原 tokenizer、8192 长度限制、去重和 assistant-only 监督检查，182 条全部保留，17 个任务全部存活。最长 4461 token，总输入与目标合计 502973 token，实际监督 20113 token；没有重复、超长或缺失监督样本。这是历史预审快照，不是本轮最终训练数据规模。

历史快照中，可靠批次前 9 个已结束任务均 rejected：6 个 Agent completed、2 个 blocked、1 个 failed，独立评测均 failed_or_blocked。部分可见测试通过仍不能替代独立评测。该批次最终为 14 个已提交任务全部收尾、3 accepted、11 rejected；21 个候选中的其余 7 个因已达到 20 个合格任务门槛而未提交，不改判旧记录。

已只读核查官方 [OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories)：公开 schema 包含 instance_id、run_id、resolved、原始 messages 和 test_result。但是云实例对 Hugging Face 下载返回 Network is unreachable，未取得文件、未完成与本项目冻结拆分的 join、未导入样本。官方标签不能直接充当本项目的正常结束及独立判分回执；若以后采用，需单独记录来源、协议转换及防泄漏审计，不得计为自行采集的成功。数据质量技能用于限定这项补充数据核查，没有改变实验源或门槛。

以下为此次只读预审的关键代码。结果是运行时快照，后续新增合格任务会改变计数；不将其当作正式训练收据。

```python
import collections, hashlib, json
from pathlib import Path
from transformers import AutoTokenizer
from scripts.fast_finish import qualified_paths
from scripts.prepare_action_sft import action_examples
from codeagentbench.runtime import AgentRuntime
from codeagentbench.train_sft import encode_example, normalise_messages, IGNORE_INDEX

b = Path('/mnt/workspace/swe-gym-plus/experiments')
roots = [b / n for n in [
    'accelerated-20260926', 'thinking-20260927', 'protocol-retry2-20260927',
    'protocol-hightext-20260927', 'native-tools-20260927',
    'native-queue-20260927', 'native-reliable-20260927',
]]
train = json.loads((roots[0] / 'pool/split.json').read_text())['train']
passed = qualified_paths(roots, train)
tok = AutoTokenizer.from_pretrained(
    '/mnt/workspace/models/Qwen2.5-Coder-3B-Instruct',
    local_files_only=True, trust_remote_code=False,
)
seen, counts, rejected = set(), collections.Counter(), collections.Counter()
lengths, raw, supervised, done = [], 0, 0, 0
for task, path in sorted(passed.items()):
    rows = action_examples(json.loads(path.read_text()),
        system_prompt=AgentRuntime._system_prompt(),
        context_chars=10000, include_done=True, history=True)
    raw += len(rows)
    for row in rows:
        messages = normalise_messages(row['messages'])
        fp = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        length = len(tok(text, add_special_tokens=False)['input_ids'])
        reason = 'duplicate' if fp in seen else 'too_long' if length > 8192 else ''
        example = encode_example(tok, messages, 8192) if not reason else None
        if not reason and (not example or not any(x != IGNORE_INDEX for x in example['labels'])):
            reason = 'no_supervision'
        if reason:
            rejected[reason] += 1
            continue
        seen.add(fp)
        counts[task] += 1
        lengths.append(length)
        supervised += sum(x != IGNORE_INDEX for x in example['labels'])
        done += bool(json.loads(row['messages'][-1]['content'])['done'])
print(dict(qualified_tasks=len(passed), raw_actions=raw, kept_actions=len(lengths),
    surviving_tasks=len(counts), done_examples=done, max_tokens=max(lengths),
    total_tokens=sum(lengths), supervised_tokens=supervised, rejected=dict(rejected)))
```

## 本轮完成清单

1. [x] 累计 20 个不同训练任务：初始提示无评测泄漏、Agent 正常结束、补丁独立评测通过；保留失败和阻塞记录。
2. [x] 冻结私有合并快照，完成去重、监督区间和 tokenizer 长度审计：250 条动作样本，没有静默截断。
3. [x] Qwen2.5-Coder-3B-Instruct BF16 LoRA、1 epoch：32 步，返回码 0，权重有效且实际更新；不带 API 密钥。
4. [x] 原冻结留出集 Base/SFT 配对评测：两路独立任务并行，每题模型顺序交替，同样预算；保留全部 24 个任务，核对 24 次独立判分。
5. [x] 核对训练收据、权重、配对结果、报告、源码身份和资产哈希，保存接续记录。没有提升也如实交付；到此停止新增实验。

### 最终关键文件 SHA256

| 文件 | SHA256 |
|---|---|
| `pairs.json` | `7cf0844b0591fd9ad8a466f68f0c506ca292793b6279c6c5eb03adc418351055` |
| `report.json` | `02ad2c43198dbd95dc0dbee1d4867707cdd0c5df4655349374e11bb61762e110` |
| `pipeline-receipt.json` | `5c35fa94419c441fbbd392966917eaf98506a09ff9cf583cfacaba5c70c417e6` |
| `verification.json` | `98c9225c44123f58f6dde9de613430ee321ac0c00c4197d72e2f3da0cc338700` |
| `final-assets.verified.json` | `935d42369945a7f4f4344813cc77233968a510aee7e92f9f49372c8b7dd444dd` |

最终核验逐一对照 frozen task 顺序、质量准入、模型 summary、预算、补丁存在性、真实 evaluation 事件和流程回执；重新计算配对通过数及不一致检验，并确认训练资产哈希和 56 个源码文件身份未变。

正式环境记录：Python 3.12.13、PyTorch 2.12.0+git6bbd260、ROCm 7.2.53211、Transformers 5.15.1、PEFT 0.20.0、Accelerate 1.14.0；显存报告 205822885888 bytes（约 191.7 GiB）。环境收据的 GPU 名称字段为空，不能据此认定具体 AMD 卡型号。

## 云端运行与证据位置

- 代码工作树：`/mnt/workspace/swe-gym-plus-completion`，HEAD `a3aa038f01ec77a6d4de0961b4fa54caf71d5ba9`；本轮修改尚未推送 GitHub。
- Flash：`/mnt/workspace/swe-gym-plus/experiments/accelerated-20260926`，已结束，合格 14 个（含 6 个种子）。
- 首轮 Pro：`/mnt/workspace/swe-gym-plus/experiments/thinking-20260927`，已结束，合并合格 16 个。
- 未调用 API 的启动失败：`/mnt/workspace/swe-gym-plus/experiments/protocol-retry-20260927`，保留完整错误记录，不能直接重启覆盖。
- 已排空的低思考文本批次：`/mnt/workspace/swe-gym-plus/experiments/protocol-retry2-20260927`，13 rejected；协调器中断新调度后自然排空，未生成完整 cohort report，不能直接重启覆盖。
- 已排空的高思考文本批次：`/mnt/workspace/swe-gym-plus/experiments/protocol-hightext-20260927`，21 rejected，保留结果和源码。
- 已排空的初版原生工具批次：`/mnt/workspace/swe-gym-plus/experiments/native-tools-20260927`，10 个已提交任务均有明确结果，不能直接重启覆盖。
- 已排空的原生工具逐条队列批次：`/mnt/workspace/swe-gym-plus/experiments/native-queue-20260927`，9 rejected，结果及源码保留。
- 当前四路可靠原生工具批次：`/mnt/workspace/swe-gym-plus/experiments/native-reliable-20260927`，云端终端 4。至少 20 个合格任务后合并各批成功导出，独占 pipeline.log 防止重复训练。
- 接续脚本：`scripts/repair_protocol.py`，规范化 LF SHA256 `c5589bbbf87232d32c4821517cd117ff12ea4d7ee8faf8619502d69359461406`。
- 独立高思考脚本：`scripts/repair_protocol_next.py`，规范化 LF SHA256 `7aa597891bb1f423bf21e2a3fef5862d0a8313e755d3049eaf866b0de444a851`。
- 初版原生工具模块：`scripts/collect_native.py`，文件 SHA256 `112ed09ab4af9d7809e97f8e5e5719570675848d043eee62d0ec0a9911ee4cfd`。
- 逐条队列模块：`scripts/collect_native_queue.py`，文件 SHA256 `8c75fe7de898ed76eda283cf328ed809a0e3c31c13201db935c521e775b22cae`，已核对云端与本机一致；identity 额外保存所有 core Python 源码哈希。
- 当前可靠模块：`scripts/collect_native_reliable.py`，文件 SHA256 `801327260cf7350adcc084278345779f702e3386e48ba71720d70206f2be293f`；测试文件 SHA256 `a07324b03a3a928029dfdd2c372d9fd79b9a1deaddd0e64d2898611a85fc4a13`，均已核对云端与本机一致。
- 可靠接续云端回归收据：`/mnt/workspace/swe-gym-plus/experiments/native-reliable-validation-20260927.xml`；此前队列收据不覆盖。
- 最新云端回归收据：`/mnt/workspace/swe-gym-plus/experiments/baseline-prefetch-validation-20260927.xml`。
- Base 预执行：正式目录中的 `base-prefetch.json`、`base-prefetch.lock`、`base-prefetch-report.json`、`logs/*-base*.receipt.json` 和 `paired/runs/*-base`；身份文件保存原始模型文件哈希及全部实际源码哈希。
- 预执行模块：`scripts/prewarm_baselines.py`，SHA256 `853b63b626d8fffe630b1cb7884ad777f2139a9d691e1d9f02bc63403008bb14`；当前训练协调器 SHA256 `2808231f8068249be5622bdb38f157de36f13a41d789cb276a3db5c06b094600`，均已核对云端与本机一致。
- 原训练协调器副本：`/mnt/workspace/swe-gym-plus/experiments/native-reliable-20260927/run_clean_training.before-prefetch.py`，SHA256 `f2dafdbe50e9af7038a26eac0e69f484199061906aa6fa13d6226d9091fc4b87`。
- 正式输出：`/mnt/workspace/swe-gym-plus/experiments/formal-accelerated-20260926`。
- 私有训练资产校验清单：正式目录中的 `training-assets.verified.json`；原始私有文件仍仅在云端工作区，外部备份暂缓。
- 私有最终判分核验及资产清单：正式目录中的 `verification.json` 和 `final-assets.verified.json`；当前状态为 verified_completed，不等于整个项目所有后续阶段都完成。
- 主要状态：两批 `collection-progress.json` / `progress.json`；Pro `snapshot/sources.json`；正式 `pipeline.log`、`pipeline-receipt.json`、`experiment.json`、`logs/*.receipt.json`、`checkpoint/train_metrics.json`、`pairs.json`、`report.json`。

本轮结束时，终端 2、3、4、5、6、7 已回到 bash；终端 4 的正式流程、终端 5 的只读监测自然退出。终端 1 的旧采集器仍暂停，不要直接恢复。不得把未更新的旧 progress.json 当成最终状态，也不要覆盖已有实验目录重新启动；以正式 report、verification、全部 summary/outcome 和回执为准。

## 完成判据与故障处理

以正式 `report.json`、训练与配对收据为准，不以 GPU 利用率、目录存在或启动命令为准。达标后先等待在途任务写完再冻结数据；如果数据不足、审计失败、未知付费结果或训练失败，保留日志并检查，不降低门槛、不自动重放不确定请求。

当前 ModelScope GPU 可用，PyTorch 可见显存约 191.7 GiB。CPU 执行沙箱命令和独立判分；GPU 已完成 Base 预执行、正式 3B LoRA 微调及全部 Base/SFT 配对推理。最终核验时实例剩余约 1 小时 37 分，属于倒计时，不是完成 ETA。私有正式资产已在 `/mnt/workspace` 落盘；实例仍开启，本轮没有替用户关闭或释放服务器。外部备份暂缓，不能把单个云工作区说成永久或独立备份。

收尾时网页截图与鼠标通道出现异常，但键盘、命令面板和终端复制仍可用。一次多行核对输入未被正确识别为终端粘贴，停在 here-document，未执行核对脚本；用正确结束标记返回 bash 后，以单行输入执行同一范围的核对，最终成功。没有重启训练或改动评测结果。
