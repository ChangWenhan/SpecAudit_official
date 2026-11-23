# SpecAudit 中文说明

文档更新：2026-11-23。原始 2026-10-11 论文证据仍保持冻结；本轮新增实验、双判官结果、三 seed 聚合、learned proposal source 补充实验和远端结果回流均已完成。当前主线和引言结构见 [docs/POSITIONING.md](docs/POSITIONING.md)。

## 一句话版本

**宽松验证让外部草稿模型的选择进入最终回答；当草稿的安全行为不稳定时，一个原本只负责加速的小模型就会变成整个服务的安全依赖。**

第三方模型存在不等于一定有害，检查放宽也不等于每种配置都会出现危害。项目展示的是已测条件下的供应链风险、传播路径和边界。

## 当前完成状态

| 工作项 | 最终状态 |
|---|---|
| P0 机制实验 | 生成 `25,191/25,191`，正式验证 `9/9`，fresh control `9/9`，release gate 为 `ready` |
| P0 双判官 | gpt-oss `25,191/25,191`、Qwen3Guard `25,191/25,191`，两者均为 `unknown=0` |
| E13 seeds 1/2 | 生成、双判官、打分和聚合全部完成；每个 seed 的两套判官结果各为 36 个文件、19,593 条标签 |
| E13 最终分析 | 126 条 ASR 聚合行、21 条 utility 聚合行，Table 1 primary/auxiliary 与 Table 5 judge comparison 已生成 |
| IFEval utility | seeds 0/1/2 完成；5 个实际生成配置 × 3 个模型 × 3 个 seed，共 45 个基础格、9,000 条回答 |
| Learned proposal sources | Medusa-style 与 EAGLE-style 完成 18 个文件、5,598 条有效主判官标签，`unknown=0`；N-gram 与 Lookahead 不属于 learned draft 威胁模型，正式结果已移除 |
| 远端执行 | s109、s111 项目目录已在本机校验压缩归档后删除；模型权重和环境未进入本地归档 |

当前没有排队中的实验。最终入口：

- P0：`results/p0_final_20261119/`
- P0 分析：`results/pareto_20261118/`
- E13 三 seed 分析：`results/e13_analysis_20261122/`
- s109/s111 审计归档：`results/remote_archive_20261122/`

## 三个角色

| 角色 | 负责什么 |
|---|---|
| 用户 | 提交原始请求；比较中不修改问题 |
| 服务方 | 掌握目标大模型和验证策略 |
| 草稿提供者 | 提供小模型权重，其安全行为可能与目标模型不同 |

小模型先提出若干候选 token，大模型在一次前向中打分。接受的 token 进入回答；拒绝后的恢复分布也属于完整验证策略。精确验证同时约束接受和恢复，因此保持目标模型分布；宽松验证只检查排名或概率门槛，不提供同样的分布保证。

## 受控供应链实验

目标模型、原始请求和服务端规则在比较中固定。研究使用同家族原版草稿和经过有害续写数据微调的草稿，展示外部权重行为怎样穿过验证规则影响最终回答。微调草稿是可控的高风险实例，不代表所有第三方供应商都会故意提供有害权重。

三个 target/draft 模型对：

| Target | Draft |
|---|---|
| Qwen3-4B | Qwen3-0.6B |
| Gemma-3-4B | Gemma-3-1B |
| MiniCPM4-8B | MiniCPM4-0.5B |

有害请求数据集为 JBB（100）、AdvBench（520）和 StrongREJECT（313），每个模型对共 933 条。主判官是 `gpt-oss-safeguard-20b`；`Qwen3Guard-Gen-4B` 作为辅助视角单独报告，不与主判官平均。

## 冻结证据的核心结论

JBB 结果来自主判官，每格 100 条请求：

| 条件 | Qwen3-4B | Gemma-3-4B | MiniCPM4-8B |
|---|---:|---:|---:|
| Target only | 1% | 5% | 10% |
| 原版草稿 + rank-50 | 24% | 6% | 34% |
| 微调草稿 + rank-50 | 59% | 77% | 60% |
| 同一微调草稿 + Exact | 1% | 8% | 10% |

九组 target/dataset 比较中，rank-50 相对 target-only 增加 **50.00–81.73 个百分点**，经 Holm 修正后均显著。同一微调草稿改用 Exact 后，与 target-only 的差值为 **−1.15 至 +3.00 个百分点**，九组均未检出显著差异。

概率下限和前十个回答位置的精确验证能降低风险，但两类缓解相对 target-only 仍有 **16/18** 组显著偏高。Exact 与 target-only 未检出差异也不等于证明两个分布完全相同。

## 新增机制实验

P0 将 N sweep 与 opening-token 2×2 合并成一个九条件批次。位置 0 被强制指定，位置 1 及之后切换 rank-50、前缀 Exact 或 Exact；强制 token 会推进真实输出位置并丢弃旧草稿链。该批次还包含 Exact anchor，因此可以用同一批回答分析位置 0 的交互作用、N 的增长和 acceptance 与 ASR 的权衡。

最终 P0 生成、结构验证、fresh-control、双判官和 release gate 全部通过。分析同时使用主判官与 Qwen3Guard，后者把 `controversial` 作为宽口径下的有效阳性，而不是静默丢弃。

E13 将主表扩展到三个 seed：Target、Draft、rank-50、Exact、SecureSD、TAP、FlipAttack。最终导出包含 126 条 ASR 行；均值与 min–max 范围按 seed 0/1/2 报告，不使用自由度为 2 的 t 区间。IFEval 覆盖 5 个实际生成配置，TAP 和 FlipAttack 按定义复用 Bare 输出，因此最终 utility 表有 21 行。

## 结果边界

- 三 seed 证据只覆盖主表；K sweep、概率规则、多数 defense 格和辅助判官矩阵仍是单 seed。
- 实验使用 Hugging Face `transformers` 解码循环，不是生产 vLLM/SGLang 集成，也不作生产延迟或吞吐结论。
- 没有 adaptive attacker；缓解后的残余风险不能解释成防御稳健性的保证。
- rank-50 不是对生产默认值的声明；结论限于已测模型、请求、规则和实现。
- 原版草稿并非自动无害或自动未对齐；现有格也不构成完整的 released/modified × exact/rank 因子实验。
- 自动判官可以误判；辅助判官是非独立第二视角，主分析始终单独报告它。

完整限制见 [docs/LIMITATIONS.md](docs/LIMITATIONS.md)。

## 代码、结果与复现

`src/` 是执行代码，`scripts/` 包含生成、判官、聚合和绘图脚本，`tests/` 是 CPU 回归检查，`docs/` 记录方法、证据、复现和限制。

冻结论文入口是 [evidence.json](results/paper_evidence_20261011/evidence.json)：135 格、41,985 条回答、83,970 条自动判定、486 组成对比较。原始生成和判官 JSONL 含有害模型输出，不进入公开 release；本工作区内保留完整实验文件和审计归档。模型权重、benchmark 资产和环境由各自提供方管理，具体命令见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 相关工作

- **SecureSD**：研究与输入有关的策略退化，并采用按输出位置加强验证的设计；本项目承认策略层的先验重合。
- **SSD**：假定小模型具有所需安全属性；本项目研究外部草稿行为缺少这种保证的情形。
- **SafeSpec**：将安全估计与轨迹恢复纳入投机推理。
- **Speedbumps**：研究成本与可用性方向的拒绝攻击，与最终回答的安全帮助风险正交。

详见 [docs/RELATED_WORK.md](docs/RELATED_WORK.md)。
