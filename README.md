# MAB 模型组合路由中的并列监督、共享参数与部署规则

本仓库研究语言模型路由中的一个具体问题：**当有限次生成得到的经验奖励并列时，训练目标怎样改变共享预测器的参数、其他问题的动作排序，以及最终答案质量。** 原始系统在六个冻结模型中选择三个，形成 20 个组合，由固定聚合器输出答案；路由器在生成前只读取问题表征。

当前论文主线为 **E17–E23 的机制研究**。经验全并列训练行的对称损失项能够改变共享参数方向和其他问题的严格动作排序；最终质量还取决于预测近并列时怎样选择动作。E23 在原 P2 评测集上，只用训练先验改进 B 的部署规则，就将准确率提高到 86.50000%，使 C−B 从 +1.32598 缩至 +0.18056 个百分点。在同一评测集上，C 的准确率比三个匹配控制的等权混合高 0.52315 个百分点，Ridge 则以 86.87500% 高于 C 的 86.68056%。[E23 完整结果](routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md)补充了训练约束、部署规则与质量差异之间的关系。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 结论 | [主张与证据](paper/CLAIMS_AND_EVIDENCE.md) |
| 论文表格 | [结果表与来源](paper/RESULTS.md) · [CSV](paper/results.csv) |
| 研究为什么从反馈采集转到监督机制 | [研究历史](docs/HISTORY.md) |
| 旧实验分支 | [分支与归档索引](docs/BRANCHES.md) |
| 从公开汇总复核到完整重训各需要什么 | [复现说明](docs/REPRODUCIBILITY.md) |
| 编号、经验冠军规则与训练监督的关系 | [E17 结果摘要](routing_idea2/research/e17_supervision_distortion_audit/EXPERIMENT_SUMMARY.md) |
| 全并列约束的理论构造与端点干预 | [E19 理论与预测](routing_idea2/research/e19_symmetric_tie_constraint/01_THEORY_AND_PREDICTIONS.md) |
| 原系统的独立新题结果 | [P2 最终结果](routing_idea2/research/e20_p2_execution/P2_FINAL_RESULTS.md) |
| 跨损失与外部单选池的机制边界 | [P3 结果](routing_idea2/research/e21_p3_external_mechanism/P3_RESULTS.md) |
| 监督总量、匹配随机降权与参数几何 | [E22 审计](routing_idea2/research/e22_mechanism_specificity/AUDIT_REPORT.md) · [几何分析](routing_idea2/research/e22_mechanism_specificity/GEOMETRY.md) |
| 训练先验部署与同一评测集的对照方法 | [E23 摘要](routing_idea2/research/e23_deployment_prior_and_references/EXPERIMENT_SUMMARY.md) · [最终汇总](routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json) |

E1–E22 来源固定为提交 `8f490a06ebe1153be2fcbeb42ad3944d63075b7d`，E23 来源固定为 `3958f9ceaac5b0b1de88bbd47d89ddd940833ff2`。入口文档随研究进展更新，各阶段的协议、数据角色和结果保留其冻结版本。

## 核心设定与对照

原模型池为六个冻结语言模型，动作是从中选取三个成员的无序组合。成员独立回答后，由冻结的 Qwen3-8B 聚合器输出最终答案。主要奖励为最终答案的正确性；候选中出现正确答案只是一项辅助诊断。

E19–E23 原系统使用冻结 MiniLM 表征、训练区拟合的 PCA16 和标准化、一个截距以及 17×20 线性头，训练参数共 340 个。E21 外部实验使用六个单模型动作，每个任务分别拟合一个头。

设 T 为训练区中所有动作经验平均奖励完全相同的问题集合。B/C 两个端点保留相同训练问题、原始归一化分母、表示、函数类和正则，只改变 T 中行的损失权重：

| 端点 | 全并列行权重 | 其他训练行权重 | 含义 |
|---|---:|---:|---|
| B，β=1 | 1 | 1 | 保留对称约束 |
| C，β=0 | 0 | 1 | 关闭该约束，原归一化分母不变 |

经验全并列不等于真实条件期望完全相同。全并列行没有指定哪个动作更好，但它仍可能约束预测分差；共享参数使这项约束影响其他问题。E22 进一步使用同总权重均匀降权和三份预固定匹配随机掩码，检验质量变化是否特属于全并列行。

E23 保留原 B/C 权重和 `tau=1e-8` 支持集合，在集合内优先选择 train600 的三个生成重复中整数成功数最高的组合；计数仍并列时均匀选择。这一训练先验规则只改变 B 的 2,048 道近并列题，C 在全部 2,400 题上保持原动作。GlobalFixed 由同一训练统计选出组合 5。Uniform 与三个 Matched 则重新拟合全量 600 题的最终头，与 E22 的五折 OOF 控制分别归档。[策略定义与冻结顺序](routing_idea2/research/e23_deployment_prior_and_references/PROTOCOL.json)

## 当前主要结果

准确率和点差均来自相应阶段的冻结策略，按数据角色、模型池和损失分别列示。

| 阶段 | 评价范围 | 主要观察 | 解释范围 |
|---|---|---|---|
| E17 | 原 600 题开发池 | 100 次重编号后，坐标冠军 SPO+ 的真实组合一致率均值约 6.13%；Ridge 与均匀冠军 TSPO+ 为 100% | 编号与冠军规则会塑造真实组合选择 |
| E18 | 2,998 道可评分独立问题 | TSPO_uniform 对 SPO_coordinate 为 +0.011749 个百分点，纠错 377、破坏 376 | 冠军方向干预的质量点差接近零 |
| E19 / P1 | 原 600 题五折开发分析 | B/C 严格换队 150 题；C−B 为 +1.442 个百分点 | 受控机制与开发结果，不是独立新题评价 |
| E20 / P2 | 2,400 道独立新题 | B 85.355%，C 86.681%，C−B 为 +1.326 个百分点；严格换队 302 题 | 原系统冻结 B/C 策略的独立质量结果；主要收益来自 BBH |
| E21 / P3 | 原池 ListNet 与外部 548 题 | 原池 ListNet 严格换队 135 题、质量差 −0.278 个百分点；外部 TSPO+ / ListNet 严格换队 113 / 41 题、质量差 +0.145 / −0.202 个百分点 | 严格排序变化可扩展，质量方向不一致 |
| E22 | 原 600 题匹配控制 | C 相对三份匹配随机掩码平均，TSPO+ 为 +0.352、ListNet 为 −0.019 个百分点 | 对全并列行定向降权的特殊质量优势证据不足 |
| E23 | 已曝光 P2 原 2,400 题的补充比较 | B 训练先验 86.50000%，C 86.68056%，差 +0.18056 个百分点；C 对 MatchedMean +0.52315；Ridge 86.87500% | 改进 B 部署规则获得 +1.14542；C 准确率高于本轮匹配控制，Ridge 总体准确率最高 |

P2 原主规则在距最大分不超过 1e−8 的动作间均匀分配，质量按冻结概率计算期望。raw-exact 使用已有浮点分数的零容差最大集合，其敏感性点差为 +0.639 个百分点；该名称只指浮点解码规则。涉及均匀策略的纠错和破坏按期望计数报告。

P3 的 62.194% 对照是**各任务分别用训练区选择固定模型，再对两个任务等权汇总**。旧摘要中的 `global_fixed` 按执行代码的任务内语义解释。两个外部主对照的 C 均低于该固定基线。

E22 显示参数变化超出共同平移或统一正比例缩放；一般训练行降权也能造成广泛严格换队。E23 进一步保留原 302 道严格换队题的 +0.26389 个百分点贡献，但原 B 近并列区在使用训练先验后，对 C 的贡献变为 −0.08333，两者合计 +0.18056。MatchedMean 是三份预定策略的等权动作概率混合，C 相对它在两个任务、三个生成 seed 上的点差均为正；C 相对 B 训练先验的任务和 seed 方向则有差异。[分组点差与区域贡献](routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)

## 复现入口与所需资产

先在仓库根目录执行公开完整性与论文表格复核：

```bash
python scripts/verify_repository.py
python scripts/build_paper_tables.py --check
```

这两条命令检查仓库文件与已公开汇总的一致性。逐题重算、模型拟合和物理生成的执行层次见 [复现说明](docs/REPRODUCIBILITY.md)。

仓库提供源码、协议、脱敏汇总、文件索引和执行账本。完整重训与生成还需原题、标准答案、逐题奖励矩阵、拟合权重和原生缓存等私有资产；E18/P2 生成链还需补齐 `typed_aggregation_execution.py`、`typed_operator_contract.py`、`stage1_review_common.py` 和 `budget_router_v2/scoring.py` 四个源文件。各阶段入口如下：

| 范围 | 复现说明 | 资产边界 |
|---|---|---|
| E17 监督失真 | [E17 REPRODUCE](routing_idea2/research/e17_supervision_distortion_audit/REPRODUCE.md) | 真实数据重放需原 train600 与冻结折包 |
| E18 独立确认 | [E18 REPRODUCE](routing_idea2/research/e18_confirmatory_execution/final_results/REPRODUCE.md) | 逐题重算需独立评价私有缓存 |
| E19 端点理论与开发分析 | [E19 REPRODUCE](routing_idea2/research/e19_symmetric_tie_constraint/REPRODUCE.md) | 人工构造与真实数据步骤分开；后者需冻结训练资产 |
| P2 独立新题 | [P2 REPRODUCE](routing_idea2/research/e20_p2_execution/REPRODUCE.md) | 需输入登记、已冻结模型及合法原生输出/评分资产 |
| P3 跨损失与外部缓存 | [P3 REPRODUCE](routing_idea2/research/e21_p3_external_mechanism/REPRODUCE.md) | 外部数据版本在协议中固定；原池对照仍依赖 train600 |
| E22 匹配与几何审计 | [E22 README](routing_idea2/research/e22_mechanism_specificity/README.md) | 需 E19、P3 冻结模型、训练资产与固定掩码来源 |
| E23 部署与对照方法 | [E23 REPRODUCE](routing_idea2/research/e23_deployment_prior_and_references/REPRODUCE.md) | 需原 P2 资产、E17 Ridge、E23 冻结策略及新增调用收据 |

执行前按各阶段协议恢复依赖、配置资产路径，并使用新的输出目录保存结果。

P2 原独立评价执行 147,825 次成员/聚合调用，使用 116,097,715 tokens。E23 补齐 1,770 次成员生成和 4,263 次聚合，共 **6,033 次调用、4,486,492 tokens**，合计累计 120,584,207 tokens。E21/E22 复用已有缓存。这里统计的是多份冻结策略支持并集的评价成本，单策略部署成本另按实际执行路径计量。[E23 成本账目](routing_idea2/research/e23_deployment_prior_and_references/RESULTS_FINAL.json)

## 范围与后续整理

当前贡献围绕经验并列监督、共享参数和部署选择之间的关系展开。优势是端点干预、严格动作变化、参数几何、原系统独立评价以及部署规则对照相互补充。当前覆盖一套三成员聚合池，使用固定线性头、正则和有限生成重复；外部实验为单模型选择，E23 的训练行控制限于三份预定掩码。

接下来的论文整理重点是统一各阶段的实验身份、部署规则和成本口径，补齐复现依赖。跨系统质量提升、同等调参机会下的性能排序与有限反馈采集效率属于后续问题，见 [主张与证据](paper/CLAIMS_AND_EVIDENCE.md)。新汇总报告准确率、配对差值、纠错与破坏、任务/seed 分解和执行成本。
