# E18.1 执行账目与GlobalFixed训练来源复核

## 研究问题与范围

复核实际生成、有效评分及排除输出的数量，按物理执行身份去重Token，并追溯GlobalFixed组合5的历史选择依据。版本基点为 `95c6c8a38a71866d6efdb224aec04b9a158adf17`。

账目来自现有最终成本账本、执行scope、有效题mask及完整独立native审计。GlobalFixed复核使用原train600输入身份和三生成组合评分。复核过程新增LLM调用、训练、标准答案源读取、sealed源读取及归档payload读取均为0。

## 实际执行与有效评分

| 项目 | 原题数或调用数 |
|---|---:|
| 请求原题 | 3000 |
| 有效原题 | 2998 |
| 不可评分原题 | 2 |
| 实际执行聚合 | 30921 |
| 有效聚合评分 | 30900 |
| 已执行的不可评分聚合输出 | 21 |
| 新成员生成 | 42569 |
| 历史付费成员缓存 | 4717 |
| 成员输出并集 | 47286 |
| 去重物理执行总数 | 78207 |

实际聚合key集合与scope的required_aggregate_keys相同。成本账本中的成员key集合覆盖required_member_keys，物理attempt identity逐项唯一。有效题mask排除的两个身份与原私有不可评分清单一致。

### 两题的策略并集和生成重复

| BBH子任务 | 题数 | 策略数 | 每题逻辑策略聚合 | 不同组合数 | 每seed实际聚合 | 三seed实际聚合 | 不同成员数 | 三seed成员生成 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| movie_recommendation | 1 | 4 | 12 | 4 | 4 | 12 | 6 | 18 |
| ruin_names | 1 | 4 | 12 | 3 | 3 | 9 | 6 | 18 |

三个成员生成seed为42、137、2024。两题四策略共需24次逻辑聚合，按真实组合并集复用后实际为21次，三个seed分别执行7次。两题的四套策略每套均具有6次逻辑聚合调用，策略之间共享的同组合输出只计算一次实际执行。逐题动作、原ID及对应key保留在私有诊断中。

| BBH子任务 | 已执行聚合Token | 已执行成员Token | 合计Token |
|---|---:|---:|---:|
| movie_recommendation | 11,196 | 7,446 | 18,642 |
| ruin_names | 6,362 | 6,257 | 12,619 |
| 合计 | 17,558 | 13,703 | 31,261 |

这31,261 Token属于完整请求队列的已发生推理费用。质量分析共同排除两题，费用账本保留其实际执行。

## Token分账

| 来源 | 物理调用 | 输入Token | 输出Token | 合计Token |
|---|---:|---:|---:|---:|
| 本轮新增实际推理 | 73,490 | 34,094,356 | 12,493,160 | 46,587,516 |
| 历史付费成员缓存 | 4,717 | 824,280 | 1,119,946 | 1,944,226 |
| 总计 | 78,207 | 34,918,636 | 13,613,106 | 48,531,742 |

| 按算子汇总 | 调用 | 输入Token | 输出Token | 合计Token |
|---|---:|---:|---:|---:|
| 成员生成 | 47,286 | 7,999,752 | 11,245,367 | 19,245,119 |
| 组合聚合 | 30,921 | 26,918,884 | 2,367,739 | 29,286,623 |

新增推理的输入、输出Token与完整独立native审计的实际值一致。历史缓存按既有物理输出实际Token入账，共用成员及同一聚合的复用以physical_attempt_id去重。

标准答案账户以canonical question与reference version组合去重，最终有3000个唯一账户。评分恢复涉及跨进程重复读取和CPU计算，其运行记录分别保留，唯一账户计数与实际读取次数分列解释。上述48,531,742为模型推理Token，CPU扫描和确定性评分的运行时费用由相应性能收据提供。

## GlobalFixed组合5的合法历史来源

### 代码与标签单位

`freeze_confirmation.py`的 `_training_targets` 读取历史train600的三个生成评分，按原题身份、canonical hash和seed顺序校验，返回600×20的三重复平均奖励。`_fixed_actions` 对这600行再求组合均值，并用冻结的语义动作顺序选择最高分。GlobalFixed的输入为这一常数动作。

输入来源有GSM8K和BBH各300个独立原题，标签张量为3×600×20，seed依次为42、137、2024。均值的分母为600道题，每题三重复等权，等价于1800条已记录生成事件的平均。

### 独立重算结果

| 历史指标 | 结果 |
|---|---:|
| 全局固定动作 | 5 |
| 对应成员索引 | [0, 2, 4] |
| 组合5正确生成事件 | 1555 / 1800 |
| 组合5历史平均正确率 | 86.388889% |
| 次名动作 | 15 |
| 次名正确生成事件 | 1552 / 1800 |
| 次名历史平均正确率 | 86.222222% |
| 组合5相对次名点差 | +0.166667个百分点 |
| 精确最高分组合数量 | 1 |

组合5为唯一经验最高分，当前选择结果无需通过真实奖励并列规则打破平局。20个历史组合的正确次数及均值均保存在本目录的聚合JSON。

动作编号沿用0起始的原20组合顺序。动作5对应成员索引[0, 2, 4]。

### 冻结与继承核对

1. E17输入及奖励NPZ的实际SHA与INPUT_AND_CACHE_MANIFEST绑定一致，训练ID和canonical hash逐项对齐。
2. E17配置、FULL600_FIT_LOG和CONFIRMATION_PREPARATION_FROZEN互相绑定。配置记录的四份源码SHA与现有源码一致。
3. FULL600_FIT_LOG记录global_action为5，与本次历史评分均值重算一致。
4. E17封存fresh400动作中的GlobalFixed400项均为5，E18冻结3000项均为5。E18前400题身份与E17输入侧动作身份相同。
5. E18最终execution scope的四策略动作与FROZEN3000_INPUT_PREDICTIONS逐项一致。
6. E18的frozen_input_budget程序从E17 FULL600_FIT_LOG读取固定动作，并由原fresh动作中的常数5扩展新2600题。

train600与E18全部3000题的question ID交集及canonical hash交集均为0。

这些绑定支持GlobalFixed5由合法历史train600评分确定。复核选择动作时使用的奖励张量来自train600。本次计算读取E18输入侧动作及执行元数据完成继承验证。E17冻结收据记录fresh_rewards_read为0、sealed_payloads_read为0、independent_evaluation_executed为false。

## 来源摘要

| 来源 | SHA256 |
|---|---|
| `completed_native_audit` | `8367d50ef3138c4c0d7327fe18ebc883a93b709d28bdfcdcb59f6c19ef8f8655` |
| `final_result` | `86a91632c460085db016bf7718b56cd58fc40808f39a487d898c0dce080db41f` |
| `train600_reward_npz` | `a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e` |
| `train600_input_npz` | `57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3` |
| `e17_full600_fit_log` | `0002e2d47f44ac05edc4e8a3a7bf50a5d0b83722d7912f9f714770e0350ee4c5` |
| `e17_input_cache_manifest` | `aafc48e54d140f75f06d1c413fa8a083d3ab3997b375f840dcb4d62b8a56d45a` |
| `e17_confirmation_frozen_receipt` | `9fd825ed752247286381f90dd70ed33696b6cecc865aa591759f808b7f5d889d` |
| `e17_prefit_config` | `0bbd6ba39b973213bf01ee37f1ff24ef32a5809bf15fff5e56c7a7e767c535dc` |
| `e17_fresh_frozen_actions` | `1e4253a0ec9e2add6a7ec83d6ca369fa20bb23240c7731bd14754eeece305ad3` |
| `e18_frozen_input_actions` | `9f87499782a8417ab1bbf7c5ef7c409ca4bf10ece6f6cf35a51e9212b97f9aee` |
| `e18_execution_scope_expected_from_manifest` | `cae646111612ceb26196d73ed4469165e77b163a332f4db81deeb6f66551f73e` |
| `e18_frozen_input_budget_source` | `00bb8a1b803e1dce63c6f9eae1b34756a55794302a20b5c5b27ced0e3ddcd479` |

### 绑定的E17源码

| 文件 | SHA256 |
|---|---|
| `freeze_confirmation.py` | `e7449538be5b7786310ee8e6307fa2f983dc96d1c49e46967b503cbd1b9d2618` |
| `confirmation.py` | `9c92dec8be59f984447da2dcf4d6491dc8b5817a2df1133dcad723f26ef58ae2` |
| `objectives.py` | `8cdabb9ce256db8ed23811de05bd53274484795704f40bc78bfd86f8be2e79a6` |
| `e17_common.py` | `a78552b426027c00753c77edc8f7dd3a974df7d61ac1319aad819acf81fedc26` |

`e18_execution_scope_expected_from_manifest`表示本次复核所用既有manifest的封存SHA引用。完整native来源使用8367审计的既有结论。本次通过账本中的执行key集合、调用数量及实际Token进行核对。

## 优势与风险

- 优势是模型执行费用、有效评分数量及历史缓存费用能够分别核验。被排除的21条聚合仍可在同一scope和物理账本中追溯。
- 优势是GlobalFixed从train600标签到固定动作、E17 fresh动作、E18输入侧动作及实际scope具有连续绑定，重算能够恢复组合5。
- 风险是历史86.388889%属于用于选择组合的开发训练统计，生成噪声和数据分布变化影响其未来表现。
- 风险是成本字段提供Token及物理执行数，CPU重复评分、文件扫描、空闲时间与货币价格需要相应独立计量。
- 风险是时间顺序证据来自既有冻结收据与版本链，其保证受记录完整性约束。当前核对恢复了该链的一致性。

## 结论

账目数量与Token核对通过。实际30921次聚合对应30900次有效评分和21条不可评分输出，全部执行费用为48,531,742 Token。GlobalFixed5的600题历史来源和进入E18的常数动作继承均通过复核。

本文件为只读账目及来源核验，未执行模型拟合或新增推理。E18.1区间与统计重采样由本阶段另行授权的统计模块处理。
