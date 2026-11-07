# D阶段统计复核

## 审查范围

对`summarize_decision.py`、`run_decision_audit.py`、`e4_metrics.py`实施只读审查，并从165条live轨迹、30个snapshot文件独立重建汇总。只在D工作区写本复核及审计费用结果。原始记录、已冻结模型代码、E盘公开产物均保持输入身份。

## 结果与资格

**未发现阻塞当前D开发诊断的数值汇总缺陷。** 实测状态为`D_STATISTICS_READONLY_QUALIFICATION_PASS`。该状态覆盖记录与统计一致性，科学判断依照以下边界。

| 检查 | 实际结果 |
|---|---|
| live身份完整性 | 165条，11方法上下文，每上下文5fold×3生成seed |
| 每题生成记录 | 每上下文600原题，每原题3条不同seed，无重复或缺失 |
| 多种子依赖 | 先按q平均，再以600原题计算配对区间 |
| 公开准确率 | 独立重建所有权重口径，逐值一致，容差1e-12 |
| 纠错/破坏 | expected correction−harm等于600原题奖励差总和 |
| NoFeedback | 同模型所有方法共享完全相同的原题初始基线 |
| live资源曲线 | 每路径120题，共同horizon，末值与正式账本一致，资源轴单调 |
| snapshot身份 | 30文件，共1200候选，每模型600 |
| 来源块 | 每模型20原始Support题块，每块3生成seed×10候选 |
| Support/Query | 每个snapshot来源题与对应100题audit Query严格分离 |
| snapshot gold | 每独立反事实揭示1题gold，main+neighbor可合法评分2个输出 |
| 邻居复用 | main三成员，one-replacement neighbor新增一成员与一次聚合 |
| 校正恒等式 | expected correction−harm等于100×balanced_v2实际增益 |
| 反事实费用 | 独立世界分别收费，新增物理LLM及gold获取均为0 |

## 统计解释与限制

1. live的paired bootstrap在GSM及各BBH子任务内按原题重采，额外BBH任务块敏感性保持语义任务依赖。历史拟合、在线学习轨迹、候选选择和已实现生成缓存均固定。顺序学习的路径变动未被重新模拟。区间用于条件开发描述。
2. D仅使用order17。5fold×3seed在每个原题处只形成三次重复，order稳定性尚未在D补充。此前E3多order结果可作为历史背景，不能增加D的独立重复数。
3. snapshot按`(fold,q)`聚类，完整保留同来源题的10候选与3生成重复。20块共享固定Query及先前logger反馈历史，source-block bootstrap不覆盖这些共同路径的不确定性。候选Spearman相关以反事实候选为描述对象。
4. `actual_gain_by_support_generation_seed`已经明确表示Support反馈的生成seed。每候选的Query奖励仍在三生成seed内平均。这一分组可检查Support噪声方向，但当前没有提供仅排除Support同seed的Query结果。
5. top-decile阈值保留边界全部并列候选，实际纳入数随ties变化。代码给出纳入数，报告同时给出分母600。量化分箱保持相等值不拆分。常值proxy会使Spearman未定义，分箱可为空，需保留这种支持限制。
6. 单独相关性CI使用固定seed顺序产生各自bootstrap draws，当前没有对“相关性之差”构造共同重采样CI。各自区间不能直接充当两代理排序能力差的检验。
7. 描述性p来自中心化bootstrap尾部，百分位CI使用原bootstrap分位点，两者并非互逆构造。离散小净纠错中可出现percentile CI单边而p大于.05。报告保留两者，并统一避免确认性显著主张。15个D开发策略比较的Holm最小值为0.263947。
8. 同gold20仍可产生不同Token及完整服务费用。同实际花费位置的曲线前缀可能含不同问题数和题目集合，当前曲线提供描述性质量—成本关系，题级等成本因果CI未构造。
9. snapshot public摘要保留收费说明，新增`D_PROXY_AUDIT_COSTS.json`从私有账本补齐service/audit/total。1200独立反事实的相加费用是审计累计会计量，演员策略每路径仍为20gold。

## 算子拆分复核

Gaussian E3的`Decision-Aware`使用原linear Gaussian KG，`Probability-DA`改为同Gaussian更新下枚举Bernoulli观测与概率效用，`Calibrated-DA`再加history-OOF Platt变换。Binary E4的DA已直接使用Bernoulli枚举。Platt仅进入采集效用及观测权重，服务头、反馈更新及固定delta梯度仍按各自actor执行。

额外描述性配对计算给出E3 Probability−linear DA为+0.2222个百分点，95%原题CI为[+0.0000, +0.5000]，p为0.124175。该额外拆分未加入既有15项Holm家族。Gaussian校准相对Probability的差值另有正式比较，后续报告独立列出。

## 优势与风险

- **优势**为身份、预算、Support/Query权限、题级统计和收费账本可独立重建，并保留正负结果。
- **风险**为已用于模型选择的开发600题、单一order、20来源块、稀疏非零增益和共享未来Query限制独立推断。
- 数字一致性置信度为高，依据完整记录复核。新题环境中采集机制泛化收益置信度为低，依据当前开发复用及广泛跨零相关性区间。

## 输入身份

- `DECISION_LIVE_RESULTS.json` 的 SHA256 为 `cf7d9060094dd08d41f4869ae4f26457d663c5d2b334d1f3c25443003a33edb6`。
- `DECISION_QUALITY_COST_CURVES.json` 的 SHA256 为 `febeb6afd17180a77442d7974f3e79d12a4e065bd3ab4c48bcd88e99b4904c06`。
- `DECISION_PROXY_CALIBRATION.json` 的 SHA256 为 `ee5f829d9526271e42f20f43c0e2fad89e238d426abb4175a454a96391bcf505`。
- `BINARY_COMPARISON_RESULTS.json` 的 SHA256 为 `99e00d582bb58894c87f37204dcf3ff17d5af3ddc0c4adeb988f13898bdf7cf2`。
