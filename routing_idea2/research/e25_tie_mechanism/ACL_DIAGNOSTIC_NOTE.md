# 可迁移的路由诊断

## 什么时候值得查

在有限生成重复下若有较多训练题的所有候选奖励相同，先检查训练目标在这些题上的具体项。TSPO+ 给全并列题施加相对分数惩罚 `2(max q−mean q)`，其尖点可把一部分动作维持在数值接近的分数上。E24 中训练全并列行的占比为 51%。阈值、训练数据和损失会影响触发频率。

| 可观测信号 | 检查动作 | 能判断什么 |
|---|---|---|
| 第一名与第二名常很接近 | 同报原始分差、除以同题分数跨度的分差、不同容差下的唯一赢家数 | 量化边界分离程度与部署容差敏感性 |
| 很多输入共享一大组近最高分动作 | 统计重复支持集合、候选共现、组内分差和相对最佳簇外动作的间隔 | 找出特定候选合并与整个分数向量变平的区别 |
| 怀疑线性参数合并 | 比较同一候选簇内权重列距离，用观测设计范数乘以最大列距 | 判断已观测输入范围内的分差是否由参数对比限制 |
| 需要解释准确率变化 | 冻结分数和支持集，比较至少两种支持内部署，拆出两支均唯一的赢家变化和题级配对质量 | 衡量部署的决定与明确选队各贡献多少 |

规则以相对量、候选身份和同分数干预为主。对具体系统可保留原预设容差，再作输入侧敏感性对照。若真实输出的缓存覆盖不全，先列共同支持与缺失账本，并用配对系数计算可识别范围。

## 为什么这个系统值得分析

SPO+ 在这里把组合选择写入训练目标，因而为观察对称并列监督怎样进入共享参数提供了可检查的案例。E23 中 Ridge 的历史准确率高于受控的 SPO+ 变体，这个性能排序与论文的训练目标、预测排名及部署诊断各自回答不同问题。ARR 指南把科学问题的价值与模型排名区分开，并要求证据覆盖实际提出的主张。[ARR 审稿指南](https://aclrollingreview.org/reviewerguidelines)

## 适合加入论文的英文段落

> A repeated near-maximal set gives a practical trigger for a routing audit. Log the top-two margin on both raw and within-question scales, tally repeated candidate supports, and compare deployment rules on the same scores. For a linear head, the norm of an action-coefficient contrast times the design-vector norm bounds the corresponding score gap. In E24, all 1,033 near-tied inputs under B share the same 17-action support. Its within-set score gap is bounded by 6.03×10⁻¹⁰ over the observed design norms, while the set remains separated from the other actions. This pattern persists across three locked regularization scales. In a small nonlinear head, adding the tied-example penalty reduces the median margin across all three optimization seeds. The number of inputs crossing the deployed tolerance varies with initialization. Together these observations motivate reporting support identities and continuous margins alongside accuracy and tolerance-specific tie counts.

这段补充结果基于 E24 已曝光题池。原独立复验的主检验及判分文件保留原版本。
