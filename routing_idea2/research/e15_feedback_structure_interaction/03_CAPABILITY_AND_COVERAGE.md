# 能力误差与独立题覆盖

## 研究问题

观察更多动作与覆盖更多独立历史题，在同一学习器中分别产生什么变化。

## 实验设置与对照

每个Token上限定义N为三结构可覆盖题数的最小值，CommonCoverage使用同一个原始题目前缀。该控制匹配题目身份和数量，执行费用逐项单列。

能力误差仅用E120留出奖励。Group使用冻结C，SaveRouter单列原生组能力表，RawCentered使用同一留出组内问题得分的均值。后两者仍分别保留完整服务得分MSE。真实组只用于诊断，服务选队输入为原预测组。三生成均值是经验目标。

## 价值差分解

邻居或随机第二组合记为A，Main记为M。同一E题集上逐事件严格核验

`Afull-Mfull=(Acommon-Mcommon)+(Afull-Acommon)-(Mfull-Mcommon)`

主预算点差分解，单位为百分点。

| 学习器及结构 | 总差 | 同题覆盖额外组合效应 | 该结构新增覆盖集合效应 | Main新增覆盖集合效应 |
|---|---:|---:|---:|---:|
| RawCentered NeighborPair | +0.3333 | +0.4074 | -0.0926 | -0.0185 |
| RawCentered RandomPair | +0.4444 | +0.4259 | +0.0000 | -0.0185 |
| Group NeighborPair | +0.8519 | +0.5370 | -0.0556 | -0.3704 |
| Group RandomPair | +0.3519 | -0.0185 | +0.0000 | -0.3704 |
| SaveRouterFixedMask NeighborPair | +0.7222 | +0.5741 | +0.5556 | +0.4074 |
| SaveRouterFixedMask RandomPair | +0.5370 | +0.9444 | +0.0000 | +0.4074 |

## 能力估计误差

主预算BBH组–动作均值误差，组等权汇总。

| 控制、结构及学习器 | 留出均值MAE | 中心化能力MSE |
|---|---:|---:|
| CommonCoverage Main Group | 0.22769 | 0.02453 |
| CommonCoverage Main RawCentered | 0.22308 | 0.02006 |
| CommonCoverage Main SaveRouterFixedMask | 0.22386 | 0.01974 |
| CommonCoverage NeighborPair Group | 0.22873 | 0.02696 |
| CommonCoverage NeighborPair RawCentered | 0.22300 | 0.02066 |
| CommonCoverage NeighborPair SaveRouterFixedMask | 0.22205 | 0.02006 |
| CommonCoverage RandomPair Group | 0.22905 | 0.02692 |
| CommonCoverage RandomPair RawCentered | 0.22299 | 0.02070 |
| CommonCoverage RandomPair SaveRouterFixedMask | 0.22222 | 0.02026 |
| TokenMatched Main Group | 0.23221 | 0.02769 |
| TokenMatched Main RawCentered | 0.22501 | 0.02077 |
| TokenMatched Main SaveRouterFixedMask | 0.22339 | 0.02031 |
| TokenMatched NeighborPair Group | 0.22827 | 0.02689 |
| TokenMatched NeighborPair RawCentered | 0.22269 | 0.02060 |
| TokenMatched NeighborPair SaveRouterFixedMask | 0.22190 | 0.02004 |
| TokenMatched RandomPair Group | 0.22905 | 0.02692 |
| TokenMatched RandomPair RawCentered | 0.22299 | 0.02070 |
| TokenMatched RandomPair SaveRouterFixedMask | 0.22222 | 0.02026 |

## 统计结论与风险

SUPPORT_AND_COVERAGE记录独立原题、标签数、组–动作支持、同题双动作分歧率。CAPABILITY_QUALITY_LINK使用同组、同E题的配对误差变化和选队变化，相关性只作描述。

新增覆盖集合效应包含具体新增题目的任务和难度构成。Group仅使用边际sum/count，结果不能作为已实现同题相关性建模的证据。低支持分组与任务混淆继续保留。CommonCoverage和TokenMatched的费用权限分别记录。

## 下一步

依据同题覆盖分解及完整Token曲线判断哪个环节值得独立确认。
