# E9 Bayes gate 输入与协议核查

## 范围与身份

本轮独立执行用户给出的 `MAB_E9_Codex_Experiment_Prompt.md`。现有 `e9_consistent_cost_sensitive_learning` 对应经验/后验均值与CSCE矩阵，方法和主比较不同。本轮使用独立的 `experiments/e9_bayes_gate` 与 `results/e9_bayes_gate`，旧研究资产保持只读。

已实际读取阶段汇报及同版本公式规范版。原始train600评分为3×600×20，转为题×生成×组合。评分、ID、canonical、MiniLM384向量、GSM/BBH各300与BBH27子任务映射一致。五折history480与evaluation120按原题和canonical互斥，历史包逐项与原矩阵一致，评价输入包没有奖励字段。

## 独立历史回放

| 冻结方法 | 准确率 | 最高频动作题数 |
|---|---:|---:|
| M0 original | 85.2778% | 55/600 |
| M4 exact SPO+ | 85.9444% | 467/600 |
| NS-SPO+ δ=0.05 | 86.0556% | 170/600 |

实际回放使用保存动作与真实最终聚合评分，不重新拟合、不生成。M0 original与tuned的600个冻结动作已逐项核对，结论见JSON。M0 tuned五外折均选择multiplier10，但其系数尺度与original仍不同。不得仅因动作相同宣称拟合程序相同。

## 合法预处理和内层配置

E8固定MiniLM-L6-v2 384维，合法拟合区PCA16(full)、StandardScaler及受罚常数，线性17×20共340参数。原正则λ=100×multiplier/(3×n×20)，候选multiplier为0.1、1、10，原M0固定raw_alpha100。内层三折由task/subtask分层，seed=20261113+outer；保存的FIT_LOG记录的是history480的局部行号。已核对每个内层训练/评价互斥、三评价分区覆盖history480。

## 生成与后验边界

原成员生成重复为42、137、2024，聚合seed固定42。同题每个生成块共享六成员输出，20组合之间存在相关性。seed编号本身不证明独立随机化。NS工作后验为有界联合截断Gaussian，噪声协方差20%对角收缩，历史区内部交叉拟合；512样本是诊断量，旧NS训练使用8样本。它不是精确Bernoulli后验，不自动提供真实潜在奖励区间校准或Bayes一致性。

## 来源及缺失事项

所有本轮依赖的原协议、源码、summary、公开manifest和私有冻结动作均记录真实路径、字节数及SHA256。所选历史公开项逐一与原manifest核验。专用sealed_test候选目录未定位不等于数据不存在，其协议来源仍保留；未搜索或读取其内容，fresh_dev400仅检查目录存在。validation200、fresh400和sealed200载荷读取次数均为0。

当前D工作区没有本地Git checkout。最近E12远端提交只引用已保存的核验收据，并非本脚本再次联网核验。分支创建留由主执行器处理，缺少本地Git不会改变实验数据或放宽评价权限。本轮不push、merge或重写历史。

## 核查状态

资产和历史主基准PASS后才允许本轮训练。数学资格、拟合和科学结论由各自真实结果给出，本文件不把输入核查PASS当作科学假设成立。若manifest列出关键MISSING项，状态为BLOCKED，应先补齐来源。
