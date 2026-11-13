# E8文献、实现与20组合选队目标的对应

## 研究问题与阅读范围

本轮检验固定题目表示、20输出线性容量与历史评分下，训练目标是否影响未见题的选队质量。借用的pairwise、ListNet、SPO+及加权最小二乘均按已有方法标注。文献核查没有读取validation200、fresh400或sealed200奖励，没有进行物理LLM推理。

已获取五篇指定研究的完整原文及Liu–Grigas补充材料，共134个PDF页面，逐文件保存原始字节、页数和SHA256。重点精读模型定义、损失、梯度、优化、并列规则和理论条件，并对RankNet、ListNet、SPO+及PyEPO的关键公式页进行图片核对。原论文实证数值没有作为本任务已复现结果引用。来源与下载记录见`E8_LITERATURE_SOURCE_MANIFEST.json`。

## 固定文献与作者代码

| 文献 | 原文与核对位置 | 代码状态 |
|---|---|---|
| Burges等，2005，Learning to Rank using Gradient Descent，DOI 10.1145/1102351.1102363 | 微软作者原版8页，PDF第3页式1–3，第4–5页反向传播，第6页并列实验 | 通过作者出版页及微软资源的有界检索未定位可固定commit的作者公开源码，本轮按原文独立实现 |
| Cao等，2007，Learning to Rank From Pairwise Approach to Listwise Approach，DOI 10.1145/1273496.1273513 | 微软完整技术报告MSR-TR-2007-40共9页，第4页Top-one probability，第5页式2–3及Algorithm 1 | 同样未定位可核实为原作者发布的源码，不将第三方同名仓库标作作者实现 |
| Elmachtoub与Grigas，2022，Smart Predict, then Optimize，DOI 10.1287/mnsc.2020.3922 | 完整作者预印本1710.08005v5共46页，第17页Definition 3，第18页Proposition 3，第21页Assumption 1，第23–24页正则化优化 | `paulgrigas/SmartPredictThenOptimize`，MIT，固定commit `4002e9fe94a539c95d1e7520afd4f84d3063adf8` |
| Liu与Grigas，NeurIPS 2021，Risk Bounds and Calibration for a Smart Predict-then-Optimize Method | 最终论文12页及补充18页，主文第4–6页一致性及多面体校准，第6–8页强凸可行域，附录A–D | `HeyuanLiu/SPO_plus_public`，MIT，固定commit `60f5ea2896ae315e541e220fb8267a7dd485e80c` |
| Tang与Khalil，2024，PyEPO，DOI 10.1007/s12532-024-00255-x | 完整作者预印本2206.14234v3共41页，该修订版日期为2026-04-17，第9–12页训练与SPO+式7–8 | `khalil-research/PyEPO`，MIT，固定commit `e5d768e2749bccdeba7494b4ec848c3eb3ffbd25` |

文献年份采用正式出版年份，下载的预印本版本单独记录。SPO作者仓库README直接建议新任务使用PyEPO，并提供Julia原始损失与SGD实现。Liu–Grigas作者仓库README明确对应NeurIPS 2021论文。三个仓库均仅保留私有原始快照，公开交付为独立核验代码、公式映射与哈希记录。[RankNet原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf) [ListNet原文](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf) [SPO作者仓库](https://github.com/paulgrigas/SmartPredictThenOptimize) [Liu–Grigas作者仓库](https://github.com/HeyuanLiu/SPO_plus_public) [PyEPO作者仓库](https://github.com/khalil-research/PyEPO)

四条DOI均通过Crossref官方精确端点核验题名、作者及出版记录。ListNet的Crossref主标题简写为Learning to rank，作者技术报告提供完整副标题。核查结果见`E8_BIBLIOGRAPHY_IDENTITY_CHECK.json`。

## 决策对象的精确映射

令K=20表示可选三模型组合的数量，r为一道题20个组合的三生成重复平均真实聚合奖励，q=φW为预测分数。部署动作是`argmax q`，每次执行所选组合的三个成员和冻结聚合器。每道题的三个生成重复始终属于同一个问题簇。

SPO原文的成本最小化对应到本任务时，可令`c=-r`、`c_hat=-q`，决策集合取20个one-hot动作的凸包。线性目标在该凸包与20个离散动作上具有相同最优值。这项表示保留了组合的真实聚合价值。将决策直接改成六成员的可加奖励，会改变训练目标与已冻结评分的含义。

| 原方法对象 | E8对应对象 | 保留的结构 | 需要实测的变化 |
|---|---|---|---|
| 检索query | 原始独立问题 | 同一问题内比较候选 | 组合效用来自真实聚合评分 |
| 候选文档 | 20个三模型组合 | 对一份输入输出候选分数 | 每个候选具有单独线性输出头 |
| relevance/cost | 三重复平均奖励或其负值 | 同题完整标签向量 | 离散、相关并有大量可能并列 |
| 预测排序/优化oracle | 最高分组合 | 输入驱动的预先选队 | 不能使用当前题真实评分选队 |
| 样本平均风险 | 原问题平均训练目标 | 每题一个外层权重 | 190对与三个重复增加题内信息量 |

## M1的RankNet对应与适配

原RankNet令`P_ij=sigmoid(q_i-q_j)`，以pair preference概率的交叉熵训练。对严格偏好，其损失化为`softplus[-sign(r_i-r_j)(q_i-q_j)]`。原论文还允许并列偏好目标0.5。E8指定的奖励差绝对值权重、零差剔除及20输出线性头属于本任务适配。

冻结实现使用

\[
L_1(W)=\frac1n\sum_x\frac1{190}\sum_{i<j}
|r_i-r_j|\log\{1+\exp[-\operatorname{sign}(r_i-r_j)(q_i-q_j)]\}
+\lambda\|W\|_F^2.
\]

零差对贡献0，分母始终为190。外层按原问题平均，内层较大的真实差值仍具有较大训练影响。该约定没有把每题非零权重重新缩放为总和1。导数为每对`-|gap|*sign(gap)*sigmoid[-sign(gap)*score_gap]/190`，分别加到两个分数的相反方向。原文的pairwise概率描述相对偏好，q本身具有自由共同偏移和尺度，其raw MSE应按E8合同单列。[RankNet第3–4节](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)

## M2的ListNet对应与适配

原ListNet先构造全排列概率，再利用Top-one probability避免枚举K!种排列。指数映射下，真实relevance和预测分数各自经过softmax，以两分布交叉熵训练。原技术报告的实证本身采用线性模型，目标按query定义。

E8采用

\[
t=\operatorname{softmax}(r/\tau_y),\quad
p=\operatorname{softmax}(q/\tau_p),\quad
L_2(W)=-\frac1n\sum_x\sum_jt_j\log p_j+\lambda\|W\|_F^2.
\]

对q的梯度为`(p-t)/tau_p`，再按题平均并加正则。两温度均为1时对应原Top-one指数形式。原始奖励落在[0,1]，温度控制目标分布的集中程度，属于需要训练区内部固定的小规模参数选择。这里的softmax目标不是三模型组合答对概率的观测值。[ListNet第4–5节](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)

## M3的差值平方目标与归一化

对任意误差向量e以及中心化矩阵`C=I-11^T/K`，直接展开得到

\[
\sum_{i<j}(e_i-e_j)^2
=K\sum_i e_i^2-(\sum_i e_i)^2
=K\|Ce\|_2^2.
\]

因此E8的等权`1/K²`差值平方目标，恰等于按K个分量平均的中心化MSE。若题权重、特征、常数项惩罚和λ一致，alpha=0的差值Ridge具有原MSE Ridge的同一中心化解和排序。共同得分方向与相对得分方向在Frobenius正则下正交分离。A2的排序一致性是后续竞争加权解释的基础。

M3以训练区OOF M0预测定义Top-3竞争集合，再将相关pair权重增大，每题190个权重均值归一为1。该机制保留线性容量并改变训练误差的重点。OOF竞争集合的生成必须遵循内外层训练边界。加权最小二乘本身属于已有数学工具，本报告只建立E8目标与未加权基线的代数关系。

## M4的奖励最大化SPO+、梯度与并列

原SPO+成本形式为

\[
\ell_+(\hat c,c)=\max_w(c-2\hat c)^Tw+2\hat c^Tw^*(c)-c^Tw^*(c).
\]

逐项代入`c=-r`、`c_hat=-q`得到E8精确形式

\[
\boxed{\ell_+(q,r)=\max_j(2q_j-r_j)-2q^Ta^*(r)+r_{\max}},
\qquad
\boxed{g_q=2\{a^*(2q-r)-a^*(r)\}}.
\]

其中`a*`是选定最优组合的one-hot向量。原Julia实现的`solver/util.jl`第160–177行计算按题平均损失，`solver/sgd.jl`第109–117行计算相应次梯度。PyEPO的`pkg/pyepo/func/surrogate.py`第124–141行显式区分MIN与MAX，其MAX分支与上述公式一致。数学最大项在20动作上可以直接枚举。[SPO定义与性质](https://arxiv.org/pdf/1710.08005v5) [PyEPO固定源码](https://github.com/khalil-research/PyEPO/blob/e5d768e2749bccdeba7494b4ec848c3eb3ffbd25/pkg/pyepo/func/surrogate.py)

令a为预测分数的最高动作，b为任一真实最优动作，则

\[
\ell_+(q,r)\ge 2q_a-r_a-2q_b+r_b
=r_{\max}-r_a+2(q_a-q_b)\ge r_{\max}-r_a\ge0.
\]

这给出对当前经验奖励选队损失的凸上界。真实最优动作的任意固定凸组合仍使上式成立。E8主实验对真实最优并列采用固定组合顺序的字典序，真实最优集合均匀平均另列敏感变体。扰动oracle也采用字典序。并列处可出现多个有效次梯度，应核对次梯度不等式，普通有限差分需要避开折点。

若某题20个奖励全相等，主字典序损失为`2(max q-q_first)`，平均真实并列变体为`2(max q-mean q)`。这类题的经验服务决策损失始终为0，却仍可影响代理目标和梯度。其频率与损失贡献需要单列诊断。主字典序可能诱导索引偏好，平均并列变体则鼓励该题全部分数相等。两种目标均属于预定对照，不能在外折查看结果后替换主版本。

## 原作者算子执行核查

使用PyEPO固定源文件中未改写的`SPOPlusFunc`，仅将优化oracle替换为精确20动作枚举器，在24道完全合成离散奖励问题上分别检查字典序和平均真实并列变体。

| 核查量 | 最大绝对误差 |
|---|---:|
| 作者forward与显式奖励式 | 2.22e-16 |
| 作者backward与显式次梯度 | 0 |
| 共同得分偏移不变性 | 8.88e-16 |

两变体均非负，均覆盖其当前经验选队损失。源文件SHA256为`23713e82c845928e6d619a0d5a2dff019c9533552038549ea4ec5983ce10914a`，完整结果见`E8_AUTHOR_SPO_OPERATOR_CHECK.json`。这项软件证据核验公式迁移，E8真实表现由固定外折实验评价。

## 正则化尺度与精确优化

若原Ridge目标为三重复全部记录的平方误差之和加`alpha||W||²`，用三重复均值后，匹配的按题、按20输出平均MSE正则为

\[
\lambda_0=\frac{\alpha}{3nK},\qquad K=20.
\]

E8所有W分量包含原常数项，按现有实现共同惩罚。原SPO作者使用`mean_question(loss)+(lambda_author/2)||W||²`，其梯度为`loss_grad+lambda_author*W`。E8写成`mean_question(loss)+lambda||W||²`时，两者参数关系是`lambda_author=2lambda`。ListNet、RankNet与SPO+的损失单位不同，相同数值λ不能解释为相同统计收缩强度，应明确固定候选网格、归一化和合法内部选择。[SPO第5节与作者SGD实现](https://github.com/paulgrigas/SmartPredictThenOptimize/blob/4002e9fe94a539c95d1e7520afd4f84d3063adf8/solver/sgd.jl)

固定特征下，SPO+可用每题一个epigraph变量表示，约束为`t_x >= 2 phi_x W_j-r_xj`，目标为题均值`mean(t-2q^Ta_true+rmax)+lambda||W||²`。该形式是有限动作凸QP的精确重写，求解器残差和目标一致性仍需核验。平滑最大值会改变训练目标，必须另行标注。

## Liu–Grigas风险界的适用范围

原SPO Fisher一致性要求条件成本分布连续、关于均值中心对称，并对原定理的唯一性结论加入非空内部及均值最优解唯一等条件。Liu–Grigas的多面体统一校准进一步限制分布族，包括相对参考正态密度的下界；较强风险界还要求不同的可行域几何与分布条件。其附录明确使用独立问题的泛化分析。

E8的三个二元生成奖励取均值后仍为离散标签，其条件分布对称性尚未建立。20维one-hot凸包满足`1^T w=1`，普通20维内部为空，原多面体宽度常数在全1方向为0。可在19维相对价值空间表述决策几何，但离散分布的条件依然需要单独证明。因此原论文的校准和样本复杂度率不能直接作为E8泛化收益保证。凸上界与可计算次梯度的代数性质仍适用于当前有限动作目标。[Liu–Grigas主文](https://papers.nips.cc/paper_files/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Paper.pdf) [完整补充证明](https://papers.nips.cc/paper_files/paper/2021/file/b943325cc7b7422d2871b345bf9b067f-Supplemental.pdf)

## 优势、风险与本轮可支持结论

本轮优势是使用同一表示、容量和问题簇，直接比较相对预测与决策损失，且SPO+在20动作上可精确求解。主要风险包括目标尺度、有限样本模型失配、经验奖励并列及跨生成随机性。训练目标变好只能支持优化成功，未见题准确率及配对统计决定是否支持选队改进。

文献来源、奖励符号、作者实现以及正则换算的核对置信度高。RankNet与ListNet的原作者公开代码可得性尚未建立，本轮以论文公式的独立实现标注。原方法没有承诺在该真实组合数据上超过Ridge，E8也不能仅根据使用已有损失宣称原创路由算法。
