# E13 一手文献与 SaveRouter 适配边界

## 来源核查

2026-11-14 实际检索并阅读作者原文 v1 的 §4.2、§4.3、附录 C.2，以及官方固定 commit 的核心源码。论文中的共享能力与逐题残差已经构成已有方法，本轮比较其在真实组合奖励上的迁移表现。[论文指定版本](https://arxiv.org/html/2609.37402v1)

官方源码固定为 `250b7ef35a741c598aaf1bb264930d1e0bfd1a61`。本轮重新通过作者 GitHub API 获取四份文件，实际字节 SHA256 与既有缓存逐项一致。

| 文件 | SHA256 |
|---|---|
| router.py | d3f97af559021f1e8b991c4f4bfe879bcbed889f4cd10e9c2a8edda531ce92b2 |
| data.py | 5c55ace20cda0deb48155b00ddf4cc8d7032178113c18521573621c86789d70c |
| grouping.py | 1af306fb87ad3ca6856dbc514dacc104a790e10c364d42b46c20a6900057ec69 |
| profiles.py | 319d97271cf05fa7853e42ef3591127b9b4566d441f84a85e396560923c8109f |

## 原生能力估计器

令当前合法已观察元组集合为 \(\Omega\)，动作 \(a\) 在本项目中表示一个三成员组合。作者实现首先拟合带截距的加性 Ridge。

\[
(\hat b,\hat u,\hat v)=\arg\min_{b,u,v}
\sum_{(i,a)\in\Omega}(r_{ia}-b-u_{g_i}-v_a)^2
+10(\|u\|^2+\|v\|^2),\quad
\pi_{ga}=\operatorname{clip}(\hat b+\hat u_g+\hat v_a,0,1).
\]

随后以当前稀疏反馈估计组能力。

\[
C_{ga}=\frac{S_{ga}+\tau\pi_{ga}}{n_{ga}+\tau}.
\]

残差标签为 \(r_{ia}-(S_{g_i a}-r_{ia}+\tau\pi_{g_i a})/(n_{g_i a}-1+\tau)\)。加性先验仍由全部当前已观察元组拟合，因此该操作移除局部统计中的直接贡献。每动作至少8条观察时拟合带截距的LSQR Ridge残差，最终预测为 \(C_{g(x),a}+\gamma h_a(x)\)，最终得分保留原始数值范围。实际运行应直接调用作者 `FixedKSparseRouter`。[固定源码](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/router.py)

## 冻结 profile

本轮锁定作者 `llmrouterbench` profile。其参数为 \(\tau=40\)、加性先验 Ridge=10、残差 Ridge=200、\(\gamma=2\)、最低残差观察数8。该选择依据文本路由 profile 的可追溯性，未使用E13评价成绩。反馈动作数由共享logger决定，必须与每次回答的三成员数分别记录。[作者 profiles.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/profiles.py)

## 稀疏接口与生成重复

原生 `SparseSupervision` 仅包含已观察的query/action/reward/cost元组。它要求每个传入问题恰好有 \(k\) 个不同动作，同一query/action至多出现一次。每个生成seed独立形成轨迹，在对应已揭示问题子集内重新编号，可以直接保留这一数据类型。若反馈问题获得的动作数不同，需要另行明确接口适配，不能静默冒用原生fixed-k数据对象。[作者 data.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/data.py)

## 共同信息权限

作者原生分组器使用训练标签拟合逻辑分类器，并以预测组处理新题。auto模式还可以回退到输入聚类。E13向各组方法提供同一训练组和同一输入预测映射，明确记录与作者完整分组pipeline的差异。[作者 grouping.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/grouping.py)

共同映射固定为 \(\hat g(x)=\arg\max_g\phi(x)T_M\)，其中 \(T_M\) 来自E12合法H480文本与训练组metadata。主Group、Group+Residual、SaveRouter及固定策略共用此映射，评价真实子任务标签只进入另列oracle诊断。Group在本轮按hard组能力查表预测，输入传播和三重复均值监督不进入该部分反馈定义。

本轮适配对象是SaveRouter能力学习部分。反馈由共同掩码提供，20个动作使用真实三模型最终聚合评分。MiniLM-PCA16为共同输入。纯质量主比较使用预测质量argmax，成本按已执行组合与成员并集另行记账。这些适配需要随实际协议和运行收据一起发布。

作者 `costs` 字段用于估计未来单动作服务费，应对应该动作的完整三成员生成和聚合费用。共享成员后的增量反馈执行账应单列。同题gold复用、成员生成、组合聚合与实际token属于本项目必须保留的费用维度。[作者稀疏接口与路由实现](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/router.py)

## 优势、风险与结论权限

**优势**是能够原样运行作者能力估计类，并在相同反馈单元上比较监督利用效率。动态导入仅依赖NumPy、SciPy、scikit-learn及四份已核对源码，无需语言模型推理。

**风险**是原论文encoder、采集器和服务前沿与本项目的共同权限不同。profile未经本项目调参，低预算动作可能不足8条观察，残差自然关闭。三次生成是题内重复，原seed编号本身没有证明独立随机化。

本轮实际结果由主报告给出。将20个组合视为20个路由动作、组共享和残差拟合各自具有已有方法基础。后续创新判断仍需真实部分反馈和联合执行成本证据。

## 必要软件检查

4项合成检查实际通过。原作者类与适配器在相同稀疏观察上给出逐元素完全相同的分数及成本预测，保存数组恢复分数满足 \(10^{-12}\) 容差。其余检查覆盖未观察训练行不进入残差拟合、拒绝稠密未揭示奖励、拒绝变长fixed-k及重复query/action生成混入。检查只使用人工数组，新增物理模型调用和真实评分读取均为0，CPU为1.140625秒、wall为2.154013秒。具体日志保存在私有qualification。
