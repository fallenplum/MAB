# E12 文献核查与方法定位

## 核查范围

核查日期为 2026-11-14。本记录完整阅读 E12 指定任务附件，并实际读取指定版本的一手原文。SaveRouter 精读 §4、附录 A.1–A.2、C.1–C.2，逐项对照官方核心源码。Menon 精读 §3.1、§4.1–4.2，并检查 PDF 第 3、5、6 页渲染。UniRoute 精读 §5.1–5.3，Avengers 精读 §3 与附录 B.6。E11 私有来源记录用于定位，以下事实均经本次原文复核。

SaveRouter 原文与官方仓库实际可访问。直接访问 Google Search 页面失败，检索由可用联网搜索工具完成，再回到一手来源确认。完整原文下载和源码读取已完成，核查范围为方法与信息权限，论文基准数值复现仍待单独执行。

## SaveRouter 的直接相关内容

论文固定为 Lai et al., *Routing Should Pay for Itself: Sparse Supervision for Economical LLM Routing*, arXiv:2609.37402v1，2026-09-29。§4 联合研究分组、稀疏反馈获取、能力估计与逐题修正。附录 C 明确训练输入及训练 task 标签的权限，测试分组来自输入预测。[指定原文](https://arxiv.org/html/2609.37402v1#S4)

本次官方源码固定到 commit `250b7ef35a741c598aaf1bb264930d1e0bfd1a61`。以下为源码与原文的逐项核对。

| 核对项 | 已确认实现及边界 |
| --- | --- |
| 训练 task 与聚类 | `GroupAssigner` 用训练 task 标签拟合逻辑分类器。auto 模式要求类别数不超过 128、训练分类准确率至少 0.8，否则采用冻结输入表示上的 MiniBatchKMeans。主 pipeline 的测试组由该映射预测。训练分组输入包含文本及训练 metadata。[grouping.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/grouping.py#L44) [pipeline.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/pipeline.py#L104) |
| 稀疏 query–model 反馈 | 每题获取 K 个不同模型的反馈。每轮重置获取统计量并打乱题序，跨轮保留已选掩码。先确定动作，再调用 `reveal`。未观测动作的可用性可见，质量与成本在选中后揭示。[supervision.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/supervision.py#L16) |
| 组先验及收缩 | 在已观测元组上拟合带正则的 `b + u_g + v_m` 加性先验，再以观测数加权融合组–模型经验均值与先验。缺少局部观测的单元依赖先验。[router.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/router.py#L66) |
| 逐题残差 | 每个模型拟合上下文 Ridge 残差。目标从局部组统计量扣除当前观测，但加性先验仍由完整已揭示集合估计，应按“移除局部直接贡献”理解。最终预测为组能力加缩放残差。[router.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/router.py#L116) |
| 学习入口与成本权限 | `fit(X, supervision)` 的监督来自已选稀疏元组。质量和成本估计共用该集合，组成本缺失时回退到已观测的模型级成本。离线适配器管理稠密缓存，向学习器交付稀疏对象。[supervision.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/supervision.py#L157) [router.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/router.py#L156) |

附录 A 的 `C0` 按获取反馈所用的模型执行费求和，主实验账的范围为反馈执行费与服务费。完整 gold、judge、训练计算及部署路由开销需要额外核算。当前源码的通用指标函数支持这些额外成本参数，主 pipeline 调用采用默认零值。[原文附录 A](https://arxiv.org/html/2609.37402v1#A1) [metrics.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/metrics.py#L52) [pipeline.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/pipeline.py#L183)

**官方源码提供四基准的固定 profile、数据 revision、稀疏观测保存及复现入口，当前版本与指定 v1 存在参数差异。** 例如 MMR-Bench profile 的收缩强度为 120、残差缩放为 0.25，而 v1 表 A1 列出 40、2。RouterBench profile 的残差 Ridge 权重为 200，而表 A1 列出 100，并仅注明 LLMRouterBench 为 200。后续复现需要同时标明论文版本、源码 commit 与实际 profile。本次已核实源码可获得及入口可读，数值复现结果尚未产生。[profiles.py](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/saverouter/profiles.py) [原文附录 C](https://arxiv.org/html/2609.37402v1#A3) [官方复现说明](https://github.com/LAMDA-Model-Reuse/SaveRouter/blob/250b7ef35a741c598aaf1bb264930d1e0bfd1a61/README.md#reproducing-the-experiments)

## 其余指定文献的支持范围

- **Menon et al., ICML 2021。** §3.1 Lemma 1 证明 Bayes 类概率监督可降低固定预测器的经验风险估计方差。§4.1 Proposition 3 给出近似教师误差与方差的权衡。§4.2 明确分析采用教师、学生不同训练样本。E12 在同一揭示集合估计组均值并拟合学生，20 个奖励维度可同时为真，该定理在 E12 中的用途为监督平滑动机，组均值有效性由实际预算曲线检验。PDF 物理页码为第 3、5、6 页。[指定 PDF](https://proceedings.mlr.press/v139/menon21a/menon21a.pdf)
- **UniRoute，arXiv:2502.08773v2。** §5.1 式 (12)–(13) 在训练输入上建簇，用验证反馈构造模型的簇平均错误向量。§5.2 进一步学习软簇归属。§5.3 Proposition 2 在正则性条件下用逐题错误与簇平均错误的差距刻画总体路由近似误差。它确立了簇能力路由基础，也保留了组内异质性问题。E12 的有限反馈收益仍需实验确认。[指定版本 §5](https://arxiv.org/html/2502.08773v2#S5)
- **The Avengers，arXiv:2505.19797v3。** §3 已将验证输入聚类并记录各模型的簇能力，测试时匹配最近簇、选 top-n 模型，再重复采样和多数投票。附录 B.6 承认能力 profile 需要保留验证数据。其校准对象为单模型的簇分数，E12 的直接监督对象为三成员组合经过实际聚合后的最终正确性。成员边际正确率与组合最终奖励之间的映射需要专门建模。[指定版本 §3](https://arxiv.org/html/2505.19797v3#S3) [附录 B.6](https://arxiv.org/html/2505.19797v3#A2.SS6)

## E12 的可支持定位

**本轮研究同一反馈掩码的利用效率。** 固定各预算的揭示集合、完整 H480 输入权限和训练组 metadata 权限，比较逐题监督、组均值目标及向全部训练输入传播的预算曲线。`P_group_revealed` 帮助判断目标平滑与拟合行扩展各自的实际作用。所有学习器的奖励入口遵循 `Y[S_B]`，组能力教师由当前预算重估。

SaveRouter 已同时研究分组能力、稀疏单元、先验收缩与逐题残差，UniRoute、Avengers 已提供簇能力路由基础。将 20 个组合整体视为 20 个动作并使用其最终奖励，是既有路由方法的自然适配。充分创新主张需要额外机制或证据。E12 现阶段的价值为固定反馈权限下的经验预算曲线，以及组平滑与输入传播的分解。主矩阵按已固定的学习方法执行，主动获取属于后续比较阶段。

| 视角 | 优势 | 劣势与风险 |
| --- | --- | --- |
| 统计估计 | 共享组能力可提高有限标签的复用程度，控制项有助于定位收益来源。 | 小预算组均值可能不稳，组内残差含可预测结构，伪目标行彼此相关。 |
| 实验设计 | 相同 mask、相同输入池和固定正则，使预算差异的归因更集中。 | 训练 metadata 是明确的信息优势，未来外部方法比较需等权授予。 |
| 部署与新颖性 | 直接用最终组合奖励学习，契合现有组合动作。 | 当前完整题包反馈限制了对真正部分动作反馈的外推，强表征部署费需独立列账。 |

## 拓展思考与后续边界

后续最值得检验的区别，是组间统计共享、成员响应复用和非可加最终奖励的联合作用。对同题同生成的组合集合 \(\mathcal A\)，可采用以下执行账。

\[
\operatorname{cost}(\mathcal A)=
\sum_{m\in\cup_{a\in\mathcal A}a}\operatorname{member\_cost}(m)
+\sum_{a\in\mathcal A}\operatorname{aggregation\_cost}(a)
+\operatorname{gold\_cost}(\mathcal A).
\]

该联合成本结构允许共享已生成的成员响应，每个组合的最终聚合反馈仍需实际执行并评分。成员的单独正确率缺少答案内容、错误相关性与聚合器行为等信息。后续部分动作反馈比较可先固定 logger 与原始题目–动作–生成掩码，再评估学习器的反馈利用能力。E12 的实验对象为完整 20×3 反馈题包的数量曲线。

**置信度评级。** 一手来源存在性、相关方法机制及本次源码读取结论为高。E12 的预算收益、机制占比与后续方案效果依赖实际实验结果。联合成本与组合奖励是否形成充分的新颖性，仍需针对该问题开展后续文献与实证核查。

## 私有复核材料

本次来源正文保存于 `saverouter_v1.html`、`uniroute_v2.html`、`avengers_v3.html` 和 `menon21a.pdf`，源码摘取于 `saverouter_source/`，均与本记录同目录。Menon PDF 为 11 页，结尾标记完整，关键页已实际渲染检查。上述材料用于私有复核，公共交付可直接采用本文的转述与固定来源链接。
