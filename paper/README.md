# 论文主线

**When Training Ties Change the Winner in LLM Ensemble Routing**

训练中把经验并列组合对称处理，仍会压缩它们的分差。E19 推导这种约束如何通过共享参数改变其他题的赢家，P2 记录了 302 道两支均有唯一赢家却选中不同组合的题。

E24 在 1,196 道独立新题上复验。C 相对 B 的准确率差在均匀处理近并列时为 **+1.189** 个百分点，在用训练历史处理近并列时为 **+0.028** 个百分点。两种部署下的差距相差 **+1.161** 个百分点。两支均唯一且换队的题有 **145** 道。[独立评价](https://github.com/fallenplum/MAB/blob/d48fef3cb1e77e876d4410be4933c5eb6cdb5738/routing_idea2/research/e24_training_deployment_replication/EXPERIMENT_SUMMARY.md) · [离线复算](../routing_idea2/paper_materials/e24_acl_v1/README.md)

E25 在同一已曝光题池上做补充诊断。B 的 1,033 道近并列题共享同一组 17 个组合。这个模式跨三档正则保持。小型 MLP 也出现分差收缩，达到固定并列阈值的题数随初始化变化。[机制诊断](https://github.com/fallenplum/MAB/blob/7345504242e1215c1d04837484227d35cf47225c/routing_idea2/research/e25_tie_mechanism/EXPERIMENT_SUMMARY.md) · [论文诊断清单](https://github.com/fallenplum/MAB/blob/7345504242e1215c1d04837484227d35cf47225c/routing_idea2/research/e25_tie_mechanism/ACL_DIAGNOSTIC_NOTE.md)
