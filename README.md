# MAB 模型组合路由

论文主线研究有限次数生成产生并列成绩时，训练目标如何改变新题的组合排序，以及部署时处理近并列的规则如何影响准确率。[论文主张与证据](paper/CLAIMS_AND_EVIDENCE.md) · [论文入口](paper/README.md)

## 最新独立复验

E24 在 **1,196 道独立新题**上比较两种冻结训练头和两种部署规则。使用均匀近并列部署时，关闭经验全并列约束的 C 比 B 高 **1.189 个百分点**；在相同支持集合内优先选训练成功数较高的组合后，差距为 **0.028 个百分点**。两种部署下的训练差距相差 **1.161 个百分点**。两支评分头各有唯一第一名、却选中不同组合的题有 **145 道**。[四格结果](paper/E24.md)

## 仓库入口

| 入口 | 用途 |
|---|---|
| [论文主线](paper/README.md) | E17–E24 的证据、方法与结果 |
| [E24 数值复算](routing_idea2/paper_materials/e24_acl_v1/README.md) | 用分层充分统计量重算四格准确率、交互和 p 值 |
| [全部实验](EXPERIMENTS.md) | 查阅各阶段的报告、代码和原始提交 |
| [分支索引](docs/BRANCHES.md) | 切换到具体实验的执行记录 |
| [复现说明](docs/REPRODUCIBILITY.md) | 归档检查、数值复算与生成依赖 |

`main` 维护论文与复现入口。各实验分支保留当时的协议、运行记录和产物。E24 原始实验记录固定在 [d48fef3](https://github.com/fallenplum/MAB/commit/d48fef3cb1e77e876d4410be4933c5eb6cdb5738)。

```bash
python scripts/verify_repository.py
python routing_idea2/paper_materials/e24_acl_v1/scripts/recompute_e24.py --sufficient routing_idea2/paper_materials/e24_acl_v1/data/e24_stratum_sufficient_statistics.csv
```
