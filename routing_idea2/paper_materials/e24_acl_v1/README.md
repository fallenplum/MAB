# E24 离线复算

使用 Python 3.11。从仓库根目录执行：

```bash
cd routing_idea2/paper_materials/e24_acl_v1
python -m pip install -r requirements-offline.txt
python -B scripts/recompute_e24.py --sufficient data/e24_stratum_sufficient_statistics.csv
```

环境准备好后即可断网复算。CSV 是 27 层充分统计量，三个生成 seed 先在每题内汇总。预期输出：

| 题数 | B–U | B–P | C–U 与 C–P | 交互 | 双侧 p |
|---:|---:|---:|---:|---:|---:|
| 1,196 | 86.464686% | 87.625418% | 87.653289% | +1.160731851 pp | 0.0000886396891 |

本入口复算已闭合的汇总结果。用原缓存重放逐题结果或重新生成答案，需要对应的私有资产、冻结模型和运行环境。[E24 原实验运行说明](https://github.com/fallenplum/MAB/blob/d48fef3cb1e77e876d4410be4933c5eb6cdb5738/routing_idea2/research/e24_training_deployment_replication/REPRODUCE.md) · [论文主线](../../../paper/README.md)
