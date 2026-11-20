# E24 离线汇总复算

本目录从 E24 冻结评分导出的 27 层充分统计量，独立重建四种策略准确率、训练×部署交互、Satterthwaite 自由度和双侧 p 值。每层保留题级交互的一阶和二阶矩、四策略效用和及样本数。三个生成重复先在题内聚合。GSM8K 占一半，保留的 26 个 BBH 子任务各占 0.5/26。

## 运行

Python 3.11 和本目录的科学计算依赖可在准备环境后断网运行：

```bash
python -m pip install -r requirements-offline.txt
python -B scripts/recompute_e24.py --sufficient data/e24_stratum_sufficient_statistics.csv
```

预期汇总为 B–U 86.464686%，B–P 87.625418%，C–U 与 C–P 均 87.653289%，交互 +1.160731851 pp，双侧 p = 0.0000886396891，自由度 462.394193508。

[论文结果说明](../../../paper/E24.md) · [历史执行记录](../../../experiments/e24-training-deployment-replication/README.md)

此入口提供汇总数值复算。实验分支保存原协议、代码和执行记录。本地材料另含匿名题次用于逐题复核。完整生成验证需要私有输入、缓存、权重、模型运行环境及历史科学源码。
