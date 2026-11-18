# 机制特异性审计复现

## 入口与依赖

本目录在仓库中的路径为 `routing_idea2/research/e22_mechanism_specificity/`。使用同级 E19、P3 代码与既有的私有 train600 资产，Python 3.11、NumPy 1.26.4、SciPy 1.11.4、Clarabel 0.11.1。Clarabel 复用 E8 私有 dependencies 目录。

先执行 `qualify_audit.py --output <新资格JSON>`。它使用人工数组检查原目标恢复、固定 mask、有效总权重、分数权重、零权重与几何关系。

正式入口分两阶段

1. `run_audit.py --phase fit --root <新运行目录>`
2. `run_audit.py --phase evaluate --root <同一运行目录>`

第一阶段逐折生成固定行权重，复用已冻结 B/C 头，拟合 40 个新控制并恢复 TSPO+ 对偶梯度。全部折冻结后第二阶段读原 train600 的外折评分，生成 `public/RESULTS.json`。目录与输出均拒绝覆盖。协议、runner、solver 与 geometry 的摘要写入冻结收据。

`summarize_audit.py --results <RESULTS.json> --root <运行目录> --output <新汇总目录>` 生成 `SUMMARY.json`、`controls.csv`、`geometry.csv`。匹配三种子的平均对照使用三套冻结策略概率的等权平均，纠错与破坏为该混合策略的期望值。

## 私有依赖

- 原 train600 input/reward NPZ，仅使用元数据、相应合法训练行和冻结后的评价行。
- E19 的五折 INPUTS_ONLY 包和 B/C 模型，各有既有冻结收据。
- P3 的原域 ListNet 五折模型，来自 `internal_run_v1/private`。
- 本轮 `private/foldN/MASKS_FROZEN.npz` 保存逐题权重，`tspo_FROZEN.npz` 与 `listnet_FROZEN.npz` 保存模型、分数及梯度分量，`FROZEN.json` 绑定全折来源。

## 结果索引

`AUDIT_REPORT.md` 给出匹配控制与最终主张，`GEOMETRY.md` 给出公式及参数解释。全部数据汇总保存在 RESULTS 与 SUMMARY，费用和记录缺口见 `COSTS.json`。旧 train600 的结果属于开发机制证据。三个生成重复随同一题处理，三个 mask 也不增加独立题数。全并列标记只来自每折训练奖励。

