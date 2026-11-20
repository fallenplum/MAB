# E25 复现

以本分支父提交 E24 `d48fef3cb1e77e876d4410be4933c5eb6cdb5738` 为来源。在本目录运行，保留与 E20、E19、E23、E24 目录的相对位置。使用 Python 3.11.14、NumPy 1.26.4、SciPy 1.11.4、scikit-learn 1.7.2、Clarabel 0.11.1。本次使用原 Windows 环境及已有 Clarabel 依赖。实验在 CPU 执行，单线程数值库变量在导入前设置。

## 输入

- `ASSETS` 为原私有研究资产根目录。训练结果和输入采用 E20 冻结绑定。 `COHORT` 为 `ASSETS/research/e24_training_deployment_replication/private/cohort_v1`。
- 原 E20 `FINAL600_MODELS_FROZEN.npz` 及预拟合配置，E24 `INPUTS_ONLY.npz`、`E24_POLICIES_FROZEN.npz` 的摘要由各脚本校验。
- 拟合读历史 train600 的冻结结果。评价另需 E24 闭合的 `PAID_SUPPORT_SCORES.jsonl` 和 `CLOSED_NATIVE_SCORING_RECEIPT.json`。两者均在 `COHORT/scoring_v1`。

## 运行

用一份指向已冻结资产的工作副本指定 `ASSETS`，为每个新产物指定空目标。令 `PRIVATE=ASSETS/research/e25_tie_mechanism/private`。

```bash
python -B linear.py --assets ASSETS --report NEW/LINEAR_FIT_REPORT.json
python -B mlp.py --assets ASSETS --output PRIVATE/MLP_HEADS.npz --audit NEW/MLP_FIT_AUDIT.json
python -B geometry.py --assets ASSETS --output NEW/GEOMETRY.json
python -B linear_geometry.py --assets ASSETS --fit-report NEW/LINEAR_FIT_REPORT.json --output NEW/LINEAR_GEOMETRY.json
```

第一条把预测和权重存到 `PRIVATE/LINEAR_HEADS.npz`。在复算工作副本中，将本次两个拟合收据放到 `evaluate.py` 同目录绑定新产物。已发布版本搭配同源私有预测数组可直接复算。运行：

```bash
python -B evaluate.py --linear PRIVATE/LINEAR_HEADS.npz --mlp PRIVATE/MLP_HEADS.npz \
  --predictions COHORT/E24_POLICIES_FROZEN.npz \
  --scores COHORT/scoring_v1/PAID_SUPPORT_SCORES.jsonl \
  --scoring-receipt COHORT/scoring_v1/CLOSED_NATIVE_SCORING_RECEIPT.json \
  --e24-protocol ../e24_training_deployment_replication/E24_PROTOCOL.json --output NEW/E25_RESULTS.json
```

读取新 JSON 的 `results.linear` 和 `results.mlp` 获取各格的任务与 seed 结果。已知奖励支持和新头支持由代码逐题合并。 `missing_cost_total_union` 按原组合及成员键去重。置信区间与 bootstrap 计数均为 0。公开目录含输入校验摘要、优化收据和汇总，私有目录保存逐题分数和拟合参数。

[实验总结](EXPERIMENT_SUMMARY.md) · [论文诊断清单](ACL_DIAGNOSTIC_NOTE.md)
