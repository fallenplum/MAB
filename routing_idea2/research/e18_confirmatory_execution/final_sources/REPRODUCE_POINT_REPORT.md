# E18 点值报告入口

`generate_point_only_report.py`从已完成的E18结果摘要及私有数值数组生成中文报告。它核对四策略的GSM8K、BBH、任务宏平均、27个子任务宏平均、三个生成seed及主比较纠错计数，保存聚合CSV和JSON。原题文本、标准答案和native缓存均由前阶段处理。

## 命令

使用原Python与NumPy环境。

```powershell
$e18Code = '<E18完整源码目录>'
$e18CompletedResult = '<已完成E18_FINAL_RESULTS.json完整路径>'
$e18NumericArrays = '<已完成FINAL_POLICY_REWARDS.npz完整路径>'

python -B -X utf8 "$e18Code/generate_point_only_report.py" `
  --results "$e18CompletedResult" `
  --results-sha256 '<完成结果SHA256>' `
  --rewards "$e18NumericArrays" `
  --rewards-sha256 '<完成数值数组SHA256>' `
  --output-root '<全新公开报告目录>'
```

该入口要求结果状态为`COMPLETE`，原finalizer源SHA为冻结9b，评价范围为3000道原题、30921个聚合键与47286个成员键。结果文件及数值数组通过显式SHA绑定。可恢复的失败运行保留其独立记录，报告由最终完成版本生成。

## 输出

- `E18_CONFIRMATION_REPORT.md`保留主问题、四策略主表、配对纠错、选队集中度、分组、费用和局限。
- `E18_POINT_RESULTS.json`保存相同聚合点值。
- `MAIN_TABLE.csv`、`PAIRED_TABLE.csv`、`GROUP_TABLE.csv`、`SEED_TABLE.csv`及`COST_TABLE.csv`用于结果核对。

主点差为TSPO_uniform相对SPO_coordinate，GSM8K和BBH各占50%。每题三个生成重复保留在同一原题簇内。报告从原始完成数值数组计算点值，并与评分阶段的质量和主纠错计数比较。

源码SHA为`665d15876667f999d6ff996d65362ec37b68f0b3d0e09b59d63f918f09bc2c1e`。现有旧报告器和原9b评分源保持各自冻结身份。

## 费用口径与局限

费用分列实际联合执行并集及每个冻结策略的逻辑服务投影，保留成员、聚合、输入Token和输出Token。历史所需成功缓存计入已知物理投入。旧失败全生命周期、独占GPU核时和货币费用需要各自的原始计量来源。

该报告器采用描述性点值。它支持已完成资产的聚合解释，正式结论仍受有限生成重复、任务来源和固定模型池范围约束。
