# E9 复现入口

## 环境

使用 `E:\语言模型组合路由\.venv\Scripts\python.exe`，Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2与psutil。全部E9计算为CPU，BLAS单线程，Python加 `-B -X utf8`。

评价恢复复用既有NS特征映射helper，其导入沿用E8私有隔离目录中的Clarabel0.11.1依赖。E9头训练使用闭式Ridge与L-BFGS，当前环境已具备所需包。

当前复现以原E盘路径及私有来源收据为前提。既有收据保留绝对来源路径，完整资产根为 `E:\语言模型组合路由`。

## 来源核验

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning\public\code\e9_reproduce.py' verify
```

该入口核对原E8/M4/NS文件清单及字节、协议初始收据、核心源锁、30个新教师、全部195个槽位、逐模型与聚合预测连接、标签来源及背景折映射。

## 独立评分重放

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning\public\code\e9_reproduce.py' scoring-replay
```

程序创建新的时间戳私有目录，复用冻结模型和教师，完整封存所有动作后评价旧train600。三个结果JSON的科学字段与原公开文件逐项一致。模型恢复误差的浮点元数据单列，验收阈值为1e-12且动作必须一致。重放的CPU和wall记录保存在新目录的 `REPRODUCTION_VERIFIED.json`。

## 全部新教师与新头重算

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning\public\code\e9_reproduce.py' full-refit
```

该入口使用原字节的冻结算子和配置，在新的私有运行目录重建30个教师区域，运行主矩阵及内容控制，然后评价。它只重定位模块输出目录。五个outer既有均值和A继续按冻结规则复用。完整重算将再次执行165个实际头拟合与30个派生槽位，正式3档网格保持固定。

重算时的运行时间和恢复误差属于当前运行记录，科学指标应与原结果一致。新运行只用于复现已曝光开发资产，原结果继续保留。

## 单阶段源码

- `e9_teachers.py --fold 0`构建对应合法教师，正式入口要求既有教师源锁。
- `e9_train.py --fold 0`执行一折的主头、内层选择及全部内容控制，要求教师已完整。
- `e9_evaluate.py --out <新目录>`要求全部五折195槽位已封存，随后统一评分。
- `qualify_csce.py --output <新建私有目录/资格文件.json>`执行数学核合成资格。
- `qualify_e9_heads.py --output <新建私有目录/资格文件.json>`执行预测头与权限资格。

教师与头文件夹使用不可覆盖的冻结收据。若出现未封存失败目录，程序保留失败证据并拒绝把它当成功缓存。实验运行和完整资格源码保存在私有执行档案中。

## 权限与统计

每个inner320教师在其内部再做三折，先验拟合区为213/214题。inner160标签仅用于选参。outer120只提供问题输入，所有内容控制完成后统一封存再评分。

主统计按outerfold×task的原题簇配对bootstrap10000次，保留同题三生成及全部策略，主族只有C−A、B−A并做Holm。普通95%区间和调整p分别展示。三次置乱仅为给定配置敏感性，标准Monte Carlo最小p为0.25。CSCE概率未作为正确率计算Brier。

全部物理LLM调用与保留题评分读取均为0。共同历史完整反馈费用、教师CPU、头优化、资格与失败时间在 `COST.json`分别保存，逐拟合wall合计与端到端时间采用不同口径。

## 成本与CSV重建

归档中保留全部资格输入。使用新输出目录可重建分账和四份CSV。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning\public\code\build_cost_and_tables.py' --asset-root 'E:\语言模型组合路由' --support-root 'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning\private\runtime' --output 'E:\语言模型组合路由\research\e9_consistent_cost_sensitive_learning\private\reproductions\cost_tables_rebuild'
```

生成器还支持 `--experiment-root`和单独资格输入路径。原始未记录的端到端时间及算子专属时间保持空值，重建过程不会把新的执行时间替代旧历史缺口。
