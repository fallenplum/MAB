# NS-SPO+机制验证复现

## 数据与环境

资产根目录为 `E:\语言模型组合路由`，本轮目录为 `research\m4_noise_aware_set_learning`。原train600的真实组合评分、题目向量、问题划分和来源哈希继续从E8合法历史区读取。所有代码使用CPU、单线程BLAS和Python `-B`。

环境沿用原 `.venv\Scripts\python.exe`，包括Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2，精确有限MC的QP求解器为E8私有隔离环境中的Clarabel0.11.1。

公开源码位于 `public/code`。原运行结构保存在 `private/runtime`，其中 `training/ns_training.py`与`training/private/fold*`同处一个训练根，噪声后验位于 `noise/private/fold*`。私有目录保留权重、逐题结果、后验样本与真实评分访问收据。

## 核查与独立评分复演

PowerShell核查命令如下。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\m4_noise_aware_set_learning\public\orchestration\reproduce_study.py' verify
```

完整评分复演使用以下命令。

```powershell
& 'E:\语言模型组合路由\.venv\Scripts\python.exe' -B -X utf8 'E:\语言模型组合路由\research\m4_noise_aware_set_learning\public\orchestration\reproduce_study.py' scoring-replay
```

程序核对两个父研究全部文件与六个核心来源哈希，恢复16个模型并检查五折后验哈希。评分复演创建新的时间戳私有目录，使用全部已冻结动作，随后读取旧train600评分。完整结果和两项主统计JSON应与原文件字节相同。

## 完整后验与训练重算

在新的输出目录先执行 `public/code/run_joint_noise.py qualify --out <新的噪声运行目录>`。随后对fold0至fold4分别执行其 `build --fold <编号> --out <同一噪声运行目录>`，最后运行 `summarize_joint_noise.py --out <同一噪声运行目录>`。

在另一个新训练根原样放置 `ns_training.py`。运行 `qualify --out <新训练根>`，并对每折运行 `fit-fold --fold <编号> --posterior-root <新噪声目录/private> --out <新训练根> --delta 0.05`。全部16个模型齐备后，使用 `evaluate_ns.py --training-root <新训练根> --posterior-root <新噪声目录/private> --output <新的评分目录>`。

每个命令均需使用同一Python环境、`-B -X utf8`、原资产根目录及原字节来源文件。已有结果目录继续保留，新输出使用新目录。math模块的`qualify_set_losses.py`保存19项合成检查，`review_training_operators.py`提供独立算子核查。

## 预声明设置与限制

主δ=0.05由用户明确选择，δ=0是敏感性。噪声协方差固定20%收缩。正则沿用原M4每折值。NS和Davidson都使用相同8个联合样本，512样本用于后验诊断。Davidson的ν为历史平局率零margin匹配，未联合拟合。

源锁与数学软件检查建立在真实模型拟合前。代码求解的是精确的有限MC凸目标，后验积分仅作8样本近似。另一批8样本引起85/120题变队，这项敏感性按预声明保留。全部统计仍是重复开发资产上的条件证据。

采样收敛、Gaussian矩预测覆盖、潜在概率校准与最终服务质量分别报告。经验集合TSPO+作为回退对照保留。当前缓存的聚合seed固定42，三次重复变化来自成员生成seed。
