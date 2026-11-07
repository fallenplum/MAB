# E4复现与文件索引

## 入口与环境

本研究使用冻结的E3资产加载器和V2源码。基线提交为45ff7ad04fa7f017ebafe27f4c3d73ffe5451cf6，E3核心身份为7995dfdc3afcff8eb8473f6da2af79d55818283d30044200a92c293bf583a499。

本机记录为Python3.11.14、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2。代码固定BLAS单线程和CPU计算。原始问题、gold、编码器向量、回答、评分矩阵和行级日志保留在本地受控资产目录。公开文件给出来源哈希、可复现代码与脱敏统计。

将本目录代码放入资产根目录的research/e4_statistical_model_audit/public。原E3目录应位于research/decision_acquisition_e3/public，V2模块位于src。环境变量MAB_ASSET_ROOT可指定资产根目录，默认E盘原项目目录。执行时保持该目录下代码与交付清单一致。

## A与B

```powershell
$assetRoot='E:\语言模型组合路由'
$python="$assetRoot\.venv\Scripts\python.exe"
$study="$assetRoot\research\e4_statistical_model_audit"
& $python -B -X utf8 "$study\public\e4_gaussian.py"
& $python -B -X utf8 "$study\public\raw_audit.py" --output-dir "$study\private\raw_replay"
& $python -B -X utf8 "$study\public\run_learning.py" --workers 6
& $python -B -X utf8 "$study\public\stability_analysis.py"
& $python -B -X utf8 "$study\public\run_posterior_audit.py" --workers 6
& $python -B -X utf8 "$study\public\aggregate_e4.py" learning
& $python -B -X utf8 "$study\public\aggregate_e4.py" posterior
& $python -B -X utf8 "$study\public\audit_calibration.py"
```

原始审计显式指向private输出目录，避免逐题证据进入公开目录。继续运行前先执行交付校验。A/B轻量缓存续跑依赖固定源配置，若改变源代码或资产，应使用隔离的新研究目录并记录新身份。

## C

先核对C_GATE记录中的B实测证据，再运行概率模型。学习器使用历史内部三折选择alpha，外折评分仅用于评价。

```powershell
& $python -B -X utf8 "$study\public\e4_binary.py"
& $python -B -X utf8 "$study\public\run_binary.py" --workers 6
& $python -B -X utf8 "$study\public\aggregate_e4.py" binary
& $python -B -X utf8 "$study\public\run_binary_online.py" --workers 4
& $python -B -X utf8 "$study\public\run_metadata_sensitivity.py"
& $python -B -X utf8 "$study\public\summarize_binary_online.py"
& $python -B -X utf8 "$study\public\audit_binary_optimization.py" --workers 4
& $python -B -X utf8 "$study\public\summarize_binary_online.py"
```

主要C结果在元数据敏感性之前冻结。辅助元数据结果单列文件，特权真实子任务标签保持辅助角色。数值优化复核入口以该脚本的help与交付版本为准，原在线演员轨迹继续保留。

## D

先核对D_GATE_AND_FREEZE记录，包括开发候选选择、order17、预算20和附加Probability-DA对照。

```powershell
& $python -B -X utf8 "$study\public\e4_decision.py"
& $python -B -X utf8 "$study\public\run_decision_audit.py" --workers 4
& $python -B -X utf8 "$study\public\run_decision_audit.py" --snapshots --workers 4
& $python -B -X utf8 "$study\public\summarize_decision.py" live
& $python -B -X utf8 "$study\public\summarize_decision.py" snapshots
```

完整前瞻服务与合法反馈记录在private/decision_live，冻结反事实记录在private/decision_snapshots。公共费用与统计文件保留实际预算和付费范围。反事实未来评分只供评价，当前服务动作先于反馈冻结。

## 主要文件

|文件|用途|
|---|---|
|01至08报告|数据、题量、后验、噪声、二值模型、元数据、采集机制和综合结论|
|E4_COMPLETE_RESULTS.json|阶段状态、主要假设与全部数值附件索引|
|E4_EXPERIMENT_CONFIG.json|A/B和C预定配置|
|E4_PROTOCOL_AMENDMENTS.json|控制反馈划分、D范围与数值诊断等执行选择|
|PUBLIC_DELIVERY_MANIFEST.json|公开文件SHA256、Git blob身份与字节数|
|LEARNING_CURVE_RESULTS.json|650组拟合按原题汇总的完整曲线与主比较|
|POSTERIOR_AUDIT_RESULTS.json|900条Gaussian受控路径和所有噪声对照|
|BINARY_COMPARISON_RESULTS.json|冻结主C模型比较及历史CV选择|
|BINARY_ONLINE_RESULTS.json|720条在线路径、纠错/破坏、full-refit与费用|
|METADATA_SENSITIVITY_RESULTS.json|真实与文本预测子任务信息的权限对照|
|DECISION_LIVE_RESULTS.json|165条先服务后反馈路径、配对结果与Holm|
|DECISION_PROXY_CALIBRATION.json|1200个候选代理与未来真实纠错关系|
|FULL_SERVICE_COST_ACCOUNTS.json|完整缓存、历史、Support与Query分项费用|
|D_PROXY_AUDIT_COSTS.json|反事实候选各自假想世界的额外审计费用|
|BINARY_OPTIMIZATION_ACCURACY.json|独立Newton精修对原MAP及重拟合参考的数值敏感性|

完成后运行verify_e4_delivery.py，输出路径应指定到private，避免覆盖已发布收据。公开manifest与验证收据分别由Git blob校验，保持内容哈希无循环依赖。

Gaussian权重空间后验与二值分类Laplace近似可参照原作者教材。当前代码的正规方程、梯度、Hessian及更新均另有实际数值资格测试。[GPML回归章节](https://gaussianprocess.org/gpml/chapters/RW2.pdf)、[GPML分类章节](https://gaussianprocess.org/gpml/chapters/RW3.pdf)
