# E5复现入口与重要文件索引

## 冻结来源

分支为codex/e5-paired-gain-validation，基于E4提交b5fa3317c77b1256037d734c93a001ecb19bfa28。原E3核心身份为7995dfdc3afcff8eb8473f6da2af79d55818283d30044200a92c293bf583a499。

将本目录代码放入资产根目录research/e5_paired_gain_validation/public，旧E3/E4依赖分别位于research/decision_acquisition_e3/public与research/e4_statistical_model_audit/public。环境变量MAB_ASSET_ROOT可指定资产根目录，默认E盘原项目目录。旧输入、输出矩阵、模型池及算子均须通过原加载器的SHA校验。

PDF和原任务书保存在private，公开发布来源哈希及章节对应信息。原题、gold、成员/聚合文本、问题向量、逐题数组、小模型参数和先导记录保持在private。数学合成脚本可以在相应NumPy/SciPy环境直接运行。

## 环境与主要命令

本轮使用已有Python3.11实验环境，NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2。全部CPU计算固定单BLAS线程。

```powershell
$assetRoot='E:\语言模型组合路由'
$python="$assetRoot\.venv\Scripts\python.exe"
$study="$assetRoot\research\e5_paired_gain_validation"
& $python -B -X utf8 "$study\public\theory_checks.py" --output "$study\public"
& $python -B -X utf8 "$study\public\run_base.py" --workers 5
& $python -B -X utf8 "$study\public\centered_diagnostics.py"
& $python -B -X utf8 "$study\public\run_gates.py" --workers 5
& $python -B -X utf8 "$study\public\run_metadata_controls.py"
& $python -B -X utf8 "$study\public\summarize_metadata.py"
& $python -B -X utf8 "$study\public\summarize_switching.py"
& $python -B -X utf8 "$study\public\audit_switch_ablation.py"
& $python -B -X utf8 "$study\public\cost_and_parameters.py"
```

run_gates先写冻结的G门控和E输入决策，再读取E配对评分。元数据版本单独保存，不参与主门控继续关口。重新生成03主报告后，需再次运行audit_switch_ablation以恢复其探索性消融及元数据附录。

## 认证与分配合成

下面的合成和开发功效脚本必须显式指定private输出位置，避免将大规模逐案数组或先导原始记录写到公开目录。三个JSON总表再复制到public，报告生成器使用这些总表及REAL_ALLOCATION_GATE。

```powershell
& $python -B -X utf8 "$study\public\run_certification_synthetic.py" --output "$study\private\synthetic_certification"
& $python -B -X utf8 "$study\public\run_allocation_synthetic.py" --output "$study\private\synthetic_allocation" --real-gate-status NOT_RUN_GATE_NOT_PASSED
& $python -B -X utf8 "$study\public\run_certification_power.py" --output "$study\private\power_development"
Copy-Item -LiteralPath "$study\private\synthetic_certification\CERTIFICATION_SYNTHETIC_RESULTS.json" -Destination "$study\public"
Copy-Item -LiteralPath "$study\private\synthetic_allocation\ALLOCATION_SYNTHETIC_RESULTS.json" -Destination "$study\public"
Copy-Item -LiteralPath "$study\private\power_development\CERTIFICATION_DEVELOPMENT_POWER.json" -Destination "$study\public"
& $python -B -X utf8 "$study\public\audit_allocation_selection.py"
& $python -B -X utf8 "$study\public\make_cert_allocation_reports.py"
```

报告生成器无参数运行，优先读取公开目录的三份flat JSON，并保留旧子目录输入回退。若存在ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json，05会保留额外零点和并列容差检查。真实分配由主C两比较的冻结关口控制，本轮关口未通过。

## 主要文件

|文件|作用|
|---|---|
|01_THEORY_IMPLEMENTATION_CHECKS.md|无并列反例、秩、总体投影/有限LS/Ridge和真实中心化资格|
|02_CENTERED_ERROR_DIAGNOSTICS.md|绝对、中心化、共同偏移误差，Ti假设和真实选队结果|
|03_SWITCH_RANKING_AND_MATCHED_CONTROLS.md|四评分、20阈值曲线、三匹配控制、直接消融和同权限标签对照|
|04_CERTIFICATION_VALIDITY_AND_POWER.md|完整与IPW证书、错误率、功效和分层样本需求|
|05_TWO_RESOURCE_ALLOCATION.md|双预算求解、目标退化、估计/选择分离、先导收费及预算偏差|
|EXPERIMENT_SUMMARY.md|主要结论、科学关口和完整局限|
|E5_COMPLETE_RESULTS.json|阶段状态和数值附件索引|
|E5_EXPERIMENT_CONFIG.json|在当前结果查看前冻结的配置|
|E5_PROTOCOL_AMENDMENTS.json|条件似然权重、阈值语义、匹配和分层证书细节|
|REFERENCE_AND_INPUTS.json|PDF、任务书、冻结矩阵与旧依赖的来源哈希|
|REAL_ALLOCATION_GATE.json|两项主比较及真实分配继续判断|
|COST_AND_REUSE_RESULTS.json|原缓存、F/G/E、部署和实测CPU的分项费用|
|ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json|原插件选择的浮点零点与并列稳健性|
|E5_INDEPENDENT_AUDIT.json|独立重放、权限和数值复核收据|
|PUBLIC_DELIVERY_MANIFEST.json|公开文件字节数、SHA256与Git blob身份|

完整真实角色文件位于private/base、private/gates、private/metadata_gates，统计重复与配对数组位于private/switching。分配MC、先导及审计派生数组分别位于对应private子目录。历史文件继续由原E1至E4目录提供。

验证前先保持所有实验代码和输入身份一致。重复运行会产生新的时长和收据，数值复现以冻结配置和随机种子为准。正式交付核验的输出路径应指向private，避免覆盖已发布的验证收据。

稿件的经验Bernstein证书常数与样本方差界对应原始理论。实际实现以导师稿件规定的区间、家族和固定评价时点为准。[Maurer与Pontil原论文](https://arxiv.org/abs/0907.3740)
