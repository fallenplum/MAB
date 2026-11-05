# 本轮重要文件索引

本轮为科学诊断检查点，未完成独立fresh检验。所有路径相对本交付目录。

|文件|用途|
|---|---|
|DELIVERY_MANIFEST_v1.json|每个交付文件SHA和压缩JSON解压后SHA；不含自身|
|PAUSE_AND_COST_PUBLIC_v1.json|暂停后调用、token与成本闭合摘要；无机器身份|
|REPORT_zh.md|先读：五个科学审查问题、暂停结论和后续边界|
|seed_stability/INTERPRETATION_zh.md|跨seed可复现性、oracle机会及局限|
|seed_stability/PUBLIC_AGGREGATE_SEED_STABILITY_v1.json.gz|全迁移方向、seed、任务和子任务聚合统计|
|seed_stability/PUBLIC_ARITHMETIC_VERIFICATION_v1.json|使用原题簇bootstrap的独立算术复核|
|selector/PUBLIC_DIAGNOSTIC_BRIEF_zh.md|选择器训练/验证差距和模型结构诊断|
|selector/PUBLIC_DIAGNOSTIC_SANITIZATION_RECEIPT_v1.json|公共导出边界及校验凭据|
|selector/PUBLIC_FILE_INDEX_zh.md|公共导出边界及校验凭据|
|selector/PUBLIC_SELECTOR_DIAGNOSTIC_v1.json.gz|五方法、各域/子任务与胜负支持数完整聚合统计|
|source_snapshot/diagnostics/export_public_diagnostic_v1.py|由私有诊断导出仅聚合统计的公共文件|
|source_snapshot/diagnostics/run_frozen_selector_diagnostic_v2.py|冻结选择器回放及既有矩阵诊断|
|source_snapshot/diagnostics/run_independent_top3_v1.py|事后探索性独立Top3 CPU对照构建|
|source_snapshot/diagnostics/run_seed_stability_diagnostics_v1.py|同题跨生成seed稳定性诊断|
|source_snapshot/scripts/v2/full_feedback/prepare_inputs.py|训练专用PCA/标准化和隔离输入资产|
|source_snapshot/src/budget_router/features.py|题目/组合特征构造|
|source_snapshot/src/budget_router/learner.py|共享岭回归及更新|
|source_snapshot/src/budget_router/ranking_evaluation.py|排序指标与Top1Regret定义|
|source_snapshot/src/budget_router_v2/full_feedback.py|本轮实际冻结的共享二阶岭选择器、基线和gate定义|
|VALIDATION_AND_COST_DECISION_v1.json|验证指标、正式gate状态与最小聚合成本|

私有完整报告/逐题数组/原始回答/SQLite账本/备份留在本机和服务器，未上传。source_snapshot是实际源码的原字节快照，完整复验还需恢复私有资产和依赖，不宣称本目录独立可运行。
