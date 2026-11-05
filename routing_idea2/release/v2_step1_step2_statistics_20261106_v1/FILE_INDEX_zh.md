# 本次补充推送：V2 Step 1 与 Step 2

上次跨种子诊断文件属于Step2 train/validation；这里单独补齐Step1真实开发诊断。两阶段题目角色不同，不能混合统计。

|文件|用途|
|---|---|
|step1/README_zh.md|Step1各方法质量和全部9方向迁移表|
|step1/STEP1_COMPLETE_SCIENTIFIC_STATISTICS_v1.json|源statistics全部字段、20组合、27子任务、配对区间和覆盖率|
|step1/STEP1_CROSS_SEED_TRANSFER_v1.json|全部6个跨seed方向及3个对角线对照、CI与支持数|
|step2/README_zh.md|独立Top3准确率、逐seed、27子任务和配对CI表|
|step2/STEP2_INDEPENDENT_TOP3_VALIDATION_v1.json|原完整Top3验证报告的路径脱敏导出|
|step2/STEP2_INDEPENDENT_TOP3_ACCURACY_BY_TASK_v1.json|五方法逐任务、逐seed和27子任务结果|
|step2/STEP2_INDEPENDENT_TOP3_PAIRED_CI_v1.json|Top3对四方法全部总体/逐seed配对95%CI|
|step2/STEP2_TOP3_FREEZE_AND_LINEAGE_PUBLIC_v1.json|模型定义、冻结身份和原谱系复核摘要|
|EXPORT_COMPLETENESS_AND_INTEGRITY_v1.json|原统计逐项相等、SHA及导出范围检查|
|DELIVERY_MANIFEST_v1.json|全部推送文件SHA|

全部为已完成的开发统计，未评价fresh400，不代表整个研究完成。只导出统计与元数据；原始回答、参考答案、逐题数组、权重和连接信息留在私有目录。相关计算源码已见上次诊断检查点的source_snapshot。
