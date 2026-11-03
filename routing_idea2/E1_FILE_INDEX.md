# E1重要文件索引

| 文件 | 用途 |
|---|---|
| results/e1/e1_diagnostics.json | 原E1结果与机会门；原字节保留。 |
| results/e1/audit_metrics.json | 投票、聚合损失、配对区间与20组合后验审计；省略逐题原文。 |
| results/e1/causal_diagnostic_summary.json | 追加600次对照、完整费用与配对区间，单列于原E1。 |
| artifacts/audit/audit_isolated_e1_causal_outputs_v1.py | 独立核对600次请求、响应、SQLite提交与固定评分。 |
| artifacts/audit/compare_causal_fusion_to_vote_v1.py | 新强聚合与原直接投票的题级配对比较，0新增模型调用。 |
| results/e1/input_reconstruction.json | 2400条成员请求、6400条聚合/独答请求及候选压缩检查的相关证据。成员请求另见audit_metrics。 |
| results/e1/option_text_diagnostic.json | 原题选项文字的唯一精确匹配诊断；不替代原评分。 |
| results/e1/physical_call_summary.json | 8800次真实调用分组、token、时间与原文件哈希。 |
| results/e1/all_fixed_teams.csv | 20个固定组合的选择集、评估集准确率。 |
| results/e1/independent_review_summary.json | 原E1与独立重放完全一致的来源证据。 |
| scripts/evaluate_real_e1.py | 从完整真实输出重建E1统计；拒绝覆盖既有输出。 |
| src/budget_router/diagnostics.py | 基线选择、分任务配对bootstrap、覆盖率和机会门。 |
| src/budget_router/e1_controls.py | 选择强模型及每题三次独立新采样控制。 |
| src/budget_router/scoring.py | 固定最终答案评分，safe_exact_v1_1。 |
| src/budget_router/member_execution.py | 成员独答提示与内容绑定。 |
| src/budget_router/execution.py | 聚合提示、匿名顺序、head128/tail384压缩、执行账本。 |
| configs/ | 固定模型版本、NF4和生成参数。 |
| protocol/E0_LOCK.json | 原E0冻结身份，不能随迁移或修复静默替换。 |
| artifacts/audit/audit_e1_failure_modes_v2.py | 本次最终数值审计；不更改原正式结果。 |
| artifacts/audit/audit_e1_input_content_v1.py | 精确重建聚合消息与检查候选最终答案保留。 |
| artifacts/audit/audit_e1_option_text_v1.py | 只读格式诊断。 |
| artifacts/audit/build_e1_teacher_report_v1.py | 从本地真实证据构建Word。 |
| tests/test_real_e1_assembly.py 等 | 软件与边界验证；测试通过不是方法性能证据。 |

报告位于仓库 reports/e1/。GitHub归档不含逐题数据和权重，因此保留完整本地E盘目录仍是复评分的前提。
