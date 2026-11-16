# E18.1复现说明

## 版本与权限

基点为`fallenplum/MAB@95c6c8a38a71866d6efdb224aec04b9a158adf17`，研究分支为`codex/e18-1-statistics-and-mechanisms`。本轮用户明确要求置信区间，E18.1_PROTOCOL.json记录该专项例外及20,000次题级分层Bootstrap设置。

E18原始结果、冻结权重与策略保持原版本。新增程序仅处理既有评分及训练来源记录。本轮统计是在看过E18结果之后补充，主模型、权重与任务比例没有据此重选。

## 环境

实际执行环境为Python3.11.14、NumPy1.26.4、SciPy1.11.4。图像渲染使用Pillow12.0.0及Windows系统字体。统计程序使用CPU，执行收据中的主分析墙钟约2.66秒，CPU约2.42秒。依赖列在requirements.txt。

## 私有输入

1. E18完成后的FINAL_POLICY_REWARDS.npz，SHA为`5c8c8702ba1dd5b74474ec69475806a7325213e6b3ac7664e16f7a9de397e70a`。含4策略×3生成×2,998题的奖励、冻结动作、任务标签和原3,000题有效掩码。
2. E18_FINAL_RESULTS.json，SHA为`86a91632c460085db016bf7718b56cd58fc40808f39a487d898c0dce080db41f`。来自E18的evaluable_confirmation_v1目录。
3. 成本及GlobalFixed复核使用既有PHYSICAL_COST_LEDGER.json、执行scope、两题不可评分清单、E17 train600输入与奖励缓存，以及FULL600_FIT_LOG和冻结收据。完整来源SHA列在05报告与ACCOUNTING_GLOBAL_FIXED_AGGREGATES.json。
4. E17机制证据来自已有冻结汇总，来源索引和脱敏摘录保存在E17_EVIDENCE_EXTRACT.json。本轮复用既有100次置换与算子对照结果。

原题、标准答案、逐题矩阵与模型权重保持私有。公开仓库提供汇总、可复现统计源码和依赖清单。复现需使用授权的私有输入。

## 运行顺序

统计程序和报告程序位于本目录。以下PowerShell示例使用原实验工作区。复现时为输出指定一个尚不存在的新目录，原记录保留。

```powershell
$pythonPath = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$codeRoot = 'D:\develop\CV\多臂老虎机\experiments\e18_1_statistics_and_mechanisms'
$studyRoot = 'E:\语言模型组合路由\research\e18_1_statistics_and_mechanisms'
$publicOut = "$studyRoot\reproduction_public"
$privateOut = "$studyRoot\reproduction_private"

& $pythonPath -B -X utf8 "$codeRoot\analyze_e18_1.py" `
  --rewards 'D:\develop\CV\多臂老虎机\.e18_cpu_runtime\evaluable_bulk_v1\FINAL_POLICY_REWARDS.npz' `
  --e18-results 'E:\语言模型组合路由\research\e18_confirmatory_execution\public\evaluable_confirmation_v1\E18_FINAL_RESULTS.json' `
  --protocol "$codeRoot\E18_1_PROTOCOL.json" `
  --output-root $publicOut --private-root $privateOut

& $pythonPath -B -X utf8 "$codeRoot\render_reports.py" `
  --results "$publicOut\E18_1_COMPLETE_RESULTS.json" --output-root $publicOut
```

程序校验两个完成输入的小文件SHA、策略顺序、问题唯一性、二值奖励、同动作复用一致性、任务支持数及原E18点值。随后使用同一批问题抽样共同计算策略、生成和交互。子任务各自在固定支持数内抽样，完整保存27组。

私有BOOTSTRAP_REPLICATES.npz保留20,000次任务与子任务统计抽样值。公开CSV和JSON保留点值、区间、原p值与纠错破坏。STATISTICAL_REVIEW.json记录独立解析核对和每任务前3个真实抽样的显式整块重建，本轮正式大Bootstrap运行一次。

## 账目复核

`accounting_global_fixed_audit.py`使用原工作区目录结构，工作区根由脚本位置的parents[2]确定，资产根为`E:\语言模型组合路由`。授权复现时将源码放在上述experiments目录结构的副本，并保留对应元数据相对路径。程序以新文件写入聚合结果和私有排除题诊断，原输出已存在时会停止，适合在新的工作区副本复核。

它只从合法train600评分重算20个训练均值，并比较E17/E18输入侧冻结动作。执行成本读取原账本，原始归档payload复查次数为0。训练过程和LLM执行没有启动。

## 图与论文材料

`make_joint_evidence_figure.py`与原字节共享渲染器`shared_scene_renderer.py`放在同目录后可直接执行。输出figure.drawio、figure.svg、figure.png、几何JSON与QA。图数据和CI绑定本次完成结果SHA。图注在figure.caption.md。

本机draw.io CLI缺失，图由同一几何场景生成可编辑XML、SVG与Pillow PNG。原生draw.io导出的视觉一致性尚待具有该软件的环境复核，当前交付经过结构与PNG视觉检查。

06_JOINT_EVIDENCE.md为联合证据分析。07_RESULTS_AND_DISCUSSION.md为中英论文草稿，需要作者审阅。原SPO+、DARS与EquiRouter的一手页面在稿内逐项链接，原理论的分布条件与本任务有限容量保持各自适用边界。

## 推断与成本边界

所有新CI为逐项边际95%近似区间，条件于冻结模型、当前题源混合及三次已记录生成。原题是独立单位，生成seed和100次E17置换保持各自重复测量身份。两个缺失题的确定性范围与Bootstrap区间分列。

TSPO离开组合5的条件子集点差按子集原题平均，主服务差固定任务50/50。熵及HHI与质量为描述性并列，现有四策略数据无法识别分散度的因果作用。

新增LLM调用、训练、模型调参和sealed200访问均为0。原48,531,742推理Token保留其新增与历史成功缓存边界。历史失败全生命周期费用、独占GPU核时和货币费用的既有缺口保持原记录。
