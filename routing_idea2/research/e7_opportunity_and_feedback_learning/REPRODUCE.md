# E7复现说明

## 资产与环境

执行环境为Python3.11、NumPy1.26、SciPy1.11、scikit-learn1.7，CPU线程固定1。默认根目录为`E:\语言模型组合路由`，`MAB_ASSET_ROOT`可覆盖。输入为既有train600 MiniLM向量、V2真实20组合×3生成评分和费用，加载器继续验证原bundle、成员池、聚合器、评分器及题目顺序。

E7继承E6提交`0c34a38bede9fcf46359316058759b27fa9f5fea`。原E3/E4/E5代码是数学/统计依赖，旧文件保留。公开库包含代码与脱敏聚合结果。本机private含原题索引、奖励、参数、反馈、快照和成本账本，完整回放需要这些历史资产。

## 冻结与权限

`prepare_e7.py`只读train600输入和旧外折索引，生成`private/E7_SPLITS_FROZEN.npz`及`public/E7_PROTOCOL.json`。每折键为`f{f}_H,I,U,E,refs,weights`。三个生成重复始终共享题目角色。I240和U240按ID哈希与语义分层固定，U按GSM/BBH交错形成嵌套子集。旧默认与新I默认分别标注。

`private/PROTOCOL_FROZEN.json`保存首次评价前的协议和划分哈希。已有协议拒绝覆盖。完整原运行记录保留，重新计算会更新指定E7输出，应在保留原产物的资产副本或明确的重算目录执行。共享输入仍需满足加载器的冻结路径与哈希校验。

## 检查现有交付

```powershell
$e7Python = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$e7Public = 'E:\语言模型组合路由\research\e7_opportunity_and_feedback_learning\public'
& $e7Python -B -X utf8 "$e7Public\verify_e7_delivery.py"
```

## 按原顺序重放

先阅读`RELATED_WORK_MAPPING.md`与`E7_PROTOCOL.json`。冻结已存在时直接使用其划分。

```powershell
& $e7Python -B -X utf8 "$e7Public\a\qualify_opportunity.py"
& $e7Python -B -X utf8 "$e7Public\a\run_opportunity.py" freeze
& $e7Python -B -X utf8 "$e7Public\a\run_opportunity.py" evaluate
& $e7Python -B -X utf8 "$e7Public\a\audit_opportunity.py"
& $e7Python -B -X utf8 "$e7Public\b\run_learning_curves.py" prepare
& $e7Python -B -X utf8 "$e7Public\b\run_learning_curves.py" run
& $e7Python -B -X utf8 "$e7Public\b\summarize_learning_curves.py"
& $e7Python -B -X utf8 "$e7Public\b\audit_learning_curves.py"
& $e7Python -B -X utf8 "$e7Public\c\active_feedback.py" qualify
& $e7Python -B -X utf8 "$e7Public\c\active_feedback.py" run --workers 3
& $e7Python -B -X utf8 "$e7Public\c\evaluate_feedback.py"
& $e7Python -B -X utf8 "$e7Public\c\make_report.py"
& $e7Python -B -X utf8 "$e7Public\finalize_e7.py"
```

A冻结全部候选集合后评价。B训练和评价分离，P3每折λ仅在I中按题三折选择，之后带宽/均值/λ保持固定。P3的20次语义内置乱种子为100—119，I和合法U前缀分别整块置乱。

C先做外折0的10包资格检查，再运行2学习器×3采集×5折×3算法种子，共90条正式路径。所有路径在E评分评价前冻结，0/20/40/60各保存模型快照。三生成先取题均值进入更新，评价时三个原始生成成绩仍完整保留。已完成的相同源路径由收据直接复用。

## 后验与单位

P1为旧直接20组合PCA16 Ridge，保持带惩罚常数。raw三个重复的alpha100等价于均值观测alpha100/3，工作GP核为XX'/100、题均值噪声1/3。完整MiniLM线性P2使用同单位。P3工作核为RBF，题均值噪声为I内选出的λ，均值偏置固定为I组合平均奖励。

完整题包提供20个三重复均值，共享标量输入后验与独立20输出工作假设。IG为10log(1+variance/noise)。DecisionAware采用同一模型下参考输入最大预测价值的期望改善，每步32对反向Gaussian幻想，20维创新在全部参考输入间共享。未对幻想奖励剪裁为正确概率。

## 私有产物

| 路径 | 内容 |
|---|---|
| `private/E7_SPLITS_FROZEN.npz` | I/U/E及参考输入原题索引 |
| `private/a` | 全部嵌套候选、200随机集合、留生成选择和5000次统计 |
| `private/b/SPEC_f*_P*.npz` | 冻结特征、核、噪声、均值和I内参数选择 |
| `private/b/FROZEN_B_f*.npz/.json` | 主模型和20置乱的评价前预测、内部CV及源映射 |
| `private/b/PRIMARY_B_COMPARISONS.npz` | 两个B主比较的题级贡献及bootstrap |
| `private/c/f*_*/PROSPECTIVE_CHOICES.json` | 每次揭示前的购买决定和预测量 |
| `private/c/f*_*/ACCESS_AND_COST_LOG.json` | 合法访问、历史估价和购买后的实际成本 |
| `private/c/f*_*/PATH_RECORDS.npz` | 购买ID、原始反馈、所有候选分数、模型预测和计时 |
| `private/c/f*_*/MODEL_B*.npz` | 四个快照，冻结Spec加rows/received_means可重构完整后验 |
| `private/c/ALL_PATHS_FROZEN.json` | 90条路径全部冻结收据 |
| `private/c/E7_C_PAIRED.npz` | 600道E题的分算法/生成成绩和题级bootstrap |
| `private/audit_replays` | 未购U/E标签及费用NaN扰动等独立核验 |

## 统计与成本

四主比较为P3增加独立训练题、P3原始对任务内置乱、P3 Gold60的DA对Random及IG。统一Holm4，另给95%和Bonferroni98.75%题级配对区间。算法种子、生成重复和置乱数不增加独立题数。区间条件于保存的预测，外折历史有重叠，全部结果属于既有开发数据。

一次完整新题包为1 Gold、18成员调用、60聚合、60评分。历史I240单列，实际Token按购买后源记录收费。相同Gold与调用数不等于相同Token。原始源缓存只计一次，本轮物理新增LLM调用0。CPU按process_time记录，Windows计时粒度可能使短操作显示0，路径wall-time另存。

## 实施细节

参考集合为I内27道GSM及27个BBH子任务各1题。权重匹配I的GSM/BBH各半和BBH题级频率。64次Gaussian幻想每步、路径种子固定。I上的参考是未来决策价值的工作代理，未建立对E分布的保证。

评价器在任何结果保存前修正了三维预测调用旧二维argmax的轴问题，改用共享模型的末轴argmax。购买序列和模型快照没有因此改变。所有最终预测到真实奖励的映射均经过独立重建。
