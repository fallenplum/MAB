# CES审计与缓存迁移复现

## 环境和冻结源

当前运行使用Python3.11、NumPy1.26、SciPy1.11，CPU线程1。原仓库README开发环境为Python3.12，本轮直接导入的数学核心使用兼容的NumPy和标准库。实验根`E:\语言模型组合路由`可用`MAB_ASSET_ROOT`指定。

CES作者代码固定`JinkunXu/CES@9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7`，旧`third_party/ces_source_audit/upstream`保持原字节。核心、projectors、MCQ initializer逐项校验SHA。原MIT版权保留。完整Zenodo ZIP39.5MB及论文PDF、官方Git树和来源元数据仅存private。公开文件以审计与可复现代码为主。

新分支从冻结E7提交`a7bc3d9a72160b3e6134cae76a1a13f47b8b992b`创建。`E7_FREEZE_RECEIPT.json`保存Decision-Aware源代码、E7配置和结论的SHA，旧数字保持原版本。

## 可复现范围

原算子与能力profile聚合算术已经执行。原论文主成绩没有公开逐题轨迹支持，本轮将其标为作者报告。14个profile算术一致、1个异常的事实不等于验证了原始MMLU预测。原样FLASK runner还存在模型profile键缺失。

当前迁移是受控作者代码移植。使用6个本地冻结模型、K3、真组合评分和I240/U240/E120。CES使用额外已计分的I成员先验，不能把全部输入权限写成与E7完全相同。

## 审计与检查

```powershell
$cesPython = 'E:\语言模型组合路由\.venv\Scripts\python.exe'
$cesPublic = 'E:\语言模型组合路由\research\ces_traceable_baseline\public'
& $cesPython -B -X utf8 "$cesPublic\verify_delivery.py"
& $cesPython -B -X utf8 "$cesPublic\native\audit_native_ces.py"
& $cesPython -B -X utf8 "$cesPublic\adapter\audit_ces_operators.py"
& $cesPython -B -X utf8 "$cesPublic\adapter\audit_profile_source.py"
```

`data/`保留原Git/归档/能力profile审计脚本，完整来源响应和925文件的ZIP交叉核查记录保存在private/public_assets。离线复核依赖已下载归档，不需要创建在线API客户端。

## 迁移执行顺序

`freeze_ces.py`先读取train600输入、真实成员score v3，并用ID过滤后只解码问题字符串生成原8维手工通道。原参考答案字段不进入该特征提取器。冻结协议现已存在，重复初始化会拒绝覆盖。

```powershell
& $cesPython -B -X utf8 "$cesPublic\run_migration.py" prepare
& $cesPython -B -X utf8 "$cesPublic\run_migration.py" run --workers 3
& $cesPython -B -X utf8 "$cesPublic\evaluate_migration.py"
& $cesPython -B -X utf8 "$cesPublic\finalize_baseline.py"
```

重算会写新研究目录，原交付可先完整保存。依赖E7的冻结U Random购买序列以及E7/private/c/E7_C_PAIRED.npz对照，不能用其他问题顺序替代。原train600数据、成员评分和全部组合成本均按既有来源校验。外折E只由评价器读取，45条路径全部冻结后才评价。

## 三条轨道

| 轨道 | theta初始化 | U每题反馈 | 三生成成员/聚合调用 |
|---|---|---|---|
| ProfileOnly-SingleCombo | 0，只有I能力profile | 当前CES选择一个组合 | 9/3 |
| HistoryFit-SingleCombo | I240×20均值标签4800次更新 | 当前CES选择一个组合 | 9/3 |
| HistoryFit-Full20 | 同上一行 | 全20组合 | 18/60 |

主选择使用原alpha3.5、ridge.1、K3，均值alpha0只作预先固定诊断。完整I额外需要14400次聚合，能力profile另有4320个成员评分操作，Gold与成员输出复用。所有Token按实际缓存记录，无法以相同Gold声称相同执行费用。

## 适配细节

- 当前冻结L6问题向量替换原多语言L12编码。手工FeatEncoder原式保留，按字符计算。
- 旧15模型的10项MMLU画像由新6模型I区GSM/BBH能力和历史费用3项替换。
- 原SVD initializer只用于求投影方向，应用时直接乘raw profile，保留原实现行为。
- `author_operators`仅执行原函数/类的AST，补齐其NumPy和标准库类型依赖，避免导入在线runner客户端。
- 三重复先取组合均值，一次原update。原参数数值固定，这一观测单位变更已经记录。raw3与mean的正则/alpha换算另有资格测试。
- 核心组合均值为成员phi加和，中心化组合表达秩最多5。真实UCB仍按整个集合方差贪心。

## 私有产物

`private/prepared`包含15组原特征、I-only profile、投影及历史状态。`private/replay`包含45条路径，每条保存前瞻购买决定、已合法执行的原始反馈、逐题账本、四个模型快照和评价前动作。`private/MIGRATION_PAIRED.npz`保留原题级三生成/算法重复与5000次配对统计。

独立审计将非I先验标签、未购买U、所有E及单组合轨道未执行组合的奖励/费用改为NaN，确认行为不变。结果源与数组权限的核验不等于操作系统级文件访问取证。

## 结果解释

主比较为完整历史/完整包CES-UCB对冻结E7 Random Ridge与核岭，Gold60，Holm2。其余轨道、预算和均值服务属于探索性对照。原题600道是统计单位，三个生成及三个算法种子留在题内。全部为旧开发数据和交叠外折历史。

本轮没有发送生成或评判API请求。原协议调用预算、缺失轨迹与最小补充材料见01/02/05。审计PASS表示可追溯数值核验完成，原论文主表复现状态继续明确标记。
