# E18源码与评分流程复现

## 布局与来源

code目录内保留两个同级模块目录。所有.py按冻结原始字节复制。INVENTORY.json记录source SHA和建议发布相对路径。资格摘要按其source_bindings回溯原始代码，外部冻结源明确标为external dependency。

## 已有公开科学源码依赖

调用者需从原MAB仓库/已封存发布提供原budget_router.inference、member_execution、execution、budget_router和budget_router_v2的scoring、typed_aggregation_execution、stage1_review_common、typed_operator_contract及原冻结76-source闭包。该包没有重复拷贝完整历史V2源，运行时原guard继续逐项核验。

还需原pool/ModelSpec科学元数据、E0/source receipt、typed operator lock、28输出合同及相应source manifest。实际机器路径/加载映射不应上传。远端可选原mapped_backend_v1.py，来源SHA为4ea0b3f213e4abe70a388e5f3d86cfe92b86eabb1db4f550ea0c57758febb845，另由用户在受控资产中提供。原本地TransformersBackend也是合法路径。

CPU侧需要原Python、NumPy、SciPy版本，GPU侧按原记录提供PyTorch、Transformers和tokenizer环境、模型revision及生成依赖。当前源包没有新造环境锁，也没有执行清洁环境验证。

## 软件检查

以下命令需使用新的输出位置，禁止覆盖资格历史。SUITE_ROOT为本包code绝对位置。

```powershell
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/qualify_incremental_aggregation_planner.py" --output NEW_PLANNER_QUALIFICATION.json
python -B -X utf8 "$SUITE_ROOT/e18_mechanism_and_confirmation_preparation/qualify_e18_reference_adapter.py" --output NEW_REFERENCE_QUALIFICATION.json
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/qualify_final_report.py" --output NEW_REPORT_QUALIFICATION.json
```

GPU worker与完整评分toy资格还会调用原Store/typed API，需要先存V2源及配置元数据。部分原fixture固定使用已公开的E盘资产默认位置，环境变量接口仅在原源码提供处适用。完整toy使用FakeBackend/FakeTokenizer，不加载真实语言模型。当前打包步骤没有运行这些命令。

## 用户自备私有资产

- 冻结3000输入顺序、canonical与cohort/source-frame元数据、4策略动作和批准scope/approval。
- 原合法六模型权重、聚合器tokenizer字节、source receipts与私有路径映射。
- 原任务plan、资源策略、lease、输出拥有者根与费用围栏。
- 47286必要成员原记录，其中4717为历史所需成功缓存；30921稀疏必要聚合原记录，以及全部已知本轮physical attempts费用。
- 获准3000题reference和显式sealed排除单向metadata。sealed答案不进入评价。
- 首16精确prompt/typed审计、原始source身份、CPU只读transport manifest与完成接纳。
- 头训练重现另需原train600/validation200完整20组合×3重复奖励、输入表示/PCA/缩放器、E17训练代码、配置、种子与初始化。

历史缓存桥接复用cache_audit原函数。缓存原记录、身份和成本仍私有。历史成本只覆盖所需成功记录，不恢复旧失败生命周期。

## 正式评分与报告入口

仅在完整执行接纳通过后，由获准操作者触发。以下是占位命令，包不携带真实manifest。

```powershell
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/e18_final_scoring.py" --manifest FINAL_MANIFEST.json --manifest-sha256 MANIFEST_SHA --preflight-only --native-read-root NATIVE_READ_ROOT --runtime-root CPU_RUNTIME_ROOT
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/e18_final_scoring.py" --manifest FINAL_MANIFEST.json --manifest-sha256 MANIFEST_SHA --execute-scoring --native-read-root NATIVE_READ_ROOT --runtime-root CPU_RUNTIME_ROOT
python -B -X utf8 "$SUITE_ROOT/e18_confirmatory_execution/generate_final_report.py" --results E18_FINAL_RESULTS.json --results-sha256 RESULT_SHA --rewards FINAL_POLICY_REWARDS.npz --rewards-sha256 REWARD_SHA --cost-ledger PHYSICAL_COST_LEDGER.json --cost-ledger-sha256 COST_SHA --input-records ADMITTED3000_INPUTS_ONLY.jsonl --input-records-sha256 INPUT_SHA --output-root NEW_REPORT_DIRECTORY
```

完整门槛为3000原题、4×3×3000策略生成奖励、30921必要聚合、47286必要成员，费用未知或缺失会阻断reference阶段。沿原评分器解析，统计沿已冻结paired_question_test及E18获批例外执行，主比较与五项Holm保持原值。没有重建新3000的完整20组合矩阵。

D盘NTFS小文件运行视图与E盘canonical私有闭合归档保存相同native字节/逻辑scope。runtime_root仅提供工程位置。按题统计将三个生成重复保留在同一原题簇，GSM/BBH各50%；BBH共池246/扩展1254和fresh400/new2600依冻结metadata单列。

## 解释边界

E17编号敏感性与E18独立质量分节。100排列不增加独立题。native wall time与未测独占GPU核时间分别表述。评分阶段protected值/答案解码与准备期获批sealed输入指纹审计分别披露。实际质量、优势和风险依正式完成产物，源码/软件资格不替代结果。
