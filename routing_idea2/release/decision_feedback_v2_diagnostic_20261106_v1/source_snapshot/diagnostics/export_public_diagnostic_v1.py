"""Export aggregate-only diagnostics; never copy row arrays or machine locations."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

ROOT=Path(r"E:\语言模型组合路由")
FF=ROOT/"artifacts/private/routing_redesign_20261104/ff_v1"
OUT=FF/"diagnostics_safe_pause_20261106_v1/selector_generalization"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
    with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def read(p):return json.loads(p.read_text(encoding="utf-8"))

private=OUT/"SELECTOR_GENERALIZATION_DIAGNOSTIC_v1.json"
original_sha=sha(private)
v=read(private)
public={
    "schema":"routing_v2_aggregate_only_selector_diagnostic_v1",
    "status":"EXPLORATORY_TRAIN_VALIDATION_DIAGNOSTIC_ONLY",
    "scientific_learnability_established":False,
    "formal_stage2_complete":False,
    "independent_test_confirmation":False,
    "counting_units":{
        "training_unique_questions":600,"validation_unique_questions":200,
        "generation_replicates_per_question":3,
        "training_paired_generation_question_instances":1800,
        "validation_paired_generation_question_instances":600,
        "validation_independent_sample_unit":"question; not generation-question instance",
        "all_generation_uses":"Generation seeds 42, 137, 2024 are repeated outcomes on the same questions, not independent new question sets."
    },
    "methods":{
        "selector":"Frozen shared second-order structured ridge on training PCA16/scaling plus constant (17-dimensional context), crossed with constant, six membership indicators and fifteen member-pair products; ridge=1. No explicit task/subtask one-hot. Not twenty independent heads.",
        "global_fixed":"Training-selected single combination with highest overall mean executed aggregate accuracy, used for every question.",
        "task_fixed":"Training-selected combination per task (GSM8K or BBH).",
        "subtask_fixed":"Training-selected combination per BBH subtask; GSM8K uses task-level combination.",
        "independent_top3":"Six independent ridge member-correctness predictors using the same frozen 17-dimensional question transform; select three highest predicted members, then look up the corresponding executed frozen aggregator result. Designed after original validation was observed; exploratory."
    },
    "metric_definitions":{
        "accuracy":"Mean exact aggregate correctness, macro-averaged across two equally represented tasks, with three generation repeats per question.",
        "top1_regret":"Mean of maximum observed binary correctness across all twenty executed combinations minus chosen combination correctness, over the same question and generation.",
        "wins_losses_ties_instances":"Paired method-quality differences over question x generation (validation: 200 x 3 = 600 instances).",
        "wins_losses_ties_questions":"Compare each method after averaging its three generation outcomes within each unique question (validation support: 200 questions).",
        "confidence_interval":"Original 2,000 bootstrap replicates, seed 20261104, sampling question clusters within task/BBH-subtask strata and preserving all three repeats; conditional on fitted model and observed generations; descriptive contrasts without multiplicity adjustment."
    },
    "train":v["train"],"validation":v["validation"],
    "train_minus_validation_accuracy":v["train_minus_validation_accuracy"],
    "model_structure":v["model_structure"],
    "floating_point_replay":v["floating_point_replay"],
    "interpretation":{
        "confirmed":"On validation, the selector has no stable gain over global/task/subtask baselines or exploratory independent Top3. Both tasks have negative point differences versus subtask baseline.",
        "relative_to_subtask":"The selector is already 1.4444 percentage points below subtask baseline in training and 1.3333 points below it in validation. This weakens an explanation based solely on overfitting. It does not deny the selector's own 3.5556-point train/validation accuracy gap.",
        "relative_to_global_and_task":"In-sample gains over global/task fixed baselines do not carry to validation; this is not confirmation of unseen-question routing.",
        "representation":"PCA16 retains 44.4222% of training-centered embedding squared norm and 41.5387% on validation centered by the training mean; this is not a measurement of retained task-relevant information. The fixed-K second-order combination basis has rank 15 within twenty combination values, but the missing-span component accounts for only 10.0298% of current validation within-question centered mean-score MSE; it is not evidence that higher-order interactions are the principal failure cause.",
        "sample_support":"Only 11-12 training and 3-4 validation questions per BBH subtask. Larger numbers of executed combinations/generations do not create additional independent questions.",
        "confidence":"High for computed counts/metrics and unchanged-byte checks; limited for causal explanations. No model, feature, or hyperparameter intervention was performed."
    },
    "limitations":v["limitations"],
    "cost":{
        "new_llm_calls":0,"new_answer_scoring_operations":0,"new_model_fits":0,"hyperparameter_search":False,
        "fresh_or_sealed_outcome_files_opened":0,
        "completed_diagnostic_cpu_s":v["runtime"]["cpu_s"],"completed_diagnostic_wall_s":v["runtime"]["wall_s"],
        "timing_scope":"Completed main routine only; import, first failed exact-bit replay assertion, inspection, and export overhead are not included.",
        "prior_pure_CPU_attempt_failed":True,
        "prior_attempt_reason":"Strict bitwise reconstructed-score assertion failed; maximum original selector difference was 8.4633e-8 and all 200 argmax selections were unchanged. Failed attempt retained. Validation metrics use original frozen score arrays."
    },
    "integrity":{
        "private_report_sha256":original_sha,
        "protected_inputs_unchanged":True,
        "source_and_asset_hashes":[{"asset_name":Path(r["path"]).name,"sha256":r["sha256"]} for r in v["source_lineage"]],
        "diagnostic_source_sha256":v["source_script"]["sha256"],
        "contains_row_level_outputs":False,"contains_question_or_canonical_identifiers":False,
        "contains_raw_questions_answers_gold_or_model_weights":False,
        "contains_local_or_remote_machine_locations":False
    }
}
public_path=OUT/"PUBLIC_SELECTOR_DIAGNOSTIC_v1.json"
brief="""# 已有训练与验证数据诊断：逐题组合选择尚未证实泛化

**当前不能把训练完成或代码通过解释为科学假设成立。** 本次仅回放冻结模型和已有评分矩阵：新增 LLM 调用、答案评分、模型训练均为0，未读取新400题或封存题答案。

## 样本与统计口径

训练集600道不同题；验证集200道不同题。每题有3个生成种子，因此验证比较包含 **200×3=600条配对实例**，独立题目数仍是200。置信区间以题目为簇保留三个重复，不把600条实例当作600道不同题。

## 训练和验证表现

| 方法 | 训练准确率 | 验证准确率 | 验证 Top-1 Regret |
|---|---:|---:|---:|
| 逐题 Selector | 89.2222% | 85.6667% | 0.105000 |
| 最佳固定组合 | 86.3889% | 86.0000% | 0.101667 |
| 任务级固定路由 | 87.1111% | 86.1667% | 0.100000 |
| BBH子任务级路由 | 90.6667% | 87.0000% | 0.091667 |
| 独立 Top-3（事后探索性） | 86.9444% | 86.3333% | 0.098333 |

Selector相对子任务路由在验证集低1.3333个百分点，95%题簇区间为[-3.6667,+1.0000]个百分点，3个生成种子均为负。它在200题中赢12题、输16题、平172题（按题内3种子平均）；对应600条配对实例赢18、输26、平556。相对其他对照的区间也跨0，没有稳定增益。区间跨0不能反向证明研究方向无效。

## 诊断指向与限制

- **不能只归因于过拟合。** Selector训练集已经比子任务路由低1.4444个百分点。不过，Selector自身仍存在3.5556个百分点的训练/验证准确率差；训练中超过全局和任务级基线的优势没有在验证中保留。
- **原模型有明确结构约束。** 它是17维题目特征与二阶组合基底构成的共享岭回归，名义374参数；不是20个独立组合预测头。固定K=3时组合基底秩为15，训练设计秩为255。
- **结构或PCA尚未被证实是主因。** PCA16对按训练均值中心化的嵌入平方范数覆盖为训练44.42%、验证41.54%，这不能说明损失了多少路由相关信息。二阶基底未覆盖方向的描述性残差仅占当前验证组合差异MSE的10.03%，不能直接推出必须换高阶模型。
- **每个BBH子任务样本很少。** 训练11–12题、验证3–4题；局部胜负和区间不足以支持稳定的子任务结论。36,000次训练组合聚合不是36,000道独立训练题。
- **排序信号变弱。** 对组合结果有差异的题，Selector题均两两排序准确率在训练为64.36%–67.23%，验证为53.11%–59.24%。这是现有冻结模型的表现，不是对所有逐题路由算法的否定。

**优势：** 零新增LLM调用，保留全部负结果，可定位样本内拟合与泛化的差别。**风险：** 诊断发生在已看过验证结果之后，仅支持开发分析；任何新特征或新选择器都需另立方案，再使用未见题确认，不能沿用本报告作为独立测试证据。

全部聚合统计、27个BBH子任务支持数、方法定义与SHA见 `PUBLIC_SELECTOR_DIAGNOSTIC_v1.json`。该公共文件不包含逐题输出、题目标识、机器地址、原始问答、标准答案或权重。
"""
index="""# 文件索引

| 文件 | 用途 | 发布建议 |
|---|---|---|
| PUBLIC_SELECTOR_DIAGNOSTIC_v1.json | 仅含训练/验证聚合统计、方法定义、支持数、限制和源SHA | 可进入GitHub候选白名单 |
| PUBLIC_DIAGNOSTIC_BRIEF_zh.md | 向研究协作者解释结论、口径和风险 | 可进入GitHub候选白名单 |
| PUBLIC_FILE_INDEX_zh.md | 公共报告文件索引 | 可进入GitHub候选白名单 |
| PUBLIC_DIAGNOSTIC_SANITIZATION_RECEIPT_v1.json | 公共文件SHA及题目标识/地址排除校验 | 可进入GitHub候选白名单 |
| SELECTOR_GENERALIZATION_DIAGNOSTIC_v1.json | 完整私有证据链与统计，含本机资产路径 | 本地保留，不直接发布 |
| EXISTING_MATRIX_SELECTED_OUTCOMES_DIAGNOSTIC_v1.npz | 行顺序对应私有输入的逐题选择与结果，用于复查；无显式qid/canonical字段 | 本地保留，不发布 |
| DIAGNOSTIC_README_zh.md | 完整私有诊断简报 | 本地保留 |
| run_frozen_selector_diagnostic_v1.py | 第一次CPU诊断源，因逐bit回放断言停止 | 保留失败记录，不发布 |
| DIAGNOSTIC_ATTEMPT_v1_FAILURE.json | 第一次CPU诊断失败原因及零新调用证据 | 本地保留 |
| run_frozen_selector_diagnostic_v2.py | 成功诊断源，使用原冻结验证分数并明示回放浮点差异；含本机路径 | 本地保留，不直接发布 |
| DELIVERY_v1.json | 完整私有诊断交付的SHA索引 | 本地保留 |
"""
brief_path=OUT/"PUBLIC_DIAGNOSTIC_BRIEF_zh.md"
index_path=OUT/"PUBLIC_FILE_INDEX_zh.md"
texts=[json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False),brief,index]
joined="\n".join(texts)
assert not re.search(r"[A-Za-z]:[\\/]",joined),"Machine drive location leaked"
assert not re.search(r"/(?:data\d?|home|Users)/",joined),"Remote/local path leaked"
assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b",joined),"IPv4 address leaked"
identifiers=[]
inputs=ROOT/"artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1"
for name in ["selector_train_inputs.npz","selector_validation_inputs.npz"]:
    with np.load(inputs/name,allow_pickle=False) as z:
        identifiers.extend(str(x) for k in ["question_ids","canonical_hashes"] for x in z[k])
assert not any(x in joined for x in identifiers),"Training/validation question identifier leaked"
put(public_path,public)
with brief_path.open("x",encoding="utf-8") as f:f.write(brief)
with index_path.open("x",encoding="utf-8") as f:f.write(index)
assert sha(private)==original_sha
receipt={"schema":"aggregate_only_public_diagnostic_sanitization_v1",
         "public_files":[{"name":p.name,"sha256":sha(p),"bytes":p.stat().st_size} for p in [public_path,brief_path,index_path]],
         "source_private_report_sha256":original_sha,"private_report_unchanged":True,
         "training_validation_identifiers_checked":len(identifiers),"identifier_hits":0,
         "machine_location_hits":0,"ip_address_hits":0,"private_row_arrays_published":False,
         "question_standard_answers_or_raw_outputs_published":False,"new_llm_calls":0,
         "scan_scope":"Whitelisted three public text files; exact comparison against all 800 training/validation qids and 800 canonical hashes, machine path and IPv4 pattern checks. No fresh outcome or answer opened."}
put(OUT/"PUBLIC_DIAGNOSTIC_SANITIZATION_RECEIPT_v1.json",receipt)
print(json.dumps(receipt,ensure_ascii=False))
