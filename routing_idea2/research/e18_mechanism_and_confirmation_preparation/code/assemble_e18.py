"""Create the six E18 preparation reports without evaluating new rewards."""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import shutil
import time

from e18_common import (ASSETS, BRANCH, CODE, E17, PARENT, PRIVATE, PUBLIC,
    POLICIES, SEEDS, preserve_e17, read_json, sha256, write_json)
from power_and_statistics import power
import numpy as np


def md(name, text):
    with (PUBLIC / name).open("x", encoding="utf-8") as stream:
        stream.write(text.rstrip() + "\n")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |"] +
        ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def copy_file(name, destination=None):
    shutil.copy2(CODE / name, PUBLIC / (destination or name))


def main():
    start = time.perf_counter()
    old_integrity = preserve_e17()
    budget = read_json(PUBLIC / "E18_FINAL_CALL_BUDGET.json")
    planning = read_json(PUBLIC / "E18_POWER_ANALYSIS.json")
    data = read_json(PUBLIC / "DATA_AUDIT.json")
    closure = read_json(PUBLIC / "HISTORICAL_INPUT_REGISTRY_CLOSURE.json")
    source_qualified = read_json(PUBLIC / "DATA_SOURCE_ALIGNMENT_AND_FINAL_VERIFICATION.json")
    repaired = read_json(PUBLIC / "INPUT_METADATA_REPAIR_v2.json")
    model = read_json(PUBLIC / "FROZEN_MODEL_QUALIFICATION_v3.json")
    member_mirror = read_json(PUBLIC / "MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json")
    adapter = read_json(CODE / "EXECUTION_ADAPTER_READINESS.json")
    dag = read_json(PUBLIC / "PROPOSED_EXECUTION_DAG_QUALIFICATION.json")
    if data["status"] != "PROPOSED_NOT_ADMITTED_INPUT_SOURCE_AUDIT_COMPLETE":
        raise ValueError("Candidate input audit has not completed")
    if closure["status"] != "NO_OUTSIDE_REGISTERED_INPUT_FOUND_IN_REVIEWED_CANONICAL_METADATA":
        raise ValueError("Historical input scope requires review")
    if not source_qualified["status"].startswith("PASS") or repaired["status"] != "PASS":
        raise ValueError("Input identity or metadata qualification failed")
    if member_mirror["postcheck"] != "PASS_ALL_EXPECTED_PRIVATE_MIRRORS":
        raise ValueError("Cache materialization has not passed its independent check")
    if dag["status"] != "PASS_APPROVAL_REQUIRED":
        raise ValueError("Review-only dependency graph has not passed qualification")
    old_results = read_json(E17 / "public/PERMUTATION_SUMMARIES.json")
    all_groups = {(row["method"], row["inference"]):row for row in old_results["groups"] if row["group"] == "all"}
    rows = []
    for method in ("Ridge", "SPO_coordinate", "TSPO_uniform"):
        row = all_groups[method, "semantic_exact"]
        rows.append(dict(method=method, inference="semantic_exact", original_questions=600,
            identity_accuracy_percent=row["identity_accuracy"],
            mean_accuracy_across100_percent=row["accuracy_across100"]["mean"],
            min_accuracy_percent=row["accuracy_across100"]["minimum"],
            max_accuracy_percent=row["accuracy_across100"]["maximum"],
            accuracy_variance_pp_squared=row["accuracy_across100"]["variance"],
            physical_choice_consistency_percent=100*row["identity_consistency_across100"]["mean"]))
    with (PUBLIC / "E17_CORE_RESULTS_TABLE.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    core_table = table(("冻结方法", "原编号准确率%", "100排列范围%", "方差pp²", "实际身份一致率%"),
        [(row["method"], f"{row['identity_accuracy_percent']:.3f}",
          f"{row['min_accuracy_percent']:.3f}—{row['max_accuracy_percent']:.3f}",
          f"{row['accuracy_variance_pp_squared']:.6f}", f"{row['physical_choice_consistency_percent']:.3f}") for row in rows])
    copy_file("A_THEORY_AND_PROOFS_ZH.md")
    copy_file("A_ACL_EXPERIMENT_STRUCTURE.md")
    copy_file("A_RELATED_WORK_AND_LIMITATIONS.md")
    copy_file("A_SOURCE_MANIFEST.json")
    copy_file("A_THEORY_QUALIFICATION.json")
    copy_file("OPERATOR_IDENTITY_QUALIFICATION.json")
    copy_file("STATISTICAL_INDEPENDENT_QUALIFICATION.json")
    copy_file("EXECUTION_ADAPTER_READINESS.md")
    copy_file("EXECUTION_ADAPTER_READINESS.json")
    if (CODE / "SPARSE_RESULT_PROJECTION_QUALIFICATION.json").exists():
        copy_file("SPARSE_RESULT_PROJECTION_QUALIFICATION.json")
    figures = PUBLIC / "figures"
    figures.mkdir(exist_ok=True)
    for path in (CODE / "figures").iterdir():
        if path.is_file():
            shutil.copy2(path, figures / path.name)
    qa_path = CODE / "MECHANISM_DIAGRAM_QA.json"
    if qa_path.exists():
        copy_file("MECHANISM_DIAGRAM_QA.json")
    write_json(PUBLIC / "ROOT_FIGURE_REVIEW.json", dict(status="PASS", reviewer="root",
        reviewed_image_sha256=sha256(figures / "e18_mechanism_audit.png"),
        readable=True, overlaps_observed=False, clipped_labels_observed=False,
        operator_and_data_scope_checked=True,
        renderer="shared geometry to native Drawio XML, standalone SVG and Pillow PNG; native draw.io CLI unavailable"))
    md("E18_MECHANISM_THEORY_REPORT.md", f"""# E18机制理论报告

## 研究问题

工作题目沿用 *When Ties Become Preferences: Auditing Supervision for LLM Ensemble Routing*。本轮将任意冠军约定、训练/推理等变性、奖励估计与代理目标、独立新题质量分别绑定到原始证据。

## E17核心结果

{core_table}

以上是旧600题、1800条生成事件的开发证据。100个排列没有增加独立题量。原SPO+平均80.09%的问题选择当前数字编号0。TSPO+在397/600题上最高两分差不超过1e-8，数字近并列推理的身份一致率降至37.308%。同8draw下LossThenMean为86.000%，MeanThenLoss为85.111%，后者净少16条生成事件。

## 数学与文献产物

完整六项命题及证明见A_THEORY_AND_PROOFS_ZH.md，内容包括任意并列冠军损失差、次梯度集合平移、同一步SGD对未来分数的内积影响、动作重命名条件、置换对称与Bayes一致性、纯全平局且全参数正L2下的唯一零模型。702项纯合成资格仅核对实现与推导。

DARS、EquiRouter与原始SPO+的逐项差异见A_RELATED_WORK_AND_LIMITATIONS.md。两份作者源码固定到明确commit，全文位置与内容边界见A_SOURCE_MANIFEST.json。已有监督稳定性、并列偏好及SPO+理论得到明确归属，本轮没有新增损失函数。[DARS](https://aclanthology.org/2026.emnlp-main.333/)，[EquiRouter v2](https://arxiv.org/html/2602.03478v2)，[原始SPO+](https://arxiv.org/html/1710.08005v5)

## 机制图与ACL组织

![监督失真机制](figures/e18_mechanism_audit.svg)

图中R表示工作后验样本；准确率由真实V2聚合评分计算。可编辑源为figures/e18_mechanism_audit.drawio，PNG为本地核验预览。主机缺少draw.io CLI，使用同一几何的本地SVG/Pillow渲染回退，结构与root视觉核验通过。

可直接用于论文实验组织的摘要、章节、三项主表、反例及附录清单见A_ACL_EXPERIMENT_STRUCTURE.md。未来确认需要的作用域接口资格在独立准备报告中列明。

## 优势、局限与下一步

优势是同数据重编号和同draw算子对照能识别具体机制，证明与实际服务负结果共同保留。局限是旧题反复开发、单一六模型池和聚合器、近并列与工作后验失配。机制证据强度为高，新题质量差异和广泛适用性需要后续确认。
""")
    md("E18_DATA_SOURCE_AND_DEDUP_REPORT.md", f"""# E18数据来源与去重报告

## 研究问题与来源角色

现有fresh400保持原身份，拟议新增2600题，GSM与BBH各1300。GSM沿官方train，BBH沿官方test。原sealed200的GSM来自官方test，该来源差异继续记录。全部新输入保持PROPOSED_NOT_ADMITTED，自动入组为0。

原注册1308题加fresh400形成1708个排除身份，覆盖train600、选择器validation200、历史聚合诊断200、选择校准100、资格8、sealed200和fresh400。原问题ID、规范化文本hash及预声明近重复规则分别核验。

## 实际审计结果

官方GSM train7473题、test1319题。官方BBH27子任务6511条，旧共同子池1078题。完成ID/canonical对齐、4个精确重复input簇、52条近重复边及保护签名筛查。边界待审的83个端点从候选中排除。数值、实体与运算符保留，模板不同实例不机械合并，近似关系不取传递闭包。

原始隔离投影为13876题，补回1个已注册重复源题作为仅去重参考后，参考池为13877题。追加补审覆盖1508个非保护注册题与200个保护题签名，候选内容、顺序、SHA均未改变。可用GSM6719、BBH5587。历史闭包另核对26研究模块、47份协议来源与50个身份/索引容器，注册外ID与canonical均为0；仅scope/父链的两模块及未审阅的大型回放、外部原生基线数据、远端副本、人工未登记审阅分别注明。

## 来源扩展边界

| 拟议新增来源 | 题数 |
| --- | --- |
| GSM官方train原来源 | 1300 |
| BBH旧共同子池 | 46 |
| 官方BBH完整题池覆盖扩展 | 1254 |

旧BBH子池按本轮规则仅余215题。保持fresh400及50/50后，原子池可支持的平衡规模上界约830题，达不到3000。当前3000题方案因此含明确的覆盖扩展，不能自动解释成原子池同分布确认。原来源与扩展层的输入清单、来源标签和未来结果表分别保存。是否接受扩展总体属于待审批内容。

## 封存例外与权限

依据用户2026年11月16日追加授权，一次性隔离程序仅处理200个保护题输入，生成私有单向HMAC签名和排除行号索引。保护题原文未展示或另存，答案未解析，向量未编码，模型未训练。正常候选构建及补审没有再次解码保护输入。fresh奖励和全部新题标准答案均未读取。

## 资格、局限与下一步

DATA_AUDIT.json、DATA_AUDIT_REGISTERED_REFERENCE_SUPPLEMENT.json、DATA_SOURCE_ALIGNMENT_AND_FINAL_VERIFICATION.json及HISTORICAL_INPUT_REGISTRY_CLOSURE.json共同构成当前审计链。候选SHA为{budget['inputs']['proposal_sha256']}。

优势是来源、身份、文本与保护签名分层核验，并保留首次漏入仅去重参考题的修正记录。局限包括语义改写、模板相关性及语言模型预训练暴露，去重程序没有证明统计独立性或基准未被预训练使用。实际评价前应批准目标总体与来源扩展解释。
""")
    variance = planning["source_variances"]
    power_rows = []
    for n in (1200,2400,3000,3700,4800,6000,10000):
        values = {row["absolute_effect_pp"]:row for row in planning["grid"] if row["n_questions"] == n}
        power_rows.append((n, f"{100*values[.5]['power']:.2f}%", f"{100*values[1.0]['power']:.2f}%",
                           f"{100*values[1.0]['variance_x2_power']:.2f}%"))
    old_power = read_json(E17 / "public/CONFIRMATION_POWER.json")["TSPO_vs_SPO"]
    observed_magnitude = abs(old_power["observed_task_balanced_difference_pp"])
    sensitivity = dict(status="PLANNING_SENSITIVITY_ONLY", observed_development_magnitude_pp=observed_magnitude,
        observed_magnitude_is_not_a_true_future_effect=True,
        power_at_3000_for_observed_magnitude=power(3000,observed_magnitude,variance),
        original_frame_max_balanced_n=830, original_frame_upper_n_power_1pp=power(830,1,variance))
    write_json(PUBLIC / "E18_POWER_SENSITIVITY.json", sensitivity)
    md("E18_STATISTICAL_POWER_REPORT.md", f"""# E18统计功效与配对推断计划

## 研究问题与主要比较

唯一主要比较为TSPO_uniform减SPO_coordinate，单位为最终聚合准确率百分点。原题是独立单位，三个成员生成seed先在题内平均，GSM与BBH各权重50%，BBH内部按题平均。理论中的条件均值目标与当前固定三seed服务协议的有限重复评价分别说明。

未来固定评价时点采用双侧5%检验及95%题级配对区间。统计量使用两个任务配对均值的等权和，方差为两任务题级样本方差按0.25/n相加，Satterthwaite自由度。零样本方差返回需审查及宽区间，不给零宽证书。方法依赖独立原题和大样本工作近似。

## 功效数值

{table(("独立题数", "0.5pp功效", "1pp功效", "1pp且方差翻倍"), power_rows)}

1个百分点、80%功效的平衡题数为2796，90%为3742。0.5个百分点、80%需11172题。推荐3000题的1pp工作功效为82.71%。E17原点差幅度为{observed_magnitude:.3f}pp，单纯将该探索性点差代入3000题，工作功效约{100*sensitivity['power_at_3000_for_observed_magnitude']:.2f}%，它没有被当作真实未来效应。

## 预定推断与停止规则

主要零假设为期望差等于0，对立假设为非零，双侧显著性既可能表示改善也可能表示损害。Ridge与GlobalFixed是次要对照，五项预声明次要比较统一Holm，普通95%区间与调整后p分开标注。任务、27个子任务、生成seed、来源层及纠错/破坏分解保留支持量，探索性分解不替代主比较。

N=1200/2400/3700等仅用于准备功效表，不是反复查看结果的中期停止点。实际N与目标总体在gold读取前一次性批准并锁定，不能根据已揭示成绩追加题目、选seed或更改策略。未来实际评价尚未开始。

## 条件、局限与权限

方差来源于E17旧600题的480题外折头，未来部署加载full600头。官方BBH覆盖扩展、来源比例、题型难度及模板关联均可能改变方差，表中方差翻倍时3000题功效降至53.70%。基于当前插件方差的82.71%属于条件性规划。

本轮明确授权覆盖E18未来独立评价的双侧检验与题级配对区间。原2026年11月14日的历史无区间约定及E17结果保持原冻结版本，本轮没有补算E17真实区间或读取fresh成绩。软件资格中仅使用合成数组检查统计公式。
""")
    protocol = dict(schema="E18_frozen_confirmation_preparation_v1", branch=BRANCH, parent_commit=PARENT,
        status="FROZEN_PREPARATION_AWAITING_TARGET_BUDGET_APPROVAL_AND_SCOPE_ADAPTER",
        client_date="2026-11-16", models=POLICIES, model_asset_sha256=model["models_sha256"],
        pca_scaler="load exact full600 E17 state; no fit", representation="frozen MiniLM384 to PCA16+StandardScaler+penalized constant",
        original_questions=dict(existing_fresh=400, proposed_new=2600, proposed_total=3000, GSM8K=1500, BBH=1500),
        dataset_admission="PROPOSED_NOT_ADMITTED", automatically_admitted_new_questions=0,
        source_target="GSM official train + BBH official test; BBH source expansion1254 separately tagged and requires approval",
        primary_comparison="TSPO_uniform minus SPO_coordinate", primary_alpha=.05, alternative="two-sided",
        confidence_interval="fixed-time95% task-stratified paired t interval with Satterthwaite df, original-question unit",
        primary_effect_for_planning_pp=1, source_variance_sha256=planning["source_sha256"],
        secondary_comparisons=planning["secondary_comparisons"], secondary_multiplicity=planning["secondary_multiplicity"],
        member_generation_seeds=SEEDS, member_sample_index=0, sampling_control=False, aggregation_invocation_seed=42,
        active_operator=model["active_operator"], service_tie="semantic_exact using immutable physical-team identity",
        source_frames_separate_reporting=True, no_implicit_original_frame_3000_claim=True,
        fixed_horizon=True, sequential_peeking=False, optional_stopping=False,
        near_tolerance_primary=0, near_tie_diagnostics="same predeclared1e-8 score sensitivity; no model or action revision",
        source_inputs_sha256=budget["inputs"]["proposal_sha256"],
        corrected_input_archive_sha256=budget["inputs"]["corrected_feature_inputs_sha256"],
        frozen_actions_sha256=budget["inputs"]["frozen_choices_sha256"],
        proposed_DAG_qualification_sha256=sha256(PUBLIC / "PROPOSED_EXECUTION_DAG_QUALIFICATION.json"),
        exact_call_budget_sha256=sha256(PUBLIC / "E18_FINAL_CALL_BUDGET.json"),
        head_fits=0, pca_refits=0, scaler_refits=0, new_llm_generation_calls=0,
        input_encoding=dict(questions=2600, batches=325, batch_size=8, device="cpu"),
        gold_reveal_rule="canonical_question plus original source/reference revision once, reused across executed outputs",
        model_judge_calls=0, score_operations=budget["deterministic_score_operations"],
        failure_rule="original cost journal and stop-on-paid-failure retained; unknown-cost attempts block unaccounted repetition",
        automatic_paid_retries=0, manual_paid_resume_total_cap=None,
        authorization_required_before_actual_generation_or_scoring=True,
        protected_input_exception=dict(authorized_date="2026-11-16", original_inputs=200,
            purpose="once-only isolated exclusion fingerprints", display=False, raw_persistence=False,
            answer_reading=False, encoding=False, model_training=False),
        fresh400_reward_reads=0, sealed_answer_reads=0,
        scope_adapter_readiness=adapter,
        old_worker_reuse="scientific lower-level factory/backend may be reused; old400/full20 worker and scorer entry points cannot be launched unchanged",
        cache_materialization_postcheck_sha256=sha256(PUBLIC / "MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json"),
        approval_topics=["Accept and name the expanded BBH target population with separate source tables",
                         "Fix N before any reward, with one-sided historical plans superseded for E18 only",
                         "Approve missing-call budget and any separately accounted manual resume",
                         "Complete the E18-specific native scope admission and sparse result adapter before the first paid call"])
    write_json(PUBLIC / "E18_EVALUATION_PROTOCOL.json", protocol)
    write_json(PRIVATE / "E18_PROTOCOL_FREEZE_RECEIPT.json", dict(protocol_sha256=sha256(PUBLIC / "E18_EVALUATION_PROTOCOL.json"),
        status=protocol["status"], underlying_E17_assets=old_integrity, source_sha256=sha256(__file__)))
    write_json(PUBLIC / "ARTIFACT_VERSION_INDEX.json", dict(
        active_model_qualification="FROZEN_MODEL_QUALIFICATION_v3.json",
        active_input_archive_private="PROPOSED2600_INPUTS_ONLY_v2.npz",
        active_input_metadata_receipt="INPUT_METADATA_REPAIR_v2.json",
        active_call_budget="E18_FINAL_CALL_BUDGET.json",
        active_member_inventory="REMOTE_CACHE_POSTCHECK_v2.json",
        active_local_cache_materialization="MEMBER_CACHE_MATERIALIZATION_POSTCHECK_v4.json",
        active_aggregate_inventory="REMOTE_AGGREGATION_POSTCHECK_v3.json",
        protocol="E18_EVALUATION_PROTOCOL.json",
        historical_engineering_records=["FROZEN_MODEL_QUALIFICATION.json", "FROZEN_MODEL_QUALIFICATION_v2.json",
            "E18_EXACT_CALL_BUDGET.json", "AGGREGATION_SCOPE_AND_RETRY_v2.json"],
        executed_budget_source_v1_private="executed_source_v1/frozen_input_budget.py",
        version_role="Earlier source/schema or local-only records are retained; active records above govern this preparation"))
    md("E18_FROZEN_EVALUATION_PROTOCOL.md", f"""# E18冻结评价协议

## 协议身份

原始父提交为{PARENT}，新分支为{BRANCH}。完整机器协议为E18_EVALUATION_PROTOCOL.json，当前状态为冻结准备方案，待目标总体、预算审批及新作用域执行适配。新数据尚未准入，实际评价尚未开始。

## 数据、策略与服务

只加载E17的Ridge、SPO_coordinate、TSPO_uniform、GlobalFixed。MiniLM、PCA16、StandardScaler、340参数线性头、原正则、typed V2 Qwen3聚合器及safe_exact_v2_0解析规则固定。full600权重、预处理和策略没有重新拟合。原fresh400的1600个决策全部恢复，新2600题只做冻结编码与输入侧预测。全部3000题动作在评分前封存，服务按原物理组合身份的semantic_exact执行。

每题选择三个成员。成员生成重复为42、137、2024，sample_index0，sampling_control=False；聚合调用的随机种子固定42，成员生成重复单独标记。四策略相同组合输出和共同成员按唯一键复用。主路由输入只有原题向量，任务/子任务元数据仅用于原有输出合同、解析、采样和统计分组。

## 评价目标与统计

主要比较是TSPO_uniform与SPO_coordinate在固定三生成协议下的题均准确率差。GSM/BBH等权，双侧5%检验和95%题级配对区间，三个seed保持同题簇。Ridge和GlobalFixed为次要对照。N3000是拟议覆盖扩展总体，1254题BBH扩展与原来源分开报告，审批前不声称它们属于原共同子池同分布样本。

## 执行顺序

先批准目标总体和预算，再完成E18专用作用域准入及稀疏四策略结果接口的非付费资格；核验冻结动作、原模型加载身份及缓存索引；仅补未满足的成员键；依赖齐备后执行选中组合聚合；一并锁定原始输出和真实资源账本；最后按一次性gold揭示与固定评分器统一评价。任何故障/重试分别记物理attempt，不重做已完成缓存。

旧native worker和full_feedback入口固定400题/全20组合，直接调用会违反本轮付费范围。准备DAG及稀疏投影资格和实际付费worker实现状态在EXECUTION_ADAPTER_READINESS.md分别列示。没有提供可误启动旧队列的命令。

## 停止与权限

本轮实际生成与评分为0。未批准、源身份不符、缓存哈希不符、依赖缺失、未知成本或新作用域资格失败均不进入付费执行。原协议自动付费重试为0，人工恢复另外计费。冻结后禁止根据fresh奖励改模型、阈值、seed、题目或样本量。保护200题仅经过已批准的一次输入指纹处理，答案继续封存。
""")
    remaining = budget["not_satisfied_by_verified_inventory"]
    budget_table = table(("范围", "所需成员键", "所需聚合键", "已验证成员复用", "成员缺口", "聚合缺口"),
        [(row["cohort"], row["required_member_keys"], row["required_aggregate_keys"],row["verified_member_reuse"],
          row["member_calls_unfulfilled_by_verified_cache"],row["aggregate_calls_unfulfilled_by_verified_cache"]) for row in budget["segments"]])
    md("E18_EXACT_CALL_BUDGET_REPORT.md", f"""# E18精确调用与实际Token预算

## 冻结调用并集

{budget_table}

合计47286次成员生成、30921次聚合的逻辑需求。单策略各需27000次成员、9000次聚合，四策略朴素相加为108000/36000；合法共享后按实际唯一键计费。结构复用为所有策略共同权限。

## 已完成缓存核验与本地准备

8份原本地导入、远端新端点pin核验、5330份成员身份/资源后检以及2130份新增缓存本地镜像均已完成。8个新增私有文件共4677193字节，SHA与原已审计远端源一致，没有覆盖旧8份文件。5330份库存中有{budget['required_verified_member_reuse']}份满足本轮四策略并集。

所需缓存对应历史实际输入{budget['verified_reused_member_actual_cost']['prompt_tokens']:,}、输出{budget['verified_reused_member_actual_cost']['completion_tokens']:,}，共{budget['verified_reused_member_actual_cost']['total_tokens']:,} Token。全部5330份已验证成功输出历史消耗2,101,075 Token，两者按集合口径分别报告。同步和本轮输入预测新增推理Token为0。

远端聚合审计检查282目录、1735条目、130份安全元数据。68个旧输出容器不属于已验证fresh计划/context，原文未解码；已扫描范围内可核验fresh聚合缓存为0。域外目录与保护命名边界仍为UNKNOWN，完整范围见B4_REMOTE_AGGREGATION_SCOPE_v3.md。

## 待执行预算

在当前已验证库存边界内，尚需满足成员{remaining['member_calls']:,}键、聚合{remaining['aggregate_calls']:,}键。发现更多合法缓存时可减少新物理调用，当前没有声称全服务器缓存全集已穷尽。标准答案最多3000次揭示，确定性判分30921次，LLM judge为0。

按原成员每次最多6144、聚合每次最多8192 Token，首次每缺失键至多一次尝试的保守上限为{budget['conservative_token_cap_initial_one_attempt_per_missing_key']:,} Token。它包含本轮首次尝试边界，原自动付费重试为0。跨后续人工恢复的总attempt/Token硬上限尚未锁定，单列null。未来实际Token保持未知，不能把cap当作实际花费。

只完成主要SPO/TSPO比较时逻辑需求39468次成员、17340次聚合，已核库存后缺35510/17340。该项仅提供预算透明度，当前拟议正式方案仍包含四策略，不据此改变研究对照。

## 编码成本与版本

新增2600题使用冻结MiniLM编码，batch8，对应325次批量前向；生成式LLM调用、PCA/Scaler/头拟合均为0。旧预算v1中的new_encoder_forwards=2600按题编码计数，最终账本明确区分题数与批次数。

最终权威预算为E18_FINAL_CALL_BUDGET.json。原本地v1预算、旧地址超时、追加远端核验和本地镜像分别保留，未覆盖历史费用。原始输出、逐题键和真实账本仅保存于私有目录。
""")
    qualification_rows = [
        ("理论与反例", "PASS", "20组702项合成检查"),
        ("双侧统计", "PASS", "8项软件检查及独立不平衡任务重算"),
        ("数据来源与去重", "PASS_WITH_LIMITATIONS", "1708排除、来源分层、近重复启发式边界"),
        ("原资产与四策略恢复", "PASS", "1600原决策恢复，权重与服务配置固定"),
        ("新输入/动作", "PASS", "2600编码，10400新决策冻结，改用Unicode v2工件"),
        ("成员缓存", "PASS", "5330核验、2130新增镜像、本地可组合复用"),
        ("聚合库存", "PASS_SCOPED", "已扫范围内fresh合法缓存0，域外未知"),
        ("候选DAG", "PASS_REVIEW_ONLY", "每聚合恰3依赖，计划不授权执行"),
        ("付费作用域入口", "NOT_IMPLEMENTED", "旧400题/20队入口不能直接用于E18"),
        ("目标总体与执行批准", "PENDING_USER_APPROVAL", "BBH覆盖扩展、固定N、预算"),
        ("独立科学结论", "NOT_EVALUATED", "新题评分与真实区间尚未执行")]
    precheck = dict(schema="E18_pre_execution_qualification_v1",
        status="PREPARATION_COMPLETE_PAID_EXECUTION_NOT_RELEASED", rows=qualification_rows,
        source_proposal_sha256=budget["inputs"]["proposal_sha256"],
        metadata_repair_sha256=sha256(PUBLIC / "INPUT_METADATA_REPAIR_v2.json"),
        corrected_input_sha256=sha256(PRIVATE / "PROPOSED2600_INPUTS_ONLY_v2.npz"),
        frozen_action_sha256=sha256(PRIVATE / "FROZEN3000_INPUT_PREDICTIONS.npz"),
        E17_preservation=old_integrity, fresh_reward_reads=0, sealed_answer_reads=0,
        protected_input_fingerprint_processing=200, new_llm_generations=0, scoring_operations=0,
        automatically_admitted_questions=0, encoder_question_encodings=2600, encoder_batches=325,
        pca_fits=0, scaler_fits=0, head_fits=0,
        actual_paid_runner_status="NOT_IMPLEMENTED_FOR_E18_SCOPE",
        execution_authorized=False, scope_adapter_detail_sha256=sha256(CODE / "EXECUTION_ADAPTER_READINESS.json"))
    write_json(PUBLIC / "E18_PRE_EXECUTION_QUALIFICATION.json", precheck)
    md("E18_PRE_EXECUTION_QUALIFICATION_REPORT.md", f"""# E18执行前资格检查

{table(("检查项", "状态", "依据"), qualification_rows)}

## 已修复的工程问题

新输入v1的GSM子任务空值曾被序列化为object，allow_pickle=False不能加载该字段。已保留v1并追加Unicode v2，不启用pickle，28项修复资格确认其余四数组字节、全部预测、调用并集和预算不变。当前发布入口也从源头规范化None为空字符串，权重/编码不重做。默认使用PROPOSED2600_INPUTS_ONLY_v2.npz。

准备阶段同时修正了PCA复载API所需状态和底层模型配置指纹/复合typed算子指纹混比；当前1600动作精确恢复、完整typed身份、28合同、76源码及tokenizer均核验。初始失败没有产生模型训练或真实评分。

## 技术闸口与审批边界

原native worker、fresh准入和全矩阵分析分别固定旧800/400身份或每题全20组合，不能直接承接3000题四策略稀疏集合。E18已有独立DAG和结果投影软件资格，真实付费作用域适配入口尚未实现。该项必须在批准后的非付费执行准入阶段完成，再进入任何模型调用。DAG PASS不代表付费worker就绪，也不代表科学假说成立。

目前报告与方案准备完成，付费执行资格未放行。需要确认BBH覆盖扩展总体、固定样本数和预算。实际生成队列、gold揭示、评分和新题置信区间均未运行。保护题只有经批准的一次输入指纹步骤，答案保持封存。
""")
    md("EXPERIMENT_SUMMARY.md", f"""# E18中文准备总结

## 本轮结论

机制理论、论文组织、输入去重、功效规划、四策略冻结、缓存核验和3000题精确调用预算已经完成。当前仅为准备，实际评价未执行，付费资格尚未放行。

## 关键数值

- 新增候选2600题，GSM/BBH各1300，加原fresh400形成拟议3000题。
- 双侧5%下，1pp差异的80%功效需2796题，3000题工作功效82.71%；0.5pp需11172题。
- BBH新增题中46来自旧子池，1254来自官方覆盖扩展。原子池合格余量215，当前3000题不能自动称原框同分布确认。
- 四策略唯一逻辑需求47286次成员、30921次聚合；已验证本地可复用成员库存5330，其中4717次命中该并集。
- 在已核验缓存边界内，待满足42569次成员、30921次聚合。首次每缺失键一次尝试的Token上限514848768，人工恢复总上限另计。
- 已验证库存历史Token2101075，本轮复用子集历史Token1944226；本次同步新增推理Token0。
- 新增2600题冻结编码，对应325批MiniLM前向；模型头/预处理拟合0，生成式LLM调用与评分0。

## 理论与论文产物

六项命题与证明、20组702项纯合成检查、可编辑机制图、E17核心结果CSV、相关工作固定源码版本和ACL实验组织均已保存。理论只支持规定条件，旧600题性能没有升级为独立确认。训练冠军与推理近并列、工作奖励估计与代理目标、对称性与Bayes一致性分别报告。

## 执行前仍需处理

拟议数据与预算需要审批，1254道BBH覆盖扩展需明确目标总体与单列结果。旧native入口不适用于3000题稀疏四策略，E18付费作用域适配尚未实现，需先完成非付费准入软件资格。原data v1 object metadata问题已追加修为Unicode v2，其余数值和预算不变。

## 优势、风险与状态

优势是所有决策在gold前冻结，重复生成不扩大题量，缓存身份与真实资源逐项核对。风险是来源框扩大、方差迁移、语义近重复和预训练暴露，以及实际付费入口的技术待办。数学与工程核验强度为高，未来质量差异尚未得到科学确认。sealed输入只按用户一次授权生成指纹，答案未读。

协议为E18_EVALUATION_PROTOCOL.json，六份主报告均以E18_开头。当前状态PREPARATION_COMPLETE_PAID_EXECUTION_NOT_RELEASED。
""")
    md("REPRODUCE.md", """# E18真实入口与复核顺序

## 依赖与私有资产

Python3.11、NumPy1.26.4、SciPy1.11.4、scikit-learn1.7.2，原冻结MiniLM使用Torch2.4.1、Transformers4.51.3及原tokenizer/safetensors。远端核验依赖Paramiko，机制图使用Pillow。资产根为原E盘MAB目录，原E17来源和完整模型权重保持不变。

私有清单包括原1708身份注册、源官方输入文件、一次性保护签名/HMAC私钥、proposed_new2600_inputs_only.jsonl、PROPOSED2600_INPUTS_ONLY_v2.npz、FROZEN3000_INPUT_PREDICTIONS.npz、E17 full600权重与预处理、原8缓存、8份新增镜像和缓存资源账本。原题、标准答案、逐题矩阵和权重没有上传。

## 本轮实际执行顺序

1. 核对E17远端父提交和本地封存哈希。
2. audit_sources.py盘点来源；candidate_proposal.py执行一次已明确授权的保护输入指纹及候选构建；complete_registered_reference_audit.py补审仅去重参考；verify_source_assets.py核对候选身份。
3. qualify_theory.py完成数学资格，make_mechanism_diagram.py生成共享几何的Drawio/SVG/PNG。
4. power_and_statistics.py产生双侧规划；qualify_operator_identity.py独立核验typed算子和统计公式。
5. cache_audit.py与token_provenance_audit.py核对原8文件；remote_cache_audit_v2.py只读远端成员；remote_cache_postcheck.py复核身份和账本。
6. frozen_input_budget.py qualify-frozen加载原权重；propose-budget仅做新输入编码和调用计数。当前source含metadata修复，原实际执行source v1快照单独留私有。
7. repair_input_metadata.py追加Unicode v2并验证四数值字段及预测不变。
8. remote_aggregation_audit_v3.py、aggregation_scope_v3_report.py核对聚合范围；materialize_remote_members_v4.py及materialization_postcheck_v4.py同步8份已验证远端成员文件。
9. final_reconciliation.py扣除已验证缓存；prepare_proposed_execution_dag.py准备审核DAG；执行作用域审计和sparse_policy_projection.py只做合成软件资格。
10. assemble_e18.py生成六主报告与协议；verify_and_seal_e18.py复核和封存交付。

带argparse的脚本提供--help，其他模块的真实入口是文件末尾的main或纯函数。写入采用不可覆盖记录，当前资产布局的普通复核命令为python -B -X utf8 verify_and_seal_e18.py verify。完整重建需要隔离输出空间及原私有输入清单，不能在现有封存目录上直接重复运行写入阶段。一次性保护指纹已处理200题，普通复核只使用已保存签名，不能再次自动解码原保护题。SSH通过交互getpass，密码不入命令行、源码或报告。原地址超时与新地址核验记录分别保留。

## 实际评价入口状态

当前不存在已经通过E18准入的付费运行命令，旧600/200/400和全20组合入口不能直接使用。EXECUTION_ADAPTER_READINESS.md列所需作用域适配，稀疏结果投影与审核DAG有独立合成/拓扑资格。任何模型执行和gold评分都等待明确批准以及对应软件闸口通过。

题级配对推断纯函数为power_and_statistics.paired_question_test，输入为[3,原题数]两策略合法评分和任务标签。当前只在合成数据上验证。没有以旧600题补算真实置信区间。
""")
    sources = {}
    for source in CODE.rglob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"))
        sources[source.relative_to(CODE).as_posix()] = sha256(source)
        destination = PUBLIC / "code" / source.relative_to(CODE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    final_state = dict(status=precheck["status"], branch=BRANCH, parent_commit=PARENT,
        protocol_sha256=sha256(PUBLIC / "E18_EVALUATION_PROTOCOL.json"),
        sources=sources, source_inputs_sha256=budget["inputs"]["proposal_sha256"],
        new_actual_generations=0, new_gold_scoring=0, automatically_admitted_questions=0,
        question_encodings=2600, batched_encoder_forwards=325, source_target_requires_approval=True,
        paid_scope_adapter="NOT_IMPLEMENTED", E17_assets=old_integrity,
        root_figure_qa=True, manuscript_draft_and_plan_only=True, wall_s=time.perf_counter()-start)
    write_json(PUBLIC / "E18_COMPLETE_PREPARATION.json", final_state)
    write_json(PRIVATE / "E18_FINAL_PREPARATION_RECEIPT.json", final_state)
    print("E18_SIX_PREPARATION_REPORTS_WRITTEN", flush=True)


if __name__ == "__main__":
    main()
