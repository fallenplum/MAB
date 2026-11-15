"""Assemble aggregate-only E17 reports and an immutable public delivery list."""
from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys

from e17_common import CODE, PRIVATE, PUBLIC, read_json, sha256, verify_protocol, write_json
import numpy as np


def csv_records(name):
    with (PUBLIC / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def markdown(name, content):
    with (PUBLIC / name).open("x", encoding="utf-8") as stream:
        stream.write(content.rstrip() + "\n")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_range(summary):
    return f"{summary['mean']:.3f} [{summary['minimum']:.3f}, {summary['maximum']:.3f}]"


def optional_number(value, digits=5):
    return "未定义" if value is None else f"{value:.{digits}f}"


def main():
    verify_protocol()
    evaluator_freeze=read_json(PRIVATE/"EVALUATOR_SOURCE_FREEZE.json")
    if evaluator_freeze["source_sha256"]!=sha256(CODE/"evaluate_e17.py"):
        raise ValueError("Evaluator differs from its pre-reward source freeze")
    for name,digest in evaluator_freeze["source_hashes"].items():
        if sha256(CODE/name)!=digest:raise ValueError("Evaluator dependency changed before reporting")
    if evaluator_freeze["protocol_sha256"]!=sha256(PUBLIC/"E17_PROTOCOL.json"):
        raise ValueError("Evaluator protocol context changed")
    if evaluator_freeze["prediction_receipt_sha256"]!=sha256(PRIVATE/"ALL_PREDICTIONS_FROZEN.json"):
        raise ValueError("Frozen prediction manifest changed")
    if evaluator_freeze["qualification_sha256"]!=sha256(PUBLIC/"FROZEN_ARTIFACT_QUALIFICATION.json"):
        raise ValueError("Frozen qualification changed")
    finalizer_hash=sha256(CODE/"finalize_e17.py")
    write_json(PRIVATE/"FINALIZER_SOURCE_FREEZE.json",dict(status="FROZEN_BEFORE_REPORT_ASSEMBLY",
        source_sha256=finalizer_hash,evaluator_receipt_sha256=sha256(PRIVATE/"EVALUATOR_SOURCE_FREEZE.json"),
        protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json")))
    complete = read_json(PUBLIC / "E17_COMPLETE_RESULTS.json")
    permutations = read_json(PUBLIC / "PERMUTATION_SUMMARIES.json")
    order = read_json(PUBLIC / "LOSS_ORDER_RESULTS.json")
    confirmation = read_json(PUBLIC / "CONFIRMATION_INPUT_ONLY_PLAN.json")
    power = read_json(PUBLIC / "CONFIRMATION_POWER.json")
    fresh = read_json(PUBLIC / "FRESH_NUMBERING_INPUT_ONLY_SUMMARY.json")
    qualification = read_json(PUBLIC / "FROZEN_ARTIFACT_QUALIFICATION.json")
    frozen = read_json(PRIVATE / "ALL_PREDICTIONS_FROZEN.json")
    csv_rows = csv_records("PERMUTATION_RUNS.csv")
    prior_rows = csv_records("NO_CONTENT_CHAMPION_PRIOR.csv")
    summary_index = {(row["method"], row["inference"], row["group"]): row for row in permutations["groups"]}
    method_names = ("Ridge", "SPO_coordinate", "TSPO_uniform")
    report_rows = []
    for method in method_names:
        for policy in ("semantic_exact", "semantic_near"):
            summary = summary_index[method, policy, "all"]
            runs = [row for row in csv_rows if row["method"] == method and row["inference"] == policy and int(row["permutation"]) > 0]
            report_rows.append((method, policy, f"{summary['identity_accuracy']:.3f}",
                format_range(summary["accuracy_across100"]),
                f"{100*summary['identity_consistency_across100']['mean']:.2f}%",
                f"{100*np.mean([float(row['numeric_first_share']) for row in runs]):.2f}%",
                f"{100*np.mean([float(row['maximum_action_share']) for row in runs]):.2f}%"))
    ties = table(("外折", "训练题数", "经验多冠军", "20组合全并列", "经验唯一冠军"),
        [(row["fold"], row["questions"], row["multiple_best"], row["all_actions_tied"], row["unique_best"]) for row in permutations["training_ties"]])
    permutation_table = table(("训练目标", "推理规则", "原编号准确率%", "100排列均值及极值%", "身份一致率", "选择新编号0", "单次最大组合占比"), report_rows)
    group_rows = []
    for group in ("tie_free", "tie_heavy", "all20_tied", "GSM8K", "BBH"):
        for method in method_names:
            row = summary_index[method, "semantic_exact", group]
            group_rows.append((group, row["questions"], method, f"{row['identity_accuracy']:.3f}",
                format_range(row["accuracy_across100"]), f"{100*row['identity_consistency_across100']['mean']:.2f}%"))
    prior_accuracy = np.array([float(row["accuracy"]) for row in prior_rows[1:]])
    prior_agreement = np.array([float(row["agreement_with_SPO"]) for row in prior_rows[1:]])
    prior_gain = np.array([float(row["SPO_minus_no_content_champion_prior_pp"]) for row in prior_rows[1:]])
    original = complete["old_transition_audit"]
    markdown("02_RANDOM_RENUMBERING.md", f"""# 随机重编号机制验证

## 研究问题

保持实际成员、组合、奖励和输入不变，只改变动作数组编号，检查冠军约定如何改变训练和选队。主服务规则固定实际组合语义顺序，另外单列仅推理规则改变的对照。

## 实验设置与对照

复用600个旧开发问题的五折，每折480题训练、120题评价。预固定100个随机排列及一个原编号控制，每排列训练Ridge、坐标首冠军SPO+、经验精确并列冠军均匀TSPO+。三个目标使用原有各自正则和340参数头。原编号恢复与全部预测先封存，之后读取外折评分。

{ties}

上述并列是三次生成的经验并列。它没有证明真实条件均值相同。

## 数值结果

{permutation_table}

方括号表示100次重编号的最小值和最大值。PERMUTATION_SUMMARIES.json同时提供总体标准差和方差，准确率方差单位为百分点平方pp²，分母为100。全部极值完整保留，未按准确率选择排列。置信区间和bootstrap均为0。

{table(("分组", "独立题数", "目标", "原编号准确率%", "排列均值及极值%", "身份一致率"), group_rows)}

## 不使用问题内容的冠军先验对照

每折仅统计训练区冠军标签的众数，在外折所有题上固定使用该实际组合。100个排列下，该基线准确率均值{prior_accuracy.mean():.3f}%，极值{prior_accuracy.min():.3f}%至{prior_accuracy.max():.3f}%。SPO+与它的选择一致率均值{100*prior_agreement.mean():.2f}%，SPO+相对它的准确率差均值{prior_gain.mean():+.3f}个百分点。

该对照与按训练平均奖励选择的全局固定组合分别保留。全局固定组合的外折准确率为{permutations['default_accuracy']:.3f}%。前者诊断冠军先验，后者提供直接服务价值参照。

## 训练与推理因素的隔离

PERMUTATION_RUNS.csv给出四种推理约定的逐排列汇总。INFERENCE_ONLY_CONTROLS.csv固定同一份已训练权重，仅改变推理优先级。canonical语义冠军训练控制对应同一个目标的坐标重表达，因此直接复用既有解，新增拟合为0。

主规则使用精确最高分，10⁻⁸近并列规则作为预固定数值敏感性分析。凸目标对称性、分数等变误差和argmax稳定性分别评价。极小数值扰动在近并列处也可能改变argmax，因此不能把全部选队变化都归因于真实冠军标签。

Ridge与均匀TSPO+的目标在重编号下保持相同语义。它们若表现为极小分数等变误差、精确argmax变化且近并列规则稳定，该现象属于数值边界敏感性。坐标SPO+另有真实冠军标签随编号改变的训练干预，两个来源分别解释。

## 历史数字复核

原SPO+选择组合0共{original['original_combo0']}题，反序版本选择组合19共{original['reverse_combo19_total']}题，两者改变{original['changed']}题。由原组合0转向反序组合19的实际题数为{original['old_combo0_to_reverse_combo19']}。这些统计口径分别保留。

## 证据边界、失败原因与下一步

100个排列共享600题及三次生成，排列数不增加独立题量。该实验可识别任意冠军规则的训练敏感性及其当前题池质量后果。它本身无法建立未来总体质量变化。tie-free评价题也仍由包含大量训练并列的历史区拟合。

优势是同步重命名、固定容量、相同求解器和推理隔离形成直接干预。风险包括近并列argmax、重复使用的开发题和未做独立准确率确认。编号敏感性结论以全部排列和分数资格为依据，外推强度以独立评价为边界。
""")
    model_rows = []
    for model in order["models"]:
        model_rows.append((model["method"], f"{model['accuracy']:.3f}",
            f"{model['versus_Ridge']['accuracy_difference_pp']:+.3f}",
            model["versus_Ridge"]["corrections_events"], model["versus_Ridge"]["corruptions_events"],
            optional_number(model["relative_gap_calibrated_MAE"]), optional_number(model["strict_pair_rank_accuracy"],4),
            model["exact_score_ties"],model["near_score_ties"]))
    primary_pair = next(row for row in order["paired_operators"] if row["MC_samples"] == 8)
    primary_a = next(row for row in order["models"] if row["method"] == "LossThenMean_B8")
    primary_b = next(row for row in order["models"] if row["method"] == "MeanThenLoss_B8")
    markdown("03_SAME_DRAWS_OPERATOR_ORDER.md", f"""# 固定奖励样本后的运算顺序

## 研究问题

固定相同联合奖励draw，分别最小化逐draw普通SPO+的平均损失，以及相同draw平均奖励上的SPO+。以此检查奖励估计与训练算子能否分开解释。

## 实验设置与对照

每折使用原缓存的8个主draw作为主要比较。预声明1、2、4、8、16的数值敏感性，其中16为原8个主draw加既有8个alt draw。每个问题始终只贡献1个问题权重，数据项除以nB，正则项只加一次。512诊断draw仅保存了均值摘要，本轮没有重抽512个样本，也没有把它与8draw目标混用。

原NS_delta0在实际8draw全部唯一最优时等价于逐draw普通SPO+，其已冻结系数经资格核对后复用。原NS_main的δ=.05集合权重保留背景身份。两个新运算顺序目标都固定δ=0。原Ridge、经验SPO+、均匀TSPO+和NS_main保持原冻结版本。

## 数值结果

{table(("模型", "准确率%", "相对Ridge pp", "纠错事件", "破坏事件", "校准相对差MAE", "严格pair排序率", "精确分数并列", "近分数并列"), model_rows)}

8draw主比较中，先平均奖励相对先平均损失改变{primary_pair['changed_questions']}道题，准确率差{primary_pair['MeanThenLoss_minus_LossThenMean']['accuracy_difference_pp']:+.3f}个百分点。两者准确率分别为{primary_b['accuracy']:.3f}%与{primary_a['accuracy']:.3f}%。逐任务、子任务、seed及MC稳定性完整保存在LOSS_ORDER_RESULTS.json。

{table(("运算顺序", "draw数", "相对同目标8draw的实际准确率差pp", "变化题数", "净纠错事件"), [(row['operator'],row['MC_samples'],f"{row['empirical_three_repeat_vs_B8']['accuracy_difference_pp']:+.3f}",row['changed_vs_B8'],row['empirical_three_repeat_vs_B8']['net_correct_events']) for row in order['MC_stability']])}

分数间隔的均值、中位数、最小值、最大值及10⁻⁸近并列数量保存在同一JSON。MC表中的准确率差直接来自三个既有生成重复的真实聚合评分，另列的共同B8拟合分数差保留模型诊断身份。

## 合成已知真值与理论解释

固定9份(.6,.5)和1份(0,1)，已知均值为(.54,.55)。先平均损失得到分差+.05并选择动作0，真实均值选择损失为.01。先平均奖励得到动作1，真实均值选择损失为0。工作均值误差为0，该差异仍存在。

二值联合奖励的对称并列反例进一步显示，均匀真实冠军权重可以得到相等的最优预测分数。数值求解残差与语义推理规则分别记录。详细证明、目标解析式和数值资格见THEORY_AND_COUNTEREXAMPLES.md及SYNTHETIC_RESULTS.json。

## 统计结论与失败边界

真实问题只有三个观测重复，无法直接观测真实条件均值及真实Bayes regret。真实结果仅报告三次经验均值选择损失和实际聚合正确率。工作后验均值相对三次均值的差异也不能称为真实去噪误差。

SPO类分数经训练区拟合的共同非负尺度校准后，才用于相对差误差诊断。原始分数、服务argmax和排序保持各自身份。校准后的误差与最终准确率分别报告。

## 优势、风险与下一步

优势是相同draw、相同函数类、相同正则下只改变运算顺序。风险是有限draw积分误差、工作后验失配和SPO+在当前离散分布下的一致性条件。先平均奖励也不能自动建立总体一致性。下一步只围绕冻结机制及独立题目确认制定预算。
""")
    budgets = confirmation["budget_plans"]
    budget_rows = []
    for name in ("all_text_only_policies", "all_policies_with_oracle_metadata", "TSPO_uniform_versus_Ridge", "all20_hindsight_diagnostic_separate"):
        plan = budgets[name]
        budget_rows.append((name, plan["from_zero"]["member_calls"], plan["from_zero"]["aggregate_calls"],
            plan["receipt_backed_cached_member_calls"], plan["unverified_or_missing_local_execution"]["member_calls"],
            plan["unverified_or_missing_local_execution"]["aggregate_calls"]))
    power_rows = []
    for comparison, plan in power.items():
        for effect in plan["effects"]:
            working_power=effect["power_by_n"][0]["normal_approximation_power"]
            power_rows.append((comparison, f"{plan['sigma_question_mean']:.5f}", effect["effect_pp"],
                "未定义" if effect["required_n"] is None else effect["required_n"],
                "未定义" if working_power is None else f"{100*working_power:.2f}%"))
    markdown("04_INDEPENDENT_CONFIRMATION_PLAN.md", f"""# 冻结独立确认与资源计划

## 研究问题与数据身份

fresh400尚未进行本轮准确率评价。当前完成输入侧动作冻结、编号敏感性新输入审计、调用并集与功效规划。sealed200继续保留最终封存身份。

## 已完成的冻结

用完整历史train600重新拟合PCA16、StandardScaler、Ridge、原SPO+和均匀TSPO+。正则固定取既有五折最终参数的中位数原值，Ridge alpha={confirmation['training']['parameters']['ridge_mean_target_alpha']:.8g}，SPO λ={confirmation['training']['parameters']['spo_regularization']:.8g}。这些配置在训练之前封存，未使用fresh奖励。

全局固定组合使用train600平均奖励选出。TaskFixedOracle和SubtaskFixedOracle需要真实任务标签，单列额外输入权限。三个模型头和GlobalFixed构成text-only主并集。预固定语义精确并列规则，记录分数及全部动作哈希。

五个外折模型在fresh400相同题目输入上的100排列结果保存在FRESH_NUMBERING_INPUT_ONLY_SUMMARY.json。这检查编号敏感性在新输入上的延续，未产生准确率证据，也未增加独立题量。

## 必要调用并集

{table(("策略集合", "从零成员调用", "从零聚合", "收据覆盖成员", "本地待核成员", "本地待核聚合"), budget_rows)}

已有8个导入收据合计3200份完整成员输出。主text-only并集可覆盖其中{budgets['all_text_only_policies']['receipt_backed_cached_member_calls']}份。收据元数据已核对，答案payload未重新打开，远端库存仍待核对。本地缺失量保留待核身份，不能直接认定必须重新生成。已存在结果在后续执行前还须验证身份与缓存完整性。

text-only并集的保守缺失部分Token上限为{budgets['all_text_only_policies']['conservative_token_upper_bound_missing_local']:,}，包含成员输入/输出6144和聚合输入/输出8192的每调用cap假设。该值没有使用fresh实际输出长度，也不作为实际Token花费。完整账本分别保留gold、成员、聚合与评分。

全20组合事后Oracle需要24000次聚合。该项单独列支，当前主策略评价所需并集为{budgets['all_text_only_policies']['from_zero']['aggregate_calls']}次聚合。

## 功效规划

{table(("配对候选", "开发题级标准差", "目标效应pp", "80%功效所需题数", "400题工作功效"), power_rows)}

这是单侧5%固定时点正收益检验的正态工作规划。假设新的独立问题GSM8K/BBH各半、题级配对方差可迁移，三次生成先在题内平均。估计正收益与证明收益下界超过1个百分点需不同的目标效应。表内题数采用当前冻结候选的开发方差，仍有迁移和估计风险。

开发方差来自五个480题训练模型的外折预测，独立确认头则使用完整600题拟合。拟合规模和模型参数的差异也是方差迁移的规划假设。若开发配对方差为0，所需题数和工作功效记为未定义，零观测方差无法建立未来零方差。

## 执行状态、风险与下一步

独立准确率执行状态为PENDING。新的物理LLM调用、fresh答案读取与评分均为0。后续先核对远端已有缓存，再依据上述功效和调用并集确定一次性确认规模与执行预算。任何补题方案按预冻结任务覆盖制定，不按当前题目正确性筛选。

优势是配置、动作与调用需求均在结果读取前固定。劣势是fresh400对小效应的检出能力有限，现有缓存尚未齐全，额外元数据基线与text-only主策略需要分别解释。
""")
    permutation_cpu = sum(sum(row["cpu_s"] for row in receipt["diagnostics"]) for fold in frozen["permutation_folds"] for receipt in fold["receipts"])
    order_cpu = sum(receipt["cpu_s"] for fold in frozen["order_folds"] for receipt in fold["receipts"])
    costs = dict(
        real_new_head_fits=1558, permutation_head_fits=1515,
        permutation_ridge_fits=505, permutation_spo_qps=1010,
        order_logical_models=50, order_new_qps=40, order_reused_models=10,
        full600_confirmation_head_fits=3, synthetic_qps=10,
        qualification_unit_tests=read_json(PUBLIC/"QUALIFICATION.json")["tests"],
        permutation_head_cpu_seconds=permutation_cpu, order_head_cpu_seconds=order_cpu,
        training_wall_seconds=frozen["wall_s"], evaluator_wall_seconds=complete["wall_s"],
        qualifier_wall_seconds=qualification["wall_s"],
        new_physical_llm_calls=0, new_embedding_forwards=0, new_fresh_scoring_ops=0,
        posterior_draws_reused_per_fold=dict(main=8,alt=8,diagnostic_mean_summary=512),
        new_teacher_fits=0,new_posterior_draws=0,
        unit_fit_cost_scope="Unit-test micro-solves are part of qualification runtime; synthetic10 QPs are separately enumerated.")
    write_json(PUBLIC/"E17_COST_SUMMARY.json", costs)
    ridge_summary = summary_index["Ridge", "semantic_exact", "all"]
    spo_summary = summary_index["SPO_coordinate", "semantic_exact", "all"]
    tspo_summary = summary_index["TSPO_uniform", "semantic_exact", "all"]
    markdown("EXPERIMENT_SUMMARY.md", f"""# E17 监督失真机制实验总结

## 两项核心机制结果

**固定真实组合后，任意冠军编号如何改变训练和选队。** 100个随机重编号中，原SPO+相对原编号的真实组合选择一致率均值为{100*spo_summary['identity_consistency_across100']['mean']:.2f}%。Ridge为{100*ridge_summary['identity_consistency_across100']['mean']:.2f}%，均匀TSPO+为{100*tspo_summary['identity_consistency_across100']['mean']:.2f}%。推理的精确并列与10⁻⁸近并列结果同时保留，分数等变资格和选择稳定性分别检查。

**相同奖励样本能否隔离目标错配。** 同一8draw下，先平均损失与先平均奖励改变{primary_pair['changed_questions']}道题的路由。后者相对前者准确率差为{primary_pair['MeanThenLoss_minus_LossThenMean']['accuracy_difference_pp']:+.3f}个百分点。已知真值反例在工作均值完全正确时仍出现不同最优动作，说明该现象能够由算子顺序单独产生。真实题池的总体奖励均值未被观测，实际结果不能据此唯一归因为后验或损失。

## 原编号表现与质量边界

{table(("冻结方法", "开发外折准确率%", "相对Ridge pp", "校准中心化MSE", "校准相对差MAE", "严格pair排序率"), [(row['method'], f"{row['accuracy']:.3f}", f"{row['versus_Ridge']['accuracy_difference_pp']:+.3f}",optional_number(row['centered_calibrated_MSE']),optional_number(row['relative_gap_calibrated_MAE']),optional_number(row['strict_pair_rank_accuracy'],4)) for row in complete['canonical_background']])}

Ridge、经验SPO+与均匀TSPO+使用各自冻结的训练区相对尺度。历史反序SPO和NS背景没有对应校准尺度，其校准误差记为未定义，原始分数排序率仍独立报告。

这些是重复使用的600道开发题及1800条生成事件。全部100排列的质量分布和正负结果均保留。编号偏好被机制干预识别，并不直接说明原准确率提升虚假，也不自动证明偏好损害准确率。

## 理论与文献定位

原SPO+对任意真冠军的选择显式依赖优化oracle。对称性、期望奖励一致性、新题泛化分别需要证据。平方损失的无约束总体最优仍为条件均值。SPO+的一致性条件及当前离散奖励的边界单列。

DARS已经研究随机生成下的路由监督稳定性。EquiRouter已经讨论确定性并列处理诱导的模型偏好。本轮贡献候选限于当前组合路由中的受控放大机制、训练/推理分离，以及奖励模型/损失算子/真实服务的分层验证。全局新颖性证据仍不完整。[DARS](https://aclanthology.org/2026.emnlp-main.333/)，[EquiRouter v2](https://arxiv.org/html/2602.03478v2)

## 独立确认与停止边界

三个full600模型头和fresh400输入侧动作已经封存。全部text-only策略并集为{budgets['all_text_only_policies']['from_zero']['member_calls']}次成员生成及{budgets['all_text_only_policies']['from_zero']['aggregate_calls']}次聚合，已有收据覆盖{budgets['all_text_only_policies']['receipt_backed_cached_member_calls']}次所需成员输出。fresh准确率尚未执行，远端缺失缓存核对和预算决定仍待完成。sealed200保持封存。

本轮不新增损失或采集公式。下一阶段围绕冻结假设、独立题目数量与必要调用并集作确认方案决策。400题对0.5—1个百分点小效应的工作功效见04_INDEPENDENT_CONFIRMATION_PLAN.md，不能仅因点值为正宣称可靠泛化。

## 优势与风险

优势是动作身份不变、相同draw、相同容量及来源资格共同控制机制。风险包括大量经验并列、极小数值间隔、工作后验失配、有限draw和旧开发数据反复暴露。数学及直接干预结论的证据强度为高，未来总体质量优势与论文全局新颖性尚待独立证据。

## 实际执行成本

完成1515个重编号预测头、40个新的运算顺序QP及3个full600冻结头，共1558个真实数据预测头。运算顺序另有10个合法复用模型。完成10个已知真值微型QP及{costs['qualification_unit_tests']}项冻结前单元测试，训练墙钟{costs['training_wall_seconds']/60:.2f}分钟。物理LLM调用、fresh评分、置信区间及bootstrap均为0。完整数值、失败及数值资格记录保留。
""")
    shutil.copy2(CODE / "LITERATURE_DRAFT.md", PUBLIC / "01_LITERATURE_AND_TARGET.md")
    shutil.copy2(CODE / "THEORY_DRAFT.md", PUBLIC / "THEORY_AND_COUNTEREXAMPLES.md")
    shutil.copy2(CODE / "synthetic_results.json", PUBLIC / "SYNTHETIC_RESULTS.json")
    shutil.copy2(CODE / "literature_sources.json", PUBLIC / "LITERATURE_SOURCES.json")
    markdown("REPRODUCE.md", """# E17复现入口

## 环境与私有资产

Python3.11、NumPy、SciPy、scikit-learn、Clarabel0.11.1。使用既有E8私有dependencies中的Clarabel。默认MAB_ASSET_ROOT为E:\\语言模型组合路由，可通过同名环境变量切换。MAB_E17_ROOT可指定新的独立结果目录。

私有资产包括train600输入与20×3评分、E13各折输入包、原M0/M4及lex/reverse/uniform权重、NS各折8+8draw与512均值摘要、fresh400输入编码及8个缓存导入收据。原题、答案、私有逐题矩阵与权重均不在公开发布中。

## 顺序

在code目录执行以下命令。

```powershell
python -B -X utf8 train_e17.py prepare
python -B -X utf8 validate_frozen.py freeze-source
python -B -X utf8 train_e17.py freeze --workers 2
python -B -X utf8 validate_frozen.py qualify
python -B -X utf8 freeze_confirmation.py freeze-inputs-only
python -B -X utf8 fresh_numbering_inputs.py audit-inputs-only
python -B -X utf8 evaluate_e17.py
python -B -X utf8 synthetic_and_theory.py
python -B -X utf8 finalize_e17.py
```

原运行使用分阶段代码冻结。训练源码先冻结，验证器及各评价/输入审计源码在各自读取前单独冻结。发布版包含随后新增的只读验证器和汇总代码，完整新运行的协议source_hash集合会因此包含更多文件。来源数据、算法、排列和配置必须保持一致，逐项结果资格不依赖运行时长或压缩文件字节一致。

prepare和封存/评价写入均采用不可覆盖记录，重新复现应使用新的MAB_E17_ROOT。训练中断可在同一协议下继续freeze，完成后需通过独立资格。输入独立确认脚本一旦封存，重新运行采用新输出目录。既有公开报告不覆盖。

## 评价边界

evaluate_e17仅在全部预测及资格完成后读取旧train600外折评分。独立fresh准确率未实现自动执行入口。后续必须先核实缓存与预算、确认一次性评价协议。原sealed200不由本轮程序访问。

统计为题级点差、纠错破坏、排列分布及任务/seed分解。没有置信区间和bootstrap。预测头重复、生成重复、重编号重复均不扩大独立题量。
""")
    markdown("README.md", """# E17监督失真审计

主要入口为EXPERIMENT_SUMMARY.md。02_RANDOM_RENUMBERING.md记录100组随机动作重编号，03_SAME_DRAWS_OPERATOR_ORDER.md记录相同draw的算子顺序，04_INDEPENDENT_CONFIRMATION_PLAN.md记录输入侧冻结、调用并集与功效。

CPU机制实验已经完成，fresh400独立准确率仍待缓存核对和执行预算决定。E17_COMPLETE_RESULTS.json的状态保留该区别。新物理LLM调用为0。原有E1—E16数据与报告保持其冻结哈希。

公开内容包括可复现源码、协议、来源哈希、脱敏汇总和资格记录。逐题私有数据及模型权重在原E17 private目录。环境、顺序、工件角色见REPRODUCE.md。
""")
    code_destination = PUBLIC / "code"
    code_destination.mkdir(exist_ok=True)
    for path in CODE.rglob("*.py"):
        target = code_destination / path.relative_to(CODE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    final_sources = {str(path.relative_to(CODE)): sha256(path) for path in CODE.rglob("*.py")}
    if final_sources["finalize_e17.py"]!=finalizer_hash:
        raise ValueError("Finalizer changed during report assembly")
    for name,digest in evaluator_freeze["source_hashes"].items():
        if sha256(CODE/name)!=digest:raise ValueError("Frozen evaluator dependency changed during report assembly")
    write_json(PUBLIC/"FINAL_SOURCE_RECEIPT.json", dict(source_hashes=final_sources,
        evaluator_source_sha256=evaluator_freeze["source_sha256"],finalizer_source_sha256=finalizer_hash,
        original_assets_verified_unchanged=True,protocol_sha256=sha256(PUBLIC/"E17_PROTOCOL.json")))
    files = []
    for path in sorted(PUBLIC.rglob("*")):
        if path.is_file():
            if path.suffix not in (".md", ".py", ".json", ".csv"):
                raise ValueError("Unexpected public format")
            files.append(dict(path=path.relative_to(PUBLIC).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
    write_json(PUBLIC/"PUBLIC_DELIVERY_MANIFEST.json", dict(status="SEALED_AGGREGATE_ONLY", files=files,
        question_texts=0,answers=0,private_question_arrays=0,model_weights=0,
        new_llm_calls=0,fresh_reward_reads=0,sealed_reward_reads=0))
    print("E17_PUBLIC_DELIVERY_SEALED",len(files)+1,flush=True)


if __name__ == "__main__":
    main()
