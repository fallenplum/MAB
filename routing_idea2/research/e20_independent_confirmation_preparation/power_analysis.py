"""Plan a fixed-time paired E20 evaluation using E19 development variance only.

No fresh/sealed payload, answer, score, or reward is read.  The three recorded
generation seeds remain within their original question.  This script computes
analytic power, without Monte Carlo repetitions or bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import nct, norm, t


DEFAULT_SOURCE = Path(
    r"E:\语言模型组合路由\research\e19_symmetric_tie_constraint"
    r"\private\OOF_EVALUATION.npz"
)
EFFECTS_PP = (0.5, 1.0, 1.5, 2.0)
QUESTION_COUNTS = (200, 400, 1200, 2400, 3000, 3700, 5000, 10000)
TASKS = ("gsm8k", "bbh")


def variance_summary(utilities: np.ndarray, labels: np.ndarray) -> dict:
    differences = (utilities[1] - utilities[0]).mean(axis=0)
    summaries = {}
    for task in TASKS:
        values = differences[labels == task]
        summaries[task] = {
            "questions": int(values.size),
            "mean_difference_pp": float(100 * values.mean()),
            "sample_variance": float(values.var(ddof=1)),
            "sample_standard_deviation": float(values.std(ddof=1)),
            "nonzero_difference_questions": int(np.count_nonzero(values)),
        }
    effective_variance = sum(summaries[task]["sample_variance"] for task in TASKS) / 2
    return {
        "independent_development_questions": int(differences.size),
        "recorded_generation_events": int(utilities.shape[1] * differences.size),
        "task_weights": {task: 0.5 for task in TASKS},
        "strata": summaries,
        "effective_variance_for_balanced_total_N": float(effective_variance),
        "effective_standard_deviation": float(np.sqrt(effective_variance)),
        "macro_mean_difference_pp": float(
            sum(summaries[task]["mean_difference_pp"] for task in TASKS) / 2
        ),
        "difference_definition": "mean_over_three_seeds(utility_C - utility_B)",
    }


def power_at(
    total_questions: int,
    effect: float,
    alpha: float,
    summary: dict,
    task_question_counts: dict | None = None,
) -> dict:
    if task_question_counts is None:
        if total_questions < 4 or total_questions % 2:
            raise ValueError("Balanced design requires an even total of at least four questions.")
        task_question_counts = {task: total_questions // 2 for task in TASKS}
    if set(task_question_counts) != set(TASKS) or sum(task_question_counts.values()) != total_questions:
        raise ValueError("Prespecified task counts must match the total independent questions.")
    if any(count < 2 for count in task_question_counts.values()):
        raise ValueError("Each task requires at least two independent questions.")
    components = np.array(
        [0.25 * summary["strata"][task]["sample_variance"] / task_question_counts[task] for task in TASKS]
    )
    sampling_variance = float(components.sum())
    standard_error = float(np.sqrt(sampling_variance))
    degrees_of_freedom = float(
        sampling_variance**2 / np.sum([
            component**2 / (task_question_counts[task] - 1)
            for task, component in zip(TASKS, components)
        ])
    )
    noncentrality = effect / standard_error
    normal_critical = norm.isf(alpha / 2)
    t_critical = t.isf(alpha / 2, degrees_of_freedom)
    normal_power = norm.sf(normal_critical - noncentrality) + norm.cdf(
        -normal_critical - noncentrality
    )
    t_power = nct.sf(t_critical, degrees_of_freedom, noncentrality) + nct.cdf(
        -t_critical, degrees_of_freedom, noncentrality
    )
    return {
        "total_independent_questions": total_questions,
        "questions_per_task": total_questions // 2 if len(set(task_question_counts.values())) == 1 else None,
        "task_question_counts": task_question_counts,
        "task_weights": {task: 0.5 for task in TASKS},
        "effect_pp": float(effect * 100),
        "two_sided_alpha": alpha,
        "standard_error_pp": standard_error * 100,
        "welch_effective_degrees_of_freedom": degrees_of_freedom,
        "normal_approximation_power": float(normal_power),
        "welch_noncentral_t_approximation_power": float(t_power),
    }


def minimum_even_sample_size(effect: float, alpha: float, target: float, summary: dict) -> dict:
    lower_half, upper_half = 2, 4
    while power_at(2 * upper_half, effect, alpha, summary)[
        "welch_noncentral_t_approximation_power"
    ] < target:
        upper_half *= 2
        if upper_half > 5_000_000:
            raise RuntimeError("Sample-size search exceeded its declared bound.")
    while lower_half < upper_half:
        middle = (lower_half + upper_half) // 2
        power = power_at(2 * middle, effect, alpha, summary)[
            "welch_noncentral_t_approximation_power"
        ]
        if power >= target:
            upper_half = middle
        else:
            lower_half = middle + 1
    result = power_at(2 * lower_half, effect, alpha, summary)
    result["target_power"] = target
    result["minimum_even_N_under_planning_approximation"] = 2 * lower_half
    return result


def numerical_checks(summary: dict) -> dict:
    null_power = power_at(400, 0.0, 0.05, summary)
    if not np.isclose(null_power["welch_noncentral_t_approximation_power"], 0.05, atol=1e-12):
        raise AssertionError("The null rejection probability must match the two-sided alpha.")
    powers = [
        power_at(n, 0.01, 0.05, summary)["welch_noncentral_t_approximation_power"]
        for n in QUESTION_COUNTS
    ]
    if not np.all(np.diff(powers) > 0):
        raise AssertionError("Power must increase with question count in this fixed-variance design.")
    requirement = minimum_even_sample_size(0.01, 0.05, 0.8, summary)
    n_required = requirement["minimum_even_N_under_planning_approximation"]
    if power_at(n_required - 2, 0.01, 0.05, summary)[
        "welch_noncentral_t_approximation_power"
    ] >= 0.8:
        raise AssertionError("The reported balanced sample size is not minimal.")
    unequal = power_at(199, 0.01, 0.05, summary, {"gsm8k": 100, "bbh": 99})
    unequal_variance = (
        0.25 * summary["strata"]["gsm8k"]["sample_variance"] / 100
        + 0.25 * summary["strata"]["bbh"]["sample_variance"] / 99
    )
    if not np.isclose(unequal["standard_error_pp"], 100 * np.sqrt(unequal_variance), atol=1e-12):
        raise AssertionError("The unequal-stratum planning variance does not preserve task weights.")
    return {
        "status": "PASS",
        "null_rejection_probability": null_power["welch_noncentral_t_approximation_power"],
        "sample_size_monotonicity": True,
        "minimum_even_sample_size_boundary_checked": True,
        "sealed199_unequal_stratum_variance_checked": True,
    }


def render_report(result: dict) -> str:
    main = result["decoders"]["tau_1e-8_uniform"]
    summary = main["variance_summary"]
    lines = [
        "# E20独立评价统计功效规划",
        "",
        "## 研究问题与数据边界",
        "",
        "主要比较固定为C−B。统计单位为独立原始问题，每题三个生成seed先取平均，GSM8K与BBH各占50%。本报告只读取E19旧开发OOF的策略效用和任务标签，没有读取fresh400奖励或sealed200答案，没有重新训练、LLM调用、bootstrap或新置信区间计算。",
        "",
        "fresh400已在历史E18进行过奖励评价，因此N=400的功效数值仅表示这种样本量的统计能力。该题池目前应归入已暴露开发资产。sealed200仍需由权限审计确认答案封存状态。",
        "",
        "## 配对方差与计算方法",
        "",
        "定义dᵢ为同题三个seed下C效用减B效用的平均。分层估计量为两任务题均值的等权平均。平衡总样本量N对应每任务N/2题，规划方差为(s²_GSM+s²_BBH)/(2N)。",
        "",
        f"E19共600道独立开发题，记录1,800次题目×生成事件。主解码的有效配对方差为{summary['effective_variance_for_balanced_total_N']:.10f}，标准差为{summary['effective_standard_deviation']:.8f}。其开发点差为{summary['macro_mean_difference_pp']:.6f}个百分点。",
        "",
        "| 任务 | 独立题数 | 题级配对方差 | 题级标准差 | 开发点差（pp） |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        s = summary["strata"][task]
        lines.append(
            f"| {task} | {s['questions']} | {s['sample_variance']:.8f} | "
            f"{s['sample_standard_deviation']:.8f} | {s['mean_difference_pp']:.6f} |"
        )
    lines.extend([
        "",
        "主分析固定双侧α=0.05。令目标差为Δ、分层标准误为SE，非中心参数为Δ/SE，使用Welch有效自由度及非中心t两侧尾概率作规划近似，同时输出正态近似。该近似依赖未来题级分布和旧开发方差的可迁移性。SciPy的非中心t参数定义和cdf/sf实现已核查。[SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html)",
        "",
        "## 主比较功效",
        "",
        "下表采用非中心t规划近似，百分比为检出率。目标效应均指真实C−B总体差，双侧检验对相反方向相同绝对效应具有对称功效。",
        "",
        "| 总独立题数 | 0.5pp | 1pp | 1.5pp | 2pp |",
        "|---:|---:|---:|---:|---:|",
    ])
    grid = main["single_primary_comparison"]["power_grid"]
    for n in QUESTION_COUNTS:
        cells = [row for row in grid if row["total_independent_questions"] == n]
        lines.append("| " + str(n) + " | " + " | ".join(
            f"{100 * row['welch_noncentral_t_approximation_power']:.2f}%" for row in cells
        ) + " |")
    lines.extend([
        "",
        "## 达到预定功效所需的平衡样本量",
        "",
        "每个数值均为本规划近似下满足目标的最小偶数总题量，每任务占一半。应据最终冻结模型、独立题池来源和实际部署评价方式再冻结方案。",
        "",
        "| 目标差（pp） | 单主比较80% | 单主比较90% | 两项Bonferroni80% | 两项Bonferroni90% |",
        "|---:|---:|---:|---:|---:|",
    ])
    for effect in EFFECTS_PP:
        values = []
        for comparison in ("single_primary_comparison", "two_comparison_bonferroni_scenario"):
            for target in (0.8, 0.9):
                row = next(r for r in main[comparison]["required_sample_sizes"]
                           if np.isclose(r["effect_pp"], effect) and r["target_power"] == target)
                values.append(str(row["minimum_even_N_under_planning_approximation"]))
        lines.append(f"| {effect:g} | " + " | ".join(values) + " |")
    raw = result["decoders"]["raw_exact_uniform"]["variance_summary"]
    lines.extend([
        "",
        "raw-exact为预先声明的敏感性诊断，保留其独立方差和完整功效网格，不按其结果改选主解码。",
        f"raw-exact的有效方差为{raw['effective_variance_for_balanced_total_N']:.10f}，开发点差为{raw['macro_mean_difference_pp']:.6f}个百分点。",
        "",
        "## sealed199待批准隔离情景",
        "",
        "输入侧近重复审计发现sealed200中一道BBH题与历史开发/fresh400题存在高相似。原sealed200题池和原200题功效规划保持完整。本节只规划经批准隔离该题后的199题情景，GSM8K为100题，BBH为99题。本轮没有自动删题，也没有读取其奖励或答案。",
        "",
        "此时GSM8K与BBH主权重仍各为50%，按0.25×s²_GSM/100+0.25×s²_BBH/99计算方差，并使用相应的Welch自由度。",
        "",
        "| 候选题池 | 0.5pp | 1pp | 1.5pp | 2pp |",
        "|---|---:|---:|---:|---:|",
    ])
    scenario = main["sealed199_pending_approval_scenario"]
    lines.append("| GSM100＋BBH99，α=0.05 | " + " | ".join(
        f"{100 * row['welch_noncentral_t_approximation_power']:.2f}%"
        for row in scenario["single_primary_comparison_power"]
    ) + " |")
    lines.extend([
        "",
        "该情景用于事前审查。隔离决定和后续评价仍需批准，题池身份与统计方案应在读取评分前冻结。完整JSON同时保留该199题情景的raw-exact及两项Bonferroni规划。",
        "",
        "## 多重比较与解释边界",
        "",
        "- 正式确认只保留一个主比较C−B，主解码tau=1e−8；α=0.05。任务分解、BBH子任务及raw-exact仅作预声明诊断。",
        "- 如果未来将raw-exact或其他效用对照另立为确认性假设，需要预先调整检验家族。JSON另给两个主假设各α=0.025的Bonferroni保守情景，未据结果扩展主家族。",
        "- 主要问题是证明非零总体差。证明收益下界超过1pp具有更严格目标，所需非中心参数应使用真实差减1pp；真实差恰为1pp时，该更严格目标没有80%功效。",
        "- 五折历史训练相互重叠，这些方差条件于旧OOF拟合。最终600题模型可能产生不同动作分布及配对方差，当前题数估计具有规划风险。",
        "- 主功效使用均匀argmax策略在完整缓存中的精确期望效用。若实际只随机抽取一个动作评价，额外动作随机性会提高方差，需要另立规划。",
        "- 独立问题数不能用三个seed或多次编号/算法重复扩张。各seed也不足以精确辨识无限生成次数下的条件均值。",
        "",
        "## 优势与风险",
        "",
        "优势是直接使用同题B/C配对差，保留生成簇并固定任务权重，能够把目标效应与资源规模明确对应。",
        "",
        "风险主要来自旧开发数据反复使用、外折训练重叠、最终600题模型与480题模型的差异，以及新题来源改变。功效数值是指定假设下的条件估计，不能保证未来复现。",
        "",
        "## 审查结论",
        "",
        "本报告仅完成P2统计设计。fresh400的历史奖励曝光和sealed200较小题量需要在独立题池方案中同时处理，不能以两者名义上的总题数替代独立性审计。新增题目、生成及真实奖励评价均等待用户批准。",
        "",
        "完整数值、软件版本和数值自检保存在E20_POWER_RESULTS.json。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as source:
        # Lazy NPZ reads deliberately decode only E19 utility arrays and task labels.
        labels = source["task_labels"].copy()
        utilities = source["utilities"].copy()
        raw_utilities = source["raw_exact_utilities"].copy()
    if utilities.shape != (2, 3, 600) or raw_utilities.shape != utilities.shape:
        raise ValueError("Unexpected E19 policy/seed/question shape.")
    if {task: int(np.sum(labels == task)) for task in TASKS} != {task: 300 for task in TASKS}:
        raise ValueError("The frozen E19 task counts do not match.")
    result = {
        "schema": "e20_power_planning_v1",
        "source_stage": "E19 train600 OOF development",
        "source_commit": "07b01361d425b089e49643f1ec5282eae0ac1ec0",
        "source_array_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "allowed_decoded_fields": ["utilities", "raw_exact_utilities", "task_labels"],
        "fresh_rewards_read": 0,
        "sealed_answers_read": 0,
        "new_llm_calls": 0,
        "new_bootstrap_replicates": 0,
        "confidence_intervals_computed": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "method": "Two-sided normal and Welch noncentral-t planning approximations with equal task weights and prespecified stratum counts",
        "primary_comparison": "C - B, tau=1e-8 uniform numerical argmax",
        "multiplicity": "One primary hypothesis at alpha=0.05; optional two-hypothesis Bonferroni sensitivity at alpha=0.025 per hypothesis",
        "official_method_reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html",
        "decoders": {},
    }
    for name, values in (("tau_1e-8_uniform", utilities), ("raw_exact_uniform", raw_utilities)):
        summary = variance_summary(values, labels)
        decoder = {"variance_summary": summary, "numerical_checks": numerical_checks(summary)}
        for comparison, alpha in (("single_primary_comparison", 0.05), ("two_comparison_bonferroni_scenario", 0.025)):
            decoder[comparison] = {
                "power_grid": [power_at(n, effect / 100, alpha, summary)
                               for n in QUESTION_COUNTS for effect in EFFECTS_PP],
                "required_sample_sizes": [minimum_even_sample_size(effect / 100, alpha, target, summary)
                                          for effect in EFFECTS_PP for target in (0.8, 0.9)],
            }
        decoder["sealed199_pending_approval_scenario"] = {
            "status": "PLANNING_ONLY_PENDING_APPROVAL",
            "original_sealed200_pool_preserved": True,
            "automatic_question_removal_performed": False,
            "reason": "One BBH input has a high-similarity match to historical development/fresh400 input; potential quarantine requires approval.",
            "task_question_counts": {"gsm8k": 100, "bbh": 99},
            "task_weights": {"gsm8k": 0.5, "bbh": 0.5},
            "variance_formula": "0.25*s_gsm_squared/100 + 0.25*s_bbh_squared/99",
            "single_primary_comparison_power": [
                power_at(199, effect / 100, 0.05, summary, {"gsm8k": 100, "bbh": 99})
                for effect in EFFECTS_PP
            ],
            "two_comparison_bonferroni_power": [
                power_at(199, effect / 100, 0.025, summary, {"gsm8k": 100, "bbh": 99})
                for effect in EFFECTS_PP
            ],
        }
        result["decoders"][name] = decoder
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "E20_POWER_RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "03_POWER_ANALYSIS.md").write_text(render_report(result), encoding="utf-8")
    main_decoder = result["decoders"]["tau_1e-8_uniform"]
    print(json.dumps({
        "status": "COMPLETE",
        "summary": main_decoder["variance_summary"],
        "one_pp_power": [row for row in main_decoder["single_primary_comparison"]["power_grid"]
                         if row["effect_pp"] == 1.0 and row["total_independent_questions"] in (200, 400)],
        "one_pp_required_N": [row for row in main_decoder["single_primary_comparison"]["required_sample_sizes"]
                              if row["effect_pp"] == 1.0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
