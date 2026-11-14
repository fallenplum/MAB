"""Cached train600 signal diagnostics with original-question uncertainty units.

No predictor is fitted. Reference policies use only each original outer history.
Same-question cross-repeat choices are explicitly hindsight diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

from common import (ASSETS, E8, INPUTS, OUTCOMES, RESULTS, choice, legal_fold,
                    legal_metadata, load_npz, prior_scores, process_state,
                    sha256, write_json)


SEEDS = (42, 137, 2024)
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261115071
PAIR_I, PAIR_J = np.triu_indices(20, 1)
TEAMS = list(itertools.combinations(range(6), 3))


def interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else None


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def ratio_samples(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def bootstrap_weights(rows, tasks, seed):
    """Within-task bootstrap of question clusters, preserving stratum counts."""
    rng = np.random.default_rng(seed)
    local_tasks = tasks[rows]
    weights = np.zeros((BOOTSTRAPS, len(rows)), dtype=np.int16)
    for task in sorted(set(local_tasks.tolist())):
        local = np.flatnonzero(local_tasks == task)
        count = len(local)
        weights[:, local] = rng.multinomial(count, np.full(count, 1.0 / count), size=BOOTSTRAPS)
    assert np.all(weights.sum(axis=1) == len(rows))
    return weights.astype(float)


def group_views(task, sub):
    views = {"overall": np.arange(len(task)), "gsm8k": np.flatnonzero(task == "gsm8k"),
             "bbh": np.flatnonzero(task == "bbh")}
    for key in sorted(set(sub[task == "bbh"].tolist())):
        views["bbh/" + key] = np.flatnonzero((task == "bbh") & (sub == key))
    return views


def ordinary_summary(y, rows):
    block = y[rows]
    successes = block.sum(axis=1)
    ordered = np.sort(successes, axis=1)
    ties = np.sum(successes == ordered[:, -1, None], axis=1)
    gaps = (ordered[:, -1] - ordered[:, -2]) / 3
    pair_gap = (successes[:, PAIR_I] - successes[:, PAIR_J]) / 3
    distinct_gap = []
    for row in successes:
        lower = row[row < row.max()]
        distinct_gap.append(float((row.max() - lower.max()) / 3) if len(lower) else None)
    all_same_each_repeat = np.all(block == block[:, :, :1], axis=(1, 2))
    return {
        "n_original_questions": len(rows), "n_recorded_generation_events": 3 * len(rows),
        "n_action_generation_records": int(block.size),
        "empirical_top_tie_count_histogram": {str(k): int(np.sum(ties == k)) for k in range(1, 21)},
        "multiple_empirical_best_questions": int(np.sum(ties > 1)),
        "all20_empirical_means_tied_questions": int(np.sum(ties == 20)),
        "all20_reward_vectors_identical_within_every_repeat_questions": int(all_same_each_repeat.sum()),
        "best_minus_second_best_gap_histogram": {str(k): int(np.sum(np.isclose(gaps, k))) for k in (0., 1/3, 2/3, 1.)},
        "best_minus_best_strictly_lower_gap_mean_defined_questions": (
            float(np.mean([v for v in distinct_gap if v is not None]))
            if any(v is not None for v in distinct_gap) else None),
        "best_minus_best_strictly_lower_support": sum(v is not None for v in distinct_gap),
        "unordered_pair_absolute_gap_histogram": {str(k): int(np.sum(np.isclose(abs(pair_gap), k))) for k in (0., 1/3, 2/3, 1.)},
        "action_reward_means": block.mean(axis=(0, 1)).tolist(),
        "action_means_by_generation": block.mean(axis=0).tolist(),
        "observed_reward_mean_levels": sorted(set((successes / 3).ravel().tolist())),
        "empirical_hindsight_max_mean_reward": float(successes.max(axis=1).mean() / 3),
        "not_a_population_or_input_only_oracle": True,
    }


def paired_action_summaries(y, baselines, rows, weights):
    block = y[rows]
    output = {}
    for name, actions in baselines.items():
        default = block[np.arange(len(rows)), :, actions[rows]]
        delta = block - default[:, :, None]
        per_question = delta.mean(axis=1)
        draw_mean = weights @ per_question / len(rows)
        positive = (delta > 0).sum(axis=1)
        negative = (delta < 0).sum(axis=1)
        conditional = ratio_samples(weights @ (positive - negative), weights @ (positive + negative))
        cases = []
        for action in range(20):
            p, n = int(positive[:, action].sum()), int(negative[:, action].sum())
            cases.append({
                "action": action, "members": TEAMS[action],
                "mean_paired_delta": float(per_question[:, action].mean()),
                "paired95_ci": interval(draw_mean[:, action]),
                "corrected_generation_events": p, "destroyed_generation_events": n,
                "equal_generation_events": int(3 * len(rows) - p - n),
                "nonzero_generation_events": p + n,
                "questions_with_any_nonzero_delta": int(np.sum(np.any(delta[:, :, action] != 0, axis=1))),
                "conditional_mean_delta_given_nonzero": safe_ratio(p - n, p + n),
                "conditional_mean95_ci": interval(conditional[:, action]),
                "nonzero_conditioning_is_outcome_based_diagnostic": True,
                "mean_delta_by_generation": delta[:, :, action].mean(axis=0).tolist(),
            })
        output[name] = {"default_accuracy": float(default.mean()),
                        "default_generation_accuracy": default.mean(axis=0).tolist(),
                        "actions": cases}
    return output


def cross_seed_pairs(y, rows, weights):
    block = y[rows]
    differences = block[:, :, PAIR_I] - block[:, :, PAIR_J]
    output = {}
    for first, second in itertools.combinations(range(3), 2):
        a, b = differences[:, first], differences[:, second]
        both = (a != 0) & (b != 0)
        matching = both & (a == b)
        denominator = both.sum(axis=1)
        numerator = matching.sum(axis=1)
        first_strict = a != 0
        # Orient each comparison by its first-repeat winner, then inspect the second.
        oriented = np.sign(a) * b
        oriented_sum = (oriented * first_strict).sum(axis=1)
        first_count = first_strict.sum(axis=1)
        key = f"{SEEDS[first]}_vs_{SEEDS[second]}"
        output[key] = {
            "n_original_questions": len(rows),
            "questions_with_both_nonzero_comparisons": int(np.sum(denominator > 0)),
            "both_nonzero_unordered_pair_events": int(denominator.sum()),
            "same_direction_pair_events": int(numerator.sum()),
            "opposite_direction_pair_events": int(denominator.sum() - numerator.sum()),
            "same_direction_fraction_given_both_nonzero": safe_ratio(numerator.sum(), denominator.sum()),
            "same_direction95_ci": interval(ratio_samples(weights @ numerator, weights @ denominator)),
            "first_seed_strict_pair_events": int(first_count.sum()),
            "second_seed_tie_given_first_strict_pair_events": int(np.sum(first_strict & (b == 0))),
            "mean_second_seed_delta_oriented_by_first_strict_winner": safe_ratio(oriented_sum.sum(), first_count.sum()),
            "oriented_delta95_ci": interval(ratio_samples(weights @ oriented_sum, weights @ first_count)),
            "denominator_is_pair_events_not_independent_sample_size": True,
        }
    return output


def cross_repeat_selection(y, baselines, rows, weights):
    block = y[rows]
    chosen = np.empty((len(rows), 3), dtype=int)
    value = np.empty((len(rows), 3), dtype=float)
    uniform_value = np.empty_like(value)
    train_advantages = {}
    for held in range(3):
        fit = [seed for seed in range(3) if seed != held]
        training_score = block[:, fit].sum(axis=1)
        selected = choice(training_score)
        best = training_score == training_score.max(axis=1, keepdims=True)
        chosen[:, held] = selected
        value[:, held] = block[np.arange(len(rows)), held, selected]
        uniform_value[:, held] = np.sum(best * block[:, held], axis=1) / best.sum(axis=1)
        for name, default_actions in baselines.items():
            default = default_actions[rows]
            advantage = (training_score[np.arange(len(rows)), selected] - training_score[np.arange(len(rows)), default]) / 2
            train_advantages.setdefault(name, np.empty((len(rows), 3)))[:, held] = advantage
    result = {"kind": "HINDSIGHT_SAME_QUESTION_TWO_REPEATS_SELECT_THIRD_EVALUATE",
              "deployment_input_only_claim": False,
              "selection_tie_rule": "exact first action in original semantic combination column order",
              "selected_heldout_accuracy": float(value.mean()),
              "selected_accuracy_by_held_generation": value.mean(axis=0).tolist(),
              "selected_accuracy95_ci": interval(weights @ value.mean(axis=1) / len(rows)),
              "uniform_best_tie_expected_heldout_accuracy": float(uniform_value.mean()),
              "uniform_best_tie_expected_accuracy_by_held_generation": uniform_value.mean(axis=0).tolist(),
              "comparisons": {}}
    for name, default_actions in baselines.items():
        default = block[np.arange(len(rows)), :, default_actions[rows]]
        delta = value - default
        uniform_delta = uniform_value - default
        positive_train = train_advantages[name] > 0
        observed = positive_train & (delta != 0)
        improvements = positive_train & (delta > 0)
        observed_q, improvement_q = observed.sum(axis=1), improvements.sum(axis=1)
        selected_region_delta_q = (delta * positive_train).sum(axis=1)
        selected_region_count_q = positive_train.sum(axis=1)
        result["comparisons"][name] = {
            "baseline_accuracy": float(default.mean()),
            "paired_delta": float(delta.mean()),
            "paired95_ci": interval(weights @ delta.mean(axis=1) / len(rows)),
            "delta_by_held_generation": delta.mean(axis=0).tolist(),
            "corrected_generation_events": int(np.sum(delta > 0)),
            "destroyed_generation_events": int(np.sum(delta < 0)),
            "changed_choice_events": int(np.sum(chosen != default_actions[rows, None])),
            "positive_training_gap_record_count": int(positive_train.sum()),
            "positive_training_gap_unique_questions": int(np.any(positive_train, axis=1).sum()),
            "heldout_mean_delta_given_positive_other_two_gap": safe_ratio(selected_region_delta_q.sum(), selected_region_count_q.sum()),
            "positive_other_two_gap_mean_delta95_ci": interval(ratio_samples(weights @ selected_region_delta_q, weights @ selected_region_count_q)),
            "heldout_nonzero_direction_accuracy_in_positive_other_two_region": safe_ratio(improvement_q.sum(), observed_q.sum()),
            "heldout_direction95_ci": interval(ratio_samples(weights @ improvement_q, weights @ observed_q)),
            "heldout_nonzero_direction_denominator": int(observed.sum()),
            "uniform_best_tie_expected_delta": float(uniform_delta.mean()),
            "uniform_best_tie_expected_delta95_ci": interval(weights @ uniform_delta.mean(axis=1) / len(rows)),
            "selection_region_uses_same_question_outcomes": True,
        }
    return result


def stable_two_seed_pair_directions(y, rows, weights):
    differences = y[rows][:, :, PAIR_I] - y[rows][:, :, PAIR_J]
    output = {}
    for held in range(3):
        first, second = [seed for seed in range(3) if seed != held]
        stable = differences[:, first] * differences[:, second] > 0
        oriented = np.sign(differences[:, first]) * differences[:, held]
        counts = stable.sum(axis=1)
        positives = (stable & (oriented > 0)).sum(axis=1)
        negatives = (stable & (oriented < 0)).sum(axis=1)
        oriented_sum = (stable * oriented).sum(axis=1)
        output[str(SEEDS[held])] = {
            "stable_other_two_nonzero_pair_events": int(counts.sum()),
            "unique_questions_in_stable_region": int(np.sum(counts > 0)),
            "heldout_same_direction_events": int(positives.sum()),
            "heldout_opposite_direction_events": int(negatives.sum()),
            "heldout_tie_events": int((counts - positives - negatives).sum()),
            "heldout_oriented_mean_delta": safe_ratio(oriented_sum.sum(), counts.sum()),
            "oriented_mean95_ci": interval(ratio_samples(weights @ oriented_sum, weights @ counts)),
            "heldout_agreement_given_nonzero": safe_ratio(positives.sum(), (positives + negatives).sum()),
            "heldout_agreement95_ci": interval(ratio_samples(weights @ positives, weights @ (positives + negatives))),
            "not_an_input_only_region_selector": True,
        }
    return output


def plot_diagnostics(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    time_font = Path("C:/Windows/Fonts/times.ttf")
    if time_font.exists():
        font_manager.fontManager.addfont(str(time_font))
        font_family = font_manager.FontProperties(fname=str(time_font)).get_name()
    else:
        font_family = "DejaVu Serif"
    matplotlib.rcParams.update({"font.family": font_family, "font.size": 10,
                               "axes.spines.top": False, "axes.spines.right": False,
                               "axes.linewidth": .8, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    colors = {"overall": "#555555", "gsm8k": "#4477AA", "bbh": "#8E6AA8"}
    labels = {"overall": "All questions", "gsm8k": "GSM8K", "bbh": "BBH"}
    bins = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 20)]
    xs = np.arange(len(bins))
    for offset, task in ((-.18, "gsm8k"), (.18, "bbh")):
        summary = results["groups"][task]["empirical"]
        hist = summary["empirical_top_tie_count_histogram"]
        values = [100 * sum(hist[str(k)] for k in range(low, high + 1)) / summary["n_original_questions"] for low, high in bins]
        axes[0].bar(xs + offset, values, width=.34, color=colors[task], label=f"{labels[task]} (n=300)")
    axes[0].set_xticks(xs, ["1", "2-4", "5-9", "10-19", "20"])
    axes[0].set_xlabel("Number of empirically best actions")
    axes[0].set_ylabel("Original questions (%)")
    axes[0].set_title("a  Empirical best-action ties", loc="left", weight="bold")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylim(0, 100)
    pair_keys = ["42_vs_137", "42_vs_2024", "137_vs_2024"]
    for offset, task in ((-.16, "overall"), (0, "gsm8k"), (.16, "bbh")):
        data = results["groups"][task]["cross_seed_unordered_pairs"]
        ys = np.array([100 * data[key]["same_direction_fraction_given_both_nonzero"] for key in pair_keys])
        ci = np.array([data[key]["same_direction95_ci"] for key in pair_keys]) * 100
        axes[1].errorbar(np.arange(3) + offset, ys,
                         yerr=np.maximum(np.array([ys - ci[:, 0], ci[:, 1] - ys]), 0),
                         fmt="o", color=colors[task], capsize=3, markersize=5,
                         label=labels[task])
    axes[1].set_xticks(np.arange(3), ["42 / 137", "42 / 2024", "137 / 2024"])
    axes[1].set_xlabel("Member-generation seed pair")
    axes[1].set_ylabel("Same direction, both differences nonzero (%)")
    axes[1].set_title("b  Cross-repeat pair directions", loc="left", weight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc="lower left", fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, bottom=.22, top=.90, wspace=.32)
    fig.text(.07, .065,
             "600 original questions; 20 actions; 3 recorded repeats. Error bars: 95% question-cluster bootstrap (2,000 resamples).\n"
             "Pair-event denominators retain within-question dependence. Same-question diagnostics do not measure input-only routing.",
             fontsize=9, ha="left", va="bottom")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return {"backend": "Python/matplotlib", "font": font_family,
            "font_substitution": not time_font.exists(), "size_inches": [11.4, 4.6], "dpi": 300,
            "claim": "Empirical ties are common and cross-repeat comparisons require independent-question uncertainty.",
            "panels": ["empirical best-action count", "both-nonzero cross-seed pair agreement"],
            "archetype": "quantitative grid", "output": path.name, "sha256": sha256(path)}


def render_report(results):
    groups = results["groups"]
    overall = groups["overall"]
    summary = overall["empirical"]
    lines = ["# E9 组合奖励可辨识性与跨生成重复诊断", "", "## 研究问题与设置", "",
             "使用已核查的原train600最终聚合评分，分析经验并列、真实记录内的组合分歧及跨生成重复方向。600道原题包含GSM8K和BBH各300道，BBH保留27个子任务。每题20组合×3生成重复始终作为同一个问题簇。", "",
             "标准误差与区间只描述这批已反复用于开发的数据。2000次bootstrap在task内重采样原题，保留同题全部动作、生成重复和比较结果。三个seed编号不证明生成误差独立，137↔2024结果与其他seed对一并保留。", "",
             "## 经验奖励分辨率", "",
             f"20组合三重复均值全平局的原题有{summary['all20_empirical_means_tied_questions']}/600道。多个经验最优组合的原题有{summary['multiple_empirical_best_questions']}/600道。", "",
             "| 任务 | 原题数 | 多个经验最优 | 20组合经验均值全平局 |", "|---|---:|---:|---:|"]
    for key in ("gsm8k", "bbh"):
        item = groups[key]["empirical"]
        lines.append(f"| {key} | {item['n_original_questions']} | {item['multiple_empirical_best_questions']} | {item['all20_empirical_means_tied_questions']} |")
    lines += ["", "经验均值只能取0、1/3、2/3、1。最佳与次佳差为0包含多个经验最优；另报告最佳与严格次优动作的差，分母仅包含存在严格较低动作的原题。单题只有三次记录，无法据此区分真实期望并列与有限生成造成的经验并列。", "",
              "## 跨生成非零差方向", "",
              "以下分母是同题无序组合对在两个生成重复中都存在非零正确性差的事件数。该条件由已有奖励定义，只用于诊断。每个事件不构成新的独立原题。", "",
              "| seed对 | 双方非零组合对事件 | 同向比例 | 题簇95%区间 |", "|---|---:|---:|---:|"]
    for key, item in overall["cross_seed_unordered_pairs"].items():
        ci = item["same_direction95_ci"]
        lines.append(f"| {key.replace('_vs_', ' ↔ ')} | {item['both_nonzero_unordered_pair_events']} | {100*item['same_direction_fraction_given_both_nonzero']:.3f}% | [{100*ci[0]:.3f}, {100*ci[1]:.3f}]% |")
    lines += ["", "第一生成严格领先的动作还在第二生成中有多少收益，按第一生成差值方向翻转后计算，含第二生成平局，详细数值及题级区间见JSON。另两次生成同向且非零的组合对在第三次生成中的收益、反向与平局计数分别列出，未将同题筛选当作可部署规则。", "",
              "## 同题两次选队、第三次评价", "",
              "三轮轮流留出一个生成重复，候选只根据另两次同题奖励选定。并列按原语义动作列序选第一个，同时给出经验最优集合均匀选择的期望结果。", "",
              f"全20组合的跨重复选择准确率为{100*overall['cross_repeat_selection']['selected_heldout_accuracy']:.4f}%，均匀并列期望结果为{100*overall['cross_repeat_selection']['uniform_best_tie_expected_heldout_accuracy']:.4f}%。", "",
              "| 默认参照 | 默认准确率 | 跨重复选择差 | 题簇95%区间（百分点） |", "|---|---:|---:|---:|"]
    for key, item in overall["cross_repeat_selection"]["comparisons"].items():
        ci = item["paired95_ci"]
        lines.append(f"| {key} | {100*item['baseline_accuracy']:.4f}% | {100*item['paired_delta']:+.4f}pp | [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] |")
    lines += ["", "这些结果利用同一道题部署前不可得的生成反馈，证据类型为hindsight/cross-repeat diagnostic。它们既不是仅凭题目输入的选队成绩，也不是真实总体能力的上界。三次经验均值的事后最大值单独记录，有限重复下更大的候选集合也不保证更高留出成绩。", "",
              "## 合法训练区默认与配对差", "",
              "Global、Task与Subtask默认均由该题所在原外折对应的history480评分计算。当前题的奖励不参与默认动作。三种默认的scores/actions先保存到私有冻结文件，再进入本诊断评价。Task/Subtask参照使用真实任务元数据，标明额外输入权限。", "",
              "对每个默认和每个组合，报告D=组合正确性−默认正确性，取值−1、0、1。JSON逐项保存纠错、破坏、非零事件数、原题支持数、三重复平均配对差与题簇95%区间，另给条件于非零分歧的平均差。零分歧区域的条件均值记null。", "",
              "## BBH子任务支持与经验平局", "",
              "| 子任务 | 原题数 | 多个经验最优 | 20组合经验均值全平局 |", "|---|---:|---:|---:|"]
    for key, group in groups.items():
        if key.startswith("bbh/"):
            item = group["empirical"]
            lines.append(f"| {key[4:]} | {item['n_original_questions']} | {item['multiple_empirical_best_questions']} | {item['all20_empirical_means_tied_questions']} |")
    lines += ["", "子任务只有11—12道不同原题，所有子任务结果仅为支持量有限的描述性诊断，未进行子任务筛选或显著性主张。", "",
              "## 输入内容对照与证据边界", "",
              "100次任务/子任务内整题奖励块置乱由主训练runner单独执行。本脚本不拟合任何预测头，也不执行置乱。语义可预测性结论必须结合主报告里的原始训练与置乱训练对比，不能由本节同题跨重复结果替代。", "",
              "**优势**为使用真实最终聚合评分、训练区默认以及保留同题相关性的区间，完整给出平局与非平局支持。", "",
              "**风险**为仅三次生成、旧600题多次开发、原五折训练历史重叠及子任务样本少。区间条件于已冻结的历史默认和当前开发资产，未覆盖重新训练整个研究流程的不确定性。", "",
              "## 下一步", "",
              "将本诊断与预先冻结的PSMR/HSMR选队成绩和100次内容负对照共同解释。跨生成重复存在可观察机会，并不保证MiniLM题目内容可以预测这些机会。", "",
              "![Empirical ties and cross-repeat directions](05_figure_tie_and_discordance.png)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    output = args.output
    targets = [output / name for name in ("02_signal_diagnostics.json", "02_signal_report.md", "05_figure_tie_and_discordance.png")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refuse to replace existing signal diagnostics")
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    source_manifest = json.loads((RESULTS / "00_asset_manifest.json").read_text(encoding="utf-8"))
    assert source_manifest["status"] == "PASS"
    raw, inputs = load_npz(OUTCOMES), load_npz(INPUTS)
    assert sha256(OUTCOMES) == "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e"
    assert sha256(INPUTS) == "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3"
    y = raw["aggregate_scores"].transpose(1, 0, 2).astype(float)
    task, sub = inputs["task_labels"], inputs["subtask_labels"]
    assert y.shape == (600, 3, 20) and np.isin(y, [0, 1]).all()
    baselines = {name: np.full(600, -1, dtype=int) for name in ("global", "task", "subtask")}
    baseline_scores = {name: np.zeros((600, 20)) for name in baselines}
    fold_id = np.full(600, -1, dtype=int)
    for fold in range(5):
        history, evaluation = legal_fold(fold)
        rows = evaluation["global_indices"]
        metadata = legal_metadata(fold, rows)
        fold_id[rows] = fold
        for name in baselines:
            scores = prior_scores(history["Y"], history["task"], history["sub"],
                                  metadata["task"], metadata["sub"], kind=name)
            baseline_scores[name][rows] = scores
            baselines[name][rows] = choice(scores)
    assert all((actions >= 0).all() for actions in baselines.values())
    private = output / "private"
    private.mkdir(parents=True, exist_ok=True)
    frozen = private / "SIGNAL_DIAGNOSTIC_BASELINES.npz"
    if frozen.exists():
        raise FileExistsError(frozen)
    np.savez_compressed(frozen, fold=fold_id,
                        **{f"actions__{name}": actions for name, actions in baselines.items()},
                        **{f"scores__{name}": scores for name, scores in baseline_scores.items()})
    freeze_hash = sha256(frozen)
    results = {"status": "COMPLETE", "schema": "e9_bayes_gate_signal_diagnostics_v1",
               "evidence": "REUSED_TRAIN600_EXPLORATORY_DIAGNOSTICS",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "seeds": SEEDS, "n_original_questions": 600, "n_recorded_generation_events": 1800,
               "actions": [{"index": index, "member_indices": team} for index, team in enumerate(TEAMS)],
               "bootstrap": {"repetitions": BOOTSTRAPS, "seed": BOOTSTRAP_SEED,
                             "unit": "original question, all actions and three repeats kept together",
                             "stratification": "task within each reported group", "interval": "ordinary percentile95, no simultaneous guarantee"},
               "baseline_freeze_sha256": freeze_hash,
               "baseline_rule": "outer-history480 only, common.prior_scores then exact semantic-column argmax",
               "baseline_metadata_permissions": {"global": "none", "task": "true task metadata", "subtask": "true task/BBH-subtask metadata"},
               "content_permutation_status": "EXECUTED_SEPARATELY_BY_MAIN_RUNNER_NOT_THIS_SCRIPT",
               "groups": {}}
    for index, (name, rows) in enumerate(group_views(task, sub).items()):
        weights = bootstrap_weights(rows, task, BOOTSTRAP_SEED + index)
        results["groups"][name] = {
            "empirical": ordinary_summary(y, rows),
            "paired_actions_versus_train_only_defaults": paired_action_summaries(y, baselines, rows, weights),
            "cross_seed_unordered_pairs": cross_seed_pairs(y, rows, weights),
            "cross_repeat_selection": cross_repeat_selection(y, baselines, rows, weights),
            "stable_two_seed_pairs_then_heldout": stable_two_seed_pair_directions(y, rows, weights),
        }
    output.mkdir(parents=True, exist_ok=True)
    figure = plot_diagnostics(results, targets[2])
    results["cost"] = {"new_predictor_fits": 0, "new_llm_calls": 0, "new_gold": 0,
                       "physical_model_calls": 0, "content_permutation_runs": 0,
                       "cpu_seconds": time.process_time() - start_cpu,
                       "wall_seconds": time.perf_counter() - start_wall, **process_state()}
    write_json(targets[0], results)
    targets[1].write_text(render_report(results), encoding="utf-8")
    log = {"command": [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--output", str(output)],
           "source_sha256": sha256(__file__), "input_sha256": sha256(INPUTS), "outcomes_sha256": sha256(OUTCOMES),
           "asset_audit_sha256": sha256(RESULTS / "00_asset_manifest.json"),
           "new_files": {path.name: sha256(path) for path in targets}, "figure_contract": figure,
           "cost": results["cost"], "private_baseline_path": frozen.name,
           "private_baseline_sha256": freeze_hash, "forbidden_payload_reads": 0}
    write_json(private / "SIGNAL_DIAGNOSTIC_RUN.json", log)
    print(json.dumps({"status": "COMPLETE", "all20_empirical_mean_ties": results["groups"]["overall"]["empirical"]["all20_empirical_means_tied_questions"],
                      "hindsight_accuracy": results["groups"]["overall"]["cross_repeat_selection"]["selected_heldout_accuracy"],
                      "cost": results["cost"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
