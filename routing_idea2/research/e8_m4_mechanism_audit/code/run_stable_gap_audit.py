"""Exploratory, question-clustered audit of frozen E8 M4 and Ridge choices.

The program freezes its definitions and source hash before opening historical
train600 outcomes. No held-out generation is used to choose its own LOSO mask.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
        else x.item()) + "\n", encoding="utf-8")


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def interval(values):
    finite = np.isfinite(values)
    return np.quantile(values[finite], [.025, .975]).tolist() if finite.any() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\语言模型组合路由"))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    output = args.out
    public = output / "public"
    private = output / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    result_path = public / "M4_STABLE_GAP_RESULTS.json"
    if result_path.exists():
        raise RuntimeError("Append-only output already contains results")
    started = time.perf_counter()
    e8 = args.root / "research/e8_decision_focused_combination_learning"
    prediction_path = e8 / "private/evaluation/PREDICTIONS_AND_CALIBRATION_FROZEN.npz"
    receipt_path = e8 / "private/evaluation/CALIBRATION_FROZEN_RECEIPT.json"
    protocol = {
        "study": "M4 small mechanism audit, stable nonzero empirical gaps",
        "status": "FROZEN_BEFORE_FIRST_NEW_REWARD_ACCESS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__),
        "source_e8_protocol_sha256": sha(e8 / "public/E8_PROTOCOL.json"),
        "predictions_sha256": sha(prediction_path),
        "scope": "Exploratory reused-development diagnostic, no independent confirmation",
        "unit": "600 original questions, all three generations attached",
        "generation_seeds": [42, 137, 2024],
        "question_bootstrap": {"repetitions": 10000, "seed": 2026111402,
            "strata": "outer fold x GSM8K/BBH, all LOSO masks and outcomes resampled jointly"},
        "masks": {
            "all_questions": "all original questions and three held-generation records",
            "two_seed_any_mean_value_gap": "two nonheld seeds have nonconstant actionwise reward sums",
            "two_seed_stable_action_pair": "some action pair has the same nonzero reward difference in both nonheld seeds",
            "two_seed_stable_selected_pair": "M4 minus Ridge is the same nonzero sign in both nonheld seeds",
            "two_seed_selected_M4_ahead": "M4 minus Ridge equals +1 in both nonheld seeds",
            "two_seed_selected_Ridge_ahead": "M4 minus Ridge equals -1 in both nonheld seeds",
            "two_seed_default_repair_opportunity": "global default fails in both nonheld seeds and some one combination succeeds in both",
        },
        "posthoc_masks": {
            "three_seed_any_mean_gap": "three-seed actionwise mean range is nonzero",
            "three_seed_nonzero_selected_mean_gap": "three-seed M4 minus Ridge mean is nonzero",
            "three_seed_selected_consistent_nonzero": "M4 minus Ridge has one strict sign in all three generations",
            "three_seed_stable_action_pair": "some action pair has one strict sign in all three generations",
        },
        "statistics": "Question-cluster paired percentile95 CI. Conditional LOSO accuracy uses selected records. Task-balanced conditional results are separate from pooled support-weighted results.",
        "multiplicity": "All subgroup and LOSO results exploratory, no claim of confirmatory p-values",
        "forbidden_outcomes": ["validation200", "fresh400", "sealed200"],
        "new_physical_llm_calls": 0,
    }
    put(public / "M4_STABLE_GAP_PREDECLARED.json", protocol)
    frozen_protocol_hash = sha(public / "M4_STABLE_GAP_PREDECLARED.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["prediction_and_calibration_sha256"] != protocol["predictions_sha256"]:
        raise AssertionError("Frozen E8 prediction receipt differs")
    predictions = archive(prediction_path)
    ridge = np.argmax(predictions["raw__M0_original"], axis=1)
    m4 = np.argmax(predictions["raw__M4"], axis=1)
    default = predictions["default_actions"].astype(int)
    fold = predictions["fold"].astype(int)
    sys.path.insert(0, str(e8 / "public/qualification"))
    from e8_assets import load_historical
    reward_open_time = datetime.now(timezone.utc).isoformat()
    data, references, _ = load_historical(args.root)
    rewards = np.asarray(data["Y"], dtype=np.int8)
    n = len(rewards)
    if rewards.shape != (600, 3, 20):
        raise AssertionError("Unexpected historical question/generation/action shape")
    rows = np.arange(n)
    y0 = rewards[rows, :, ridge]
    y4 = rewards[rows, :, m4]
    ydefault = rewards[rows, :, default]
    difference = y4 - y0
    changed = m4 != ridge
    if int(changed.sum()) != 543:
        raise AssertionError("Historical E8 543 changed actions did not reproduce")
    if int(difference.sum()) != 12:
        raise AssertionError("Historical E8 net12 question-generation events did not reproduce")
    task = np.asarray(data["task"])
    subtask = np.asarray(data["sub"])
    rng = np.random.default_rng(2026111402)
    blocks = []
    for fold_id in range(5):
        for label in ("gsm8k", "bbh"):
            group = np.flatnonzero((fold == fold_id) & (task == label))
            blocks.append(rng.choice(group, size=(10000, len(group)), replace=True))
    samples = np.concatenate(blocks, axis=1)

    def summarize(mask, question_subset=None):
        mask = np.asarray(mask, dtype=bool)
        subset = np.ones(n, dtype=bool) if question_subset is None else np.asarray(question_subset, dtype=bool)
        active = mask & subset[:, None]
        count = active.sum(axis=1)
        selected = count > 0
        total = int(count.sum())
        sums = (difference * active).sum(axis=1)
        plus = ((difference == 1) & active).sum(axis=1)
        minus = ((difference == -1) & active).sum(axis=1)
        ties = ((difference == 0) & active).sum(axis=1)
        resampled_count = count[samples].sum(axis=1)
        resampled_sum = sums[samples].sum(axis=1)
        gain_draws = np.divide(resampled_sum, resampled_count,
            out=np.full(10000, np.nan), where=resampled_count > 0)
        item = {
            "unique_questions": int(selected.sum()),
            "held_generation_records": total,
            "M4_accuracy": float((y4 * active).sum() / total) if total else None,
            "Ridge_accuracy": float((y0 * active).sum() / total) if total else None,
            "M4_minus_Ridge": float(sums.sum() / total) if total else None,
            "paired95_question_cluster": interval(gain_draws),
            "corrected_question_generation_events": int(plus.sum()),
            "destroyed_question_generation_events": int(minus.sum()),
            "tied_question_generation_events": int(ties.sum()),
            "net_corrected_question_generation_events": int(sums.sum()),
            "questions_net_positive": int((sums[selected] > 0).sum()),
            "questions_net_negative": int((sums[selected] < 0).sum()),
            "questions_net_zero": int((sums[selected] == 0).sum()),
            "questions_with_any_correction": int((plus > 0).sum()),
            "questions_with_any_destruction": int((minus > 0).sum()),
            "support_records_per_generation": active.sum(axis=0).tolist(),
            "generation": [],
        }
        for seed_index, seed in enumerate((42, 137, 2024)):
            keep = active[:, seed_index]
            item["generation"].append({"seed": seed, "questions": int(keep.sum()),
                "M4_minus_Ridge": float(difference[keep, seed_index].mean()) if keep.any() else None,
                "corrected": int(np.sum(difference[keep, seed_index] == 1)),
                "destroyed": int(np.sum(difference[keep, seed_index] == -1))})
        return item

    pairs = np.array(list(itertools.combinations(range(20), 2)))
    pair_difference = rewards[:, :, pairs[:, 0]] - rewards[:, :, pairs[:, 1]]
    masks = {key: np.ones((n, 3), dtype=bool) if key == "all_questions"
        else np.zeros((n, 3), dtype=bool) for key in protocol["masks"]}
    for held in range(3):
        fit = [index for index in range(3) if index != held]
        action_sum = rewards[:, fit, :].sum(axis=1)
        masks["two_seed_any_mean_value_gap"][:, held] = np.ptp(action_sum, axis=1) > 0
        stable_pairs = pair_difference[:, fit[0], :] * pair_difference[:, fit[1], :] > 0
        masks["two_seed_stable_action_pair"][:, held] = stable_pairs.any(axis=1)
        selected_stable = difference[:, fit[0]] * difference[:, fit[1]] > 0
        masks["two_seed_stable_selected_pair"][:, held] = selected_stable
        masks["two_seed_selected_M4_ahead"][:, held] = np.all(difference[:, fit] == 1, axis=1)
        masks["two_seed_selected_Ridge_ahead"][:, held] = np.all(difference[:, fit] == -1, axis=1)
        masks["two_seed_default_repair_opportunity"][:, held] = (
            np.all(ydefault[:, fit] == 0, axis=1)
            & np.any(np.all(rewards[:, fit, :] == 1, axis=1), axis=1))
    posthoc_question_masks = {
        "three_seed_any_mean_gap": np.ptp(rewards.sum(axis=1), axis=1) > 0,
        "three_seed_nonzero_selected_mean_gap": difference.sum(axis=1) != 0,
        "three_seed_selected_consistent_nonzero": np.all(difference == 1, axis=1) | np.all(difference == -1, axis=1),
        "three_seed_stable_action_pair": (np.all(pair_difference == 1, axis=1)
            | np.all(pair_difference == -1, axis=1)).any(axis=1),
    }
    labels = {"all": np.ones(n, bool), "gsm8k": task == "gsm8k", "bbh": task == "bbh"}
    for name in sorted(set(subtask[task == "bbh"].tolist())):
        labels["bbh/" + str(name)] = (task == "bbh") & (subtask == name)
    results = {}
    for name, mask in masks.items():
        groups = {label: summarize(mask, subset) for label, subset in labels.items()}
        gsm, bbh = groups["gsm8k"], groups["bbh"]
        balanced = None
        if gsm["held_generation_records"] and bbh["held_generation_records"]:
            balanced = .5 * (gsm["M4_minus_Ridge"] + bbh["M4_minus_Ridge"])
        results[name] = {"definition": protocol["masks"][name], "groups": groups,
            "task50_50_conditional_point_gain": balanced,
            "conditional_weight_note": "Pooled row weights selected records; task50_50 computes each task's own conditional mean then equal-weights, provided both have support"}
    posthoc = {}
    for name, keep in posthoc_question_masks.items():
        mask = np.repeat(keep[:, None], 3, axis=1)
        posthoc[name] = {"definition": protocol["posthoc_masks"][name],
            "scope": "Posthoc outcome-defined stratum, selected and evaluated on the same three repeats",
            "groups": {label: summarize(mask, subset) for label, subset in labels.items()}}
    stable_positive = masks["two_seed_selected_M4_ahead"]
    stable_negative = masks["two_seed_selected_Ridge_ahead"]
    transitions = {}
    for name, mask, expected_sign in (("M4_ahead_on_two", stable_positive, 1),
                                      ("Ridge_ahead_on_two", stable_negative, -1)):
        total = int(mask.sum())
        transitions[name] = {
            "nonheld_definition_sign": expected_sign,
            "selected_records": total,
            "unique_questions": int(mask.any(axis=1).sum()),
            "held_same_sign": int(((difference == expected_sign) & mask).sum()),
            "held_reversed": int(((difference == -expected_sign) & mask).sum()),
            "held_tied": int(((difference == 0) & mask).sum()),
        }
        transitions[name]["held_same_sign_fraction"] = transitions[name]["held_same_sign"] / total if total else None
    output_results = {
        "status": "COMPLETE_EXPLORATORY_DIAGNOSTIC",
        "protocol_sha256": frozen_protocol_hash,
        "source_sha256": protocol["source_sha256"],
        "inputs": {"predictions_sha256": protocol["predictions_sha256"],
            "historical_train_outcomes_sha256": references["outcomes"]["sha256"]},
        "historical_reproduction": {"questions": n, "changed_actions": int(changed.sum()),
            "question_generation_records": n * 3,
            "M4_accuracy": float(y4.mean()), "Ridge_accuracy": float(y0.mean()),
            "net_corrected_events": int(difference.sum())},
        "loso": results,
        "selected_pair_held_transitions": transitions,
        "posthoc": posthoc,
        "limitations": [
            "Each fixed router was trained using other questions and all three historical repeats; only stratum definition excludes its own held generation",
            "LOSO strata use same-question reward feedback unavailable to an input-only router",
            "Three-repeat consistency is empirical, not a precise conditional expectation or population stability guarantee",
            "Question bootstrap retains all generation and LOSO dependence but conditions on reused-development folds with overlapping training histories",
            "All strata are exploratory and cannot certify a selected new router or unique failure mechanism",
        ],
        "new_physical_llm_calls": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    put(result_path, output_results)
    np.savez_compressed(private / "STABLE_GAP_QUESTION_RECORDS.npz",
        question_ids=data["ids"], task=task, subtask=subtask, fold=fold,
        actions_m4=m4, actions_ridge=ridge, actions_default=default,
        reward_m4=y4, reward_ridge=y0, reward_default=ydefault,
        difference=difference, **{"mask__" + key: value for key, value in masks.items()},
        **{"posthoc__" + key: value for key, value in posthoc_question_masks.items()})
    put(private / "REWARD_ACCESS_LOG.json", {
        "first_reward_read_at_utc": reward_open_time,
        "predeclared_protocol_sha256": frozen_protocol_hash,
        "frozen_before_reward_read": True,
        "decoded_reward_split": "selector_train600 only",
        "loader": str(e8 / "public/qualification/e8_assets.py"),
        "references": references,
        "forbidden_outcomes_accessed": [],
        "private_question_records_sha256": sha(private / "STABLE_GAP_QUESTION_RECORDS.npz"),
    })
    write_report(public, output_results)
    print(json.dumps({"status": "COMPLETE", "result": str(result_path),
        "historical_reproduction": output_results["historical_reproduction"],
        "selected_pair_transitions": transitions}, ensure_ascii=False))


def write_report(public, results):
    reproduction = results["historical_reproduction"]
    lines = ["# M4真实非零组合差与跨生成诊断", "", "## 研究问题", "",
        "检查冻结M4在存在经验非零组合差的题上，是否比冻结原Ridge选队更好。以下结果来自已复用的600题开发交叉验证。", "",
        "## 实验设置与对照", "",
        "六个模型、K=3、20组合、V2聚合器及三个生成重复均沿用E8缓存。M4和Ridge的动作完全冻结。每个留出生成的分层规则只查看另外两个生成的评分。", "",
        "主置信区间按原始题目成簇，并在外折×任务内重采样10000次。同一道题的三个留出记录一起重采样。该区间以当前开发数据和冻结模型为条件。", "",
        "## 数值结果", "",
        f"600题中有{reproduction['changed_actions']}题更换了组合。1800条题目×生成记录中，M4准确率为{100*reproduction['M4_accuracy']:.3f}%，Ridge为{100*reproduction['Ridge_accuracy']:.3f}%。", "",
        "| 分层规则 | 独立原题数 | 留出记录数 | 纠错 | 破坏 | 净事件数 | M4−Ridge百分点 | 题簇配对95%区间 |",
        "|---|---:|---:|---:|---:|---:|---:|---|" ]
    names = {
        "all_questions": "全部题目",
        "two_seed_any_mean_value_gap": "另外两生成存在平均组合差",
        "two_seed_stable_action_pair": "另外两生成存在同向非零组合对",
        "two_seed_stable_selected_pair": "M4与Ridge在另外两生成差值同向非零",
        "two_seed_selected_M4_ahead": "M4在另外两生成均领先",
        "two_seed_selected_Ridge_ahead": "Ridge在另外两生成均领先",
        "two_seed_default_repair_opportunity": "默认组合在另两生成均错且有稳定答对组合",
    }
    for name, section in results["loso"].items():
        item = section["groups"]["all"]
        ci = item["paired95_question_cluster"]
        ci_text = f"[{100*ci[0]:.3f}, {100*ci[1]:.3f}]" if ci else "未定义"
        gain = f"{100*item['M4_minus_Ridge']:.3f}" if item["M4_minus_Ridge"] is not None else "未定义"
        lines.append(f"| {names[name]} | {item['unique_questions']} | {item['held_generation_records']} | {item['corrected_question_generation_events']} | {item['destroyed_question_generation_events']} | {item['net_corrected_question_generation_events']} | {gain} | {ci_text} |")
    lines += ["", "三个留出生成可以使同一道题出现多次，因此表中同时报告原题数与留出记录数。支持量条件分层的总体行按所选记录加权，JSON另列GSM8K/BBH各50%的条件点值。", "",
        "### 分任务与生成方向", "",
        "| 分层 | 任务 | 原题数 | 记录数 | M4−Ridge百分点 | 生成42 / 137 / 2024百分点 |", "|---|---|---:|---:|---:|---|" ]
    for name in ("all_questions", "two_seed_stable_action_pair", "two_seed_stable_selected_pair"):
        for label in ("gsm8k", "bbh"):
            item = results["loso"][name]["groups"][label]
            directions = " / ".join(f"{100*g['M4_minus_Ridge']:.3f}" if g["M4_minus_Ridge"] is not None else "未定义" for g in item["generation"])
            gain = f"{100*item['M4_minus_Ridge']:.3f}" if item["M4_minus_Ridge"] is not None else "未定义"
            lines.append(f"| {names[name]} | {label} | {item['unique_questions']} | {item['held_generation_records']} | {gain} | {directions} |")
    lines += ["", "27个BBH子任务的完整支持量、配对区间、逐生成结果及纠错/破坏保存在M4_STABLE_GAP_RESULTS.json。", "",
        "## 统计结论", "",
        "全体结果与E8冻结值一致。分层结果均为探索性诊断，跨生成分层仍使用同一道题的事后反馈。三生成全一致层同时使用其评分定义并评价，因此另置于posthoc结果。", "",
        "## 失败原因与局限", "",
        "两生成中观察到非零差不保证第三生成继续非零。只有三次重复，无法精确识别每个组合的条件期望奖励。不同问题的样本量、内容表示和优化目标均可能影响结果，本分析不据此指定唯一原因。", "",
        "## 下一步", "",
        "结合组合分布与真实并列标签干预，再确定需要确认的冻结候选。独立确认应在未读评分的新题上执行，并按原始题目配对统计。", ""]
    (public / "02_STABLE_NONZERO_GAPS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
