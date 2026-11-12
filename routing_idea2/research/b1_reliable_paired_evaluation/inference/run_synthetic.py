"""B1 known-population paired-binary evaluation, without language-model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from b1_inference import population_many


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def protocol():
    common = dict(weights=[.25, .25, .25, .25], disagreement=[.04, .10, .22, .44],
                  expected_tokens=[8000., 8000., 8000., 8000.], union_size=[4, 4, 4, 4])
    worlds = [dict(name=name, gain_by_stratum=[gain]*4, uncertainty_reversed=False, **common)
              for name, gain in (("null_zero", 0.), ("negative_2pp", -.02),
                                  ("positive_1pp", .01), ("positive_3pp", .03))]
    heterogeneous = dict(weights=[.4, .3, .2, .1], disagreement=[.01, .04, .20, .65],
                         gain_by_stratum=[0., .02, .07, .10],
                         expected_tokens=[3000., 8000., 18000., 40000.], union_size=[4, 4, 5, 6])
    worlds.extend([dict(name="heterogeneous", uncertainty_reversed=False, **heterogeneous),
                   dict(name="heterogeneous_wrong_uncertainty", uncertainty_reversed=True, **heterogeneous)])
    return {
        "scope": "Known iid synthetic populations; no alteration or access to real validation rewards",
        "worlds": worlds, "n_questions": [200, 1000, 5000], "monte_carlo_repetitions": 2000,
        "generation_repetitions_per_question": 3,
        "generation_correlation": ["conditionally_independent", "perfectly_correlated"],
        "primary_budget": {"answer_fraction": .25, "expected_execution_fraction": .25},
        "methods": ["U0", "U1", "ASI-adapted", "Robust-ASI-adapted", "T0", "T1"],
        "p_min": .05, "alpha": .05, "gain_thresholds": [0., .01],
        "history_questions": 600, "history_fit": "Stratum mean with 20-observation global shrinkage; five question-level folds",
        "history_seed": 2026111309, "evaluation_seed": 2026111310,
        "uncertainty_fit": "Historical held-out squared residual stratum shrinkage; case 6 reverses the four stratum proxy values",
        "public_tuning": "ASI author 100-point tau grid and Robust author six-candidate c grid on historical held-out residuals",
        "pair_generation": "P(D=+1)=(rho+gain)/2, P(D=-1)=(rho-gain)/2; D=0 gives common Bernoulli(.6) correctness",
        "actual_execution_cost": "Predicted stratum tokens multiplied by independent Uniform(.75,1.25); hidden until purchased",
        "population_design": "Fixed stratum probabilities from known input/cost population mass, independently of each evaluated question sample",
        "history_cost_role": "600 common historical paired packages, separately reported; current evaluation has zero pilot labels",
        "inference": "Fixed-time iid question-cluster CLT and finite bounded-variance-upper Bernstein, separately labeled",
        "always_reject_reference": "zero false approval and zero detection; included without an efficiency claim",
        "no_optional_stopping": True,
    }


def paired_rewards(rng, strata, world, perfectly_correlated=False):
    rho = np.asarray(world["disagreement"])[strata]
    gain = np.asarray(world["gain_by_stratum"])[strata]
    repetitions = 1 if perfectly_correlated else 3
    u = rng.random(strata.shape + (repetitions,))
    plus = (rho + gain)[..., None] / 2
    minus = (rho - gain)[..., None] / 2
    diff = np.where(u < plus, 1, np.where(u < plus + minus, -1, 0))
    common_correct = rng.random(diff.shape) < .6
    default = np.where(diff > 0, False, np.where(diff < 0, True, common_correct))
    candidate = np.where(diff > 0, True, np.where(diff < 0, False, common_correct))
    if perfectly_correlated:
        default = np.repeat(default, 3, axis=-1)
        candidate = np.repeat(candidate, 3, axis=-1)
    return default.astype(np.int8), candidate.astype(np.int8)


def shrunk_by_stratum(values, strata, prior_weight=20.):
    global_mean = float(np.mean(values))
    return np.array([(values[strata == h].sum() + prior_weight * global_mean)
                     / ((strata == h).sum() + prior_weight) for h in range(4)])


def fit_history(world, history_seed, tune_asi_tau, tune_robust_radius):
    rng = np.random.default_rng(history_seed)
    weights = np.asarray(world["weights"])
    counts = np.rint(600 * weights).astype(int)
    strata = np.repeat(np.arange(4), counts)
    default, candidate = paired_rewards(rng, strata, world)
    d = (candidate-default).mean(axis=-1)
    folds = np.empty(600, dtype=int)
    for h in range(4):
        locations = np.flatnonzero(strata == h)
        folds[locations] = rng.permutation(np.arange(len(locations)) % 5)
    oof_m = np.empty(600)
    oof_v = np.empty(600)
    calibration_folds = []
    for fold in range(5):
        train = folds != fold
        heldout = ~train
        means = shrunk_by_stratum(d[train], strata[train])
        residuals = d[train] - means[strata[train]]
        v = np.maximum(shrunk_by_stratum(residuals**2, strata[train]), 1e-6)
        if world["uncertainty_reversed"]:
            v = v[::-1]
        oof_m[heldout] = means[strata[heldout]]
        oof_v[heldout] = v[strata[heldout]]
        calibration_folds.append(dict(predicted_e2=oof_v[heldout], residual=d[heldout]-oof_m[heldout]))
    m = np.clip(shrunk_by_stratum(d, strata), -1, 1)
    v = np.maximum(shrunk_by_stratum((d-oof_m)**2, strata), 1e-6)
    if world["uncertainty_reversed"]:
        v = v[::-1]
    raw_second_moment = np.maximum(shrunk_by_stratum(d**2, strata), 1e-6)
    tau = tune_asi_tau(np.sqrt(oof_v), d-oof_m, .25)
    robust = tune_robust_radius(calibration_folds, .25)
    actual_tokens = np.asarray(world["expected_tokens"])[strata] * rng.uniform(.75, 1.25, size=600)
    return {"m": m, "v": v, "raw_second_moment": raw_second_moment,
            "tau": tau, "robust_tuning": robust,
            "oof_residual_mse": float(np.mean((d-oof_m)**2)),
            "history_actual_tokens": float(actual_tokens.sum()),
            "history_gold": 600, "history_member_calls": int(np.sum(3*np.asarray(world["union_size"])[strata])),
            "history_aggregation_calls": 3600,
            "history_grading_calls": 3600,
            "arrays": dict(strata=strata, default=default, candidate=candidate,
                           d=d, oof_m=oof_m, oof_v=oof_v, folds=folds, actual_tokens=actual_tokens)}


def weighted_optimum(v, costs, weights, optimal_probabilities):
    counts = np.rint(100*np.asarray(weights)).astype(int)
    labels = np.repeat(np.arange(len(weights)), counts)
    p, diagnostics = optimal_probabilities(
        np.asarray(v)[labels], np.asarray(costs)[labels], .25*len(labels),
        .25*np.asarray(costs)[labels].sum(), pmin=.05)
    grouped = np.array([p[labels == h].mean() for h in range(len(weights))])
    assert np.max(np.abs(p-grouped[labels])) < 1e-10
    diagnostics["weighted_stratum_expansion"] = counts.tolist()
    return grouped, diagnostics


def designs(world, history, n, methods, optimizer):
    weights = np.asarray(world["weights"])
    costs = np.asarray(world["expected_tokens"])
    bann, bexec = .25, .25*(weights @ costs)
    uniform = methods.scale_to_budgets(np.ones(4), costs, bann, bexec, weights=weights)
    asi = methods.asi_probabilities(history["v"], costs, bann, bexec,
                                   tau=history["tau"]["tau"], weights=weights)
    robust = methods.robust_probabilities(history["v"], costs, bann, bexec,
                                         c_radius=history["robust_tuning"]["c_radius"],
                                         weights=weights, n_population=n)
    t0p, t0diag = weighted_optimum(history["raw_second_moment"], costs, weights, optimizer)
    t1p, t1diag = weighted_optimum(history["v"], costs, weights, optimizer)
    return {
        "U0": dict(p=uniform["p"], m=np.zeros(4), design=uniform),
        "U1": dict(p=uniform["p"], m=history["m"], design=uniform),
        "ASI-adapted": dict(p=asi["p"], m=history["m"], design=asi),
        "Robust-ASI-adapted": dict(p=robust["p"], m=history["m"], design=robust),
        "T0": dict(p=t0p, m=history["m"], design=t0diag),
        "T1": dict(p=t1p, m=history["m"], design=t1diag),
    }


def wilson(successes, repetitions):
    z = norm.ppf(.975)
    p = successes/repetitions
    denominator = 1+z*z/repetitions
    middle = (p+z*z/(2*repetitions))/denominator
    radius = z*np.sqrt(p*(1-p)/repetitions+z*z/(4*repetitions**2))/denominator
    return [float(middle-radius), float(middle+radius)]


def summarize(values, truth, threshold):
    estimate = values["estimate"]
    error = estimate-truth
    result = {"bias": float(error.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
              "mse": float(np.mean(error**2)), "mean_gold": float(values["gold"].mean()),
              "mean_actual_tokens": float(values["actual_tokens"].mean()),
              "mean_member_calls": float(values["member_calls"].mean()),
              "mean_aggregation_calls": float(values["aggregation_calls"].mean()),
              "zero_labels_count": int((values["gold"] == 0).sum()),
              "answer_budget_exceedance": float((values["gold"] > values["answer_budget"]).mean()),
              "actual_token_budget_exceedance": float((values["actual_tokens"] > values["token_budget"]).mean())}
    repetitions = len(estimate)
    for family in ("population_bernstein", "population_wald"):
        lower, upper = values[family+"_lower"], values[family+"_upper"]
        covered = (lower <= truth+1e-12) & (upper >= truth-1e-12)
        adopt = values[family+"_one_lower"] > threshold
        below = values[family+"_one_upper"] <= threshold
        result[family] = {
            "coverage": float(covered.mean()), "coverage_mc95": wilson(covered.sum(), repetitions),
            "mean_width": float(np.mean(upper-lower)),
            "adopt_probability": float(adopt.mean()), "adopt_mc95": wilson(adopt.sum(), repetitions),
            "below_probability": float(below.mean()),
            "insufficient_probability": float((~adopt & ~below).mean()),
            "false_approval_probability": float(adopt.mean()) if truth <= threshold else None,
            "detection_probability": float(adopt.mean()) if truth > threshold else None,
        }
    return result


def run_setting(world, history, design, n, repetitions, correlated, output, evaluation_seed):
    rng = np.random.default_rng(evaluation_seed)
    weights = np.asarray(world["weights"])
    truth = float(weights @ np.asarray(world["gain_by_stratum"]))
    output.mkdir(parents=True, exist_ok=True)
    names = list(design)
    records = {name: {} for name in names}
    for name in names:
        for key in ("estimate", "gold", "actual_tokens", "member_calls", "aggregation_calls",
                    "population_bernstein_lower", "population_bernstein_upper",
                    "population_bernstein_one_lower", "population_bernstein_one_upper",
                    "population_wald_lower", "population_wald_upper",
                    "population_wald_one_lower", "population_wald_one_upper"):
            records[name][key] = []
    start = time.perf_counter()
    for batch_start in range(0, repetitions, 100):
        batch = min(100, repetitions-batch_start)
        strata = rng.choice(4, size=(batch, n), p=weights)
        default, candidate = paired_rewards(rng, strata, world, correlated)
        d = (candidate-default).mean(axis=-1)
        random_buys = rng.random((batch, n))
        actual_tokens = np.asarray(world["expected_tokens"])[strata] * rng.uniform(.75, 1.25, size=(batch, n))
        members = 3*np.asarray(world["union_size"])[strata]
        for name, spec in design.items():
            p = np.asarray(spec["p"])[strata]
            m = np.asarray(spec["m"])[strata]
            purchased = random_buys < p
            q = m + purchased*(d-m)/p
            estimate = q.mean(axis=1)
            gold = purchased.sum(axis=1)
            p_support, m_support = np.asarray(spec["p"]), np.asarray(spec["m"])
            global_bound = float(max(np.max(np.abs(m_support)),
                                     np.max(np.abs(m_support+(1-m_support)/p_support)),
                                     np.max(np.abs(m_support+(-1-m_support)/p_support))))
            intervals = population_many(estimate, np.sum(q*q, axis=1), gold, n, global_bound)
            values = dict(estimate=estimate, gold=gold,
                          actual_tokens=np.sum(purchased*actual_tokens, axis=1),
                          member_calls=np.sum(purchased*members, axis=1), aggregation_calls=6*gold,
                          **{key: value for key, value in intervals.items() if key in records[name]})
            for key, value in values.items():
                records[name][key].append(value)
    results = {}
    for name in names:
        values = {key: np.concatenate(value) for key, value in records[name].items()}
        values["answer_budget"] = .25*n
        values["token_budget"] = .25*n*(weights @ np.asarray(world["expected_tokens"]))
        np.savez_compressed(output / (name.replace("-", "_")+".npz"), **values)
        results[name] = {"threshold_0": summarize(values, truth, 0.),
                         "threshold_1pp": summarize(values, truth, .01),
                         "design": plain(design[name])}
        records[name] = values
    contrasts = {}
    for baseline in ("U1", "ASI-adapted", "Robust-ASI-adapted"):
        delta = ((records["T1"]["estimate"]-truth)**2
                 -(records[baseline]["estimate"]-truth)**2)
        half = 1.96*delta.std(ddof=1)/np.sqrt(repetitions)
        contrasts[baseline] = {"mse_difference_T1_minus_baseline": float(delta.mean()),
                               "paired_monte_carlo_ci95": [float(delta.mean()-half), float(delta.mean()+half)],
                               "interpretation": "Monte Carlo uncertainty over fresh iid synthetic question samples and purchase randomization"}
    return {"world": world["name"], "n": n, "repetitions": repetitions,
            "correlation": "perfectly_correlated" if correlated else "conditionally_independent",
            "true_population_gain": truth, "results": results, "paired_mse_contrasts": contrasts,
            "runtime_seconds": time.perf_counter()-start,
            "always_reject_reference": {"false_approval_probability": 0., "detection_probability": 0.,
                                         "insufficient_probability": 1., "method_success_claim": False}}


def write_report(results, output):
    lines = ["# B1 已知总体合成实验", "", "## 研究问题", "",
             "冻结评估系统后，检查真实总体收益估计、区间覆盖、错误批准和正收益检出能力。每个数值由2000批独立合成问题簇及购买随机化产生。", "",
             "## 实验设置与对照", "",
             "六类总体、200/1000/5000题、每题三个生成重复。主实验题内条件独立，敏感性实验题内完全相关。辅助模型仅用独立历史600题。ASI和Robust为明确标注的B1采样适配，使用共同AIPW估计与推断。", "",
             "预算为总体期望Gold与执行费用各25%。当前评价输入按已知总体类别质量制定固定概率函数。实际开销可以超过期望预算。", "",
             "## 数值结果", "",
             "| 环境 | 题数 | 相关性 | 方法 | RMSE | 渐近覆盖 | 保守覆盖 | 保守采用率 | 渐近采用率 | 平均Gold | 平均Token |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for setting in results:
        for method, methods_result in setting["results"].items():
            row = methods_result["threshold_0"]
            finite, asym = row["population_bernstein"], row["population_wald"]
            lines.append(f"| {setting['world']} | {setting['n']} | {setting['correlation']} | {method} | {row['rmse']:.5f} | {asym['coverage']:.4f} | {finite['coverage']:.4f} | {finite['adopt_probability']:.4f} | {asym['adopt_probability']:.4f} | {row['mean_gold']:.2f} | {row['mean_actual_tokens']:.0f} |")
    lines += ["", "## 统计结论与失败原因", "",
              "有限样本保守区间和渐近Wald区间分开报告。保守界可能因全支持逆概率幅度而明显变宽，误差较小不自动对应更强检出能力。零收益和负收益环境的采用率为错误批准，正收益环境的采用率为检出能力。", "",
              "总是拒绝采用的参照具有零错误批准和零检出能力。此参照没有被解释为评估方法成功。", "",
              "每种采样设计在相同历史与权限下评估。主候选的风险是历史残差方向错误时重新分配预算会提高方差，公开鲁棒采样的收益也必须由实测比较决定。", "",
              "## 80%检出率资源", "",
              "| 环境 | 相关性 | 方法 | 推断 | 首个达到80%的题数 | 平均Gold | 平均Token |", "|---|---|---|---|---|---|---|"]
    for world in dict.fromkeys(x["world"] for x in results):
        for correlation in ("conditionally_independent", "perfectly_correlated"):
            settings = [x for x in results if x["world"] == world and x["correlation"] == correlation]
            if not settings or settings[0]["true_population_gain"] <= 0:
                continue
            for method in settings[0]["results"]:
                for inference in ("population_bernstein", "population_wald"):
                    reached = [x for x in settings if x["results"][method]["threshold_0"][inference]["adopt_probability"] >= .8]
                    if reached:
                        first = min(reached, key=lambda x: x["n"])
                        row = first["results"][method]["threshold_0"]
                        cells = [str(first["n"]), f"{row['mean_gold']:.2f}", f"{row['mean_actual_tokens']:.0f}"]
                    else:
                        cells = ["未达到，最大5000题", "未确定", "未确定"]
                    lines.append("| "+" | ".join([world, correlation, method, inference]+cells)+" |")
    lines += ["", "## 下一步", "", "这些结果验证合成总体中的条件性机制。真实有限题池的恢复精度与未来问题总体的覆盖分别解释，后续独立真实题评价需要另行冻结预算。"]
    (output / "SYNTHETIC_RELIABILITY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(args):
    sys.path.insert(0, str(Path(args.methods_directory)))
    sys.path.insert(0, str(Path(args.design_directory)))
    import public_methods as methods
    from b1_design import optimal_probabilities
    output = Path(args.output)
    private = Path(args.private)
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    spec = protocol()
    if args.repetitions is not None:
        spec["monte_carlo_repetitions"] = args.repetitions
    encoded = json.dumps(spec, ensure_ascii=False, sort_keys=True, indent=2).encode()
    protocol_path = output / "SYNTHETIC_PROTOCOL.json"
    if protocol_path.exists() and protocol_path.read_bytes() != encoded:
        raise RuntimeError("Synthetic protocol already exists with a different configuration")
    protocol_path.write_bytes(encoded)
    print(json.dumps({"protocol_sha256": hashlib.sha256(encoded).hexdigest(), "status": "FROZEN"}), flush=True)
    histories, results = {}, []
    for index, world in enumerate(spec["worlds"]):
        history_index = min(index, 4)
        history = fit_history(world, spec["history_seed"]+history_index,
                              methods.tune_asi_tau, methods.tune_robust_radius)
        np.savez_compressed(private / (world["name"]+"_history600.npz"), **history.pop("arrays"))
        histories[world["name"]] = plain(history)
        for n in spec["n_questions"]:
            design = designs(world, history, n, methods, optimal_probabilities)
            for correlated in (False, True):
                setting_path = private / world["name"] / str(n) / ("correlated" if correlated else "independent")
                cache = setting_path / "SUMMARY.json"
                if cache.exists():
                    result = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    result = run_setting(world, history, design, n, spec["monte_carlo_repetitions"], correlated,
                                         setting_path, spec["evaluation_seed"]+history_index*100+n+int(correlated)*100000)
                    cache.write_text(json.dumps(plain(result), ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(result)
                print(json.dumps({"world": world["name"], "n": n, "correlation": result["correlation"],
                                  "seconds": result["runtime_seconds"], "status": "COMPLETE"}), flush=True)
                (output / "SYNTHETIC_COMPLETE_RESULTS.json").write_text(
                    json.dumps({"protocol_sha256": hashlib.sha256(encoded).hexdigest(), "histories": histories,
                                "settings": results, "complete": len(results) == 36}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, output)
    print(json.dumps({"status": "COMPLETE", "settings": len(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--private", required=True)
    parser.add_argument("--methods-directory", default=str(Path(__file__).parents[1] / "methods"))
    parser.add_argument("--design-directory", default=str(Path(__file__).parents[1] / "main"))
    parser.add_argument("--repetitions", type=int)
    main(parser.parse_args())
