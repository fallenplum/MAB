"""Read completed E4 assets and summarize repeated paths at original-question level.

Outputs default to this staging directory. Existing E results, including the
sixteen-configuration C selection, are never modified by this script.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from e4_common import PRIVATE, PUBLIC, TEAMS, load_train, semantic, sha
from e4_metrics import paired_summary, prediction_metrics, scope_means


METRICS = ("accuracy", "top1_regret", "brier", "mean_mse", "raw_mse")
SCOPES = ("all3", "samegen", "crossseed")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _mean_resources(items):
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]} if items else {}


def _service_cost(data, q, chosen, generation, earlier=None):
    members = np.asarray([[i in TEAMS[int(action)] for i in range(6)] for action in chosen])
    aggregates = np.eye(20, dtype=bool)[chosen]
    if earlier is not None:
        members |= np.asarray([[i in TEAMS[int(action)] for i in range(6)] for action in earlier])
        aggregates |= np.eye(20, dtype=bool)[earlier]
    result = {"member_calls": 0, "aggregator_calls": 0, "input_tokens": 0,
              "output_tokens": 0, "tokens": 0, "source_model_wall_s": 0.}
    for stem, mask, calls in (("member", members, "member_calls"), ("aggregate", aggregates, "aggregator_calls")):
        for suffix, field in (("execution_counts", calls), ("prompt_tokens", "input_tokens"),
                              ("completion_tokens", "output_tokens"), ("wall_time_s", "source_model_wall_s")):
            value = (data["cost"][stem + "_" + suffix][generation, q] * mask).sum()
            result[field] += float(value) if suffix == "wall_time_s" else int(value)
    result["tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _ci_columns(columns, data, q):
    result = paired_summary(np.column_stack(columns), data["task"][q], data["sub"][q])
    def column(index):
        answer = {}
        for scope, values in result.items():
            if scope == "metadata":
                continue
            answer[scope] = {name: (np.asarray(value)[index].tolist()) for name, value in values.items()}
        answer["metadata"] = result["metadata"]
        return answer
    return column


def online_summary(data, full_cost):
    buffers, refits, source_hashes = {}, [], []
    seen = set()
    for path in sorted((PRIVATE / "binary_online").glob("*.json")):
        case = read(path)
        meta = case["meta"]
        key = (meta["predictor"], meta["mode"], meta["family"], meta["permission"])
        unique = key + (meta["fold"], meta["generation_index"], meta["orderseed"])
        if unique in seen:
            raise ValueError("Duplicate binary online path")
        seen.add(unique)
        source_hashes.append(sha(path))
        if key not in buffers:
            buffers[key] = {"count": np.zeros(600), "before": {}, "after": {},
                            "curve": {metric: np.zeros((21, 600)) for metric in METRICS},
                            "correct": np.zeros(600), "harm": np.zeros(600),
                            "seed_count": np.zeros((3, 600)), "seed_gain": np.zeros((3, 600)),
                            "paths": [], "support_costs": [], "deployment_costs": [],
                            "evaluation_costs": [], "paired_union_costs": [], "trace": [],
                            "negative_events": [], "runtime": []}
            for scope in SCOPES:
                buffers[key]["before"][scope] = {metric: np.zeros(600) for metric in METRICS}
                buffers[key]["after"][scope] = {metric: np.zeros(600) for metric in METRICS}
        b = buffers[key]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
            if str(a["signature"]) != case["signature"]:
                raise ValueError("Binary online metadata and arrays have different signatures")
            q = a["query"].astype(int)
            if len(q) != 100 or a["accuracy"].shape != (21, 100):
                raise ValueError("Incomplete binary online evaluator")
            generation = int(meta["generation_index"])
            b["count"][q] += 1
            b["seed_count"][generation, q] += 1
            b["seed_gain"][generation, q] += a["accuracy"][-1] - a["accuracy"][0]
            for metric in METRICS:
                b["curve"][metric][:, q] += a[metric]
                for step, side in ((0, "before"), (-1, "after")):
                    all3, same = a[metric][step], a["samegen_" + metric][step]
                    if metric == "mean_mse":
                        pred = a["initial_pred"] if step == 0 else a["final_pred"]
                        target = data["Y"][q][:, [g for g in range(3) if g != generation]].mean(axis=1)
                        cross = ((pred - target) ** 2).mean(axis=1)
                    else:
                        cross = (3 * all3 - same) / 2
                    for scope, values in (("all3", all3), ("samegen", same), ("crossseed", cross)):
                        b[side][scope][metric][q] += values
            chosen = a["chosen"]
            rewards = data["Y"][q]
            before = rewards[np.arange(100)[:, None], np.arange(3)[None, :], chosen[0, :, None]]
            after = rewards[np.arange(100)[:, None], np.arange(3)[None, :], chosen[-1, :, None]]
            b["correct"][q] += ((before == 0) & (after == 1)).mean(axis=1)
            b["harm"][q] += ((before == 1) & (after == 0)).mean(axis=1)
            b["trace"].append({"initial": float(a["trace"][0]), "final": float(a["trace"][-1]),
                                "mean_step_mean_change": float(a["mean_change"].mean())})
            shrinking = np.diff(a["trace"]) < -1e-14
            b["negative_events"].append({"updates": 20, "trace_decreases": int(shrinking.sum()),
                "regret_increases_while_trace_decreases": int(np.sum(shrinking & (np.diff(a["top1_regret"].mean(axis=1)) > 1e-12))),
                "mean_mse_increases_while_trace_decreases": int(np.sum(shrinking & (np.diff(a["mean_mse"].mean(axis=1)) > 1e-12)))})
            b["deployment_costs"].append(_service_cost(data, q, chosen[-1], generation))
            repeated = [_service_cost(data, q, chosen[-1], g) for g in range(3)]
            b["evaluation_costs"].append({field: sum(item[field] for item in repeated) for field in repeated[0]})
            union = [_service_cost(data, q, chosen[-1], g, chosen[0]) for g in range(3)]
            b["paired_union_costs"].append({field: sum(item[field] for item in union) for field in union[0]})
        b["support_costs"].append(case["cost"]["ledger"]["total"])
        b["runtime"].append(case["runtime"])
        b["paths"].append(unique)
        for comparison in case["full_refit_comparisons"]:
            refits.append({"predictor": key[0], "mode": key[1], "family": key[2], **comparison})
    if len(seen) != 720 or len(buffers) != 16:
        raise ValueError("Expected 720 completed paths across sixteen online contexts")
    columns, locations, raw = [], [], {}
    common_q = None
    for key in sorted(buffers):
        b = buffers[key]
        q = np.flatnonzero(b["count"])
        if len(q) != 500 or len(b["paths"]) != 45 or not np.all(b["count"][q] == 9):
            raise ValueError("Repeated paths do not cover the same 500 original Query questions nine times")
        if common_q is None:
            common_q = q
        if not np.array_equal(q, common_q):
            raise ValueError("Online contexts have different independent Query populations")
        raw[key] = {side: {scope: {metric: b[side][scope][metric][q] / b["count"][q] for metric in METRICS}
                           for scope in SCOPES} for side in ("before", "after")}
        for scope in SCOPES:
            for metric in METRICS:
                locations.append((key, scope, metric))
                columns.append(raw[key]["after"][scope][metric] - raw[key]["before"][scope][metric])
    extract = _ci_columns(columns, data, common_q)
    intervals = {location: extract(index) for index, location in enumerate(locations)}
    table = {}
    labels = semantic(data)[common_q]
    for key in sorted(buffers):
        b, values = buffers[key], raw[key]
        gain = values["after"]["all3"]["accuracy"] - values["before"]["all3"]["accuracy"]
        per_task = {str(task): float(np.mean(gain[labels == task])) for task in sorted(set(labels))}
        negative = [task for task, value in per_task.items() if task != "gsm8k" and value < -1e-12]
        name = "_".join(key)
        table[name] = {"config": dict(zip(("predictor", "mode", "family", "permission"), key)),
            "original_query_questions": 500, "path_count": 45, "repeats_per_query": 9,
            "before": {metric: scope_means(values["before"]["all3"][metric], data["task"][common_q], data["sub"][common_q]) for metric in METRICS},
            "after": {metric: scope_means(values["after"]["all3"][metric], data["task"][common_q], data["sub"][common_q]) for metric in METRICS},
            "changes": {metric: intervals[(key, "all3", metric)] for metric in METRICS},
            "samegen_changes": {metric: intervals[(key, "samegen", metric)] for metric in METRICS},
            "crossseed_changes": {metric: intervals[(key, "crossseed", metric)] for metric in METRICS},
            "expected_corrected_questions": float(np.sum(b["correct"][common_q] / b["count"][common_q])),
            "expected_harmed_questions": float(np.sum(b["harm"][common_q] / b["count"][common_q])),
            "per_semantic_task_accuracy_gain": per_task, "negative_BBH_tasks": negative,
            "negative_BBH_task_fraction": len(negative) / 27,
            "per_support_seed_accuracy_gain": {str((42, 137, 2024)[g]): scope_means(b["seed_gain"][g, common_q] / b["seed_count"][g, common_q], data["task"][common_q], data["sub"][common_q]) for g in range(3)},
            "quality_curves": {metric: scope_means((b["curve"][metric][:, common_q] / b["count"][common_q]).T, data["task"][common_q], data["sub"][common_q]) for metric in METRICS},
            "parameter_summary": {field: float(np.mean([item[field] for item in b["trace"]])) for field in b["trace"][0]},
            "overlapping_update_event_summary": {field: float(np.mean([item[field] for item in b["negative_events"]])) for field in b["negative_events"][0]},
            "cost_roles_mean_per_path": {"support_logger": _mean_resources(b["support_costs"]),
                "query100_one_generation_deployment": _mean_resources(b["deployment_costs"]),
                "query100_three_generation_repeated_evaluation": _mean_resources(b["evaluation_costs"]),
                "query100_before_after_union_three_generation_evaluation": _mean_resources(b["paired_union_costs"]),
                "query_gold_for_evaluation_only": 100,
                "recorded_replay_process_cpu_s": float(np.mean([item["process_cpu_s"] for item in b["runtime"]])),
                "recorded_full_refit_wall_s": float(np.mean([item["full_refit_wall_s"] for item in b["runtime"]]))}}
    refit_summary = []
    fields = ("sequential_batch_mean_l2", "sequential_batch_covariance_relative_frobenius",
              "query_probability_mean_absolute_difference", "query_service_disagreement_count",
              "gradient_infinity", "cpu_s", "wall_s")
    for family in ("logistic", "probit"):
        for step in (1, 5, 20):
            subset = [row for row in refits if row["family"] == family and row["step"] == step]
            if len(subset) != 20:
                raise ValueError("Full-refit comparison grid is incomplete")
            refit_summary.append({"family": family, "step": step, "cases": len(subset),
                **{field: {"mean": float(np.mean([row[field] for row in subset])),
                            "maximum": float(np.max([row[field] for row in subset]))} for field in fields}})
    return {"status": "COMPLETE", "table": table, "paths": 720, "original_query_questions": 500,
            "ci_numerical_zero_tolerance": 1e-12,
            "positive_accuracy_ci_contexts": [key for key, item in table.items()
                                               if item["changes"]["accuracy"]["balanced_v2"]["ci95"][0] > 1e-12],
            "full_refit_comparisons": refit_summary, "full_refit_records": len(refits),
            "source_manifest_sha256": _hash_manifest(source_hashes),
            "historical_preparation_by_fold": full_cost["history_preparation_by_fold"],
            "cost_reference": {"file": "FULL_SERVICE_COST_ACCOUNTS.json", "sha256": sha(PUBLIC / "FULL_SERVICE_COST_ACCOUNTS.json")},
            "inference": "Average all generation/order repetitions within each original Query before paired descriptive confidence intervals; conditional on fixed historical fits and legal feedback paths",
            "cost_note": "Alternative paths are not one deployment bill. Single-generation deployment overlaps the three-generation evaluation account. Before/after evaluation uses execution unions. Historical preparation, Support and evaluator gold remain separate.",
            "actual_gpu_kernel_time": None, "source_wall_is_gpu_kernel_time": False,
            "new_physical_model_calls": 0}


def _hash_manifest(values):
    import hashlib
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def metadata_summary(data, frozen_primary):
    metrics, infos, counts = {}, defaultdict(list), {}
    families = ("gaussian", "gaussian-correlated", "logistic", "probit")
    for mode in ("text-only", "task-known", "subtask-oracle", "subtask-predicted"):
        for fold in range(5):
            path = PRIVATE / "binary" / f"f{fold}_P2_{mode}.json"
            case = read(path)
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as a:
                q = a["query"].astype(int)
                for index, detail in enumerate(case["models"]):
                    family = detail["family"]
                    if family not in families:
                        continue
                    key = f"P2_{mode}_{family}"
                    if key not in metrics:
                        metrics[key] = {metric: np.full(600, np.nan) for metric in METRICS}
                        metrics[key]["log_loss"] = np.full(600, np.nan)
                        counts[key] = np.zeros(600)
                    counts[key][q] += 1
                    computed = prediction_metrics(a["predictions"][index], data["Y"][q])["per_question"]
                    for metric in metrics[key]:
                        metrics[key][metric][q] = computed[metric]
                    infos[key].append({"fold": fold, "alpha": detail["alpha"], "metadata": detail["metadata"],
                                       "fit_wall_s": detail["fit_info"].get("wall_s")})
    if any(not np.all(value == 1) for value in counts.values()):
        raise ValueError("Metadata OOF evaluation must cover every original question exactly once")
    table = {key: {"metrics": {metric: scope_means(value, data["task"], data["sub"]) for metric, value in values.items()},
                   "fits": infos[key], "original_questions": 600} for key, values in metrics.items()}
    for key, entry in table.items():
        if "_text-only_" in key or "_task-known_" in key:
            for metric in METRICS:
                if abs(entry["metrics"][metric]["balanced_v2"] - frozen_primary["table"][key]["metrics"][metric]["balanced_v2"]) > 1e-12:
                    raise ValueError("Primary C cached predictions differ from its frozen published values")
    columns, names = [], []
    for mode in ("subtask-oracle", "subtask-predicted"):
        for family in families:
            key = f"P2_{mode}_{family}"
            for baseline in ("text-only", "task-known"):
                name = key + "_minus_" + baseline
                for metric in METRICS:
                    names.append((name, metric))
                    columns.append(metrics[key][metric] - metrics[f"P2_{baseline}_{family}"][metric])
    extract = _ci_columns(columns, data, np.arange(600))
    comparisons = defaultdict(dict)
    for index, (name, metric) in enumerate(names):
        comparisons[name][metric] = extract(index)
    classifier = []
    for fold in range(5):
        info = infos["P2_subtask-predicted_gaussian"][fold]["metadata"]
        confusion = np.asarray(info["oof_confusion_matrix"])
        labels = info["oof_confusion_order"]
        gsm = labels.index("gsm8k")
        bbh = np.arange(len(labels)) != gsm
        classifier.append({"fold": fold, "history_questions": info["history_questions"],
                           "history_OOF_accuracy": info["oof_accuracy"],
                           "history_OOF_GSM_accuracy": float(confusion[gsm, gsm] / confusion[gsm].sum()),
                           "history_OOF_BBH_micro_accuracy": float(np.diag(confusion)[bbh].sum() / confusion[bbh].sum()),
                           "feature_cpu_s": info["cpu_s"], "feature_wall_s": info["wall_s"]})
    return {"status": "COMPLETE", "table": table, "comparisons": dict(comparisons),
            "additional_fit_groups": 10, "additional_model_fits": 40, "original_questions": 600,
            "history_metadata_classifier": classifier,
            "primary_C_selection_unchanged": True,
            "privilege": "Oracle has true external BBH subtask labels; predicted metadata uses historical OOF and full-history classifiers only",
            "multiplicity": "Exploratory descriptive intervals, unadjusted; no new winner replaces the frozen primary sixteen configurations",
            "new_physical_model_calls": 0}


def _percent(value):
    return f"{100 * value:.3f}%"


def _interval(item, scope="balanced_v2", scale=100.):
    point, bounds = item[scope]["difference"], item[scope]["ci95"]
    def display(value):
        return "0.000" if abs(value) <= 1e-12 else f"{value * scale:+.3f}"
    return f"{display(point)} [{display(bounds[0])}, {display(bounds[1])}]"


def reports(primary, online, metadata, optimization=None):
    report05 = ["# 05 二值奖励模型与在线适应", "", "## 研究问题", "",
        "检验Logistic/Probit能否改善真实组合奖励预测，以及20道合法Support反馈能否改善后续选队。预测尺度、排序和服务准确率分别评价。", "",
        "## 实验设置与对照", "",
        "C主比较保持原16配置冻结，使用600道历史开发题的五折题级OOF。每折480题拟合，120题评价，α只按历史内部三折Brier选择。二值组合独立似然为composite working likelihood。", "",
        "在线适应另用每折固定20道Support与100道Query，合计500道独立Query。P1/P2、text-only/task-known、Logistic/Probit、main/pair共16配置，每配置45条seed/order路径。每题的九次重复先平均，再计算配对描述性95%区间。", "",
        "主口径为GSM8K与BBH各50%的V2任务宏平均。BBH内部按题加权，同时并列列出BBH子任务等权后的两任务宏平均。C的600题与在线的500题属于不同评价人口。", "",
        "## 数值结果", "", "### 冻结C主比较", "",
        "| 配置 | V2准确率 | 子任务宏平均 | Brier | Log loss |", "|---|---:|---:|---:|---:|"]
    for key, entry in sorted(primary["table"].items()):
        m = entry["metrics"]
        report05.append(f"| {key} | {_percent(m['accuracy']['balanced_v2'])} | {_percent(m['accuracy']['balanced_subtask'])} | {m['brier']['balanced_v2']:.8f} | {m['log_loss']['balanced_v2']:.6f} |")
    report05.extend(["", f"冻结的D候选为`{primary['chosen_D_candidate']}`，依据开发OOF Brier选择。P2 text-only的Logistic及Probit相对Gaussian的Brier主比较，Holm调整值均为1。数个百万分之一的Brier差异不能建立优越性。", "",
        "### 20道Support后的独立Query变化", "",
        "下表增益与区间以百分点表示。纠错/破坏为500题上对生成与路径重复平均后的期望题数。", "",
        "| 配置 | 更新前→后V2准确率 | 准确率差及95%CI | 跨seed准确率差及95%CI | 纠错/破坏 | 负迁移BBH子任务 |",
        "|---|---:|---:|---:|---:|---:|"])
    for key, item in sorted(online["table"].items()):
        report05.append(f"| {key} | {_percent(item['before']['accuracy']['balanced_v2'])} → {_percent(item['after']['accuracy']['balanced_v2'])} | {_interval(item['changes']['accuracy'])} | {_interval(item['crossseed_changes']['accuracy'])} | {item['expected_corrected_questions']:.2f}/{item['expected_harmed_questions']:.2f} | {len(item['negative_BBH_tasks'])}/27 |")
    report05.extend(["", "完整逐任务、samegen/crossseed、三种Support seed方向、Brier/MSE/Regret变化及21点曲线保存于`BINARY_ONLINE_RESULTS.json`。同题多seed和多order均未计作独立新题。", "",
        "### 顺序近似与完整重拟合", "",
        "只在预定seed42/order17/pair病例的第1、5、20步进行数值重拟合。参考解保留优化停止误差，未替换顺序演员。", "",
        "| family | 步数 | 参数L2差均值 | 协方差相对Frobenius差均值 | Query预测MAE均值 | 重拟合CPU均值秒 | 重拟合wall均值秒 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for item in online["full_refit_comparisons"]:
        report05.append(f"| {item['family']} | {item['step']} | {item['sequential_batch_mean_l2']['mean']:.3e} | {item['sequential_batch_covariance_relative_frobenius']['mean']:.3e} | {item['query_probability_mean_absolute_difference']['mean']:.3e} | {item['cpu_s']['mean']:.4f} | {item['wall_s']['mean']:.4f} |")
    supported = online["positive_accuracy_ci_contexts"]
    largest = max(online["table"], key=lambda key: online["table"][key]["changes"]["accuracy"]["balanced_v2"]["difference"])
    largest_item = online["table"][largest]
    residual_gradient = max(item["gradient_infinity"]["maximum"] for item in online["full_refit_comparisons"])
    deployment_tokens = [item["cost_roles_mean_per_path"]["query100_one_generation_deployment"]["tokens"]
                         for item in online["table"].values()]
    report05.extend(["", "## 统计结论", "",
        f"在16个在线开发比较中，准确率差描述区间下界超出数值零容差1e-12的配置数为{len(supported)}。最大点增益来自`{largest}`，为{_interval(largest_item['changes']['accuracy'])}个百分点，其下界实际贴近0，浮点量级2e-19不能当作正向显著证据。", "",
        "这些区间条件于已拟合模型和已实现反馈路径，未包含重新抽取历史训练集的误差，也未做本组多重比较校正。", "",
        "Brier/Log loss改善并不自动意味着组合排序或路由准确率改善。收缩的Laplace协方差也不证明后续Regret降低，逐步反例频率在机器可读结果中保留。", "",
        "## 成本与失败原因", "",
        "成本沿用`FULL_SERVICE_COST_ACCOUNTS.json`的角色分隔。每折480题历史准备、20题Support日志、100题单seed部署、三seed重复评价及评价gold分开记录。Binary实际选队的成员/聚合Token费用从原缓存重新核算，前后配对评价使用同题执行并集。单seed部署已包含在三seed评价中，二者不能相加作为总账。", "",
        "| 单条假想执行路径中的角色 | 成员调用 | 聚合调用 | gold揭示 |", "|---|---:|---:|---:|",
        "| 历史480题全20组合×3seed准备 | 8640 | 28800 | 480 |",
        "| 20题main Support | 60 | 20 | 20 |", "| 20题pair Support | 80 | 40 | 20 |",
        "| 100题单seed部署 | 300 | 100 | 部署无需评价gold |",
        "| 100题三seed重复评价 | 900 | 300 | 100，跨seed复用 |", "",
        f"Binary各配置每条100题单seed部署的平均Token为{min(deployment_tokens):.0f}至{max(deployment_tokens):.0f}，按实际选择组合分别核算，未复用Gaussian选择的Token账单。", "",
        f"完整重拟合参考解记录的最大残余梯度为{residual_gradient:.3e}。所列参数/协方差差异同时包含顺序近似与优化停止误差，不能将全部数值差归因于Laplace更新。", "",
        "本轮新增物理模型调用为0。source wall是原调用耗时和，真实GPU kernel时间未知。顺序更新与参考重拟合CPU单列。近似误差、有限样本及较小的稳定组合收益均可限制适应，当前结果不足以仅凭概率范围修正声称路由机制成立。", "",
        "## 下一步", "",
        "保留冻结C候选与所有负结果，按预定D机制审计检查预测增益是否对应后续真实决策改善。本文未使用fresh400或sealed200作确认性结论。", ""])
    report06 = ["# 06 子任务元数据敏感性审计", "", "## 研究问题", "",
        "区分外部真实子任务标识的特权收益与仅靠题目文本预测子任务的可部署收益。", "",
        "## 实验设置与对照", "",
        "本阶段只补P2的subtask-oracle及subtask-predicted，各五折、四种奖励模型，共10组与40个最终拟合。text-only及task-known来自冻结主比较。全部在600道开发题OOF上评价，未改写主比较16配置或D候选。", "",
        "text-only使用历史拟合PCA16与标准化。task-known额外提供真实GSM/BBH任务标识。oracle额外提供真实27种BBH子任务onehot，属于特权信息。predicted使用冻结384维问题向量与历史LogisticRegression C=1分类器，历史概率为三折OOF，外部概率由完整历史分类器生成，不读取外部元标签或奖励。", "",
        "## 数值结果", "", "| 模式与family | V2准确率 | 子任务宏平均 | Brier | 相对同family text-only准确率差及95%CI |", "|---|---:|---:|---:|---:|"]
    for key, item in sorted(metadata["table"].items()):
        m = item["metrics"]
        comparison = metadata["comparisons"].get(key + "_minus_text-only")
        interval = _interval(comparison["accuracy"]) if comparison else "主比较参照"
        report06.append(f"| {key} | {_percent(m['accuracy']['balanced_v2'])} | {_percent(m['accuracy']['balanced_subtask'])} | {m['brier']['balanced_v2']:.8f} | {interval} |")
    report06.extend(["", "### 历史元分类器OOF诊断", "",
        "| 历史折 | 全体分类准确率 | GSM分类准确率 | BBH子任务micro准确率 | 最终特征构造CPU秒 |", "|---|---:|---:|---:|---:|"])
    for row in metadata["history_metadata_classifier"]:
        report06.append(f"| {row['fold']} | {_percent(row['history_OOF_accuracy'])} | {_percent(row['history_OOF_GSM_accuracy'])} | {_percent(row['history_OOF_BBH_micro_accuracy'])} | {row['feature_cpu_s']:.4f} |")
    metadata_positive = sum(metadata["comparisons"][key + "_minus_text-only"]["accuracy"]["balanced_v2"]["ci95"][0] > 1e-12
                            for key in metadata["table"] if "subtask-" in key)
    bbh_classification = [row["history_OOF_BBH_micro_accuracy"] for row in metadata["history_metadata_classifier"]]
    report06.extend(["", "各历史折相互重叠，分类诊断未把5×480题当作2400道独立问题。完整混淆矩阵、OOF缺类、来源哈希与最终特征构造CPU保留在各折元数据记录。", "",
        "## 统计结论", "",
        f"八个新增元数据配置相对同family text-only的准确率差中，描述区间下界超出1e-12的数量为{metadata_positive}。当前结果尚未建立稳定路由准确率收益。历史OOF的GSM识别率均为100%，BBH细分子任务micro识别率为{_percent(min(bbh_classification))}至{_percent(max(bbh_classification))}。", "",
        "表中区间先以原题为单位配对，再按GSM/BBH与BBH子任务分层。元数据比较为探索性描述区间，未调整本组多重比较。oracle的任何收益只能支持任务身份信息的价值，不能当作预测子任务方法的收益。", "",
        "## 失败原因与成本边界", "",
        "BBH子类样本少，OOF某些拟合折可能缺类，对应概率列按0处理。预测特征在历史使用OOF概率、外部使用full-fit概率，两者存在刻意保留的stacking分布差异。主C及本敏感性比较不能凭微小Brier变化宣称胜利。", "",
        "语言模型缓存全部复用，新增物理推理为0。额外成本来自元分类器、历史内部模型选择与CPU拟合，最终特征构造CPU记录不等于整个超参数搜索的总CPU。部署及重复评价按完整服务账本角色分别报告，真实GPU kernel时间未知。", "",
        "## 下一步", "",
        "将特权oracle与历史预测元数据分别保留为敏感性结果。本阶段不替换冻结主候选，不重复使用fresh400或sealed200进行挑选。", ""])
    if optimization is not None:
        initial = optimization["initial_models"]
        references = optimization["checkpoint_references"]
        supplement = ["## 独立数值敏感性审计", "",
            "原L-BFGS参考解的残余梯度未达到1e-7精度目标。本次在独立optimization目录保留原参考记录，追加Newton与Armijo线搜索校正，并在校正点重新计算完整Hessian及Laplace协方差。上文旧参考表保留作为历史记录，主C选择和原在线统计均未替换。", "",
            f"40个初始模型和120个预定检查点均在最多{max(initial['newton_iterations_max'], references['newton_iterations_max'])}步Newton后达到目标。初始模型最大梯度由{initial['gradient_inf_before_max']:.3e}降至{initial['gradient_inf_after_max']:.3e}，检查点参考由{references['gradient_inf_before_max']:.3e}降至{references['gradient_inf_after_max']:.3e}。", "",
            f"polish前后，40个初始模型在各自120题上的argmax改变数为{initial['query_argmax_changed_decisions']}，120个参考点在各自100题上的argmax改变数为{references['query_argmax_changed_decisions']}。最大概率变化分别为{initial['probability_max_absolute_difference']:.3e}与{references['probability_max_absolute_difference']:.3e}，最大协方差相对变化分别为{initial['covariance_relative_frobenius_max']:.3e}与{references['covariance_relative_frobenius_max']:.3e}。这些是模型病例上的重复数值检查，未增加独立问题数量。", "",
            "| family | 步数 | 顺序演员与校正参考参数L2差均值 | 协方差相对差均值 | Query概率MAE均值 |", "|---|---:|---:|---:|---:|"]
        for item in optimization["by_family_step"]:
            values = item["sequential_vs_refined"]
            supplement.append(f"| {item['family']} | {item['step']} | {values['mean_l2']['mean']:.3e} | {values['covariance_relative_frobenius']['mean']:.3e} | {values['probability_mean_absolute_difference']['mean']:.3e} |")
        supplement.extend(["",
            f"Newton校正记录的CPU合计{initial['polish_cpu_s_sum'] + references['polish_cpu_s_sum']:.3f}秒，四进程全流程wall为{optimization['wall_elapsed_s']:.3f}秒。前者仅含polish，后者还包含基础模型及原参考复现，二者口径不同。", "",
            "校正没有改变原初始或原参考的已检查选队，故本审计未重跑演员。顺序近似与校正全量MAP仍是两种不同更新，差异保留。完整结果见`BINARY_OPTIMIZATION_ACCURACY.json`，未使用Query奖励拟合或校正。", ""])
        location = report05.index("## 下一步")
        report05[location:location] = supplement
    return "\n".join(report05), "\n".join(report06)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    primary_path = PUBLIC / "BINARY_COMPARISON_RESULTS.json"
    before_hash = sha(primary_path)
    primary, full_cost = read(primary_path), read(PUBLIC / "FULL_SERVICE_COST_ACCOUNTS.json")
    if len(primary["table"]) != 16:
        raise ValueError("The frozen primary C comparison must retain its sixteen configurations")
    data, _ = load_train()
    online = online_summary(data, full_cost)
    metadata = metadata_summary(data, primary)
    optimization_path = PUBLIC / "BINARY_OPTIMIZATION_ACCURACY.json"
    optimization = read(optimization_path) if optimization_path.exists() else None
    report05, report06 = reports(primary, online, metadata, optimization)
    for name, value in (("BINARY_ONLINE_RESULTS.json", online), ("METADATA_SENSITIVITY_RESULTS.json", metadata)):
        write(args.output / name, value)
    (args.output / "05_BINARY_REWARD_MODELS.md").write_text(report05, encoding="utf-8")
    (args.output / "06_SUBTASK_METADATA_AUDIT.md").write_text(report06, encoding="utf-8")
    if sha(primary_path) != before_hash:
        raise AssertionError("The frozen primary comparison changed")
    print(json.dumps({"status": "COMPLETE", "online_paths": online["paths"],
                      "original_query_questions": online["original_query_questions"],
                      "metadata_additional_groups": metadata["additional_fit_groups"],
                      "primary_C_sha256_unchanged": before_hash, "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
