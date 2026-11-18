#!/usr/bin/env python3
"""Rebuild paper tables from archived public aggregates using Python 3.11 stdlib.

This script does not load question-level data, fit models, score answers, download
data, or calculate inference statistics. It selects archived fields, converts
fractions to percentages, and restores explicitly declared task-macro averages.
The CSV records source fields and arithmetic provenance. --check never writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any


RESEARCH = "routing_idea2/research/"
SOURCES = {
    "e18": RESEARCH + "e18_confirmatory_execution/final_results/E18_POINT_RESULTS.json",
    "e19": RESEARCH + "e19_symmetric_tie_constraint/E19_COMPLETE_RESULTS.json",
    "p2": RESEARCH + "e20_p2_execution/P2_POINT_RESULTS.json",
    "p3_internal": RESEARCH + "e21_p3_external_mechanism/P3_INTERNAL_LISTNET_RESULTS.json",
    "p3_external": RESEARCH + "e21_p3_external_mechanism/P3_EXTERNAL_DARS_RESULTS.json",
    "e22_summary": RESEARCH + "e22_mechanism_specificity/SUMMARY.json",
    "e22_controls": RESEARCH + "e22_mechanism_specificity/controls.csv",
    "e23": RESEARCH + "e23_deployment_prior_and_references/RESULTS_FINAL.json",
}
COLUMNS = (
    "row_id", "section", "stage", "cohort_role", "pool", "scope", "loss",
    "candidate", "reference", "n_questions", "accuracy_pct",
    "reference_accuracy_pct", "difference_pp", "strict_switches",
    "corrected_events", "damaged_events", "event_kind", "decoder",
    "aggregation", "source_file", "source_fields", "notes",
)
NUMERIC_COLUMNS = {
    "accuracy_pct", "reference_accuracy_pct", "difference_pp",
    "corrected_events", "damaged_events",
}
COUNT_COLUMNS = {"n_questions", "strict_switches"}
MACRO = "GSM8K 与 BBH 各 50%；题内生成先平均，BBH 内按原题平均"
TAU = "uniform maximum support; tau=1e-8"
RAW = "raw floating-point maximum support; tau=0; not mathematical exact oracle"
ENSEMBLE = "原六模型选三，20 组合，冻结 V2 聚合器"
EXTERNAL = "DARS 六模型单选；GPQA 与 MATH-500 分别拟合"


def number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"Expected finite numeric value, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value in archived source")
    return result


def close(actual: Any, expected: Any, context: str) -> None:
    if not math.isclose(number(actual), number(expected), rel_tol=1e-11, abs_tol=1e-9):
        raise ValueError(f"Archived aggregate mismatch: {context}: {actual} != {expected}")


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot restore an empty declared task macro")
    return math.fsum(values) / len(values)


def read_sources(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    data, hashes = {}, {}
    for key, relative in SOURCES.items():
        path = root / relative
        raw = path.read_bytes()
        hashes[key] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8-sig")
        if path.suffix == ".csv":
            data[key] = list(csv.DictReader(io.StringIO(text)))
        else:
            data[key] = json.loads(text)
            if not isinstance(data[key], dict):
                raise ValueError(f"Expected JSON object: {relative}")
    return data, hashes


def build_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(row_id: str, section: str, source: str, fields: dict[str, Any], **values: Any) -> None:
        unknown = set(values) - set(COLUMNS)
        if unknown:
            raise ValueError(f"Unknown result columns: {sorted(unknown)}")
        row = {key: "" for key in COLUMNS}
        row.update(row_id=row_id, section=section, source_file=SOURCES[source],
                   source_fields=json.dumps(fields, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")))
        row.update(values)
        for key in NUMERIC_COLUMNS:
            if row[key] != "":
                row[key] = number(row[key])
        for key in COUNT_COLUMNS:
            if row[key] != "":
                numeric = number(row[key])
                if numeric < 0 or not numeric.is_integer():
                    raise ValueError(f"Invalid count {row_id}/{key}")
                row[key] = int(numeric)
        for key in ("accuracy_pct", "reference_accuracy_pct"):
            if row[key] != "" and not 0 <= row[key] <= 100:
                raise ValueError(f"Invalid percentage {row_id}/{key}")
        if all(row[key] != "" for key in ("accuracy_pct", "reference_accuracy_pct", "difference_pp")):
            close(row["accuracy_pct"] - row["reference_accuracy_pct"], row["difference_pp"], row_id)
        if row["strict_switches"] != "" and row["strict_switches"] > row["n_questions"]:
            raise ValueError(f"Strict changes exceed original-question count: {row_id}")
        rows.append(row)

    e18 = data["e18"]
    quality = {item["policy"]: (index, item) for index, item in enumerate(e18["quality"])}
    for policy in ("GlobalFixed", "Ridge", "SPO_coordinate", "TSPO_uniform"):
        index, item = quality[policy]
        add(f"e18_{policy}", "e18_absolute", "e18",
            {"accuracy_pct": f"/quality/{index}/task_macro_pct", "n_questions": f"/quality/{index}/questions"},
            stage="E18", cohort_role="独立确认", pool=ENSEMBLE, scope="两任务等权",
            candidate=policy, n_questions=item["questions"], accuracy_pct=item["task_macro_pct"],
            decoder="E18 archived frozen policy", aggregation=MACRO,
            notes="原计划题池中不可评分题已共同排除；不与 P2 合并。")
    matches = [(index, item) for index, item in enumerate(e18["paired"])
               if item["comparison"] == e18["primary_comparison"]]
    if len(matches) != 1:
        raise ValueError("E18 primary comparison is not unique")
    pair_index, pair = matches[0]
    candidate, reference = e18["primary_comparison"].split(" minus ")
    add("e18_primary", "e18_comparisons", "e18",
        {"comparison": f"/paired/{pair_index}", "candidate": f"/quality/{quality[candidate][0]}",
         "reference": f"/quality/{quality[reference][0]}"},
        stage="E18", cohort_role="独立确认", pool=ENSEMBLE, scope="两任务等权",
        candidate=candidate, reference=reference, n_questions=pair["questions"],
        accuracy_pct=quality[candidate][1]["task_macro_pct"],
        reference_accuracy_pct=quality[reference][1]["task_macro_pct"],
        difference_pp=pair["gain_pp"], corrected_events=pair["corrected_generation_events"],
        damaged_events=pair["damaged_generation_events"], event_kind="整数题目×生成事件",
        decoder="E18 archived frozen policy", aggregation=MACRO,
        notes="方向干预确认；不是 E19/P2 的约束强度 B/C 比较。")
    for index, item in enumerate(e18["groups"]):
        if item["group_kind"] != "task":
            continue
        add(f"e18_task_{item['group']}", "e18_comparisons", "e18", {"row": f"/groups/{index}"},
            stage="E18", cohort_role="独立确认", pool=ENSEMBLE, scope=item["group"],
            candidate=candidate, reference=reference, n_questions=item["questions"],
            accuracy_pct=item[candidate + "_pct"], reference_accuracy_pct=item[reference + "_pct"],
            difference_pp=item["gain_pp"], corrected_events=item["corrected_generation_events"],
            damaged_events=item["damaged_generation_events"], event_kind="整数题目×生成事件",
            decoder="E18 archived frozen policy", aggregation="任务内原题平均；题内生成先平均")

    e19 = data["e19"]
    for key, label, decoder in (("primary", "两任务等权", TAU),
                                ("raw_exact_uniform_diagnostic", "两任务等权 raw-exact", RAW)):
        item = e19[key]
        extra = {}
        if key == "primary":
            extra = dict(strict_switches=e19["mechanism"]["strict_changed_questions"],
                         corrected_events=item["expected_corrections"],
                         damaged_events=item["expected_damages"], event_kind="冻结均匀策略耦合期望")
        add("e19_" + key, "e19", "e19", {"row": "/" + key, "n_questions": "/original_questions",
            "strict_switches": "/mechanism/strict_changed_questions" if key == "primary" else None},
            stage="E19/P1", cohort_role="重复开发 train600 OOF", pool=ENSEMBLE, scope=label,
            loss="TSPO+", candidate="C", reference="B", n_questions=e19["original_questions"],
            accuracy_pct=item["C_accuracy_pct"], reference_accuracy_pct=item["B_accuracy_pct"],
            difference_pp=item["gain_pp"], decoder=decoder, aggregation=MACRO,
            notes="只提取点值；忽略旧文件中的区间字段。", **extra)
    for index, item in enumerate(e19["task_effects"]):
        add("e19_task_" + item["task"], "e19", "e19", {"row": f"/task_effects/{index}"},
            stage="E19/P1", cohort_role="重复开发 train600 OOF", pool=ENSEMBLE, scope=item["task"],
            loss="TSPO+", candidate="C", reference="B", n_questions=item["questions"],
            accuracy_pct=item["C_accuracy_pct"], reference_accuracy_pct=item["B_accuracy_pct"],
            difference_pp=item["gain_pp"], decoder=TAU, aggregation="任务内原题平均；题内生成先平均")

    p2 = data["p2"]
    weights = p2["primary_task_weights"]
    close(math.fsum(weights.values()), 1, "P2 declared task weights")
    for key, label, decoder in (("primary", "两任务等权", TAU),
                                ("raw_exact_sensitivity", "两任务等权 raw-exact", RAW)):
        item = p2[key]
        groups = item["task_groups"]
        if set(groups) != set(weights):
            raise ValueError("P2 group keys differ from declared task weights")
        b = math.fsum(number(weights[t]) * number(groups[t]["B_accuracy_pct"]) for t in weights)
        c = math.fsum(number(weights[t]) * number(groups[t]["C_accuracy_pct"]) for t in weights)
        extra = {}
        if key == "primary":
            extra = dict(strict_switches=p2["strict_changed_questions"],
                         corrected_events=p2["expected_corrected_generation_events"],
                         damaged_events=p2["expected_damaged_generation_events"],
                         event_kind="冻结均匀策略耦合期望")
        add("p2_" + key, "p2", "p2",
            {"accuracy_pct": f"sum(/primary_task_weights[t] * /{key}/task_groups[t]/C_accuracy_pct)",
             "reference_accuracy_pct": f"sum(/primary_task_weights[t] * /{key}/task_groups[t]/B_accuracy_pct)",
             "difference_pp": f"/{key}/difference_pp", "n_questions": f"/{key}/original_questions",
             "strict_switches": "/strict_changed_questions" if key == "primary" else None,
             "event_counts": "/expected_*_generation_events" if key == "primary" else None},
            stage="E20/P2", cohort_role="独立新题", pool=ENSEMBLE, scope=label, loss="TSPO+",
            candidate="C", reference="B", n_questions=item["original_questions"], accuracy_pct=c,
            reference_accuracy_pct=b, difference_pp=item["difference_pp"], decoder=decoder,
            aggregation=MACRO, notes="准确率按归档任务权重恢复；差值直接读取归档字段。", **extra)
        if key == "primary":
            for task, group in groups.items():
                add("p2_task_" + task, "p2", "p2", {"row": f"/primary/task_groups/{task}"},
                    stage="E20/P2", cohort_role="独立新题", pool=ENSEMBLE, scope=task,
                    loss="TSPO+", candidate="C", reference="B", n_questions=group["questions"],
                    accuracy_pct=group["C_accuracy_pct"], reference_accuracy_pct=group["B_accuracy_pct"],
                    difference_pp=group["difference_pp"], decoder=TAU,
                    aggregation="任务内原题平均；题内生成先平均")

    internal = data["p3_internal"]
    item = internal["primary"]
    add("p3_internal_macro", "p3_internal", "p3_internal",
        {"row": "/primary", "accuracy_conversion": "fraction * 100"},
        stage="E21/P3", cohort_role="重复开发 train600 OOF", pool=ENSEMBLE, scope="两任务等权",
        loss="ListNet", candidate="C", reference="B", n_questions=item["questions"],
        accuracy_pct=100 * item["C_task_macro_accuracy"], reference_accuracy_pct=100 * item["B_task_macro_accuracy"],
        difference_pp=item["C_minus_B_pp"], strict_switches=item["strict_changed_questions"],
        corrected_events=item["expected_corrections"], damaged_events=item["expected_damages"],
        event_kind="冻结策略耦合期望；本比较两端均唯一", decoder=TAU, aggregation=MACRO)
    for task, group in item["by_task"].items():
        add("p3_internal_" + task, "p3_internal", "p3_internal",
            {"row": f"/primary/by_task/{task}", "accuracy_conversion": "fraction * 100"},
            stage="E21/P3", cohort_role="重复开发 train600 OOF", pool=ENSEMBLE, scope=task,
            loss="ListNet", candidate="C", reference="B", n_questions=group["questions"],
            accuracy_pct=100 * group["C_accuracy"], reference_accuracy_pct=100 * group["B_accuracy"],
            difference_pp=group["gain_pp"], decoder=TAU, aggregation="任务内原题平均；题内生成先平均")

    external = data["p3_external"]
    tasks = external["by_task"]
    if set(tasks) != {"gpqa", "math-500"}:
        raise ValueError("Unexpected P3 external task family; update the table specification explicitly")
    temperature = str(external["primary_temperature"])
    close(external["primary_temperature"], 0.7, "P3 primary cache configuration")
    external_aggregation = "GPQA 与 MATH-500 各 50%；主测试温度 0.7；每个原题只计一次"
    for cell, loss in (("tspo_L3", "TSPO+ L3"), ("listnet_L3", "ListNet L3")):
        macro = external["macro_at_0_7"][cell]
        for policy in ("B", "C"):
            close(macro[policy + "_accuracy_pct"],
                  mean([group["cells"][cell]["by_temperature"][temperature][policy + "_accuracy_pct"]
                        for group in tasks.values()]), f"P3 {cell} archived {policy} macro")
        add("p3_external_" + cell, "p3_external", "p3_external",
            {"row": "/macro_at_0_7/" + cell, "n_questions": "/unique_external_test_questions",
             "strict_switches": f"sum(/by_task[t]/cells/{cell}/strict_switches)"},
            stage="E21/P3", cohort_role="外部原题测试", pool=EXTERNAL, scope="两任务等权",
            loss=loss, candidate="C", reference="B", n_questions=external["unique_external_test_questions"],
            accuracy_pct=macro["C_accuracy_pct"], reference_accuracy_pct=macro["B_accuracy_pct"],
            difference_pp=macro["C_minus_B_pp"],
            strict_switches=sum(group["cells"][cell]["strict_switches"] for group in tasks.values()),
            decoder=TAU, aggregation=external_aggregation,
            notes="未复现 DARS 原算法；不是第二个三模型聚合系统。")
        for task, group in tasks.items():
            cell_data = group["cells"][cell]
            value = cell_data["by_temperature"][temperature]
            add("p3_external_" + cell + "_" + task, "p3_external", "p3_external",
                {"row": f"/by_task/{task}/cells/{cell}/by_temperature/{temperature}",
                 "strict_switches": f"/by_task/{task}/cells/{cell}/strict_switches",
                 "n_questions": f"/by_task/{task}/test_questions"},
                stage="E21/P3", cohort_role="外部原题测试", pool=EXTERNAL, scope=task, loss=loss,
                candidate="C", reference="B", n_questions=group["test_questions"],
                accuracy_pct=value["C_accuracy_pct"], reference_accuracy_pct=value["B_accuracy_pct"],
                difference_pp=value["C_minus_B_pp"], strict_switches=cell_data["strict_switches"],
                corrected_events=value["expected_corrections"], damaged_events=value["expected_damages"],
                event_kind="冻结均匀策略耦合期望；主原题解码", decoder=TAU,
                aggregation="任务内原题平均；主测试温度 0.7")
        raw = [group["cells"][cell]["raw_exact_at_0_7"] for group in tasks.values()]
        add("p3_external_" + cell + "_raw", "p3_external_raw", "p3_external",
            {"row": f"equal_task_mean(/by_task[t]/cells/{cell}/raw_exact_at_0_7)",
             "n_questions": "/unique_external_test_questions"},
            stage="E21/P3", cohort_role="外部原题测试", pool=EXTERNAL, scope="两任务等权 raw-exact",
            loss=loss, candidate="C", reference="B", n_questions=external["unique_external_test_questions"],
            accuracy_pct=mean([v["C_accuracy_pct"] for v in raw]),
            reference_accuracy_pct=mean([v["B_accuracy_pct"] for v in raw]),
            difference_pp=mean([v["C_minus_B_pp"] for v in raw]), decoder=RAW,
            aggregation=external_aggregation, notes="按预定两任务等权恢复原有敏感性汇总。")
    fixed_values = []
    for task, group in tasks.items():
        value = group["train_only_global_fixed_accuracy_pct"][temperature]
        fixed_values.append(value)
        add("p3_fixed_" + task, "p3_fixed", "p3_external",
            {"accuracy_pct": f"/by_task/{task}/train_only_global_fixed_accuracy_pct/{temperature}",
             "n_questions": f"/by_task/{task}/test_questions"},
            stage="E21/P3", cohort_role="外部原题测试", pool=EXTERNAL, scope=task,
            candidate="TaskFixed", n_questions=group["test_questions"], accuracy_pct=value,
            decoder=TAU, aggregation="本任务训练区选择固定模型；本任务测试原题平均",
            notes="源字段虽名为 global_fixed，但执行代码在每个任务内部单独选择。")
    add("p3_fixed_macro", "p3_fixed", "p3_external",
        {"accuracy_pct": f"equal_task_mean(/by_task[t]/train_only_global_fixed_accuracy_pct/{temperature})",
         "n_questions": "/unique_external_test_questions"},
        stage="E21/P3", cohort_role="外部原题测试", pool=EXTERNAL, scope="两任务等权",
        candidate="TaskFixed", n_questions=external["unique_external_test_questions"],
        accuracy_pct=mean(fixed_values), decoder=TAU, aggregation=external_aggregation,
        notes="各任务分别选固定模型，再等权；不是两个任务共同使用一个全局固定模型。")

    summary = data["e22_summary"]
    expected = {(item["family"], item["condition"]): item for item in summary["condition_table"]}
    observed = {(item["family"], item["condition"]): item for item in data["e22_controls"]}
    if len(observed) != len(data["e22_controls"]) or set(expected) != set(observed):
        raise ValueError("E22 CSV and JSON condition identities disagree or contain duplicates")
    for item in data["e22_controls"]:
        family, condition = item["family"], item["condition"]
        if family not in ("tspo", "listnet"):
            raise ValueError("Unexpected E22 loss family")
        for field in ("accuracy_pct", "gain_over_B_pp", "corrected", "damaged", "strict_switches"):
            close(item[field], expected[(family, condition)][field], f"E22 {family}/{condition}/{field}")
        add(f"e22_{family}_{condition}", "e22_conditions", "e22_controls",
            {"row": f"family={family};condition={condition}",
             "reference_accuracy_pct": f"family={family};condition=B;column=accuracy_pct",
             "n_questions": SOURCES["e22_summary"] + "#/training_questions"},
            stage="E22", cohort_role="重复开发 train600 OOF", pool=ENSEMBLE, scope="两任务等权",
            loss="TSPO+" if family == "tspo" else "ListNet", candidate=condition, reference="B",
            n_questions=summary["training_questions"], accuracy_pct=item["accuracy_pct"],
            reference_accuracy_pct=observed[(family, "B")]["accuracy_pct"],
            difference_pp=item["gain_over_B_pp"], strict_switches=item["strict_switches"],
            corrected_events=item["corrected"], damaged_events=item["damaged"],
            event_kind="冻结策略耦合期望；B 行为恒等自比较", decoder=TAU, aggregation=MACRO,
            notes="CSV 与 SUMMARY 的对应字段先核对；不按结果选取随机掩码。")
    for family, comparisons in summary["primary_controls"].items():
        for comparison in ("C_minus_Uniform", "C_minus_MatchedMean"):
            item = comparisons[comparison]
            reference = comparison.removeprefix("C_minus_")
            add(f"e22_{family}_{comparison}", "e22_primary", "e22_summary",
                {"row": f"/primary_controls/{family}/{comparison}", "n_questions": "/training_questions"},
                stage="E22", cohort_role="重复开发 train600 OOF", pool=ENSEMBLE, scope="两任务等权",
                loss="TSPO+" if family == "tspo" else "ListNet", candidate="C", reference=reference,
                n_questions=summary["training_questions"], accuracy_pct=item["candidate_accuracy_pct"],
                reference_accuracy_pct=item["baseline_accuracy_pct"], difference_pp=item["difference_pp"],
                corrected_events=item["expected_corrected_events"],
                damaged_events=item["expected_damaged_events"], event_kind="冻结策略耦合期望",
                decoder=TAU, aggregation=MACRO,
                notes="MatchedMean 为三份预固定掩码策略等权平均，不是选取最优掩码。")
    append_e23_rows(add, data)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate output row IDs")
    return rows


def append_e23_rows(add: Any, data: dict[str, Any]) -> None:
    """Append published E23 aggregates; retain the existing CSV schema and rows."""
    e23, p2 = data["e23"], data["p2"]
    if (e23["schema"] != "mab_e23_final_reference_results_v1"
            or e23["status"] != "COMPLETE_FULL2400_REFERENCE_EVALUATION"):
        raise ValueError("E23 requires the complete final reference evaluation")
    if "already-exposed P2" not in e23["evaluation_scope"]:
        raise ValueError("E23 evaluation-cohort identity changed")
    policies = {key: dict(value) for key, value in e23["baseline_strategies"].items()}
    matched = tuple(key for key in e23["matching"]["controls"] if key.startswith("Matched"))
    expected = {"B_uniform", "C_uniform", "B_prior", "C_prior", "B_raw", "C_raw",
                "GlobalFixed", "Ridge", "Uniform", *matched}
    if len(matched) != 3 or set(policies) != expected:
        raise ValueError("E23 frozen reference-policy identities changed")
    weights = e23["task_weights"]
    if weights != p2["primary_task_weights"]:
        raise ValueError("E23 and P2 task weights differ")
    questions, events = e23["questions"], e23["original_generation_events_per_policy"]
    close(questions, p2["primary"]["original_questions"], "E23 same P2 questions")
    task_counts = {task: p2["primary"]["task_groups"][task]["questions"] for task in weights}
    close(sum(task_counts.values()), questions, "E23 task question counts")
    for task in weights:
        close(task_counts[task] / questions, weights[task], "E23 balanced-task event denominator")
    seeds = tuple(policies["B_uniform"]["by_seed"])
    subtasks = tuple(policies["B_uniform"]["by_bbh_subtask"])
    close(events, questions * len(seeds), "E23 generation events per policy")
    for name, item in policies.items():
        if item["status"] != "FULL_COHORT_SCORED":
            raise ValueError(f"E23 incomplete strategy: {name}")
        close(item["questions"], questions, f"E23 {name} questions")
        close(item["generation_events"], events, f"E23 {name} events")
        if (set(item["by_task"]) != set(weights) or set(item["by_seed"]) != set(seeds)
                or set(item["by_bbh_subtask"]) != set(subtasks)):
            raise ValueError(f"E23 grouping identities differ: {name}")
        close(item["accuracy_pct"], math.fsum(number(weights[t]) * number(item["by_task"][t])
              for t in weights), f"E23 {name} task macro")
        close(item["accuracy_pct"], mean(list(item["by_seed"].values())), f"E23 {name} seed mean")
    # Restore the declared equal policy mixture; do not average correction counts.
    if "equal mixture of all three" not in e23["mixed_control_interpretation"]:
        raise ValueError("E23 MatchedMean definition changed")
    policies["MatchedMean"] = {
        "accuracy_pct": mean([policies[name]["accuracy_pct"] for name in matched]),
        **{group: {key: mean([policies[name][group][key] for name in matched])
                   for key in policies[matched[0]][group]}
           for group in ("by_task", "by_seed", "by_bbh_subtask")},
    }
    for policy, p2_key, side in (("B_uniform", "primary", "B"), ("C_uniform", "primary", "C"),
                                ("B_raw", "raw_exact_sensitivity", "B"),
                                ("C_raw", "raw_exact_sensitivity", "C")):
        for task in weights:
            close(policies[policy]["by_task"][task],
                  p2[p2_key]["task_groups"][task][side + "_accuracy_pct"],
                  f"E23 retained P2 {policy}/{task}")
    for group in ("by_task", "by_seed", "by_bbh_subtask"):
        for key in policies["C_uniform"][group]:
            close(policies["C_prior"][group][key], policies["C_uniform"][group][key],
                  f"E23 unchanged C prior {group}/{key}")
    comparisons = e23["paired_comparisons"]
    deployment_pairs = ("C_uniform_minus_B_uniform", "B_prior_minus_B_uniform",
                        "C_prior_minus_B_prior", "C_raw_minus_B_raw")
    reference_pairs = tuple("C_uniform_minus_" + name
                            for name in ("GlobalFixed", "Ridge", "Uniform", *matched, "MatchedMean"))
    if set(comparisons) != set(deployment_pairs + reference_pairs):
        raise ValueError("E23 frozen paired-comparison identities changed")
    for field, comparison in (("original_gap_pp", deployment_pairs[0]),
                              ("B_deployment_gain_pp", deployment_pairs[1]),
                              ("remaining_gap_pp", deployment_pairs[2])):
        close(e23[field], comparisons[comparison]["difference_pp"], "E23 " + field)
    close(e23["original_gap_pp"], e23["B_deployment_gain_pp"] + e23["remaining_gap_pp"],
          "E23 deployment decomposition")
    close(e23["original_gap_pp"], p2["primary"]["difference_pp"], "E23 original P2 difference")
    close(comparisons["C_raw_minus_B_raw"]["difference_pp"],
          p2["raw_exact_sensitivity"]["difference_pp"], "E23 original P2 raw difference")
    original_regions = comparisons[deployment_pairs[0]]["regions"]
    close(original_regions["original_strict_switched"]["questions"], p2["strict_changed_questions"],
          "E23 original B/C strict-change region")
    common = dict(stage="E23", cohort_role="已曝光 P2 new2400 后续比较", pool=ENSEMBLE)
    prior_decoder = "training-success prior within tau=1e-8 support; remaining prior ties uniform"

    def decoder(name: str) -> str:
        if name.endswith("_prior"):
            return prior_decoder
        if name.endswith("_raw"):
            return RAW
        if name == "GlobalFixed":
            return "train-only globally fixed action; training ties uniform"
        if name == "MatchedMean":
            return "equal action-probability mixture of three frozen Matched policies"
        return TAU

    def accuracy_fields(name: str, suffix: str) -> dict[str, Any]:
        if name == "MatchedMean":
            return {"accuracy_pct": "equal_policy_mean(" + ";".join(
                f"/baseline_strategies/{item}/{suffix}" for item in matched) + ")",
                "mixture_definition": "/mixed_control_interpretation"}
        return {"accuracy_pct": f"/baseline_strategies/{name}/{suffix}"}

    for name, item in policies.items():
        notes = ("全部三个预定 Matched 策略的等权动作概率混合。" if name == "MatchedMean"
                 else "E23 冻结策略；训练侧采用全量历史，评价沿用原 P2 题池。")
        add(f"e23_policy_{name}_macro", "e23_policies", "e23",
            {**accuracy_fields(name, "accuracy_pct"), "n_questions": "/questions",
             "generation_events": "/original_generation_events_per_policy"},
            **common, scope="两任务等权", candidate=name, n_questions=questions,
            accuracy_pct=item["accuracy_pct"], decoder=decoder(name), aggregation=MACRO, notes=notes)
        for group, keys in (("by_task", tuple(weights)), ("by_seed", seeds)):
            for key in keys:
                scope = key if group == "by_task" else "seed=" + key
                count = task_counts[key] if group == "by_task" else questions
                count_source = (SOURCES["p2"] + f"#/primary/task_groups/{key}/questions"
                                if group == "by_task" else "/questions")
                add(f"e23_policy_{name}_{group}_{key}", "e23_policy_groups", "e23",
                    {**accuracy_fields(name, f"{group}/{key}"), "n_questions": count_source},
                    **common, scope=scope, candidate=name, n_questions=count,
                    accuracy_pct=item[group][key], decoder=decoder(name),
                    aggregation="任务内原题平均；三个生成重复平均" if group == "by_task"
                                else "固定生成 seed；两任务等权；同一批原题", notes=notes)
    for comparison in deployment_pairs + reference_pairs:
        candidate, reference = comparison.split("_minus_", 1)
        item = comparisons[comparison]
        pointer = "/paired_comparisons/" + comparison
        close(item["corrected_events"] - item["damaged_events"], item["net_corrected_events"],
              f"E23 {comparison} net events")
        close(100 * item["net_corrected_events"] / events, item["difference_pp"],
              f"E23 {comparison} event accounting")
        for group, keys in (("by_task", tuple(weights)), ("by_seed", seeds),
                            ("by_bbh_subtask", subtasks)):
            if set(item[group]) != set(keys):
                raise ValueError(f"E23 comparison groups differ: {comparison}/{group}")
            for key in keys:
                close(policies[candidate][group][key] - policies[reference][group][key], item[group][key],
                      f"E23 {comparison}/{group}/{key}")
        event_kind = "同题缓存最大重叠耦合期望；题目×生成事件"
        pair_decoder = (decoder(candidate) if decoder(candidate) == decoder(reference)
                        else candidate + ": " + decoder(candidate) + "; " + reference + ": " + decoder(reference))
        strict = (dict(strict_switches=original_regions["original_strict_switched"]["questions"])
                  if comparison == deployment_pairs[0] else {})
        add("e23_pair_" + comparison + "_macro",
            "e23_deployment_comparisons" if comparison in deployment_pairs else "e23_reference_comparisons",
            "e23", {"row": pointer, "n_questions": "/questions",
                    "candidate_accuracy": accuracy_fields(candidate, "accuracy_pct"),
                    "reference_accuracy": accuracy_fields(reference, "accuracy_pct"),
                    "net_corrected_events": pointer + "/net_corrected_events",
                    "strict_switches": pointer + "/regions/original_strict_switched/questions"
                                       if strict else None},
            **common, scope="两任务等权", candidate=candidate, reference=reference, n_questions=questions,
            accuracy_pct=policies[candidate]["accuracy_pct"],
            reference_accuracy_pct=policies[reference]["accuracy_pct"], difference_pp=item["difference_pp"],
            corrected_events=item["corrected_events"], damaged_events=item["damaged_events"],
            event_kind=event_kind, decoder=pair_decoder, aggregation=MACRO,
            notes="归档净纠错生成事件=" + format(number(item["net_corrected_events"]), ".15g"), **strict)
        for group, keys in (("by_task", tuple(weights)), ("by_seed", seeds)):
            for key in keys:
                count = task_counts[key] if group == "by_task" else questions
                add(f"e23_pair_{comparison}_{group}_{key}", "e23_comparison_groups", "e23",
                    {"difference_pp": f"{pointer}/{group}/{key}",
                     "candidate_accuracy": accuracy_fields(candidate, f"{group}/{key}"),
                     "reference_accuracy": accuracy_fields(reference, f"{group}/{key}"),
                     "n_questions": SOURCES["p2"] + f"#/primary/task_groups/{key}/questions"
                                    if group == "by_task" else "/questions"},
                    **common, scope=key if group == "by_task" else "seed=" + key,
                    candidate=candidate, reference=reference, n_questions=count,
                    accuracy_pct=policies[candidate][group][key],
                    reference_accuracy_pct=policies[reference][group][key], difference_pp=item[group][key],
                    decoder=pair_decoder, aggregation="任务内原题平均；三个生成重复平均" if group == "by_task"
                                                    else "固定生成 seed；两任务等权；同一批原题")
        regions = item["regions"]
        if set(regions) != set(original_regions):
            raise ValueError("E23 frozen region identities differ")
        for region, value in regions.items():
            close(value["questions"], original_regions[region]["questions"], f"E23 region size {region}")
            close(100 * (value["corrected_events"] - value["damaged_events"]) / events,
                  value["full_cohort_contribution_pp"], f"E23 {comparison}/{region} contribution")
            add(f"e23_region_{comparison}_{region}", "e23_regions", "e23",
                {"row": f"{pointer}/regions/{region}",
                 "difference_pp": f"{pointer}/regions/{region}/full_cohort_contribution_pp"},
                **common, scope=region, candidate=candidate, reference=reference,
                n_questions=value["questions"], difference_pp=value["full_cohort_contribution_pp"],
                corrected_events=value["corrected_events"], damaged_events=value["damaged_events"],
                event_kind=event_kind, decoder=pair_decoder,
                aggregation="区域对全题池任务宏平均的贡献；不是区域内准确率差",
                notes="沿用原 B/C 输入侧区域；原严格换队区域是 B 唯一区域的子集，不是本比较的新换队计数。")
        partition = (regions["original_B_unique"], regions["original_B_near_tied"])
        close(sum(value["questions"] for value in partition), questions, "E23 region partition")
        for field in ("corrected_events", "damaged_events"):
            close(math.fsum(number(value[field]) for value in partition), item[field],
                  f"E23 {comparison} partition {field}")
        close(math.fsum(number(value["full_cohort_contribution_pp"]) for value in partition),
              item["difference_pp"], f"E23 {comparison} partition difference")


def csv_text(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = row.copy()
        for key in NUMERIC_COLUMNS:
            if serialized[key] != "":
                serialized[key] = format(serialized[key], ".15g")
        writer.writerow(serialized)
    return output.getvalue()


def cell(value: Any, signed: bool = False) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return format(value, "+.6f" if signed else ".6f")
    return str(value).replace("|", "\\|").replace("\n", " ")


def append_e23_markdown(lines: list[str], rows: list[dict[str, Any]]) -> None:
    index = {row["row_id"]: row for row in rows}
    policies = {row["candidate"]: row for row in rows if row["section"] == "e23_policies"}
    paired = [row for row in rows if row["section"] in
              ("e23_deployment_comparisons", "e23_reference_comparisons")]
    matched = tuple(name for name in policies if name.startswith("Matched") and name != "MatchedMean")
    seed_keys = tuple(row["scope"].removeprefix("seed=") for row in rows
                      if row["section"] == "e23_policy_groups" and row["candidate"] == "B_uniform"
                      and row["scope"].startswith("seed="))
    labels = {"B_uniform": "B · 原均匀并列", "C_uniform": "C · 原均匀并列",
              "B_prior": "B · 训练先验", "C_prior": "C · 训练先验",
              "B_raw": "B · raw-exact", "C_raw": "C · raw-exact",
              "Uniform": "Uniform · 同总权重", "MatchedMean": "MatchedMean · 三策略等权混合"}

    def label(name: str) -> str:
        return labels.get(name, name)

    def pair(name: str) -> dict[str, Any]:
        return index["e23_pair_" + name + "_macro"]

    original = pair("C_uniform_minus_B_uniform")
    deployment = pair("B_prior_minus_B_uniform")
    remaining = pair("C_prior_minus_B_prior")
    mixed = pair("C_uniform_minus_MatchedMean")
    ridge = pair("C_uniform_minus_Ridge")
    lines.extend([
        "## E23 部署规则与同题池参照", "",
        f"本阶段在已经曝光结果的 P2 {original['n_questions']:,} 道题上追加比较。"
        "B/C 与 Ridge 沿用冻结预测头，Uniform 和 Matched 使用全量训练历史拟合最终头；"
        "E22 的开发五折结果继续单独保留。",
        f"**训练先验把 B 的准确率从 {cell(policies['B_uniform']['accuracy_pct'])}% "
        f"提高到 {cell(policies['B_prior']['accuracy_pct'])}%，部署收益为 "
        f"{cell(deployment['difference_pp'], signed=True)} pp。"
        f"C−B 点差由 {cell(original['difference_pp'], signed=True)} pp "
        f"缩至 {cell(remaining['difference_pp'], signed=True)} pp。**",
        f"同题池中，Ridge 为 {cell(policies['Ridge']['accuracy_pct'])}%，"
        f"C 为 {cell(policies['C_uniform']['accuracy_pct'])}%；"
        f"C−Ridge 为 {cell(ridge['difference_pp'], signed=True)} pp，"
        f"C−MatchedMean 为 {cell(mixed['difference_pp'], signed=True)} pp。", "",
    ])

    def policy_table(title: str, names: tuple[str, ...], note: str) -> None:
        lines.extend(["### " + title, "", note, "",
                      "| 策略 | 总体 % | GSM8K % | BBH % | "
                      + " | ".join("seed " + seed + " %" for seed in seed_keys) + " |",
                      "|---|---:|---:|---:|" + "---:|" * len(seed_keys)])
        for name in names:
            values = [label(name), cell(policies[name]["accuracy_pct"])]
            values.extend(cell(index[f"e23_policy_{name}_by_task_{task}"]["accuracy_pct"])
                          for task in ("gsm8k", "bbh"))
            values.extend(cell(index[f"e23_policy_{name}_by_seed_{seed}"]["accuracy_pct"])
                          for seed in seed_keys)
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    policy_table("部署规则的准确率",
                 ("B_uniform", "B_prior", "C_uniform", "C_prior", "B_raw", "C_raw"),
                 "训练先验只在原 1e−8 支持集合内选择训练成功次数最高的组合；"
                 "训练次数仍并列时均匀选择。raw-exact 沿用原零容差规则。")
    policy_table("同一评价集的参照",
                 ("GlobalFixed", "Ridge", "Uniform", *matched, "MatchedMean", "B_prior", "C_uniform"),
                 "GlobalFixed 按训练历史选择固定组合；Uniform 表示所有训练行同幅降权。"
                 "MatchedMean 是全部三个预定策略的等权动作概率混合，其准确率由三份归档准确率恢复。")
    lines.extend(["### 配对点差、任务与生成 seed", "",
                  "点差、纠错和破坏直接读取归档比较。纠错与破坏采用同题缓存的最大重叠耦合；"
                  "混合策略的期望计数可以是小数。",
                  "每个 seed 仍是同一批原题的生成重复。", "",
                  "| 比较 | 总点差 pp | GSM8K pp | BBH pp | "
                  + " | ".join("seed " + seed + " pp" for seed in seed_keys) + " | 纠错 | 破坏 |",
                  "|---|---:|---:|---:|" + "---:|" * (len(seed_keys) + 2)])
    for row in paired:
        comparison = row["candidate"] + "_minus_" + row["reference"]
        values = [label(row["candidate"]) + " − " + label(row["reference"]),
                  cell(row["difference_pp"], signed=True)]
        values.extend(cell(index[f"e23_pair_{comparison}_by_task_{task}"]["difference_pp"], signed=True)
                      for task in ("gsm8k", "bbh"))
        values.extend(cell(index[f"e23_pair_{comparison}_by_seed_{seed}"]["difference_pp"], signed=True)
                      for seed in seed_keys)
        values.extend((cell(row["corrected_events"]), cell(row["damaged_events"])))
        lines.append("| " + " | ".join(values) + " |")
    region_labels = {"original_B_unique": "原 B 唯一区域",
                     "original_B_near_tied": "原 B 近并列区域",
                     "original_strict_switched": "原 B/C 严格换队区域"}
    lines.extend(["", "### 冻结区域的贡献", "",
                  "以下差值是区域对全题池准确率的贡献。B 唯一区域与近并列区域合计构成全部原题；"
                  "原 B/C 严格换队区域属于唯一区域的子集，不能再相加，也不是新参照比较的严格换队计数。",
                  "CSV 保留全部配对比较的三个区域。", "",
                  "| 比较 | 原输入侧区域 | 原题数 | 对全题池贡献 pp | 纠错 | 破坏 |",
                  "|---|---|---:|---:|---:|---:|"])
    for comparison in ("C_prior_minus_B_prior", "C_uniform_minus_MatchedMean"):
        for region, region_label in region_labels.items():
            row = index[f"e23_region_{comparison}_{region}"]
            values = [label(row["candidate"]) + " − " + label(row["reference"]), region_label,
                      cell(row["n_questions"]), cell(row["difference_pp"], signed=True),
                      cell(row["corrected_events"]), cell(row["damaged_events"])]
            lines.append("| " + " | ".join(values) + " |")
    source = SOURCES["e23"]
    lines.extend(["", f"来源：[RESULTS_FINAL.json](../{source})。", ""])


def markdown_text(rows: list[dict[str, Any]], hashes: dict[str, str]) -> str:
    lines = [
        "# 论文结果表", "",
        "本页由 `python scripts/build_paper_tables.py` 从归档 JSON/CSV 生成。",
        "脚本仅提取已归档的汇总字段、换算百分比单位，并按原协议恢复任务宏平均；",
        "不访问原题，不重训或重新评分，不计算新统计检验、置信区间或 bootstrap。",
        "逐行来源字段、计算口径与事件单位见 [results.csv](results.csv)。", "",
        "**不同阶段使用不同题池、训练目标或模型池，不构成可直接排序的统一算法榜单。**",
        "E19/P2/P3/E22 主解码为 1e−8 容差内最大动作集合的均匀策略；E18 沿用原冻结策略。",
        "raw-exact 是原浮点分数的零容差最大集合，",
        "不表示数学精确最优解或可用真值的 oracle。原题数不因生成重复、模型动作或匹配掩码增加。", "",
    ]

    def table(section: str, title: str, note: str, absolute: bool = False) -> None:
        selected = [row for row in rows if row["section"] == section]
        if not selected:
            raise ValueError(f"Missing section: {section}")
        lines.extend(["## " + title, "", note, ""])
        if absolute:
            lines.extend(["| 范围 | 策略 | 原题数 | 准确率 % |",
                          "|---|---|---:|---:|"])
            for row in selected:
                lines.append("| " + " | ".join(cell(row[key]) for key in
                             ("scope", "candidate", "n_questions", "accuracy_pct")) + " |")
        else:
            lines.extend(["| 范围 | 比较 | 原题数 | 参照准确率 % | 候选准确率 % | 差值 pp | 严格换队 |",
                          "|---|---|---:|---:|---:|---:|---:|"])
            for row in selected:
                comparison = (row["loss"] + " " if row["loss"] else "") + row["candidate"] + " − " + row["reference"]
                values = [cell(row["scope"]), cell(comparison), cell(row["n_questions"]),
                          cell(row["reference_accuracy_pct"]), cell(row["accuracy_pct"]),
                          cell(row["difference_pp"], signed=True), cell(row["strict_switches"])]
                lines.append("| " + " | ".join(values) + " |")
        sources = list(dict.fromkeys(row["source_file"] for row in selected))
        lines.extend(["", "来源：" + "；".join(f"[{Path(source).name}](../{source})" for source in sources) + "。", ""])

    table("e18_absolute", "E18 独立确认的冻结策略", "任务等权；已共同排除不可评分题。", absolute=True)
    table("e18_comparisons", "E18 冠军方向干预", "这是 TSPO_uniform 与 SPO_coordinate 的历史主比较，不能替换为 P2 的 B/C 干预。")
    table("e19", "E19 对称约束开发分析", "原 train600 五折外折分析，训练历史重叠且题池长期用于开发。只保留归档点值。")
    table("p2", "P2 原系统独立新题", "同一原系统的冻结 B/C 比较。总体准确率按源文件中的任务权重恢复；总体差值直接读取归档。主效果包含预定解码的影响。")
    table("p3_internal", "P3 原池 ListNet", "原 train600 的跨损失开发机制对照，不是独立新题。")
    table("p3_external", "P3 外部单模型池 L3", "DARS 缓存上的 GPQA 与 MATH-500；部署选择单模型，测试配置固定为温度 0.7。总体严格换队数为两个任务计数之和。")
    table("p3_external_raw", "P3 外部零容差敏感性", "按原协议将两个任务的已归档 raw-exact 点值等权汇总；没有重新求解模型。")
    table("p3_fixed", "P3 按任务固定模型基线", "各任务分别从自己的训练区选择固定模型，再对两个任务等权汇总。源字段名含 global_fixed，但不表示跨任务共用一个全局模型。", absolute=True)
    table("e22_conditions", "E22 全部行权重控制", "B、C、同总权重 Uniform 与每个预固定匹配掩码全部保留。每行差值及严格换队的参照均为同损失 B；B 的零值是自比较。")
    table("e22_primary", "E22 特异性主对照", "MatchedMean 为三份预固定掩码策略的等权平均；不把最优掩码替代平均对照。原开发题池的质量影响因损失和降权位置而异。")
    append_e23_markdown(lines, rows)

    main_ids = {"e18_primary", "e19_primary", "p2_primary", "p3_internal_macro"}
    event_rows = [row for row in rows if row["row_id"] in main_ids or row["section"] == "e22_primary"]
    lines.extend(["## 主比较的纠错与破坏", "",
                  "事件计数读取归档。整数观测与均匀策略耦合期望分别标注；期望值可能为小数。",
                  "任务等权准确率差不能一律由净事件数除以合并分母恢复。", "",
                  "| 阶段 | 比较 | 纠错 | 破坏 | 事件单位 |",
                  "|---|---|---:|---:|---|"])
    for row in event_rows:
        comparison = (row["loss"] + " " if row["loss"] else "") + row["candidate"] + " − " + row["reference"]
        lines.append("| " + " | ".join([cell(row["stage"]), cell(comparison), cell(row["corrected_events"]),
                                         cell(row["damaged_events"]), cell(row["event_kind"])]) + " |")
    lines.extend(["", "## 输入文件校验值", "",
                  "以下 SHA-256 来自本次构表实际读取的文件字节。表格内容与输入哈希均为确定性输出。", "",
                  "| 输入文件 | SHA-256 |", "|---|---|"])
    for key, source in SOURCES.items():
        lines.append(f"| [{source}](../{source}) | `{hashes[key]}` |")
    lines.extend(["", "## 复核命令", "", "```bash", "python scripts/build_paper_tables.py",
                  "python scripts/build_paper_tables.py --check", "```", "",
                  "`--check` 只在内存中重建并逐字节比较两个输出，不创建目录、不修改文件。",
                  "退出码 0 表示一致；1 表示缺少或漂移的输出；2 表示输入、字段或参数错误。",
                  "这项检查验证公开汇总到论文表格的映射，不等于复现原始训练、推理与逐题评价。", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root; default is the parent of scripts/")
    parser.add_argument("--check", action="store_true", help="Compare committed outputs without writing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        data, hashes = read_sources(root)
        rows = build_rows(data)
        outputs = {Path("paper/results.csv"): csv_text(rows).encode("utf-8"),
                   Path("paper/RESULTS.md"): markdown_text(rows, hashes).encode("utf-8")}
        if args.check:
            mismatches = []
            for relative, expected in outputs.items():
                path = root / relative
                if not path.is_file():
                    mismatches.append((relative, "missing"))
                elif path.read_bytes() != expected:
                    mismatches.append((relative, "content differs"))
            if mismatches:
                for relative, reason in mismatches:
                    print(f"FAIL {relative}: {reason}", file=sys.stderr)
                return 1
            print(f"PASS: {len(rows)} archived-result rows; both paper tables match; no files modified.")
            return 0
        for relative, expected in outputs.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
            print(f"WROTE {relative}")
        print(f"Rebuilt {len(rows)} rows from {len(SOURCES)} archived inputs; no model or statistical computation.")
        return 0
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, csv.Error, IndexError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
