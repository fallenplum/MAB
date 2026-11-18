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
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate output row IDs")
    return rows


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
