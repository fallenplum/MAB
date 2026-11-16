"""Generate a de-identified E18 confirmation report from completed artifacts.

Only completed scores and existing inference results are consumed. This module
does not import a scorer, reference provider, language model or inference method.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata

import numpy as np


POLICIES = ("Ridge", "SPO_coordinate", "TSPO_uniform", "GlobalFixed")
SEEDS = (42, 137, 2024)
TEAMS = tuple(combinations(range(6), 3))
FINALIZER_SHA = "9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8"
INPUTS_SHA = "d1a90c3d034b5293e74cef5194004151248d2376d9ee02d96997c5e45962dad6"
MECHANISM_REGISTRY_SHA = (
    "6859b4ac663d50b32ea772eaeace6388624ed529dab6778b8d00954811536c66"
)
OLD_BBH = "bbh_old_llmrouterbench_common_subpool"
EXT_BBH = "bbh_official_full_source_frame_expansion"
SECONDARY = (
    "TSPO_uniform minus Ridge",
    "SPO_coordinate minus Ridge",
    "TSPO_uniform minus GlobalFixed",
    "SPO_coordinate minus GlobalFixed",
    "Ridge minus GlobalFixed",
)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def checked(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Explicit artifact path and SHA256 required")
    path = Path(reference["path"]).resolve(strict=True)
    if file_sha(path) != reference["sha256"]:
        raise ValueError("Report input bytes differ " + path.name)
    return path


def read_json(reference):
    return json.loads(checked(reference).read_text(encoding="utf-8-sig"))


def require_close(actual, expected, label):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1e-10:
        raise ValueError("Completed result disagrees with private artifacts " + label)


def validate_inputs(
    results_ref, rewards_ref, costs_ref, inputs_ref, *, software_test_only=False
):
    result = read_json(results_ref)
    if (
        result.get("schema") != "e18_final_confirmation_results_v1"
        or result.get("status") != "COMPLETE"
    ):
        raise PermissionError("Only COMPLETE E18 confirmation results can be reported")
    if result.get("finalizer_source_sha256") != FINALIZER_SHA:
        raise ValueError("The completed finalizer source identity differs")
    gate = result["execution_gate"]
    if gate.get("status") != "PASS" or result.get("generation_seeds") != list(SEEDS):
        raise ValueError(
            "Complete execution admission and frozen three repeats required"
        )
    if (
        gate.get("original_questions"),
        gate.get("unique_aggregate_keys"),
        gate.get("unique_member_keys"),
    ) != (
        result["original_questions"],
        result["unique_aggregate_keys"],
        result["unique_member_keys"],
    ) or result.get("distinct_gold_accounts") != result["original_questions"]:
        raise ValueError(
            "Completed public status and execution/reference counts disagree"
        )
    if not software_test_only and (
        result.get("original_questions"),
        result.get("unique_aggregate_keys"),
        result.get("unique_member_keys"),
    ) != (3000, 30921, 47286):
        raise ValueError("The approved production scope cardinalities differ")
    with np.load(checked(rewards_ref), allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in (
                "rewards",
                "policy_ids",
                "question_ids",
                "canonical_hashes",
                "task_labels",
                "subtask_labels",
                "source_frames",
                "frozen_policy_actions",
            )
        }
    n = result["original_questions"]
    rewards = arrays["rewards"]
    actions = arrays["frozen_policy_actions"]
    if rewards.shape != (4, 3, n) or not np.isin(rewards, (0, 1)).all():
        raise ValueError(
            "Require binary four-policy / three-repeat / original-question rewards"
        )
    if (
        arrays["policy_ids"].tolist() != list(POLICIES)
        or actions.shape != (4, n)
        or actions.dtype.kind not in "iu"
        or np.any(actions < 0)
        or np.any(actions >= 20)
    ):
        raise ValueError("Frozen policy identity or selected team actions differ")
    for first in range(4):
        for second in range(first + 1, 4):
            shared = actions[first] == actions[second]
            if not np.array_equal(
                rewards[first, :, shared], rewards[second, :, shared]
            ):
                raise ValueError(
                    "Policies sharing one executed action have different rewards"
                )
    for key in (
        "question_ids",
        "canonical_hashes",
        "task_labels",
        "subtask_labels",
        "source_frames",
    ):
        if arrays[key].shape != (n,) or arrays[key].dtype.kind != "U":
            raise ValueError(
                "Explicit Unicode original-question metadata required " + key
            )
    ids, hashes = arrays["question_ids"].tolist(), arrays["canonical_hashes"].tolist()
    if len(set(ids)) != n or len(set(hashes)) != n:
        raise ValueError("Report source contains repeated original identities")
    if not software_test_only and inputs_ref["sha256"] != INPUTS_SHA:
        raise ValueError("The frozen input-side metadata identity differs")
    forbidden = {"gold", "answer", "reference", "target", "reward", "score", "scores"}
    with checked(inputs_ref).open(encoding="utf-8-sig", newline="") as stream:
        inputs = [json.loads(line) for line in stream if line.strip()]
    if len(inputs) != n:
        raise ValueError("Input-side metadata question count differs")
    for index, row in enumerate(inputs):
        canonical = hashlib.sha256(
            " ".join(unicodedata.normalize("NFKC", row["question"]).split()).encode()
        ).hexdigest()
        if (
            forbidden & set(row)
            or row["question_id"] != ids[index]
            or row["canonical_question_sha256"] != hashes[index]
            or canonical != hashes[index]
            or row["task"] != arrays["task_labels"][index]
            or (row.get("subtask") or "") != arrays["subtask_labels"][index]
            or row["source_frame"] != arrays["source_frames"][index]
            or ("input_position" in row and row["input_position"] != index)
        ):
            raise ValueError("Input-side metadata and reward axes are detached")
    roles = np.asarray([row["cohort_role"] for row in inputs])
    counts = Counter(roles.tolist())
    if dict(counts) != gate["cohort_role_counts"] or set(counts) != {
        "original_fresh400",
        "approved_new2600",
    }:
        raise ValueError(
            "Fresh/new provenance differs from the frozen input-side metadata"
        )
    labels = arrays["task_labels"]
    frames = arrays["source_frames"]
    if (
        set(labels) != {"gsm8k", "bbh"}
        or dict(Counter(frames.tolist())) != gate["source_frame_counts"]
    ):
        raise ValueError("Task/source-frame provenance differs")
    if not software_test_only:
        if (
            Counter(labels.tolist()) != Counter(gsm8k=1500, bbh=1500)
            or counts != Counter(original_fresh400=400, approved_new2600=2600)
            or int(np.sum(frames == OLD_BBH)) != 246
            or int(np.sum(frames == EXT_BBH)) != 1254
            or len(set(arrays["subtask_labels"][labels == "bbh"])) != 27
        ):
            raise ValueError(
                "Production task, BBH frame or cohort-role composition differs"
            )
    aggregates = {
        (ids[q], seed, int(action))
        for q in range(n)
        for action in set(actions[:, q])
        for seed in SEEDS
    }
    members = {
        (qid, seed, member)
        for qid, seed, action in aggregates
        for member in TEAMS[action]
    }
    if (len(aggregates), len(members)) != (
        result["unique_aggregate_keys"],
        result["unique_member_keys"],
    ):
        raise ValueError("Completed scope does not equal the sparse four-policy union")
    costs = read_json(costs_ref)
    seen, successful = set(), {"aggregation": set(), "member_generation": set()}
    for row in costs["rows"]:
        key = tuple(row["task_key"])
        allowed = (
            aggregates
            if row["stage"] == "aggregation"
            else members
            if row["stage"] == "member_generation"
            else set()
        )
        if key not in allowed or row["physical_attempt_id"] in seen:
            raise ValueError(
                "Physical cost records have an extra key or duplicated attempt"
            )
        seen.add(row["physical_attempt_id"])
        if row["status"] not in {"success", "failure"} or any(
            type(row[field]) is not int or row[field] < 0
            for field in ("prompt_tokens", "completion_tokens")
        ):
            raise ValueError("Measured settled execution usage required")
        if type(row["reused_historical_source"]) is not bool:
            raise ValueError("Historical/new physical cost partition must be explicit")
        if row["reused_historical_source"] and (
            row["stage"] != "member_generation" or row["status"] != "success"
        ):
            raise ValueError(
                "Historical cache scope is selected successful member records only"
            )
        if row["status"] == "success":
            if key in successful[row["stage"]]:
                raise ValueError(
                    "Multiple successes supplied one semantic execution key"
                )
            successful[row["stage"]].add(key)
    if successful != {"aggregation": aggregates, "member_generation": members}:
        raise ValueError("Cost evidence is missing a required successful execution")
    historical = sum(row["reused_historical_source"] for row in costs["rows"])
    if (
        len(costs["rows"]) != gate["physical_attempts"]
        or historical != result["original_source_cached_member_records"]
        or (not software_test_only and historical != 4717)
    ):
        raise ValueError("Physical or historical known-record counts differ")
    for stage in ("aggregation", "member_generation"):
        rows = [row for row in costs["rows"] if row["stage"] == stage]
        observed = dict(
            physical_attempts=len(rows),
            prompt_tokens=sum(row["prompt_tokens"] for row in rows),
            completion_tokens=sum(row["completion_tokens"] for row in rows),
        )
        if (
            observed != costs["totals"][stage]
            or observed != result["physical_cost_totals"][stage]
        ):
            raise ValueError(
                "Published physical cost aggregate differs from its private ledger"
            )
    if (
        result.get("sealed_reference_values_decoded") != 0
        or result.get("sealed_input_values_decoded") != 0
        or result["reference_provider_stats"].get("protected_input_values_decoded") != 0
        or result["reference_provider_stats"].get("unapproved_reference_values_decoded")
        != 0
    ):
        raise PermissionError(
            "The completed scoring-phase protected-value boundary differs"
        )
    if (
        result["reference_provider_stats"].get("software_test_only")
        is not software_test_only
    ):
        raise PermissionError("Synthetic and formal report inputs cannot be mixed")
    return result, arrays, roles, costs


def describe_group(rewards, labels, mask):
    tasks = [task for task in ("gsm8k", "bbh") if np.any(mask & (labels == task))]
    return dict(
        questions=int(mask.sum()),
        generation_events=3 * int(mask.sum()),
        accuracy=[
            float(
                np.mean(
                    [
                        rewards[policy, :, mask & (labels == task)].mean()
                        for task in tasks
                    ]
                )
            )
            for policy in range(4)
        ],
        averaging="50/50 task macro"
        if len(tasks) == 2
        else "within-task question mean",
    )


def wall_measurements(reference, costs):
    if reference is None:
        return dict(
            available=False,
            measured_attempts=0,
            total_attempts=len(costs["rows"]),
            seconds=None,
        )
    payload = read_json(reference)
    if payload.get("schema") != "e18_native_call_walltime_metadata_v1":
        raise ValueError("Explicit native wall-time metadata schema required")
    expected = {row["physical_attempt_id"]: row for row in costs["rows"]}
    seen, seconds = set(), 0.0
    origins = {
        name: dict(measured_attempts=0, seconds=0.0)
        for name in ("new_e18", "required_historical_success")
    }
    for row in payload["rows"]:
        physical = row["physical_attempt_id"]
        if (
            physical in seen
            or physical not in expected
            or row["native_record_sha256"] != expected[physical]["native_record_sha256"]
            or type(row["wall_time_s"]) not in (int, float)
            or not math.isfinite(row["wall_time_s"])
            or row["wall_time_s"] < 0
        ):
            raise ValueError(
                "Native wall-time metadata is detached from the physical ledger"
            )
        seen.add(physical)
        seconds += row["wall_time_s"]
        origin = (
            "required_historical_success"
            if expected[physical]["reused_historical_source"]
            else "new_e18"
        )
        origins[origin]["measured_attempts"] += 1
        origins[origin]["seconds"] += row["wall_time_s"]
    return dict(
        available=True,
        measured_attempts=len(seen),
        total_attempts=len(expected),
        seconds=seconds,
        full_coverage=len(seen) == len(expected),
        origins=origins,
    )


def cost_rows(result, costs, wall=None):
    output = []
    for category, selected, note in (
        (
            "新增E18实际物理记录",
            [row for row in costs["rows"] if not row["reused_historical_source"]],
            "含本轮已知费用的成功与失败",
        ),
        (
            "历史所需缓存成功记录",
            [row for row in costs["rows"] if row["reused_historical_source"]],
            "已知历史成功投入，旧失败生命周期未恢复",
        ),
        ("已知物理执行并集", costs["rows"], "前两类按physical_attempt_id去重合计"),
    ):
        output.append(
            dict(
                口径=category,
                策略="联合评价",
                成员调用数=sum(row["stage"] == "member_generation" for row in selected),
                聚合调用数=sum(row["stage"] == "aggregation" for row in selected),
                已知失败数=sum(row["status"] == "failure" for row in selected),
                输入Token=sum(row["prompt_tokens"] for row in selected),
                输出Token=sum(row["completion_tokens"] for row in selected),
                说明=note,
            )
        )
    for policy in POLICIES:
        row = result["selected_policy_service_costs"][policy]
        output.append(
            dict(
                口径="每策略逻辑服务投影",
                策略=policy,
                成员调用数=row["member_calls"],
                聚合调用数=row["aggregation_calls"],
                已知失败数="未纳入服务投影",
                输入Token=row["measured_prompt_tokens"],
                输出Token=row["measured_completion_tokens"],
                说明="复用所选成功记录计量，实际联合实验费用见物理并集",
            )
        )
    for row in output:
        row["总Token"] = row["输入Token"] + row["输出Token"]
        row["独占GPU核秒"] = "未计量"
        if row["口径"] == "每策略逻辑服务投影":
            row["Native调用wall秒"] = "逻辑投影未计量"
            row["wall覆盖物理次数"] = "不适用"
        elif not wall or not wall["available"]:
            row["Native调用wall秒"] = "未提供汇总"
            row["wall覆盖物理次数"] = "未提供汇总"
        elif row["口径"] == "已知物理执行并集":
            row["Native调用wall秒"] = wall["seconds"]
            row["wall覆盖物理次数"] = wall["measured_attempts"]
        else:
            origin = (
                "new_e18"
                if row["口径"] == "新增E18实际物理记录"
                else "required_historical_success"
            )
            row["Native调用wall秒"] = wall["origins"][origin]["seconds"]
            row["wall覆盖物理次数"] = wall["origins"][origin]["measured_attempts"]
    return output


def inference_text(primary):
    p = primary.get("p_value")
    estimate = primary["estimate"]
    lower, upper = primary["confidence_interval"]
    if p is None:
        zero = "预定解析推断处于需复核状态，当前输出未给出可解释的p值。"
    elif p < 0.05:
        zero = (
            "预定双侧检验支持相对0的" + ("正向" if estimate > 0 else "负向") + "差异。"
        )
    else:
        zero = "预定双侧检验相对0的证据不足。等效性需要独立的预定等效边界与检验。"
    if lower > 0.01:
        minimum = "现有95%区间下界超过预先规划的1pp参考门槛。"
    elif upper < 0.01:
        minimum = "现有95%区间整体低于预先规划的1pp参考门槛。"
    else:
        minimum = "现有95%区间跨越预先规划的1pp参考门槛。"
    return zero, minimum


def validate_inference(row):
    estimate = row["estimate"]
    interval = row["confidence_interval"]
    if (
        type(estimate) not in (int, float)
        or not math.isfinite(estimate)
        or len(interval) != 2
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in interval
        )
        or not -1 <= interval[0] <= estimate <= interval[1] <= 1
    ):
        raise ValueError("Stored paired inference values are malformed")
    for key in ("p_value", "holm_adjusted_p"):
        value = row.get(key)
        if value is not None and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Stored paired inference probability is malformed")


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def report_artifacts(
    result,
    arrays,
    roles,
    costs,
    provenance,
    *,
    wall=None,
    mechanism=None,
    software_test_only=False,
):
    rewards = arrays["rewards"].astype(np.float64)
    labels, frames, actions = (
        arrays["task_labels"],
        arrays["source_frames"],
        arrays["frozen_policy_actions"],
    )
    n = rewards.shape[2]
    stored = result["results"]
    if stored["primary_comparison"] != "TSPO_uniform minus SPO_coordinate" or set(
        stored["secondary"]
    ) != set(SECONDARY):
        raise ValueError("Predeclared primary or five-comparison family differs")
    primary = stored["primary"]
    validate_inference(primary)
    for comparison in stored["secondary"].values():
        validate_inference(comparison)
    if stored.get("primary_alpha") != 0.05 or stored.get("secondary_family_size") != 5:
        raise ValueError("Frozen alpha or secondary-comparison family differs")
    macro = [
        float(
            np.mean(
                [rewards[index, :, labels == task].mean() for task in ("gsm8k", "bbh")]
            )
        )
        for index in range(4)
    ]
    require_close(primary["estimate"], macro[2] - macro[1], "primary point estimate")
    corrected = int(((rewards[2] == 1) & (rewards[1] == 0)).sum())
    damaged = int(((rewards[2] == 0) & (rewards[1] == 1)).sum())
    if (
        primary["corrected_generation_outcomes"],
        primary["damaged_generation_outcomes"],
        primary["net_corrected_generation_outcomes"],
    ) != (corrected, damaged, corrected - damaged):
        raise ValueError("Paired correction events differ from the frozen result")
    delta = (rewards[2] - rewards[1]).mean(axis=0)
    quality_rows, selection = [], []
    for policy, name in enumerate(POLICIES):
        row = stored["quality"][name]
        for task in ("gsm8k", "bbh"):
            require_close(
                row["task_accuracy"][task],
                rewards[policy, :, labels == task].mean(),
                name + task,
            )
        require_close(row["task_macro_accuracy"], macro[policy], name + " macro")
        subtasks = sorted(set(arrays["subtask_labels"][labels == "bbh"]))
        submean = float(
            np.mean(
                [
                    rewards[policy, :, arrays["subtask_labels"] == subtask].mean()
                    for subtask in subtasks
                ]
            )
        )
        require_close(
            row["bbh_subtask_macro_accuracy"], submean, name + " subtask macro"
        )
        counts = np.bincount(actions[policy], minlength=20)
        fractions = counts / n
        top = np.flatnonzero(counts == counts.max()).tolist()
        selection.append(
            dict(
                counts=counts.tolist(),
                top=top,
                top_share=float(fractions.max()),
                hhi=float(np.square(fractions).sum()),
            )
        )
        quality_rows.append(
            dict(
                策略=name,
                GSM8K准确率_pct=100 * row["task_accuracy"]["gsm8k"],
                BBH准确率_pct=100 * row["task_accuracy"]["bbh"],
                任务宏平均_pct=100 * macro[policy],
                BBH子任务宏平均_pct=100 * submean,
                seed42_pct=100 * row["by_generation_seed"]["42"],
                seed137_pct=100 * row["by_generation_seed"]["137"],
                seed2024_pct=100 * row["by_generation_seed"]["2024"],
                原题数=n,
                生成事件数=3 * n,
                已使用组合数=int(np.sum(counts > 0)),
                最高频组合="/".join(map(str, top)),
                最高频占比=float(fractions.max()),
                HHI=selection[-1]["hhi"],
            )
        )
        for generation, seed in enumerate(SEEDS):
            require_close(
                row["by_generation_seed"][str(seed)],
                np.mean(
                    [
                        rewards[policy, generation, labels == task].mean()
                        for task in ("gsm8k", "bbh")
                    ]
                ),
                "seed accuracy",
            )
    success = {
        (row["stage"], *row["task_key"]): row
        for row in costs["rows"]
        if row["status"] == "success"
    }
    for policy, name in enumerate(POLICIES):
        prompt = completion = 0
        for question, qid in enumerate(arrays["question_ids"]):
            action = int(actions[policy, question])
            for seed in SEEDS:
                selected = [success[("aggregation", qid, seed, action)]]
                selected.extend(
                    success[("member_generation", qid, seed, index)]
                    for index in TEAMS[action]
                )
                prompt += sum(row["prompt_tokens"] for row in selected)
                completion += sum(row["completion_tokens"] for row in selected)
        service = result["selected_policy_service_costs"][name]
        if (
            service["member_calls"],
            service["aggregation_calls"],
            service["measured_prompt_tokens"],
            service["measured_completion_tokens"],
        ) != (9 * n, 3 * n, prompt, completion):
            raise ValueError(
                "Logical service cost differs from selected measured records"
            )
    zero, minimum = inference_text(primary)
    lo, hi = primary["confidence_interval"]
    p = (
        "需复核"
        if primary.get("p_value") is None
        else format(primary["p_value"], ".8g")
    )
    inference_detail = (
        f"已冻结结果记录标准误{100 * primary['standard_error']:.6f}pp、t值{primary['statistic']:.6f}和Satterthwaite自由度{primary['degrees_of_freedom']:.3f}。"
        if primary.get("standard_error") is not None
        else "解析推断的退化状态按原始结果保留。"
    )
    lines = ["# E18 冻结组合路由独立质量确认报告", ""]
    if software_test_only:
        lines += ["**合成软件资格样例，未使用正式E18结果。**", ""]
    lines += [
        "## 1 评估边界",
        "",
        f"本报告使用已完成且来源闭合的{n}道原题，四套冻结策略、三个生成重复。每个策略有{3 * n}条生成事件，配对统计的独立单位为{n}道原题。主服务保持六个冻结模型中选三个，聚合器和评分器沿V2冻结身份。",
        "",
        "## 2 四策略主表",
        "",
        "| 策略 | GSM8K | BBH | 任务宏平均 | BBH子任务宏平均 | seed42 | seed137 | seed2024 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quality_rows:
        values = [
            row[key]
            for key in (
                "GSM8K准确率_pct",
                "BBH准确率_pct",
                "任务宏平均_pct",
                "BBH子任务宏平均_pct",
                "seed42_pct",
                "seed137_pct",
                "seed2024_pct",
            )
        ]
        lines.append(
            "| "
            + row["策略"]
            + " | "
            + " | ".join(f"{value:.3f}%" for value in values)
            + " |"
        )
    lines += [
        "",
        "整体任务宏平均按GSM8K和BBH各50%计算。BBH子任务宏平均单列，三个生成重复保留在同一原题簇内。",
        "",
        "## 3 预定配对推断",
        "",
        f"主比较为TSPO_uniform相对SPO_coordinate，点差{100 * primary['estimate']:+.3f}pp，已冻结解析t方法的95%区间为[{100 * lo:+.3f}, {100 * hi:+.3f}]pp，双侧p为{p}。",
        "",
        inference_detail,
        "",
        zero,
        "",
        minimum,
        "",
        "本报告直接使用冻结结果中的p值和区间。主问题相对0的差异与1pp参考门槛分别解释。",
        "",
        "| 五项次要比较 | 点差pp | 95%区间pp | 原p | Holm调整p |",
        "|---|---:|---|---:|---:|",
    ]
    for name in SECONDARY:
        row = stored["secondary"][name]
        a, b = name.split(" minus ")
        require_close(
            row["estimate"], macro[POLICIES.index(a)] - macro[POLICIES.index(b)], name
        )
        interval = row["confidence_interval"]
        fmt = lambda value: "需复核" if value is None else format(value, ".8g")
        lines.append(
            f"| {name} | {100 * row['estimate']:+.3f} | [{100 * interval[0]:+.3f}, {100 * interval[1]:+.3f}] | {fmt(row['p_value'])} | {fmt(row['holm_adjusted_p'])} |"
        )
    lines += [
        "",
        "次要比较保留原五项家族及其Holm结果。分层表使用描述性准确率，主检验身份保持原位。",
        "",
        "## 4 纠错与破坏",
        "",
        f"TSPO相对SPO在{3 * n}条生成事件上纠正{corrected}次、破坏{damaged}次，净纠错{corrected - damaged:+d}次。原题三重复平均差为正的有{int(np.sum(delta > 0))}题，为负的有{int(np.sum(delta < 0))}题，为零的有{int(np.sum(delta == 0))}题。",
        "",
        "## 5 选队频次与集中度",
        "",
        "频次按原题计数。HHI为各组合选择占比平方之和，数值越大表示选择更集中。",
        "",
        "| 策略 | 使用组合数 | 最高频组合索引 | 最高频占比 | HHI |",
        "|---|---:|---|---:|---:|",
    ]
    for row in quality_rows:
        lines.append(
            f"| {row['策略']} | {row['已使用组合数']} | {row['最高频组合']} | {100 * row['最高频占比']:.2f}% | {row['HHI']:.6f} |"
        )
    for index, name in enumerate(POLICIES):
        lines += [
            "",
            f"{name}的组合0至19原题频数为 `"
            + ", ".join(map(str, selection[index]["counts"]))
            + "`。",
        ]
    lines += [
        "",
        "## 6 来源、任务与子任务分解",
        "",
        "以下分组由已冻结输入侧元数据按有序ID和canonical hash对齐，分组规则保持原输入来源。",
        "",
        "| 分组 | 原题数 | 生成事件数 | Ridge | SPO | TSPO | GlobalFixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [
        ("GSM8K整体", labels == "gsm8k"),
        ("BBH原共池", frames == OLD_BBH),
        ("BBH官方扩展", frames == EXT_BBH),
        ("原fresh400", roles == "original_fresh400"),
        ("新增2600", roles == "approved_new2600"),
    ]
    for name, mask in groups:
        group = describe_group(rewards, labels, mask)
        lines.append(
            f"| {name} | {group['questions']} | {group['generation_events']} | "
            + " | ".join(f"{100 * value:.3f}%" for value in group["accuracy"])
            + " |"
        )
    lines += [
        "",
        "| BBH子任务 | 原题数 | Ridge | SPO | TSPO | GlobalFixed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for subtask in sorted(set(arrays["subtask_labels"][labels == "bbh"])):
        mask = (labels == "bbh") & (arrays["subtask_labels"] == subtask)
        lines.append(
            f"| {subtask} | {int(mask.sum())} | "
            + " | ".join(
                f"{100 * rewards[index, :, mask].mean():.3f}%" for index in range(4)
            )
            + " |"
        )
    costs_table = cost_rows(result, costs, wall=wall)
    lines += [
        "",
        "## 7 执行成本与计量边界",
        "",
        "| 口径 | 策略 | 成员调用 | 聚合调用 | 输入Token | 输出Token |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in costs_table:
        lines.append(
            f"| {row['口径']} | {row['策略']} | {row['成员调用数']} | {row['聚合调用数']} | {row['输入Token']} | {row['输出Token']} |"
        )
    lines += [
        "",
        "历史部分只计所需缓存成功记录的已知投入，旧实验全生命周期失败尝试未在本账本恢复。新增E18按owner记录结算实际物理尝试。每策略逻辑服务投影与联合评价的物理并集分别报告。货币费用未计量。",
        "",
    ]
    if wall and wall["available"]:
        lines.append(
            f"提供的native调用wall元数据覆盖{wall['measured_attempts']}/{wall['total_attempts']}次物理尝试，已覆盖部分累计{wall['seconds']:.3f}秒。"
        )
        lines.append(
            f"其中新增E18覆盖{wall['origins']['new_e18']['measured_attempts']}次、累计{wall['origins']['new_e18']['seconds']:.3f}秒；历史所需缓存成功记录覆盖{wall['origins']['required_historical_success']['measured_attempts']}次、累计{wall['origins']['required_historical_success']['seconds']:.3f}秒。"
        )
    else:
        lines.append("当前报告输入未附native调用wall汇总，该项未在本报告输入中计量。")
    lines += [
        "",
        "Native调用wall采用原始调用记录。并行和共享GPU下，调用wall累计量与实验经过时间各有口径。独占GPU核时间未计量。正式执行包含共享GPU和同卡多进程，时长解读保留该资源条件。",
        "",
        "## 8 机制证据与独立质量结果",
        "",
    ]
    if mechanism:
        lines.append(
            "已冻结E17登记在旧600题的100次语义保持重编号中记录SPO semantic_exact实际组合身份一致率6.135%、TSPO semantic_exact实际组合身份一致率100%；TSPO numeric_near实际组合身份一致率37.308%。这些证据用于量化编号与推理规则敏感性，原600题仍为同一批开发资产。"
        )
    else:
        lines.append(
            "E17的语义保持重编号对照用于量化监督规则的编号敏感性。该批开发题及其重复重编号保留原有样本身份。"
        )
    lines += [
        "",
        "E18在冻结模型、动作和统计方案后评价独立题目质量。固定编号下的选队频率描述服务分布；编号敏感性由语义保持重编号对照检验。当前独立质量结论采用本报告第3节的预定结果。",
        "",
        "## 9 访问、存储与公开范围",
        "",
        "评分阶段记录sealed标准答案解码为0，protected input值解码为0。准备期曾获批准的一次sealed输入指纹审计单独留档，两阶段访问口径分别记录。官方混合源的已授权原始字节SHA/结构扫描与目标答案值解码分别计量。",
        "",
        "native和评分小文件使用D盘临时物理工作区，闭合归档保存在E盘canonical private资产。该工程路径安排保留原始字节、逻辑scope和算子身份。",
        "",
        "公开范围包含低敏源码、配置、来源SHA和去标识聚合统计。逐题数组、原始模型输出、评分receipt和reference资产保留在private边界。访问凭据沿独立授权机制管理。",
        "",
        "## 10 优势、风险与本轮决策",
        "",
        "优势包括同一冻结模型池与聚合器、评分前冻结的逐题动作、必要稀疏执行并集，以及原题簇内保留三个生成重复的配对设计。",
        "",
        "风险包括解析t推断对独立原题与方差模型的依赖、BBH扩展来源占比，以及固定六模型和单聚合器下的外推范围。历史已知缓存成本与共享GPU时长也有各自计量边界。",
        "",
        "本轮结论截至上述E18固定确认结果。后续研究方案按本轮完整正负结果另行形成，报告生成阶段保持冻结评价范围。",
        "",
        "## 11 来源核对",
        "",
    ]
    for label, reference in provenance.items():
        lines.append(f"- {label} SHA256 `{reference['sha256']}`。")
    reproduce = [
        "# E18 报告复现草稿",
        "",
        "使用Python 3.11和NumPy。报告读取已冻结解析t/Holm结果，指标推导仅含描述性核对。",
        "",
        "`MAB_CODE`指向实际包含`e18_final_scoring.py`、`generate_final_report.py`及其冻结依赖的源码目录。原始reference/scorer、统计、scope和数据引用由最终manifest管理。",
        "",
        "```powershell",
        'python -B -X utf8 "$MAB_CODE/e18_final_scoring.py" --manifest FINAL_MANIFEST.json --manifest-sha256 MANIFEST_SHA --preflight-only --native-read-root NATIVE_READ_ROOT --runtime-root CPU_RUNTIME_ROOT',
        'python -B -X utf8 "$MAB_CODE/e18_final_scoring.py" --manifest FINAL_MANIFEST.json --manifest-sha256 MANIFEST_SHA --execute-scoring --native-read-root NATIVE_READ_ROOT --runtime-root CPU_RUNTIME_ROOT',
        'python -B -X utf8 "$MAB_CODE/generate_final_report.py" --results E18_FINAL_RESULTS.json --results-sha256 RESULT_SHA --rewards FINAL_POLICY_REWARDS.npz --rewards-sha256 REWARDS_SHA --cost-ledger PHYSICAL_COST_LEDGER.json --cost-ledger-sha256 COST_SHA --input-records ADMITTED3000_INPUTS_ONLY.jsonl --input-records-sha256 INPUT_SHA --output-root NEW_REPORT_DIRECTORY',
        "```",
        "",
        "正式入口在完整执行门槛通过后使用。`--walltime-ledger/--walltime-ledger-sha256`可绑定原生wall元数据，`--mechanism-registry/--mechanism-registry-sha256`可绑定既有E17机制登记。",
        "",
        "native只读展开和评分小文件位于D工作区，root完成核对后将闭合归档存E。报告输入依照以下SHA绑定。",
        "",
    ]
    for label, reference in provenance.items():
        reproduce.append(f"- {label} `{reference['sha256']}`。")
    return {
        "E18_CONFIRMATION_REPORT.md": "\n".join(lines) + "\n",
        "MAIN_TABLE.csv": csv_text(quality_rows),
        "COST_TABLE.csv": csv_text(costs_table),
        "REPRODUCE.md": "\n".join(reproduce) + "\n",
    }


def generate_report(
    results_ref,
    rewards_ref,
    costs_ref,
    inputs_ref,
    output_root,
    *,
    walltime_ref=None,
    mechanism_registry_ref=None,
    software_test_only=False,
):
    result, arrays, roles, costs = validate_inputs(
        results_ref,
        rewards_ref,
        costs_ref,
        inputs_ref,
        software_test_only=software_test_only,
    )
    mechanism = None
    if mechanism_registry_ref is not None:
        if mechanism_registry_ref["sha256"] != MECHANISM_REGISTRY_SHA:
            raise ValueError("Frozen mechanism registry identity differs")
        mechanism = read_json(mechanism_registry_ref)
        if not {"N01", "N02", "N03", "N07"} <= {
            row["id"] for row in mechanism["fact_registry"]
        }:
            raise ValueError("Required historical mechanism evidence is missing")
    wall = wall_measurements(walltime_ref, costs)
    provenance = dict(
        完成结果=results_ref,
        私有原题奖励=rewards_ref,
        物理成本账本=costs_ref,
        冻结输入侧元数据=inputs_ref,
        报告源码=dict(path=__file__, sha256=file_sha(__file__)),
    )
    if walltime_ref is not None:
        provenance["Native调用wall元数据"] = walltime_ref
    if mechanism_registry_ref is not None:
        provenance["E17既有机制登记"] = mechanism_registry_ref
    artifacts = report_artifacts(
        result,
        arrays,
        roles,
        costs,
        provenance,
        wall=wall,
        mechanism=mechanism,
        software_test_only=software_test_only,
    )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError(
            "Report artifacts are versioned outputs; choose a fresh report directory"
        )
    with tempfile.TemporaryDirectory(prefix=".e18-report-", dir=output) as staging:
        for name, text in artifacts.items():
            path = Path(staging) / name
            with path.open(
                "w",
                encoding="utf-8-sig" if name.endswith(".csv") else "utf-8",
                newline="",
            ) as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for name in artifacts:
            os.replace(Path(staging) / name, output / name)
    return dict(
        status="TOY_REPORT_COMPLETE" if software_test_only else "REPORT_COMPLETE",
        artifacts=list(artifacts),
        original_questions=result["original_questions"],
        generation_events=3 * result["original_questions"],
        new_gold_reads=0,
        new_scorer_calls=0,
        new_inference_tests=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "rewards", "cost-ledger", "input-records"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    for name in ("walltime-ledger", "mechanism-registry"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    def reference(name):
        path = getattr(args, name)
        sha = getattr(args, name + "_sha256")
        if path is None:
            if sha is not None:
                raise ValueError("An optional SHA requires its matching artifact")
            return None
        if sha is None:
            raise ValueError("Every supplied artifact requires its explicit SHA")
        return dict(path=str(path), sha256=sha)

    result = generate_report(
        reference("results"),
        reference("rewards"),
        reference("cost_ledger"),
        reference("input_records"),
        args.output_root,
        walltime_ref=reference("walltime_ledger"),
        mechanism_registry_ref=reference("mechanism_registry"),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
