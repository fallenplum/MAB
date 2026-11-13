"""Rebuild E9 aggregate cost and CSV deliverables without fitting or rescoring.

The historical loader is restricted to the original train600 asset.  All other
inputs are completed aggregate results, software qualifications, or receipts.
Existing experiment metrics and sources are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


GENERATION_SEEDS = (42, 137, 2024)
FAMILIES = ("B", "C", "D")
PERMUTATION_SEEDS = (20261114091, 20261114092, 20261114093)
MULTIPLIERS = (0.1, 1.0, 10.0)
OBSERVED_PROCESS_WALL_SECONDS = {
    "A_first_failed_audit_process": 2.1316688,
    "A_successful_replay_process": 2.861057,
    "six_prior_recomputation_audit_process": 3.3297257,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(condition, message):
    if not condition:
        raise AssertionError(message)


def write_csv(path, rows):
    checked(bool(rows), "CSV rows must be nonempty")
    fields = list(rows[0])
    checked(all(list(row) == fields for row in rows), "CSV schemas must match")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        restored = list(csv.DictReader(stream))
    checked(len(restored) == len(rows), "Saved CSV row count changed")
    for expected, actual in zip(rows, restored):
        for key, value in expected.items():
            if isinstance(value, (int, float)):
                checked(float(actual[key]) == float(value), "CSV numeric roundtrip failed")
            else:
                checked(actual[key] == ("" if value is None else str(value)), "CSV text roundtrip failed")
    return {"rows": len(rows), "columns": len(fields), "sha256": sha256(path)}


def comparison_fields(record):
    fields = {
        "gain_fraction": record["gain"],
        "gain_pp": 100.0 * record["gain"],
        "ordinary_paired95_lower_pp": 100.0 * record["ordinary_paired95"][0],
        "ordinary_paired95_upper_pp": 100.0 * record["ordinary_paired95"][1],
        "unadjusted_two_sided_centered_bootstrap_p": record["unadjusted_two_sided_centered_bootstrap_p"],
        "corrected_generation_events": record["corrected_events"],
        "destroyed_generation_events": record["destroyed_events"],
        "net_generation_events": record["net_events"],
        "independent_questions": record["independent_questions"],
        "recorded_generation_events": record["recorded_generation_events"],
        "gsm8k_gain_pp": 100.0 * record["task_gain"]["gsm8k"],
        "bbh_question_gain_pp": 100.0 * record["task_gain"]["bbh"],
    }
    fields.update({f"generation_{seed}_gain_pp": 100.0 * value
                   for seed, value in zip(GENERATION_SEEDS, record["seed_gain"])})
    fields.update({f"outer_{outer}_gain_pp": 100.0 * value
                   for outer, value in enumerate(record["outer_gain"])})
    return fields


def build_tables(main, primary, controls, protocol):
    summary = []
    leading_names = ("A", "B", "C", "D", "M0_original", "M4", "TSPO", "NS", "Davidson", "GlobalFixed")
    names = list(leading_names) + [name for name in main["metrics"] if name not in leading_names]
    for name in names:
        metric = main["metrics"][name]
        role = "content_control" if "_perm" in name else "inherited_reference"
        if name in ("A", "B", "C", "D"):
            role = "main_family"
        row = {
            "method": name,
            "role": role,
            "accuracy_fraction": metric["accuracy"],
            "accuracy_pct": 100.0 * metric["accuracy"],
            "gsm8k_accuracy_pct": 100.0 * metric["task_accuracy"]["gsm8k"],
            "bbh_question_accuracy_pct": 100.0 * metric["task_accuracy"]["bbh"],
            "bbh_27subtask_macro_accuracy_pct": 100.0 * metric["bbh_subtask_macro"],
        }
        row.update({f"generation_{seed}_accuracy_pct": 100.0 * value
                    for seed, value in zip(GENERATION_SEEDS, metric["seed_accuracy"])})
        row.update({f"outer_{outer}_accuracy_pct": 100.0 * value
                    for outer, value in enumerate(metric["outer_accuracy"])})
        distribution = metric["distribution"]
        row.update({
            "empirical_three_repeat_mean_selection_loss_pp": 100.0 * metric["empirical_mean_selection_loss"],
            "largest_action_selection_fraction": distribution["top_fraction"],
            "effective_combinations": distribution["effective_combinations"],
            "exact_score_tie_questions": distribution.get("exact_ties"),
            "raw_uncalibrated_minimum_top_score_gap": distribution.get("top_gap_min"),
            "raw_uncalibrated_median_top_score_gap": distribution.get("top_gap_median"),
            "raw_uncalibrated_maximum_top_score_gap": distribution.get("top_gap_max"),
            "independent_questions": 600,
            "recorded_generation_events": 1800,
            "source": "E9_MAIN_RESULTS.json",
        })
        summary.append(row)

    comparisons = []
    for name, record in main["comparisons"].items():
        candidate, reference = name.split("-", 1)
        if name in primary:
            for key, value in primary[name].items():
                if key in record:
                    checked(record[key] == value, "Primary and main statistics disagree")
        comparisons.append({
            "comparison": name,
            "candidate": candidate,
            "reference": reference,
            "comparison_role": "primary" if name in primary else "descriptive_mechanism_or_reference",
            **comparison_fields(record),
            "Holm_p_two_primary_tests": primary[name]["Holm_p"] if name in primary else None,
            "positive_Holm_support": int(primary[name]["positive_Holm_support"]) if name in primary else None,
            "interval_scope": "ordinary paired95, fixed fitted models, reused development CV",
            "source": "E9_PRIMARY_STATISTICS.json" if name in primary else "E9_MAIN_RESULTS.json",
        })

    grid = []
    for outer, families in main["selected_hyperparameters"].items():
        for family, selection in families.items():
            for multiplier, accuracy in zip(protocol["lambda_multipliers"], selection["grid_accuracy"]):
                grid.append({
                    "outer_fold": int(outer),
                    "family": family,
                    "lambda_multiplier": multiplier,
                    "raw_alpha": 100.0 * multiplier,
                    "inner_fit_questions": 320,
                    "inner_validation_questions": 160,
                    "inner_folds": 3,
                    "inner_lambda": 100.0 * multiplier / (3.0 * 320.0 * 20.0),
                    "inner_routed_accuracy_fraction": accuracy,
                    "inner_routed_accuracy_pct": 100.0 * accuracy,
                    "selected_for_outer_refit": int(multiplier == selection["multiplier"]),
                    "selected_multiplier_is_grid_boundary": int(selection["grid_boundary"]),
                    "selection_weighting": "gsm8k50_bbh_question50",
                    "selection_tie_tolerance": 1e-12,
                    "selection_tie_preference": "stronger_regularization",
                    "source": "E9_MAIN_RESULTS.json",
                })

    content = []
    for family, control in controls.items():
        for seed, accuracy, record in zip(control["seeds"], control["permuted_accuracy"], control["main_minus_permuted"]):
            content.append({
                "family": family,
                "permutation_seed": seed,
                "main_accuracy_pct": 100.0 * control["main_accuracy"],
                "permuted_accuracy_pct": 100.0 * accuracy,
                **comparison_fields(record),
                "main_minus_three_permutation_mean_point_pp": 100.0 * control["main_minus_mean_permutation_point"],
                "minimum_standard_permutation_MC_p": control["minimum_possible_standard_MC_p"],
                "configuration_retuned_after_permutation": 0,
                "inference_scope": control["inference"],
                "source": "E9_CONTENT_CONTROLS.json",
            })
    checked(len(summary) == 19 and len(comparisons) == 17 and len(grid) == 45 and len(content) == 9,
            "Unexpected aggregate table coverage")
    return {"MAIN_SUMMARY.csv": summary, "COMPARISONS.csv": comparisons,
            "GRID.csv": grid, "CONTENT_CONTROLS.csv": content}


def historical_feedback(asset_root):
    loader = asset_root / "research/decision_acquisition_e3/public/assets.py"
    sys.path.insert(0, str(loader.parent))
    from assets import load_assets

    data, _ = load_assets(asset_root)
    costs = data["cost"]
    member_calls = int(np.sum(costs["member_execution_counts"]))
    aggregate_calls = int(np.sum(costs["aggregate_execution_counts"]))
    input_tokens = int(np.sum(costs["member_prompt_tokens"]) + np.sum(costs["aggregate_prompt_tokens"]))
    output_tokens = int(np.sum(costs["member_completion_tokens"]) + np.sum(costs["aggregate_completion_tokens"]))
    unique_questions = len(set(data["canonical"].tolist()))
    checked(len(data["ids"]) == unique_questions == 600, "Historical train600 identity count changed")
    checked(member_calls == 10800 and aggregate_calls == 36000, "Historical physical call counts changed")
    checked(input_tokens + output_tokens == 37274127, "Historical token count changed")
    return {
        "scope": "one shared historical full-feedback train600 preparation, counted once across overlapping folds and methods",
        "canonical_unique_questions": unique_questions,
        "gold_questions": unique_questions,
        "generation_repeats_per_question": 3,
        "member_calls": member_calls,
        "aggregator_calls": aggregate_calls,
        "all_historical_LLM_calls": member_calls + aggregate_calls,
        "cached_aggregate_score_events": int(np.asarray(costs["aggregate_scores"]).size),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
        "source_call_wall_seconds_sum": float(np.sum(costs["member_wall_time_s"]) + np.sum(costs["aggregate_wall_time_s"])),
        "source_call_wall_scope": "sum of historical recorded call wall times, including receipt-accounted failures",
        "gpu_kernel_seconds": None,
        "monetary_cost": None,
        "monetary_cost_reason": "No currency-denominated billing receipt was supplied",
        "feedback_information": "cached full 20-action x3-generation feedback with shared paid gold",
        "upstream_encoder_pretraining_cost": None,
        "upstream_encoder_pretraining_cost_reason": "Original encoder pretraining resources are outside available receipts",
        "loader_source_sha256": sha256(loader),
        "live_verified": True,
    }


def build_cost(arguments, main, teacher_qualification, a_qualification, teacher_audit, heads_qualification,
               math_qualification, math_failure, source_paths):
    experiment_root = arguments.experiment_root
    receipt_paths = sorted((experiment_root / "private/teachers").glob("fold*/*/FROZEN_RECEIPT.json"))
    receipts = [read_json(path) for path in receipt_paths]
    checked(len(receipts) == 30, "Expected thirty new teacher regions")
    checked(all(record["evaluation_reward_access"] is False for record in receipts), "Teacher receipt accessed evaluation outcomes")
    checked(all(record["physical_LLM_calls"] == 0 for record in receipts), "Unexpected new teacher model calls")
    prior_count = sum(record["total_prior_fits"] for record in receipts)
    sampling_count = sum(record["total_sampling_calls"] for record in receipts)
    checked(prior_count == sampling_count == 90, "Teacher prior or sampling count changed")

    cost = main["computational_cost"]
    logs = cost["fit_logs"]
    actual = [record for record in logs if record["actual_head_fit"]]
    derived = [record for record in logs if not record["actual_head_fit"]]
    checked(len(logs) == 195 and len(actual) == 165 and len(derived) == 30, "Head execution count changed")
    checked(all(record["accepted"] for record in logs), "A formal head was rejected")
    checked(sum(record["cpu_seconds"] for record in logs) == cost["head_cpu_seconds"], "Head CPU cost does not reconcile")
    checked(abs(sum(record["wall_seconds"] for record in logs) - cost["head_wall_seconds_sum"]) < 1e-12,
            "Head wall cost does not reconcile")
    phases = {}
    for name in ("inner", "outer", "controls"):
        selected = [record for record in logs if record["phase"].startswith(name)]
        phases[name] = {"slots": len(selected),
                        "actual_fits": sum(bool(record["actual_head_fit"]) for record in selected),
                        "derived_heads": sum(not record["actual_head_fit"] for record in selected),
                        "cpu_seconds_sum": sum(record["cpu_seconds"] for record in selected),
                        "wall_seconds_sum": sum(record["wall_seconds"] for record in selected)}

    toy_receipt_path = experiment_root / "private/qualification/variable_teacher61/FROZEN_RECEIPT.json"
    toy = read_json(toy_receipt_path)
    checked(toy["total_prior_fits"] == teacher_qualification["actual_synthetic_prior_fits"] == 3,
            "Teacher qualification prior count changed")
    checked(toy["total_sampling_calls"] == teacher_qualification["actual_synthetic_sampling_calls"] == 3,
            "Teacher qualification sampling count changed")
    head_toys = heads_qualification["fit_diagnostics"]
    checked(len(head_toys) == 12 and all(record["accepted"] for record in head_toys), "Head software qualification changed")
    a_cost = a_qualification["qualification_cost"]
    checked(a_cost["total_qualification_ridge_fits"] == 51, "A replay count changed")
    prior_audit = teacher_audit["qualification_cost"]
    checked(prior_audit["prior_moment_recomputations"] == 6, "Teacher replay audit count changed")
    checked(math_qualification["status"] == "PASS" and math_failure["status"] == "FAIL",
            "Mathematical qualification history changed")
    mock = heads_qualification["actual_gradient_rejection_probe"]
    checked(not mock["accepted"] and mock["optimizer_success"], "Expected mock gradient rejection missing")
    receipt_manifest = [sha256(path) for path in receipt_paths]
    receipt_manifest_sha = hashlib.sha256(json.dumps(receipt_manifest, separators=(",", ":")).encode()).hexdigest()
    result = {
        "schema": "E9_aggregate_cost_ledger_v1",
        "status": "COMPLETE_LEDGER_WITH_EXPLICIT_UNTIMED_COMPONENTS",
        "scope": "completed frozen exploratory E9, aggregate summaries and receipts only",
        "units": {"time": "seconds", "memory": "bytes", "rates": "fractions unless header specifies pct or pp"},
        "new_physical_resources": {"LLM_calls": 0, "API_model_calls": 0, "new_data_questions": 0,
                                   "new_gold_acquisitions": 0, "new_GPU_LLM_training_runs": 0},
        "shared_historical_feedback": historical_feedback(arguments.asset_root),
        "formal_heads": {
            "planned_candidate_slots": 195, "saved_slots": len(logs), "actual_fits": len(actual),
            "derived_Ridge_heads": len(derived), "failures": 0,
            "cpu_seconds_sum": cost["head_cpu_seconds"],
            "wall_seconds_sum": cost["head_wall_seconds_sum"],
            "actual_fit_cpu_seconds_sum": sum(record["cpu_seconds"] for record in actual),
            "actual_fit_wall_seconds_sum": sum(record["wall_seconds"] for record in actual),
            "derived_validation_cpu_seconds_sum": sum(record["cpu_seconds"] for record in derived),
            "derived_validation_wall_seconds_sum": sum(record["wall_seconds"] for record in derived),
            "phase_breakdown": phases,
            "maximum_actual_gradient_infinity_norm": max(record["gradient_infinity_norm"] for record in logs),
            "maximum_recorded_process_peak_ram_bytes": max(record.get("peak_ram_bytes", 0) for record in actual),
            "peak_ram_scope": "maximum of Windows process-lifetime working-set peaks observed by head fits",
            "timing_scope": "sum of head fitting and derived-head validation entries, excludes feature preparation, source validation and artifact I/O",
            "CPU_clock_note": "Windows process CPU readings show 0.015625-second increments and some short entries report zero",
            "full_pipeline_elapsed_wall_seconds": None,
            "full_pipeline_elapsed_reason": "A complete process-start-to-finish timer is absent from these receipts",
        },
        "formal_teachers": {
            "new_regions": len(receipts), "inner320_regions": 15, "permuted480_regions": 15,
            "prior_fits": prior_count, "posterior_sampling_calls": sampling_count,
            "draws_per_question": 512, "fallback_question_instances": sum(record["fallback_questions"] for record in receipts),
            "B_D_share_targets": True, "outer_mean_assets_reused": len(teacher_audit["outer_512_mean_reuse"]),
            "cpu_seconds_sum": sum(record["cpu_seconds"] for record in receipts),
            "region_wall_seconds_sum": sum(record["wall_seconds"] for record in receipts),
            "wall_scope": "sum of thirty individual teacher-region timers",
            "region_timer_includes": "internal splitting, prior estimation, posterior sampling, array checks, compressed output writes and artifact hashing through receipt construction",
            "process_concurrency_limit": 2,
            "parallel_pipeline_elapsed_wall_seconds": None,
            "parallel_pipeline_elapsed_reason": "Region wall sums and completion timestamps do not provide a complete concurrent pipeline elapsed timer",
            "peak_ram_bytes": None,
            "peak_ram_reason": "Teacher receipts do not contain a peak-RAM measurement",
            "receipt_count": len(receipts), "receipt_manifest_sha256": receipt_manifest_sha,
        },
        "qualification_and_replay": {
            "A_reuse": {
                "successful_Ridge_replays": a_cost["successful_replay_ridge_fits"],
                "earlier_failed_attempt_Ridge_fits": a_cost["earlier_audit_attempt_ridge_fits"],
                "all_Ridge_qualification_fits": a_cost["total_qualification_ridge_fits"],
                "successful_replay_audit_cpu_seconds": a_cost["cpu_seconds"],
                "successful_replay_audit_wall_seconds": a_cost["wall_seconds"],
                "successful_timer_scope": "audit main() through report construction, including source verification, input reads, feature fitting and fifty Ridge replays, excluding interpreter startup",
                "successful_solver_only_cpu_seconds": None,
                "successful_solver_only_wall_seconds": None,
                "successful_process_elapsed_wall_seconds": OBSERVED_PROCESS_WALL_SECONDS["A_successful_replay_process"],
                "failed_process_elapsed_wall_seconds": OBSERVED_PROCESS_WALL_SECONDS["A_first_failed_audit_process"],
                "failed_solver_cpu_seconds": None,
                "process_timing_provenance": "coordinator-observed execution durations, including process startup and surrounding work",
                "failed_solver_cpu_reason": "The failed process has no separate solver CPU timer",
                "failure_and_repair": a_cost["earlier_audit_attempt_reason"],
            },
            "teacher_variable61": {
                "synthetic_questions": 61, "prior_fits": toy["total_prior_fits"],
                "sampling_calls": toy["total_sampling_calls"], "draws_per_question": 512,
                "cpu_seconds": toy["cpu_seconds"], "wall_seconds": toy["wall_seconds"],
                "scope": "one synthetic software-qualification region, separate from thirty formal regions",
            },
            "teacher_prior_independent_replay": {
                "prior_moment_recomputations": prior_audit["prior_moment_recomputations"],
                "new_sampling_calls": prior_audit["new_posterior_sampling_calls"],
                "process_elapsed_wall_seconds": OBSERVED_PROCESS_WALL_SECONDS["six_prior_recomputation_audit_process"],
                "process_timing_provenance": "coordinator-observed entire audit process including startup, validation and I/O",
                "prior_only_cpu_seconds": None, "prior_only_wall_seconds": None,
                "timing_reason": "No timer isolates the six moment recomputations",
            },
            "heads": {
                "normal_synthetic_fits": len(head_toys), "expected_mock_rejection_calls": 1,
                "mock_real_optimizer_fits": 0,
                "normal_fit_cpu_seconds_sum": sum(record["cpu_seconds"] for record in head_toys),
                "normal_fit_wall_seconds_sum": sum(record["wall_seconds"] for record in head_toys),
                "mock_validation_cpu_seconds": mock["cpu_seconds"],
                "mock_validation_wall_seconds": mock["wall_seconds"],
                "full_qualification_cpu_seconds": None,
                "full_qualification_wall_seconds": None,
                "timing_scope": "fit-entry timers exclude finite differences and other software checks",
            },
            "mathematical_operator": {
                "current_checks_passed": math_qualification["passed_count"],
                "current_check_count": math_qualification["check_count"],
                "normal_synthetic_fits_per_invocation": 2,
                "invocation_count": None, "all_synthetic_fit_count": None,
                "cpu_seconds": None, "wall_seconds": None,
                "timing_status": "not independently timed, repeated invocation count not completely retained",
                "retained_failed_qualification_records": 1,
                "failed_checks": [name for name, record in math_failure["checks"].items() if not record["passed"]],
                "repair": "subtract per-row maximum logit before the identical CSCE data term, preserving original L2 and denominator",
            },
        },
        "failure_ledger": {
            "formal_head_fit_failures": 0,
            "mathematical_stability_qualification_failures_retained": 1,
            "A_reuse_audit_failed_attempts_retained": 1,
            "expected_mock_gradient_rejections": 1,
            "expected_invalid_input_rejections_are_failures": False,
        },
        "unavailable_total_costs": {
            "end_to_end_CPU_seconds": None, "end_to_end_elapsed_wall_seconds": None,
            "whole_pipeline_peak_RAM_bytes": None, "GPU_kernel_seconds": None,
            "reason": "Feature preparation, I/O, source verification, all audit work and repeated mathematical qualifications lack mutually complete timers",
        },
        "public_content_policy": "aggregate metrics, counts and configurations only, private identities, per-question outcomes and fitted coefficients remain private",
        "source_sha256": {path.name: sha256(path) for path in source_paths},
        "coordinator_process_observations": OBSERVED_PROCESS_WALL_SECONDS,
    }
    checked(result["formal_teachers"]["outer_mean_assets_reused"] == 5, "Outer teacher reuse count changed")
    return result, receipt_paths + [toy_receipt_path]


def cost_markdown(cost):
    history = cost["shared_historical_feedback"]
    heads = cost["formal_heads"]
    teachers = cost["formal_teachers"]
    qualifications = cost["qualification_and_replay"]
    a = qualifications["A_reuse"]
    toy = qualifications["teacher_variable61"]
    head_toy = qualifications["heads"]
    return f"""# E9 成本与执行口径

## 正式执行

**195 个候选头槽位由 165 次实际拟合与 30 个经几何、梯度及决策核验的派生 Ridge 头构成。** 新建教师区域为 30 个，包含 90 次 prior 估计与 90 次 posterior sampling。B/D 按合法区域共享标签，另复用 5 个既有 outer 后验均值资产。

| 项目 | 计数 | CPU 秒 | wall 秒 | 计时范围 |
|---|---:|---:|---:|---|
| 正式头拟合与派生核验 | 195 槽位 | {heads['cpu_seconds_sum']:.6f} | {heads['wall_seconds_sum']:.9f} | 全部头日志求和 |
| 其中实际头拟合 | 165 | {heads['actual_fit_cpu_seconds_sum']:.6f} | {heads['actual_fit_wall_seconds_sum']:.9f} | 实际求解入口求和 |
| 其中派生头核验 | 30 | {heads['derived_validation_cpu_seconds_sum']:.6f} | {heads['derived_validation_wall_seconds_sum']:.9f} | 派生与复算入口求和 |
| 新教师区域 | 30 | {teachers['cpu_seconds_sum']:.6f} | {teachers['region_wall_seconds_sum']:.9f} | 逐区域 receipt 求和 |
| 既有 outer 均值复用 | 5 | 未独立计时 | 未独立计时 | 本轮新增采样为 0，验证与读取开销归入外围流程 |

内层头槽位为 135，其中实际拟合 105、派生 30。外层最终头为 15 次实际拟合，固定配置内容对照为 45 次实际拟合。全部槽位实际梯度通过 1e-7 门限，最大值为 {heads['maximum_actual_gradient_infinity_norm']:.9g}。

头日志计时覆盖求解或派生复算，feature preparation、PCA/scaler 估计、源验证与产物 I/O 的开销需另外计量。Windows CPU 记录呈现 0.015625 秒步进，短调用的 CPU 字段可能记录为 0。应保留该分辨率对 CPU 总和与 wall 总和比较的影响。

教师 wall 为 30 个区域各自计时之和。区域计时包括内部划分、prior 估计、posterior sampling、数组检查、压缩产物写入及 receipt 构建前的哈希。运行并发上限为两个进程，整个并发流程的启动至完成时间在现有 receipt 中缺少完整独立计时。正式头的完整 pipeline elapsed 同样未独立记录。CPU、逐区域 wall 与并发 elapsed 应按各自定义解释。

## 资格验证与独立审计

| 项目 | 实际计数 | 已记录 CPU 秒 | 已记录 wall 秒 | 范围与缺口 |
|---|---|---:|---:|---|
| A 复用成功审计 | 45 inner + 5 outer Ridge | {a['successful_replay_audit_cpu_seconds']:.6f} | {a['successful_replay_audit_wall_seconds']:.9f} | 审计 main() 至报告构建，含源验证、输入读取、特征估计及 50 次 replay |
| A 成功 replay 完整进程 | 同上 50 次 | 未独立记录 | {a['successful_process_elapsed_wall_seconds']:.7f} | 协调器观察，包含 startup 与外围工作 |
| A 最早失败审计进程 | 1 次 Ridge | 求解 CPU 未独立记录 | {a['failed_process_elapsed_wall_seconds']:.7f} | 协调器观察，包含 startup，精确断言运算顺序已修正 |
| 61 题教师软件资格 | 3 prior + 3 sampler | {toy['cpu_seconds']:.6f} | {toy['wall_seconds']:.9f} | 合成区域，与 30 个正式区域分别计数 |
| 教师 prior 独立审计进程 | 6 次矩重算、0 次采样 | prior 专属 CPU 未独立记录 | {OBSERVED_PROCESS_WALL_SECONDS['six_prior_recomputation_audit_process']:.7f} | 协调器观察的完整审计进程，包含启动、验证与 I/O |
| 预测头合成资格正常拟合 | 12 次 | {head_toy['normal_fit_cpu_seconds_sum']:.6f} | {head_toy['normal_fit_wall_seconds_sum']:.9f} | 仅正常拟合入口求和 |
| 预测头 mock 梯度门探针 | 1 次 mock，实际优化拟合 0 | {head_toy['mock_validation_cpu_seconds']:.6f} | {head_toy['mock_validation_wall_seconds']:.9f} | 验证 optimizer success 且高梯度时正确拒绝 |
| 数学核合成资格 | 每次调用含 2 次正常拟合 | 未独立计时 | 未独立计时 | 实际重复调用次数缺少完整计时记录 |

A 资格求解合计 51 次，其中成功 50 次、早期断言失败 1 次。A 成功审计的内部计时从 main() 开始，覆盖源验证、输入读取、特征估计与 replay，求解专属时间未独立记录。预测头完整软件资格还包括有限差分等工作，其全流程时间未独立记录。数学核资格保留 1 份稳定性失败记录，采用题内减最大 logit 的等价式修正后，当前 14 项检查全部通过。两个失败记录均保留，正式头拟合失败为 0。mock 的预期拒绝单列为软件探针。

## 历史完整反馈资源

通过原 train600-only loader 现场复核，历史共享资产包含 600 个唯一 canonical 问题与 600 份 gold。全部方法和重叠折共享这一份历史完整反馈，以下成本计一次。

| 历史资源 | 已验证总量 |
|---|---:|
| 成员 LLM 调用 | {history['member_calls']:,} |
| 聚合 LLM 调用 | {history['aggregator_calls']:,} |
| 历史 LLM 调用合计 | {history['all_historical_LLM_calls']:,} |
| 输入 Token | {history['input_tokens']:,} |
| 输出 Token | {history['output_tokens']:,} |
| Token 合计 | {history['tokens']:,} |
| 已缓存聚合评分事件 | {history['cached_aggregate_score_events']:,} |
| 历史调用 wall 秒求和 | {history['source_call_wall_seconds_sum']:.6f} |

这些是六成员、二十动作、每题三次生成的完整反馈准备资源，gold 按 canonical 去重。它们提供了本轮监督信息预算，属于已发生的历史成本。缓存复用降低本轮增量物理调用，完整反馈的既有资源仍须保留在总账中。历史调用 wall 是记录值求和，GPU kernel 时间与货币账单均缺少独立证据。

## 本轮增量与缺失计量

本轮新增物理 LLM 调用为 0，新增 API 模型调用为 0，新增数据问题为 0，新增 gold 采集为 0。上述零值与历史反馈资源分项呈现。

头日志记录到的最大进程工作集高水位为 {heads['maximum_recorded_process_peak_ram_bytes']:,} bytes。其口径为 Windows 进程累计峰值。教师峰值 RAM、全流程峰值 RAM、全流程 CPU 与 elapsed wall、全部数学资格重复调用时间，以及上游 encoder 原始预训练资源均未独立取得。COST.json 对这些字段保存 null，并给出原因。

## CSV 与来源核对

MAIN_SUMMARY.csv 保存 19 个方法及对照的整体、任务、生成和外折聚合值。COMPARISONS.csv 保存 17 项既有比较，Holm 只用于冻结的两项主比较。GRID.csv 保存 B/C/D 的 45 行外折×家族×正则配置。CONTENT_CONTROLS.csv 保存 9 项固定配置置乱对照，其均值差仅作点估计，三次置乱的标准 Monte Carlo 最小 p 为 0.25。

四份 CSV 仅包含聚合指标与配置。数值使用源 JSON 的精度写出，比例、百分数与 pp 由列名区分，缺失值保留空字段。分数间隔保留各头未经校准的原始尺度。生成器逐值核对写出后的数值及行数，并对全部输入执行前后 SHA-256 一致性检查。COST.json 记录输入摘要与输出核对结果。

## 优势与劣势（风险）

- **优势**来自正式求解、派生、教师重建、资产复用、合成资格与审计 replay 的分项记录，历史完整反馈按唯一问题计一次，失败记录保留。
- **劣势（风险）**集中在外围流程及部分早期审计缺少独立计时，分项已知时间不足以重建完整端到端总耗时。历史调用 wall 与当前 CPU 时间分别反映各自执行环节。

对现场核实的计数、Token、逐入口时间和 CSV 源值一致性的置信度为高。完整端到端资源总量仍缺少计量依据。

## 拓展思考

历史监督信息预算与本轮增量计算量回答不同的成本问题。保留两者可以支持后续在预先冻结的信息预算下比较研究方案。

## 你可能想知道

本轮 195 个头槽位都具有执行记录，其中 30 个由闭式 Ridge 几何派生并重新核验。这解释了候选槽位数、实际优化拟合数与计时条目数的对应关系。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--support-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    for name in ("a_qualification", "teacher_audit", "heads_qualification", "math_qualification", "math_failure"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    arguments = parser.parse_args()
    arguments.experiment_root = arguments.experiment_root or arguments.asset_root / "research/e9_consistent_cost_sensitive_learning"
    public = arguments.experiment_root / "public"
    support = arguments.support_root
    paths = {
        "main": public / "E9_MAIN_RESULTS.json",
        "primary": public / "E9_PRIMARY_STATISTICS.json",
        "controls": public / "E9_CONTENT_CONTROLS.json",
        "protocol": public / "E9_PROTOCOL.json",
        "teacher_qualification": public / "TEACHER_SOFTWARE_QUALIFICATION.json",
        "a_qualification": support / "audit/E9_ASSET_A_REUSE_QUALIFICATION.json",
        "teacher_audit": support / "audit/E9_TEACHER_ARTIFACT_AUDIT.json",
        "heads_qualification": support / "public_qualification/E9_HEADS_QUALIFICATION.json",
        "math_qualification": support / "math/public/CSCE_QUALIFICATION.json",
        "math_failure": support / "math/private/qualification/CSCE_QUALIFICATION_V1.json",
    }
    for name in ("a_qualification", "teacher_audit", "heads_qualification", "math_qualification", "math_failure"):
        paths[name] = getattr(arguments, name) or paths[name]
    receipt_paths = list((arguments.experiment_root / "private/teachers").glob("fold*/*/FROZEN_RECEIPT.json"))
    receipt_paths.append(arguments.experiment_root / "private/qualification/variable_teacher61/FROZEN_RECEIPT.json")
    tracked_paths = list(paths.values()) + receipt_paths
    before = {path: sha256(path) for path in tracked_paths}
    source = {name: read_json(path) for name, path in paths.items()}
    tables = build_tables(source["main"], source["primary"], source["controls"], source["protocol"])
    cost, _ = build_cost(arguments, source["main"], source["teacher_qualification"], source["a_qualification"],
                         source["teacher_audit"], source["heads_qualification"], source["math_qualification"],
                         source["math_failure"], list(paths.values()))
    checked(all(sha256(path) == digest for path, digest in before.items()), "Read-only source changed during aggregation")
    arguments.output.mkdir(parents=True, exist_ok=True)
    csv_validation = {name: write_csv(arguments.output / name, rows) for name, rows in tables.items()}
    cost["deliverable_validation"] = {"source_files_unchanged": True, "csv_numeric_roundtrip_exact": True,
                                      "csv_files": csv_validation}
    (arguments.output / "COST_AND_EXECUTION.md").write_text(cost_markdown(cost), encoding="utf-8")
    cost["deliverable_validation"]["cost_markdown_sha256"] = sha256(arguments.output / "COST_AND_EXECUTION.md")
    checked(all(sha256(path) == digest for path, digest in before.items()), "Read-only source changed during delivery")
    cost["generator_source_sha256"] = sha256(Path(__file__))
    (arguments.output / "COST.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "csv_rows": {name: len(rows) for name, rows in tables.items()},
                      "teacher_cpu_seconds": cost["formal_teachers"]["cpu_seconds_sum"],
                      "teacher_region_wall_seconds_sum": cost["formal_teachers"]["region_wall_seconds_sum"],
                      "source_files_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
