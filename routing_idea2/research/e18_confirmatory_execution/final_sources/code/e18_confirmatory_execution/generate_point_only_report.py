"""Render completed E18 point results without inferential statistics.

Only completed result summaries and private numeric policy arrays are consumed.
No model, scorer, reference provider, raw question or native cache is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


POLICIES = ("Ridge", "SPO_coordinate", "TSPO_uniform", "GlobalFixed")
SEEDS = (42, 137, 2024)
FINALIZER_SHA = "9b164c40ad6b3a1b21f84c7dc446438a11ce6421d56991d82d224ae2c4b0daa8"
PAIRS = ((2, 1), (2, 0), (1, 0), (2, 3), (1, 3), (0, 3))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            checksum.update(block)
    return checksum.hexdigest()


def checked(reference):
    path = Path(reference["path"]).resolve(strict=True)
    require(path.is_file() and file_sha(path) == reference["sha256"],
            "Explicit completed report input SHA differs")
    return path


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) < 1e-10,
            "Completed summary and numeric arrays disagree: " + label)


def macro(reward, labels, mask):
    tasks = [task for task in ("gsm8k", "bbh") if np.any(mask & (labels == task))]
    require(tasks, "Empty report group")
    return float(np.mean([reward[:, mask & (labels == task)].mean() for task in tasks]))


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_completed(results_ref, rewards_ref):
    result = json.loads(checked(results_ref).read_text(encoding="utf-8-sig"))
    require(result.get("schema") == "e18_final_confirmation_results_v1"
            and result.get("status") == "COMPLETE"
            and result.get("finalizer_source_sha256") == FINALIZER_SHA,
            "A completed frozen-finalizer E18 result is required")
    gate = result["execution_gate"]
    require(gate.get("status") == "PASS"
            and result["generation_seeds"] == list(SEEDS)
            and (result["original_questions"], result["unique_aggregate_keys"],
                 result["unique_member_keys"]) == (3000, 30921, 47286)
            and result["distinct_gold_accounts"] == 3000,
            "Completed admission or approved scope cardinalities differ")
    require(result.get("sealed_reference_values_decoded") == 0
            and result.get("sealed_input_values_decoded") == 0,
            "Completed result sealed boundary differs")
    with np.load(checked(rewards_ref), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in (
            "rewards", "policy_ids", "task_labels", "subtask_labels",
            "source_frames", "frozen_policy_actions")}
    rewards = arrays["rewards"]
    require(rewards.shape == (4, 3, 3000) and np.isin(rewards, (0, 1)).all()
            and arrays["policy_ids"].tolist() == list(POLICIES),
            "Four-policy, three-repeat completed rewards are required")
    actions = arrays["frozen_policy_actions"]
    require(actions.shape == (4, 3000) and actions.dtype.kind in "iu"
            and np.all((actions >= 0) & (actions < 20)), "Frozen actions differ")
    for name in ("task_labels", "subtask_labels", "source_frames"):
        require(arrays[name].shape == (3000,) and arrays[name].dtype.kind == "U",
                "Frozen metadata axes differ")
    labels = arrays["task_labels"]
    require(np.sum(labels == "gsm8k") == np.sum(labels == "bbh") == 1500,
            "Approved task balance differs")
    require(len(set(arrays["subtask_labels"][labels == "bbh"])) == 27,
            "Approved BBH subtask support differs")
    for first in range(4):
        for second in range(first + 1, 4):
            shared = actions[first] == actions[second]
            require(np.array_equal(rewards[first][:, shared], rewards[second][:, shared]),
                    "Shared executed action has inconsistent rewards")
    return result, arrays


def paired_row(rewards, actions, labels, first, second, mask):
    difference = rewards[first][:, mask].astype(float) - rewards[second][:, mask]
    return dict(
        comparison=POLICIES[first] + " minus " + POLICIES[second],
        questions=int(mask.sum()), generation_events=3 * int(mask.sum()),
        gain_pp=100 * (macro(rewards[first], labels, mask) - macro(rewards[second], labels, mask)),
        changed_actions=int(np.sum(actions[first, mask] != actions[second, mask])),
        corrected_generation_events=int(np.sum(difference > 0)),
        damaged_generation_events=int(np.sum(difference < 0)),
        net_corrected_generation_events=int(difference.sum()),
        improved_questions=int(np.sum(difference.mean(axis=0) > 0)),
        worsened_questions=int(np.sum(difference.mean(axis=0) < 0)),
        unchanged_questions=int(np.sum(difference.mean(axis=0) == 0)),
    )


def build_tables(result, arrays):
    rewards = arrays["rewards"]
    labels, subtasks, frames, actions = (
        arrays["task_labels"], arrays["subtask_labels"], arrays["source_frames"],
        arrays["frozen_policy_actions"])
    all_questions = np.ones(3000, dtype=bool)
    quality, selection = [], []
    for index, policy in enumerate(POLICIES):
        stored = result["results"]["quality"][policy]
        row = dict(policy=policy, questions=3000, generation_events=9000,
                   gsm8k_pct=100 * rewards[index][:, labels == "gsm8k"].mean(),
                   bbh_pct=100 * rewards[index][:, labels == "bbh"].mean(),
                   task_macro_pct=100 * macro(rewards[index], labels, all_questions))
        subtask_mean = np.mean([rewards[index][:, (labels == "bbh") & (subtasks == name)].mean()
                               for name in sorted(set(subtasks[labels == "bbh"]))])
        row["bbh_subtask_macro_pct"] = 100 * float(subtask_mean)
        close(stored["task_macro_accuracy"], row["task_macro_pct"] / 100, policy + " macro")
        close(stored["bbh_subtask_macro_accuracy"], subtask_mean, policy + " subtask macro")
        for task in ("gsm8k", "bbh"):
            close(stored["task_accuracy"][task], row[task + "_pct"] / 100, policy + task)
        for generation, seed in enumerate(SEEDS):
            value = np.mean([rewards[index, generation, labels == task].mean()
                             for task in ("gsm8k", "bbh")])
            row["seed" + str(seed) + "_pct"] = 100 * float(value)
            close(stored["by_generation_seed"][str(seed)], value, policy + " seed")
        quality.append(row)
        counts = np.bincount(actions[index], minlength=20)
        shares = counts / 3000
        selection.append(dict(policy=policy, used_combinations=int(np.sum(counts > 0)),
                              most_selected_combinations="/".join(map(str, np.flatnonzero(counts == counts.max()))),
                              largest_share=float(shares.max()), hhi=float(np.square(shares).sum()),
                              effective_combinations=float(1 / np.square(shares).sum()),
                              counts=counts.tolist()))
    paired = [paired_row(rewards, actions, labels, first, second, all_questions)
              for first, second in PAIRS]
    primary = result["results"]["primary"]
    close(primary["estimate"], paired[0]["gain_pp"] / 100, "primary gain")
    for stored_key, row_key in (
        ("corrected_generation_outcomes", "corrected_generation_events"),
        ("damaged_generation_outcomes", "damaged_generation_events"),
        ("net_corrected_generation_outcomes", "net_corrected_generation_events"),
    ):
        require(primary[stored_key] == paired[0][row_key], "Primary correction count differs")
    groups = []
    masks = [("task", task, labels == task) for task in ("gsm8k", "bbh")]
    masks += [("source_frame", frame, frames == frame) for frame in sorted(set(frames))]
    masks += [("bbh_subtask", subtask, (labels == "bbh") & (subtasks == subtask))
              for subtask in sorted(set(subtasks[labels == "bbh"]))]
    for kind, name, mask in masks:
        row = dict(group_kind=kind, group=name, questions=int(mask.sum()),
                   generation_events=3 * int(mask.sum()))
        for index, policy in enumerate(POLICIES):
            row[policy + "_pct"] = 100 * macro(rewards[index], labels, mask)
        delta = paired_row(rewards, actions, labels, 2, 1, mask)
        row.update({key: delta[key] for key in (
            "gain_pp", "corrected_generation_events", "damaged_generation_events",
            "net_corrected_generation_events", "improved_questions", "worsened_questions")})
        groups.append(row)
    seeds = []
    for generation, seed in enumerate(SEEDS):
        row = dict(seed=seed, questions=3000)
        for item in quality:
            row[item["policy"] + "_pct"] = item["seed" + str(seed) + "_pct"]
        difference = rewards[2, generation].astype(int) - rewards[1, generation]
        row.update(primary_gain_pp=row["TSPO_uniform_pct"] - row["SPO_coordinate_pct"],
                   corrected=int(np.sum(difference > 0)), damaged=int(np.sum(difference < 0)),
                   net_corrected=int(difference.sum()))
        seeds.append(row)
    costs = []
    for stage in ("member_generation", "aggregation"):
        row = result["physical_cost_totals"][stage]
        costs.append(dict(accounting="known_physical_union", policy="joint_evaluation", stage=stage,
                          calls=row["physical_attempts"],
                          member_calls=row["physical_attempts"] if stage == "member_generation" else 0,
                          aggregation_calls=row["physical_attempts"] if stage == "aggregation" else 0,
                          prompt_tokens=row["prompt_tokens"],
                          completion_tokens=row["completion_tokens"],
                          total_tokens=row["prompt_tokens"] + row["completion_tokens"]))
    for policy in POLICIES:
        row = result["selected_policy_service_costs"][policy]
        require(row["member_calls"] == 27000 and row["aggregation_calls"] == 9000,
                "Per-policy logical service counts differ")
        costs.append(dict(accounting="selected_policy_logical_service", policy=policy,
                          stage="members_and_aggregation", calls=row["member_calls"] + row["aggregation_calls"],
                          member_calls=row["member_calls"], aggregation_calls=row["aggregation_calls"],
                          prompt_tokens=row["measured_prompt_tokens"],
                          completion_tokens=row["measured_completion_tokens"],
                          total_tokens=row["measured_prompt_tokens"] + row["measured_completion_tokens"]))
    return dict(quality=quality, paired=paired, selection=selection, groups=groups,
                generation_seeds=seeds, costs=costs)


def render(result, tables, provenance):
    main = tables["paired"][0]
    direction = "提高" if main["gain_pp"] > 0 else "降低" if main["gain_pp"] < 0 else "持平"
    lines = ["# E18 冻结路由器独立评价结果", "", "## 主结果", "",
             f"四套策略在3,000道独立原题、每题三个生成重复上完成评价。TSPO_uniform相对SPO_coordinate的任务宏平均{direction}，配对点差为{main['gain_pp']:+.3f}个百分点。",
             "", f"9,000条已记录生成事件中，TSPO纠错{main['corrected_generation_events']}次、破坏{main['damaged_generation_events']}次，净纠错{main['net_corrected_generation_events']:+d}次。按原题三重复均值计，改善{main['improved_questions']}题、下降{main['worsened_questions']}题、持平{main['unchanged_questions']}题。",
             "", "| 策略 | GSM8K | BBH | 任务宏平均 | BBH子任务宏平均 | seed42 | seed137 | seed2024 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["quality"]:
        fields = ("gsm8k_pct", "bbh_pct", "task_macro_pct", "bbh_subtask_macro_pct",
                  "seed42_pct", "seed137_pct", "seed2024_pct")
        lines.append("| " + row["policy"] + " | " + " | ".join(f"{row[key]:.3f}%" for key in fields) + " |")
    lines += ["", "GSM8K与BBH各占50%，BBH内部按题平均。27个BBH子任务的宏平均单列。三次生成保留在同一原题簇内。本报告使用点值与纠错统计。", "", "## 配对结果", "",
              "| 比较 | 点差pp | 换队题数 | 纠错事件 | 破坏事件 | 净纠错 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in tables["paired"]:
        lines.append(f"| {row['comparison']} | {row['gain_pp']:+.3f} | {row['changed_actions']} | {row['corrected_generation_events']} | {row['damaged_generation_events']} | {row['net_corrected_generation_events']:+d} |")
    lines += ["", "## 选队分布", "", "| 策略 | 使用组合数 | 最高频组合 | 最高频占比 | HHI | 有效组合数 |",
              "|---|---:|---|---:|---:|---:|"]
    for row in tables["selection"]:
        lines.append(f"| {row['policy']} | {row['used_combinations']} | {row['most_selected_combinations']} | {100*row['largest_share']:.2f}% | {row['hhi']:.6f} | {row['effective_combinations']:.3f} |")
    lines += ["", "各策略组合0至19的原题频数如下。", ""]
    for row in tables["selection"]:
        lines.append(f"- {row['policy']}为`{', '.join(map(str, row['counts']))}`。")
    lines += ["", "## 任务与来源分组", "", "| 分组 | 原题数 | Ridge | SPO | TSPO | GlobalFixed | 主点差pp | 纠错 | 破坏 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["groups"]:
        lines.append(f"| {row['group']} | {row['questions']} | " + " | ".join(
            f"{row[name+'_pct']:.3f}%" for name in POLICIES) +
            f" | {row['gain_pp']:+.3f} | {row['corrected_generation_events']} | {row['damaged_generation_events']} |")
    lines += ["", "## 实际执行费用", "", "| 口径 | 策略 | 阶段 | 成员调用 | 聚合调用 | 输入Token | 输出Token | 总Token |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in tables["costs"]:
        lines.append(f"| {row['accounting']} | {row['policy']} | {row['stage']} | {row['member_calls']} | {row['aggregation_calls']} | {row['prompt_tokens']} | {row['completion_tokens']} | {row['total_tokens']} |")
    lines += ["", f"联合评价实际并集包含{result['original_source_cached_member_records']:,}份历史所需成员成功缓存。每策略服务投影按其所选成员及聚合输出计量，多个策略的逻辑投影存在共享。Gold按原题去重，共{result['distinct_gold_accounts']:,}份。历史失败全生命周期、独占GPU核时及货币费用保留各自计量边界。",
              "", "## 机制解释与后续判断", "",
              "本轮测量冻结策略在新题上的质量和选择分布。E17的语义保持重编号实验提供编号偏好机制证据，两阶段结果分别支持各自的研究问题。固定编号下的准确率点差需要结合任务、生成重复和组合集中度解释。",
              "", "优势包括同一冻结模型池、聚合器与评分器，以及四策略共享原题和生成重复的配对设计。风险包括BBH官方扩展题来源、有限生成重复以及固定六模型和单聚合器的外推范围。当前点值描述本次冻结评价，未来分布上的重复性仍需另行验证。",
              "", "## 数据与访问边界", "",
              "评分完成记录中的sealed答案解码和sealed输入解码均为0。报告器只读取完成结果与私有数值数组，输出聚合统计和来源SHA。原题、答案、逐题数组及权重继续保存在私有工件。", "", "## 来源", ""]
    for name, row in provenance.items():
        lines.append(f"- {name} SHA256为`{row['sha256']}`。")
    return "\n".join(lines) + "\n"


def generate(results_ref, rewards_ref, output_root):
    result, arrays = load_completed(results_ref, rewards_ref)
    tables = build_tables(result, arrays)
    provenance = dict(completed_results=results_ref, private_numeric_policy_arrays=rewards_ref,
                      report_source=dict(path=__file__, sha256=file_sha(__file__)))
    public = dict(schema="e18_point_only_report_summary_v1", status="COMPLETE",
                  original_questions=3000, generation_events_per_policy=9000,
                  primary_comparison="TSPO_uniform minus SPO_coordinate", **tables,
                  source_sha256s={name: row["sha256"] for name, row in provenance.items()})
    artifacts = {
        "E18_CONFIRMATION_REPORT.md": render(result, tables, provenance),
        "E18_POINT_RESULTS.json": json.dumps(public, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        "MAIN_TABLE.csv": csv_text(tables["quality"]),
        "PAIRED_TABLE.csv": csv_text(tables["paired"]),
        "GROUP_TABLE.csv": csv_text(tables["groups"]),
        "SEED_TABLE.csv": csv_text(tables["generation_seeds"]),
        "COST_TABLE.csv": csv_text(tables["costs"]),
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    for name, text in artifacts.items():
        with (output / name).open("x", encoding="utf-8-sig" if name.endswith(".csv") else "utf-8", newline="") as stream:
            stream.write(text)
    return dict(status="POINT_REPORT_COMPLETE", output_root=str(output.resolve()),
                artifacts=list(artifacts), original_questions=3000,
                new_gold_reads=0, new_scorer_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "rewards"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(dict(path=args.results, sha256=args.results_sha256),
                              dict(path=args.rewards, sha256=args.rewards_sha256),
                              args.output_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
