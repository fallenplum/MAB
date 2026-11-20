"""Recompute the frozen E24 four-policy comparison from detached numerical evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from scipy.stats import t


POLICIES = ("B_U", "B_P", "C_U", "C_P")
SEEDS = (42, 137, 2024)
N_QUESTIONS = 1196
NUMERIC_COLUMNS = (
    "weight", "mean", "sample_variance", "sum", "sum_of_squares",
    *(f"{name}_sum" for name in POLICIES),
    *(f"interaction_{seed}_sum" for seed in SEEDS),
    "U_B_unique_sum", "P_B_unique_sum", "U_B_near_tied_sum", "P_B_near_tied_sum",
    "strict_corrected", "strict_damaged", "P_corrected", "P_damaged",
)
INTEGER_COLUMNS = ("n", "B_unique", "C_unique", "strict_switch")


def stratum(row):
    task, subtask = row["task"], row["subtask"]
    if task == "gsm8k" and not subtask:
        return task
    if task == "bbh" and subtask and subtask != "penguins_in_a_table":
        return f"bbh/{subtask}"
    raise ValueError("Unexpected evaluation stratum")


def interaction(row):
    b_u, b_p, c_u, c_p = (float(row[key]) for key in POLICIES)
    return (c_u - b_u) - (c_p - b_p)


def build_statistics(path):
    questions = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["generation_seed"])
            identity = row["row_id"]
            if not identity or seed not in SEEDS or seed in questions[identity]:
                raise ValueError("Each anonymous question and seed must appear exactly once")
            if not all(-1e-14 <= float(row[name]) <= 1 + 1e-14 for name in POLICIES):
                raise ValueError("Policy utilities must lie in [0, 1]")
            questions[identity][seed] = row
    if len(questions) != N_QUESTIONS:
        raise ValueError("Wrong number of independent E24 questions")

    strata = defaultdict(list)
    metadata = ("task", "subtask", "B_support_size", "C_support_size", "strict_switch", "B_margin", "C_margin")
    for records in questions.values():
        if set(records) != set(SEEDS):
            raise ValueError("Three generation seeds per question are required")
        rows = tuple(records[seed] for seed in SEEDS)
        if any(tuple(row[key] for key in metadata) != tuple(rows[0][key] for key in metadata) for row in rows):
            raise ValueError("Each question must preserve one stratum and routing metadata across all seeds")
        b, c, strict = (int(rows[0][key]) for key in ("B_support_size", "C_support_size", "strict_switch"))
        if not (1 <= b <= 20 and 1 <= c <= 20 and strict in (0, 1) and (not strict or b == c == 1)):
            raise ValueError("Invalid frozen support size or strict-switch flag")
        strata[stratum(rows[0])].append((rows, b == 1, c == 1, bool(strict)))

    result = []
    for group, items in sorted(strata.items(), key=lambda x: (x[0] != "gsm8k", x[0])):
        n = len(items)
        values = [math.fsum(map(interaction, rows)) / len(SEEDS) for rows, *_ in items]
        total = math.fsum(values)
        mean = total / n
        weight = 0.5 if group == "gsm8k" else 0.5 / 26
        entry = dict(stratum=group, n=n, weight=weight, mean=mean,
                     sample_variance=math.fsum((v - mean) ** 2 for v in values) / (n - 1),
                     sum=total, sum_of_squares=math.fsum(v * v for v in values))
        for key in POLICIES:
            entry[f"{key}_sum"] = math.fsum(float(row[key]) for rows, *_ in items for row in rows)
        for idx, seed in enumerate(SEEDS):
            entry[f"interaction_{seed}_sum"] = math.fsum(interaction(rows[idx]) for rows, *_ in items)
        for name, is_unique in (("B_unique", True), ("B_near_tied", False)):
            for dep in ("U", "P"):
                entry[f"{dep}_{name}_sum"] = math.fsum(
                    float(row[f"C_{dep}"]) - float(row[f"B_{dep}"])
                    for rows, b_unique, *_ in items if b_unique == is_unique for row in rows
                )
        entry.update(B_unique=sum(b for _, b, _, _ in items), C_unique=sum(c for _, _, c, _ in items),
                     strict_switch=sum(strict for _, _, _, strict in items))
        for metric in ("corrected", "damaged"):
            entry[f"P_{metric}"] = math.fsum(float(row[f"P_{metric}"]) for rows, *_ in items for row in rows)
            entry[f"strict_{metric}"] = math.fsum(
                float(row[f"U_{metric}"]) for rows, _, _, strict in items if strict for row in rows
            )
        result.append(entry)
    return result


def read_statistics(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {"stratum": row["stratum"], **{key: int(row[key]) for key in INTEGER_COLUMNS},
             **{key: float(row[key]) for key in NUMERIC_COLUMNS}}
            for row in csv.DictReader(stream)
        ]


def write_statistics(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stratum", *INTEGER_COLUMNS, *NUMERIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    if (len(rows) != 27 or len({row["stratum"] for row in rows}) != 27
            or sum(row["n"] for row in rows) != N_QUESTIONS):
        raise ValueError("Expected GSM8K and all 26 E24 BBH strata")
    for row in rows:
        group = row["stratum"]
        if group != "gsm8k" and (not group.startswith("bbh/") or group == "bbh/penguins_in_a_table"):
            raise ValueError("Unexpected E24 stratum")
        expected_n, expected_weight = (598, 0.5) if group == "gsm8k" else (23, 0.5 / 26)
        if row["n"] != expected_n or not math.isclose(row["weight"], expected_weight, rel_tol=0, abs_tol=1e-15):
            raise ValueError("Stratum size and weight must match the frozen test")
        if not all(math.isfinite(row[key]) for key in NUMERIC_COLUMNS):
            raise ValueError("Nonfinite sufficient statistic")
        if not math.isclose(row["mean"], row["sum"] / row["n"], rel_tol=0, abs_tol=1e-14):
            raise ValueError("Stratum first moment is inconsistent")
        variance = (row["sum_of_squares"] - row["sum"] ** 2 / row["n"]) / (row["n"] - 1)
        if row["sample_variance"] < 0 or not math.isclose(variance, row["sample_variance"], rel_tol=0, abs_tol=1e-14):
            raise ValueError("Stratum second moment is inconsistent")

    components = [(row["weight"] ** 2 * row["sample_variance"] / row["n"], row["n"] - 1) for row in rows]
    estimate = math.fsum(row["weight"] * row["mean"] for row in rows)
    variance = math.fsum(v for v, _ in components)
    if variance <= 0:
        raise ValueError("The frozen E24 test requires positive sampling variance")
    degrees = variance ** 2 / math.fsum(v ** 2 / df for v, df in components)
    statistic = estimate / math.sqrt(variance)
    accuracies = {
        name: 100 * math.fsum(row["weight"] * row[f"{name}_sum"] / (3 * row["n"]) for row in rows)
        for name in POLICIES
    }
    gaps = {"U": accuracies["C_U"] - accuracies["B_U"], "P": accuracies["C_P"] - accuracies["B_P"]}
    if not math.isclose(gaps["U"] - gaps["P"], 100 * estimate, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Policy mean differences disagree with the direct interaction moment")
    return dict(
        questions=N_QUESTIONS, question_seed_records=N_QUESTIONS * len(SEEDS),
        accuracies_pct=accuracies, training_gap_pp=gaps, interaction_pp=100 * estimate,
        variance_proportion_squared=variance, t_statistic=statistic, degrees_of_freedom=degrees,
        p_value_two_sided=float(2 * t.sf(abs(statistic), degrees)), alpha=0.05, strata=len(rows),
        interaction_by_task_pp={
            key: 200 * math.fsum(row["weight"] * row["mean"] for row in rows
                                if (row["stratum"] == "gsm8k") == (key == "gsm8k"))
            for key in ("gsm8k", "bbh")
        },
        interaction_by_seed_pp={
            str(seed): 100 * math.fsum(row["weight"] * row[f"interaction_{seed}_sum"] / row["n"] for row in rows)
            for seed in SEEDS
        },
        region_contributions_pp={
            dep: {name: 100 * math.fsum(row["weight"] * row[f"{dep}_{name}_sum"] / (3 * row["n"]) for row in rows)
                  for name in ("B_unique", "B_near_tied")} for dep in ("U", "P")
        },
        B_unique=sum(row["B_unique"] for row in rows),
        B_near_tied=N_QUESTIONS - sum(row["B_unique"] for row in rows),
        C_unique=sum(row["C_unique"] for row in rows), strict_switch=sum(row["strict_switch"] for row in rows),
        strict_corrected=math.fsum(row["strict_corrected"] for row in rows),
        strict_damaged=math.fsum(row["strict_damaged"] for row in rows),
        P_corrected=math.fsum(row["P_corrected"] for row in rows),
        P_damaged=math.fsum(row["P_damaged"] for row in rows),
        bootstrap_repetitions=0, confidence_intervals_computed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--utilities", type=Path)
    inputs.add_argument("--sufficient", type=Path)
    parser.add_argument("--write-sufficient", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.write_sufficient and args.utilities is None:
        parser.error("--write-sufficient requires --utilities")
    rows = build_statistics(args.utilities) if args.utilities else read_statistics(args.sufficient)
    result = summarize(rows)
    if args.write_sufficient:
        write_statistics(args.write_sufficient, rows)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
