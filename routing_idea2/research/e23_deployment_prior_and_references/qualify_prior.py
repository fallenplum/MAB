"""Synthetic support, tie, missing-value and cache accounting checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_references import train_prior, uniform, utility, write, binding
from cache_budget import budget


def qualify():
    scores = np.full((2, 20), -1.0)
    scores[0, :3] = 0
    scores[1, 3] = 1
    counts = np.arange(20)
    prior = train_prior(scores, counts)
    checks = dict(
        within_support_only=bool(np.all((prior > 0) <= (uniform(scores) > 0))),
        chooses_train_best_inside_support=bool(prior[0, 2] == 1),
        unique_action_unchanged=bool(prior[1, 3] == 1),
    )
    tied_counts = counts.copy()
    tied_counts[:3] = 7
    checks["remaining_prior_ties_uniform"] = bool(np.allclose(train_prior(scores, tied_counts)[0, :3], 1/3))
    permutation = np.random.default_rng(20261118023).permutation(20)
    mapped = np.empty_like(prior)
    mapped[:, permutation] = train_prior(scores[:, permutation], counts[permutation])
    checks["action_relabeling_equivariant"] = bool(np.array_equal(prior, mapped))
    values = np.full((3, 2, 20), np.nan)
    values[:, 0, 2], values[:, 1, 3] = 1, 0
    checks["unknown_unused_outputs_do_not_change_projection"] = bool(np.array_equal(utility(prior, values), [[1, 0]]*3))
    missing = prior.copy()
    missing[0, 2], missing[0, 1] = 0, 1
    try:
        utility(missing, values)
    except ValueError:
        checks["missing_positive_support_rejected"] = True
    else:
        checks["missing_positive_support_rejected"] = False
    first, second = np.zeros((1, 20)), np.zeros((1, 20))
    first[0, 0], second[0, 1] = 1, 1
    existing_member = {("test", seed, member) for seed in (42, 137, 2024) for member in (0, 1, 2)}
    existing_aggregate = {("test", seed, 0) for seed in (42, 137, 2024)}
    record, _ = budget(["test"], {"first": first, "second": second}, existing_member, existing_aggregate)
    checks["shared_two_members_reused"] = record["union"]["missing_member_calls"] == 3
    checks["one_additional_team_per_seed"] = record["union"]["missing_aggregate_calls"] == 3
    checks["incremental_token_cap"] = record["union"]["conservative_incremental_token_cap"] == 3*(6144+8192)
    if not all(checks.values()):
        raise AssertionError("Supplemental tie deployment qualification failed")
    return dict(schema="mab_prior_reference_qualification_v1", status="PASS", checks=checks,
                source_binding=binding(), real_evaluation_values_read=0, new_llm_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify()
    write(args.output, result)
    print(json.dumps(dict(status=result["status"], checks=len(result["checks"]))))
