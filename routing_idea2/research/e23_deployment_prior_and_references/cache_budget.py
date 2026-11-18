"""Count incremental execution keys for frozen policies against a verified cache."""
from __future__ import annotations

from itertools import combinations

import numpy as np


SEEDS = (42, 137, 2024)
TEAMS = tuple(combinations(range(6), 3))
MEMBER_TOKEN_CAP = 6144
AGGREGATE_TOKEN_CAP = 8192


def _cache_keys(values, maximum_index):
    normalized = set()
    for item in values:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError("Cache keys require question ID, generation seed and action/member index")
        question, generation, index = item
        if not isinstance(question, str) or not question:
            raise ValueError("Cache question IDs must be nonempty strings")
        if (not isinstance(generation, (int, np.integer)) or isinstance(generation, bool)
                or int(generation) not in SEEDS
                or not isinstance(index, (int, np.integer)) or isinstance(index, bool)
                or not 0 <= int(index) < maximum_index):
            raise ValueError("Cache generation seed or physical action/member index differs")
        normalized.add((question, int(generation), int(index)))
    return normalized


def _requirements(question_ids, probabilities):
    aggregates, members = set(), set()
    for question, row in zip(question_ids, probabilities):
        for action in np.flatnonzero(row > 0):
            for seed in SEEDS:
                aggregates.add((question, seed, int(action)))
                members.update((question, seed, member) for member in TEAMS[action])
    return members, aggregates


def _subtract(members, aggregates, existing_members, existing_aggregates):
    missing_aggregates = aggregates - existing_aggregates
    dependencies = {
        (question, seed, member)
        for question, seed, action in missing_aggregates
        for member in TEAMS[action]
    }
    if not dependencies <= members:
        raise AssertionError("Incremental dependencies escaped the frozen policy union")
    missing_members = dependencies - existing_members
    missing_questions = {key[0] for key in missing_members | missing_aggregates}
    public = dict(
        required_member_keys=len(members),
        required_aggregate_keys=len(aggregates),
        cached_required_aggregate_keys=len(aggregates & existing_aggregates),
        cached_required_member_keys=len(members & existing_members),
        member_dependencies_for_missing_aggregates=len(dependencies),
        cached_member_dependencies_for_missing_aggregates=len(dependencies & existing_members),
        missing_member_calls=len(missing_members),
        missing_aggregate_calls=len(missing_aggregates),
        missing_physical_calls=len(missing_members) + len(missing_aggregates),
        questions_with_missing_aggregates=len({key[0] for key in missing_aggregates}),
        questions_with_missing_member_calls=len({key[0] for key in missing_members}),
        questions_requiring_new_execution=len(missing_questions),
        all_policy_outputs_cached=not missing_aggregates,
        additional_deterministic_scoring_ops=len(missing_aggregates),
        conservative_incremental_token_cap=(
            MEMBER_TOKEN_CAP * len(missing_members)
            + AGGREGATE_TOKEN_CAP * len(missing_aggregates)
        ),
        full_policy_from_zero_token_cap=(
            MEMBER_TOKEN_CAP * len(members) + AGGREGATE_TOKEN_CAP * len(aggregates)
        ),
    )
    private = dict(
        missing_member_keys=sorted(missing_members),
        missing_aggregate_keys=sorted(missing_aggregates),
        missing_question_ids=sorted(missing_questions),
    )
    return public, private


def budget(question_ids, policies, existing_member_keys, existing_agg_keys):
    """Return a de-identified count ledger and private missing-key manifests.

    Positive policy probability defines required support. An existing aggregate
    needs no member regeneration, even if its component member files are absent.
    Cache sets must be supplied by a caller that has verified model, generation,
    prompt, input and aggregate-operator identities. This function has no reward
    input and does not authorize model calls.
    """
    question_ids = list(question_ids)
    if (not question_ids or any(not isinstance(value, str) or not value for value in question_ids)
            or len(set(question_ids)) != len(question_ids)):
        raise ValueError("Unique nonempty original question IDs are required")
    if not isinstance(policies, dict) or not policies:
        raise ValueError("At least one named frozen policy is required")
    members_cache = _cache_keys(existing_member_keys, 6)
    aggregates_cache = _cache_keys(existing_agg_keys, 20)
    per_policy, private_per_policy = {}, {}
    union_members, union_aggregates = set(), set()
    for name, values in policies.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Each policy needs a nonempty name")
        probabilities = np.asarray(values, dtype=np.float64)
        if (probabilities.shape != (len(question_ids), 20)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)):
            raise ValueError("Every frozen policy needs finite normalized question-by-20 probabilities")
        required_members, required_aggregates = _requirements(question_ids, probabilities)
        summary, missing = _subtract(
            required_members, required_aggregates, members_cache, aggregates_cache)
        summary["support_question_action_pairs"] = len(required_aggregates) // len(SEEDS)
        per_policy[name] = summary
        private_per_policy[name] = missing
        union_members.update(required_members)
        union_aggregates.update(required_aggregates)
    union, missing_union = _subtract(
        union_members, union_aggregates, members_cache, aggregates_cache)
    question_set = set(question_ids)
    public = dict(
        schema="mab_frozen_reference_cache_budget_v1",
        original_questions=len(question_ids),
        generation_seeds=list(SEEDS),
        number_of_policies=len(policies),
        per_policy=per_policy,
        union=union,
        ignored_cache_member_keys_outside_question_pool=sum(key[0] not in question_set for key in members_cache),
        ignored_cache_aggregate_keys_outside_question_pool=sum(key[0] not in question_set for key in aggregates_cache),
        member_token_cap=MEMBER_TOKEN_CAP,
        aggregate_token_cap=AGGREGATE_TOKEN_CAP,
        token_scope="Per-call input plus maximum generated output caps, not realized new Token usage",
        cache_identity_scope="Caller must verify frozen scientific identities before providing positive cache keys",
        cache_key_costs_charged_for_shared_members_once=True,
        scores_read=0,
        inference_calls_started=0,
        new_gold_acquisitions_inferred_from_output_cache=False,
    )
    private = dict(per_policy=private_per_policy, union=missing_union)
    return public, private
