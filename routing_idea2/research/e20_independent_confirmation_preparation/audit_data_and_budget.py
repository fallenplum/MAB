"""E20 input-only permissions, provenance and exact support-union budget audit.

The mixed source reader projects only explicitly allowed input properties.
Reference, correctness and reward values are syntax-skipped and never decoded.
No scorer, model trainer, inference backend or queue launcher is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from itertools import combinations

import numpy as np

CODE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("MAB_WORKSPACE_ROOT", str(CODE.parents[1])))
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
STUDY = ASSETS / "research/e20_independent_confirmation_preparation"
PRIVATE = STUDY / "private"
OLD = ASSETS / "research/e18_mechanism_and_confirmation_preparation"
EXECUTION = ASSETS / "research/e18_confirmatory_execution"
E18CODE = WORKSPACE / "experiments/e18_mechanism_and_confirmation_preparation"
if not E18CODE.is_dir():
    E18CODE = CODE.parent / "e18_mechanism_and_confirmation_preparation"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) == value:
            return
        raise FileExistsError("Immutable audit output already exists with different content")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_projection():
    path = E18CODE / "audit_sources.py"
    spec = importlib.util.spec_from_file_location("e18_input_projection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sealed_inputs():
    """Prepare only the 200 known sealed inputs, without decoding answers."""
    started = time.perf_counter()
    projection = load_projection()
    registered = read_json(OLD / "private/REGISTERED_EXCLUSION_METADATA.json")["records"]
    wanted = {row["question_id"]: row for row in registered if row["split"] == "sealed_test"}
    if len(wanted) != 200:
        raise ValueError("Expected exactly 200 externally registered sealed identities")
    found = {}
    counts = Counter()
    sources = [ASSETS / "data/derived/available_records.jsonl",
               ASSETS / "data/derived/gsm8k_questions.jsonl"]
    for path in sources:
        with path.open("rb") as stream:
            for raw in stream:
                header = projection.project(raw, {"question_id"})
                question_id = header.get("question_id")
                if question_id not in wanted or question_id in found:
                    continue
                row = projection.project(raw, projection.INPUT_FIELDS)
                if set(row) != projection.INPUT_FIELDS:
                    raise ValueError("Input-only projection schema mismatch")
                expected = wanted[question_id]
                canonical = projection.canonical_hash(row["question"])
                if canonical != expected["canonical_question_sha256"]:
                    raise ValueError("Sealed source input does not match registered canonical identity")
                if row["task"] != expected["task"] or row["source_split"] != expected["source_split"]:
                    raise ValueError("Sealed source/task identity mismatch")
                row["canonical_question_sha256"] = canonical
                found[question_id] = row
                counts[row["task"]] += 1
    if set(found) != set(wanted) or counts != {"gsm8k": 100, "bbh": 100}:
        raise ValueError("Sealed input projection incomplete")
    target = PRIVATE / "inputs/SEALED200_INPUTS_ONLY.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for question_id in sorted(found):
            stream.write(json.dumps(found[question_id], ensure_ascii=False) + "\n")
    receipt = dict(status="INPUT_ONLY_PROJECTION_COMPLETE", questions=200, task_counts=dict(counts),
                   current_authorization="E20 preparation permits sealed input-side audit and budget; sealed answers remain prohibited",
                   source_projection=source_ref(E18CODE / "audit_sources.py"),
                   projected_fields=sorted(projection.INPUT_FIELDS),
                   source_files=[dict(path=str(path), bytes=path.stat().st_size) for path in sources],
                   reference_answer_values_decoded=0, reward_values_decoded=0,
                   protected_plaintext_printed=False, private_input_output=source_ref(target),
                   sealed_answer_paths_opened=0, new_llm_calls=0, wall_s=time.perf_counter()-started)
    write_json(PRIVATE / "inputs/SEALED200_INPUT_PROJECTION_RECEIPT.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"private_input_output", "source_files"}}, ensure_ascii=False))


def metadata_audit():
    projection = load_projection()
    registered_path = OLD / "private/REGISTERED_EXCLUSION_METADATA.json"
    registered = read_json(registered_path)["records"]
    fresh_path = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/fresh_dev_test_inputs.npz"
    train_path = fresh_path.with_name("selector_train_inputs.npz")
    with np.load(fresh_path, allow_pickle=False) as data:
        fresh_ids = data["question_ids"].tolist()
        fresh_hashes = data["canonical_hashes"].tolist()
        fresh_tasks = data["task_labels"].tolist()
    with np.load(train_path, allow_pickle=False) as data:
        train_ids = set(data["question_ids"].tolist())
        train_hashes = set(data["canonical_hashes"].tolist())
    sealed = [row for row in registered if row["split"] == "sealed_test"]
    sealed_ids = {row["question_id"] for row in sealed}
    sealed_hashes = {row["canonical_question_sha256"] for row in sealed}
    scope_path = EXECUTION / "private/EXECUTION_SCOPE.json"
    scope = read_json(scope_path)
    e18_ids = {row["question_id"] for row in scope["question_keys"]}
    e18_hashes = {row["canonical_question_sha256"] for row in scope["question_keys"]}
    if not set(fresh_ids) <= e18_ids or sealed_ids & e18_ids:
        raise ValueError("E18 identity scope inconsistent with protected cohorts")
    final_path = EXECUTION / "public/evaluable_confirmation_v1/E18_FINAL_RESULTS.json"
    access = projection.project(final_path.read_bytes(), {
        "schema", "status", "requested_questions", "valid_questions", "excluded_questions",
        "distinct_gold_accounts", "sealed_reference_values_decoded", "scored_aggregate_keys",
        "original_questions", "original_source_cached_member_records"})
    historical = read_json(OLD / "public/DATA_AUDIT.json")
    index = read_json(OLD / "private/protected_input_fingerprints/OFFICIAL_BBH_INPUT_IDENTITY_INDEX.json")["records"]
    duplicate = read_json(OLD / "private/INPUT_DUPLICATE_AUDIT_DETAILS.json")
    excluded_ids = {row["question_id"] for row in registered}
    excluded_hashes = {row["canonical_question_sha256"] for row in registered}
    duplicate_hashes = {row["canonical_question_sha256"] for row in duplicate["official_exact_duplicate_groups"]}
    near_ids = set()
    for edge in duplicate["unprotected_high_similarity_edges"]:
        near_ids.update((edge["first_question_id"], edge["second_question_id"]))
    for edge in duplicate["protected_signature_edges"]:
        near_ids.add(edge["candidate_question_id"])
    for group in duplicate["format_equivalence_clusters"]:
        near_ids.update(qid for qid in group["question_ids"] if qid != group.get("representative"))
    eligible_bbh = [row for row in index if row["question_id"] not in excluded_ids
                    and row["canonical_question_sha256"] not in excluded_hashes
                    and row["canonical_question_sha256"] not in duplicate_hashes
                    and row["question_id"] not in near_ids]
    # Prior E18 screening excluded both locations of an exact repeated source input.
    # Compare its original eligible count before subtracting newly consumed E18 inputs.
    if len(eligible_bbh) != historical["eligible_after_all_fixed_input_screens"]["bbh"]:
        raise ValueError("Cached BBH exclusion reconstruction does not recover E18 eligible count")
    remaining_bbh = [row for row in eligible_bbh if row["question_id"] not in e18_ids
                     and row["canonical_question_sha256"] not in e18_hashes]
    old_ids = excluded_ids
    extra_gsm = sum(row["task"] == "gsm8k" and row["question_id"] not in old_ids for row in scope["question_keys"])
    gsm_remaining = historical["eligible_after_all_fixed_input_screens"]["gsm8k"] - extra_gsm
    public = dict(
        status="INPUT_IDENTITY_AND_HISTORICAL_ACCESS_AUDITED", new_reward_values_decoded=0,
        new_sealed_answer_values_decoded=0, new_llm_calls=0,
        train600=dict(questions=600, task_counts={"gsm8k": 300, "bbh": 300}),
        fresh400=dict(questions=400, task_counts=dict(Counter(fresh_tasks)),
                      source_frame="GSM8K official train and BBH official test",
                      distinct_ids=len(set(fresh_ids)), distinct_canonical_hashes=len(set(fresh_hashes)),
                      train_id_overlap=len(set(fresh_ids) & train_ids),
                      train_canonical_overlap=len(set(fresh_hashes) & train_hashes),
                      sealed_id_overlap=len(set(fresh_ids) & sealed_ids),
                      e18_executed_identity_overlap=len(set(fresh_ids) & e18_ids),
                      historical_reward_access="All400 belong to E18 completed scored3000; the cohort has already been reward-exposed",
                      e20_role="EXPOSED_DEVELOPMENT_COHORT_NOT_INDEPENDENT_CONFIRMATION"),
        sealed200=dict(questions=200, task_counts=dict(Counter(row["task"] for row in sealed)),
                       source_frame="GSM8K official test and BBH official test",
                       train_id_overlap=len(sealed_ids & train_ids),
                       train_canonical_overlap=len(sealed_hashes & train_hashes),
                       e18_executed_identity_overlap=len(sealed_ids & e18_ids),
                       historical_answer_access_from_e18=access["sealed_reference_values_decoded"],
                       historical_input_access="E18 authorized200-input fingerprint pass; E20 input-only canonical-verified projection and frozen encoder forwarding",
                       answer_status="SEALED", e20_role="RESERVED_CONFIRMATORY_POOL_WITH_LOW_POWER_AND_GSM_SOURCE_SHIFT"),
        e18_completion_metadata=access,
        prior_dedup=dict(official_exact_duplicate_groups=historical["official_input_duplicate_groups"],
                         high_similarity_review_edges=historical["high_similarity_review_edges"],
                         protected_signature_screen=historical["protected_signature_screen"],
                         near_duplicate_limit=historical["near_duplicate_limitation"],
                         repeated_full_text_rescan=False),
        additional_pool_plan=dict(status="INVENTORY_ONLY_NOT_SELECTED_OR_ADMITTED",
                                  remaining_gsm8k_train=gsm_remaining, remaining_bbh_test=len(remaining_bbh),
                                  remaining_balanced_max=2*min(gsm_remaining, len(remaining_bbh)),
                                  bbh_subtasks=len({row["subtask"] for row in remaining_bbh}),
                                  bbh_subtask_counts=dict(sorted(Counter(row["subtask"] for row in remaining_bbh).items())),
                                  original_bbh_common_subpool_remaining=historical["safe_old_bbh_remaining_after_screens"]-46,
                                  qualification="Cached full-official-source input screens reused; new question selection requires approval and source-frame prespecification",
                                  pretraining_exposure_limit=historical["pretraining_exposure_warning"]),
        evidence_refs=[source_ref(path) for path in (registered_path, scope_path, final_path,
                       OLD / "public/DATA_AUDIT.json", OLD / "private/INPUT_DUPLICATE_AUDIT_DETAILS.json")])
    write_json(CODE / "E20_DATA_PERMISSIONS.json", public)
    write_json(PRIVATE / "inputs/E20_REMAINING_BBH_IDENTITY_INDEX.json", dict(records=remaining_bbh))
    print(json.dumps({"status":public["status"], "fresh_exposed":400,"sealed_answers_read":0,
                      "remaining_gsm":gsm_remaining,"remaining_bbh":len(remaining_bbh)}, ensure_ascii=False))


def cohort_near_audit():
    started = time.perf_counter()
    projection = load_projection()
    registered = read_json(OLD / "private/REGISTERED_EXCLUSION_METADATA.json")["records"]
    metadata = {row["question_id"]: row for row in registered}
    rows = {}
    admitted_path = EXECUTION / "private/ADMITTED3000_INPUTS_ONLY.jsonl"
    for raw in admitted_path.read_bytes().splitlines():
        row = projection.project(raw, projection.INPUT_FIELDS | {"canonical_question_sha256"})
        if projection.canonical_hash(row["question"]) != row["canonical_question_sha256"]:
            raise ValueError("Historical admitted input identity mismatch")
        rows[row["question_id"]] = row
    for source in (ASSETS / "data/derived/available_records.jsonl", ASSETS / "data/derived/gsm8k_questions.jsonl"):
        with source.open("rb") as stream:
            for raw in stream:
                question_id = projection.project(raw, {"question_id"}).get("question_id")
                if question_id not in metadata or question_id in rows:
                    continue
                row = projection.project(raw, projection.INPUT_FIELDS)
                row["canonical_question_sha256"] = projection.canonical_hash(row["question"])
                if row["canonical_question_sha256"] != metadata[question_id]["canonical_question_sha256"]:
                    raise ValueError("Registered input canonical mismatch")
                rows[question_id] = row
    if not set(metadata) <= set(rows):
        raise ValueError("Historical input registry has missing source projections")
    row_list = list(rows.values())
    groups, pending, edges = projection._screen_near_duplicates(row_list)
    role = {}
    for row in registered:
        role[row["question_id"]] = "sealed200" if row["split"] == "sealed_test" else "fresh400" if row["split"] == "fresh_dev400" else "historical_development"
    role.update({question_id: "e18_added2600" for question_id in rows if question_id not in role})
    cross = [edge for edge in edges if role[edge["first_question_id"]] != role[edge["second_question_id"]]]
    protected = [edge for edge in cross if "sealed200" in {role[edge["first_question_id"]],role[edge["second_question_id"]]}]
    relevant = [edge for edge in cross if {role[edge["first_question_id"]],role[edge["second_question_id"]]} & {"fresh400","sealed200"}]
    pairs = Counter("--".join(sorted((role[edge["first_question_id"]],role[edge["second_question_id"]]))) for edge in relevant)
    public = dict(status="INPUT_ONLY_CROSS_COHORT_NEAR_SCREEN_COMPLETE", original_input_identities=len(rows),
                  roles=dict(Counter(role.values())), exact_rule=projection.NEAR_RULES["exact"],
                  critical_slot_gated_rules=projection.NEAR_RULES,
                  cross_role_high_similarity_edges_involving_fresh_or_sealed=len(relevant),
                  by_role_pair=dict(sorted(pairs.items())), sealed_cross_cohort_high_similarity_edges=len(protected),
                  format_equivalence_groups=len(groups),
                  limitation="Frozen high-threshold syntactic screen detects input overlap; semantic independence and pretraining exposure remain unverified",
                  reward_values_decoded=0, answer_values_decoded=0, private_input_text_printed=False,
                  new_llm_calls=0, wall_s=time.perf_counter()-started,
                  action="Any flagged independent-pool overlap requires exclusion or explicit protocol disposition before evaluation")
    write_json(CODE / "E20_COHORT_NEAR_DUPLICATE_AUDIT.json", public)
    write_json(PRIVATE / "inputs/COHORT_NEAR_DUPLICATE_DETAILS.json", dict(role=role, relevant_cross_edges=relevant, format_groups=[[row_list[index]["question_id"] for index in group] for group in groups]))
    print(json.dumps(public, ensure_ascii=False))


def budget():
    seeds = (42, 137, 2024)
    teams = tuple(combinations(range(6), 3))
    scope_path = EXECUTION / "private/EXECUTION_SCOPE.json"
    scope = read_json(scope_path)
    expected_operator = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
    if scope["operator_fingerprint"] != expected_operator or tuple(scope["generation_seeds"]) != seeds or scope["aggregation_seed"] != 42:
        raise ValueError("Historical cache operator or generation identity mismatch")
    cached_identity = {row["question_id"]: row["canonical_question_sha256"] for row in scope["question_keys"]}
    closed_path = WORKSPACE / "experiments/e18_confirmatory_execution/CLOSED_NATIVE_INDEPENDENT_AUDIT_FINAL.json"
    closed = read_json(closed_path)
    ledger_path = WORKSPACE / ".e18_cpu_runtime/evaluable_bulk_v1/PHYSICAL_COST_LEDGER.json"
    ledger = read_json(ledger_path)
    cache = {"member": {}, "aggregate": {}}
    for row in ledger["rows"]:
        if row["status"].casefold() != "success":
            continue
        stage = "member" if row["stage"] == "member_generation" else "aggregate" if row["stage"] == "aggregation" else None
        if stage is None:
            raise ValueError("Unexpected native cost stage")
        key = tuple(row["task_key"])
        if key in cache[stage] and cache[stage][key] != row:
            raise ValueError("Multiple physical successful records for one cache key")
        cache[stage][key] = row
    if set(cache["member"]) != {tuple(key) for key in scope["required_member_keys"]} or set(cache["aggregate"]) != {tuple(key) for key in scope["required_aggregate_keys"]}:
        raise ValueError("Verified cost keys do not equal frozen E18 native execution union")
    sys.path.insert(0, str(E18CODE))
    spec = importlib.util.spec_from_file_location("e18_frozen_operator_budget", E18CODE / "frozen_input_budget.py")
    frozen_operator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen_operator)
    operator = frozen_operator.operator_sources()
    if operator["aggregator_fingerprint"] != expected_operator:
        raise ValueError("Frozen operator source qualification mismatch")
    model_caps = operator["member_caps"]
    aggregate_cap = operator["aggregate_cap"]
    output = dict(schema="e20_exact_uniform_support_union_budget_v1", status="PREPARATION_ONLY_NO_EXECUTION",
                  generation_seeds=list(seeds), aggregator_invocation_seed=42,
                  operator_fingerprint=expected_operator, cohorts={},
                  identity_qualification=dict(scope_ref=source_ref(scope_path), closed_native_audit_ref=source_ref(closed_path),
                                              metadata_only_cost_ledger_ref=source_ref(ledger_path),
                                              historical_native_tasks=closed["native_tasks_checked"],
                                              current_all_archive_rehash_performed=False,
                                              qualification="Reuse E18 closed native identity audit, exact scope/operator/seed/input keys and physical-cost records; no correctness payload accessed"),
                  token_caps=dict(member_input_plus_completion=model_caps, aggregation_input_plus_completion=aggregate_cap,
                                  source_qualification=operator,
                                  interpretation="Conditional per-success execution envelope from byte-frozen max_input_tokens plus max_new_tokens; retries and changed execution operators need a new budget"),
                  llm_calls_launched=0, fresh_rewards_read=0, sealed_answers_read=0)
    for name, file_prefix, expected_n in (("fresh400", "FRESH400", 400), ("sealed200", "SEALED200", 200)):
        path = PRIVATE / f"final_models/{file_prefix}_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz"
        with np.load(path, allow_pickle=False) as data:
            probabilities = data["probabilities"].copy()
            raw_probabilities = data["raw_exact_probabilities"].copy()
            ids = data["question_ids"].tolist()
            tasks = data["task_labels"].tolist()
            hashes = data["canonical_hashes"].tolist()
        if probabilities.shape != (2, expected_n, 20) or len(set(ids)) != expected_n:
            raise ValueError("Frozen input-only prediction schema mismatch")
        if not np.allclose(probabilities.sum(axis=2), 1, atol=1e-14):
            raise ValueError("Primary policy probabilities are not normalized")
        for question_id, canonical_hash in zip(ids, hashes):
            if question_id in cached_identity and canonical_hash != cached_identity[question_id]:
                raise ValueError("Cached question ID matches but canonical input content differs")
        primary_support = np.any(probabilities > 0, axis=0)
        raw_support = np.any(raw_probabilities > 0, axis=0)
        complete_support = primary_support | raw_support
        members, aggregates = set(), set()
        for row, question_id in enumerate(ids):
            for action in np.flatnonzero(complete_support[row]):
                for seed in seeds:
                    aggregates.add((question_id, seed, int(action)))
                    members.update((question_id, seed, member) for member in teams[int(action)])
        existing_members = members & set(cache["member"])
        existing_aggregates = aggregates & set(cache["aggregate"])
        missing_members = members - existing_members
        missing_aggregates = aggregates - existing_aggregates
        available_tokens = {}
        for stage, selected in (("member", existing_members), ("aggregate", existing_aggregates)):
            available_tokens[stage] = dict(prompt_tokens=sum(cache[stage][key]["prompt_tokens"] for key in selected),
                                           completion_tokens=sum(cache[stage][key]["completion_tokens"] for key in selected))
            available_tokens[stage]["total_tokens"] = sum(available_tokens[stage].values())
        by_task = {}
        for task in ("gsm8k", "bbh"):
            task_ids = {question_id for question_id, task_label in zip(ids, tasks) if task_label == task}
            by_task[task] = dict(questions=len(task_ids),
                                 member_calls=sum(key[0] in task_ids for key in members),
                                 aggregate_calls=sum(key[0] in task_ids for key in aggregates),
                                 new_member_calls=sum(key[0] in task_ids for key in missing_members),
                                 new_aggregate_calls=sum(key[0] in task_ids for key in missing_aggregates))
        cohort = dict(questions=expected_n, primary_support_team_count=int(primary_support.sum()),
                      raw_exact_support_team_count=int(raw_support.sum()), raw_extra_team_count=int((raw_support & ~primary_support).sum()),
                      combined_support_team_count=int(complete_support.sum()),
                      per_question_union_support_histogram=dict(sorted(Counter(complete_support.sum(axis=1).tolist()).items())),
                      per_policy_support=dict(B=dict(mean=float((probabilities[0]>0).sum(axis=1).mean()),
                                                     minimum=int((probabilities[0]>0).sum(axis=1).min()),maximum=int((probabilities[0]>0).sum(axis=1).max())),
                                              C=dict(mean=float((probabilities[1]>0).sum(axis=1).mean()),
                                                     minimum=int((probabilities[1]>0).sum(axis=1).min()),maximum=int((probabilities[1]>0).sum(axis=1).max()))),
                      total_required_member_calls=len(members), total_required_aggregate_calls=len(aggregates),
                      verified_reused_member_calls=len(existing_members), verified_reused_aggregate_calls=len(existing_aggregates),
                      new_member_calls=len(missing_members), new_aggregate_calls=len(missing_aggregates),
                      actual_cached_tokens=available_tokens,
                      new_actual_tokens=None,
                      conservative_new_token_cap=sum(model_caps[key[2]] for key in missing_members)+aggregate_cap*len(missing_aggregates),
                      conservative_from_zero_token_cap=sum(model_caps[key[2]] for key in members)+aggregate_cap*len(aggregates),
                      by_task=by_task, input_prediction_ref=source_ref(path),
                      exact_expectation="Execute each positive-probability B/C support action for each generation seed once; raw-exact adds only actions outside that support",
                      primary_and_sensitivity_share_outputs=True, reward_or_answer_values_read=0)
        output["cohorts"][name] = cohort
        private_payload = dict(question_ids=ids, canonical_hashes=hashes,
                               required_member_keys=[list(key) for key in sorted(members)],
                               required_aggregate_keys=[list(key) for key in sorted(aggregates)],
                               missing_member_keys=[list(key) for key in sorted(missing_members)],
                               missing_aggregate_keys=[list(key) for key in sorted(missing_aggregates)])
        write_json(PRIVATE / f"budget/{file_prefix}_EXACT_CALL_UNION.json", private_payload)
    write_json(CODE / "E20_EXACT_EXECUTION_BUDGET_QUALIFIED.json", output)
    print(json.dumps({name:{key:cohort[key] for key in ("total_required_member_calls","total_required_aggregate_calls","verified_reused_member_calls","verified_reused_aggregate_calls","new_member_calls","new_aggregate_calls","conservative_new_token_cap")} for name,cohort in output["cohorts"].items()}, ensure_ascii=False))


def optional_scenario():
    details = read_json(PRIVATE / "inputs/COHORT_NEAR_DUPLICATE_DETAILS.json")
    flagged = set()
    for edge in details["relevant_cross_edges"]:
        for key in (edge["first_question_id"], edge["second_question_id"]):
            if details["role"][key] == "sealed200":
                flagged.add(key)
    if len(flagged) != 1:
        raise ValueError("Expected exactly one flagged protected input")
    complete = read_json(PRIVATE / "budget/SEALED200_EXACT_CALL_UNION.json")
    original_budget = read_json(CODE / "E20_EXACT_EXECUTION_BUDGET_QUALIFIED.json")
    members = [key for key in complete["required_member_keys"] if key[0] not in flagged]
    aggregates = [key for key in complete["required_aggregate_keys"] if key[0] not in flagged]
    if original_budget["cohorts"]["sealed200"]["verified_reused_member_calls"] or original_budget["cohorts"]["sealed200"]["verified_reused_aggregate_calls"]:
        raise ValueError("Scenario cache assumption changed")
    with np.load(PRIVATE / "final_models/SEALED200_INPUT_ONLY_PREDICTIONS_ORIGINAL_OPERATOR_FROZEN.npz", allow_pickle=False) as archive:
        ids = archive["question_ids"].tolist()
        tasks = archive["task_labels"].tolist()
    task_of = dict(zip(ids, tasks))
    by_task = {}
    for task in ("gsm8k", "bbh"):
        selected = {key for key in ids if key not in flagged and task_of[key] == task}
        by_task[task] = dict(questions=len(selected), member_calls=sum(key[0] in selected for key in members),
                             aggregate_calls=sum(key[0] in selected for key in aggregates))
    public = dict(status="OPTIONAL_PLANNING_SCENARIO_REQUIRES_APPROVAL", original_cohort_unchanged=True,
                  exclusion_count=1, original_questions=200, hypothetical_questions=199,
                  hypothetical_task_counts={"gsm8k":100,"bbh":99},
                  main_weighting="GSM8K and BBH each50percent after any approved exclusion",
                  required_member_calls=len(members), required_aggregate_calls=len(aggregates),
                  verified_reused_member_calls=0, verified_reused_aggregate_calls=0,
                  new_member_calls=len(members), new_aggregate_calls=len(aggregates),
                  conservative_new_token_cap=6144*len(members)+8192*len(aggregates),
                  by_task=by_task, removed_member_calls=3504-len(members), removed_aggregate_calls=9387-len(aggregates),
                  sealed_answers_read=0, new_llm_calls=0, actual_selection_or_pool_mutation=False)
    write_json(CODE / "E20_OPTIONAL_199_QUESTION_BUDGET.json", public)
    projection = load_projection()
    case_path = WORKSPACE / ".e18_cpu_runtime/evaluable_bulk_v1/UNSCORABLE_CASES_PRIVATE.json"
    raw = case_path.read_text(encoding="utf-8-sig")
    cases = []
    position = projection._skip_space(raw, 0)
    if raw[position] != "[":
        raise ValueError("Unexpected unscorable metadata container")
    position += 1
    while True:
        position = projection._skip_space(raw, position)
        if raw[position] == "]":
            break
        end = projection.skip_value(raw, position)
        cases.append(projection.project(raw[position:end], {"question_id", "task", "subtask", "reason"}))
        position = projection._skip_space(raw, end)
        if raw[position] == ",":
            position += 1
    with np.load(ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/fresh_dev_test_inputs.npz", allow_pickle=False) as archive:
        fresh_ids = set(archive["question_ids"].tolist())
    unscorable = dict(status="HISTORICAL_METADATA_ONLY_CHECK_COMPLETE", e18_unscorable_questions=len(cases),
                     fresh400_unscorable_questions=sum(row["question_id"] in fresh_ids for row in cases),
                     affected_subtasks=dict(Counter(row.get("subtask") for row in cases)),
                     scores_or_references_decoded=0,
                     interpretation="Native cache success demonstrates output existence; score eligibility remains a separate check",
                     source_ref=source_ref(case_path))
    write_json(CODE / "E20_HISTORICAL_UNSCORABLE_METADATA_CHECK.json", unscorable)
    print(json.dumps(dict(optional199=public, historical_unscorable=unscorable),ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sealed-inputs", "audit", "near", "budget", "optional-scenario"), required=True)
    args = parser.parse_args()
    if args.phase == "sealed-inputs":
        sealed_inputs()
    elif args.phase == "audit":
        metadata_audit()
    elif args.phase == "near":
        cohort_near_audit()
    elif args.phase == "budget":
        budget()
    else:
        optional_scenario()
