"""Reconcile frozen logical calls against independently verified caches."""
from __future__ import annotations

from importlib import util
import json
import time

from e18_common import (ASSETS, E17, PRIVATE, PUBLIC, POLICIES, SEEDS,
    preserve_e17, read_json, sha256, write_json)
import numpy as np


def main():
    started = time.perf_counter()
    preserve_e17()
    postcheck = read_json(PUBLIC / "REMOTE_CACHE_POSTCHECK_v2.json")
    if postcheck["status"] != "PASS":
        raise ValueError("Remote member keys have not passed independent checks")
    aggregate_path = PRIVATE / "REMOTE_AGGREGATION_INVENTORY_v3.json"
    aggregation = read_json(aggregate_path)
    if not aggregation.get("authenticated") or "inventory" not in aggregation:
        raise ValueError("Remote aggregation inventory did not complete")
    for key in ("fresh_gold_or_reward_value_decodes", "sealed_paths_opened", "remote_commands_started", "remote_writes"):
        if aggregation["inventory"][key] != 0:
            raise ValueError("Aggregation cache audit violated its read-only scope")
    frozen_path = PRIVATE / "FROZEN3000_INPUT_PREDICTIONS.npz"
    with np.load(frozen_path, allow_pickle=False) as archive:
        actions = archive["actions"].copy()
        canonical = archive["canonical_hashes"].tolist()
        qids = archive["question_ids"].tolist()
    original_union = read_json(PRIVATE / "FROZEN_CALL_UNION_3000.json")
    positions = {key:index for index,key in enumerate(canonical)}
    member_ledger = read_json(PRIVATE / "RECONCILED_MEMBER_PHYSICAL_LEDGER_v2.json")["rows"]
    member_records = {}
    for row in member_ledger:
        if row["canonical_id"] not in positions:
            continue
        position = positions[row["canonical_id"]]
        if row["question_id"] != qids[position]:
            raise ValueError("Question-dependent sampling identity differs")
        key = (position, int(row["generation_seed"]), int(row["member_index"]))
        if key in member_records:
            raise ValueError("Duplicate member key in reconciled physical ledger")
        member_records[key] = row
    teams = np.array(original_union["teams"], dtype=np.int64)
    team_to_index = {tuple(team):index for index,team in enumerate(teams.tolist())}
    aggregate_records = {}
    for row in aggregation["inventory"]["reusable_aggregation_keys"]:
        if row["canonical_id"] not in positions:
            continue
        position = positions[row["canonical_id"]]
        members = tuple(sorted(row["members"]))
        key = (position, int(row["generation_seed"]), team_to_index[members])
        if key in aggregate_records:
            raise ValueError("Duplicate verified aggregate key")
        aggregate_records[key] = row
    spec = util.spec_from_file_location("e17_counter", E17 / "public/code/confirmation.py")
    counter = util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    rebuilt = counter.input_call_union(actions, teams, SEEDS)
    if json.dumps(rebuilt,sort_keys=True) != json.dumps(original_union,sort_keys=True):
        raise ValueError("Frozen logical execution union changed")
    remainder = counter.subtract_verified_cache_keys(rebuilt, list(member_records), list(aggregate_records))
    member_keys = set(map(tuple, remainder["verified_required_member_keys"]))
    aggregate_keys = set(map(tuple, remainder["verified_required_aggregate_keys"]))
    member_new = set(map(tuple, remainder["new_member_keys"]))
    aggregate_new = set(map(tuple, remainder["new_aggregate_keys"]))
    operator = read_json(PUBLIC / "FROZEN_MODEL_QUALIFICATION_v3.json")["active_operator"]
    member_caps = {int(key):value for key,value in operator["member_caps"].items()}
    cap = sum(member_caps[member] for _,_,member in member_new) + operator["aggregate_cap"]*len(aggregate_new)
    def spent(rows):
        return dict(prompt_tokens=sum(row["prompt_tokens"] for row in rows),
                    completion_tokens=sum(row["completion_tokens"] for row in rows),
                    total_tokens=sum(row["tokens"] for row in rows),
                    observed_source_model_wall_s=sum(row["source_model_wall_s"] for row in rows))
    member_spent = spent([member_records[key] for key in sorted(member_keys)])
    aggregate_spent = spent([aggregate_records[key] for key in sorted(aggregate_keys)]) if aggregate_keys else dict(
        prompt_tokens=0, completion_tokens=0, total_tokens=0, observed_source_model_wall_s=0)
    segments = []
    for name, begin, end in (("fresh400", 0, 400), ("proposed2600", 400, 3000)):
        segments.append(dict(cohort=name, questions=end-begin,
            required_member_keys=sum(begin<=q<end for q,_,_ in rebuilt["member_keys"]),
            required_aggregate_keys=sum(begin<=q<end for q,_,_ in rebuilt["aggregate_keys"]),
            verified_member_reuse=sum(begin<=q<end for q,_,_ in member_keys),
            verified_aggregate_reuse=sum(begin<=q<end for q,_,_ in aggregate_keys),
            member_calls_unfulfilled_by_verified_cache=sum(begin<=q<end for q,_,_ in member_new),
            aggregate_calls_unfulfilled_by_verified_cache=sum(begin<=q<end for q,_,_ in aggregate_new)))
    pair = counter.input_call_union(actions[[1,2]], teams, SEEDS)
    primary_remainder = counter.subtract_verified_cache_keys(pair, list(member_records), list(aggregate_records))
    materialization_path = PRIVATE / "MEMBER_CACHE_MATERIALIZATION_v4.json"
    materialization = read_json(materialization_path) if materialization_path.exists() else None
    materialization_summary = dict(status="REMOTE_KEYS_VERIFIED_MIRROR_PENDING", new_remote_keys=2130,
        condition="Transfer and byte validation precede local consumption, or the server consumes its own verified raw source")
    if materialization is not None:
        materialization_summary = dict(status=materialization["status"],
            receipt_sha256=sha256(materialization_path),
            independently_verified_remote_key_target=2130,
            full_private_file_and_key_map_published=False)
    write_json(PRIVATE / "FINAL_VERIFIED_EXECUTION_REMAINDER.json", dict(
        member_records_source_sha256=sha256(PRIVATE / "RECONCILED_MEMBER_PHYSICAL_LEDGER_v2.json"),
        aggregation_inventory_sha256=sha256(aggregate_path), remainder=remainder,
        primary_pair_remainder=primary_remainder))
    result = dict(schema="E18_final_scoped_cache_call_budget_v1", status="PREPARED_AWAITING_APPROVAL",
        policies=POLICIES, independent_question_target=3000, existing_fresh=400, proposed_new=2600,
        generation_seeds=SEEDS, frozen_logical_calls=rebuilt["counts"],
        verified_member_inventory_total=len(member_ledger), required_verified_member_reuse=len(member_keys),
        verified_aggregate_inventory_total=len(aggregate_records), required_verified_aggregate_reuse=len(aggregate_keys),
        not_satisfied_by_verified_inventory=dict(member_calls=len(member_new), aggregate_calls=len(aggregate_new)),
        complete_if_no_more_cache=dict(new_member_generation_calls=len(member_new), new_aggregate_calls=len(aggregate_new)),
        unlimited_global_cache_completeness_claim=False,
        cache_scope="Original local imports and pinned new-endpoint known MAB member/aggregation roots; uninspected listed boundaries remain unknown",
        verified_reused_member_actual_cost=member_spent, verified_reused_aggregate_actual_cost=aggregate_spent,
        all_verified_5330_member_actual_cost=spent(member_ledger),
        remaining_execution_actual_tokens=None, actual_new_physical_generation_calls=0,
        actual_new_scoring_ops=0, gold_reveals_to_authorize=3000, deterministic_score_operations=rebuilt["counts"]["scoring_ops"],
        conservative_token_cap_initial_one_attempt_per_missing_key=cap,
        original_automatic_paid_retries=0,
        total_cap_after_future_manual_resumes=None,
        retry_audit_sha256=sha256(PRIVATE / "ORIGINAL_RETRY_BOUNDS_AUDIT.json"),
        retry_scope="Original STOP_REQUIRES_REVIEW and cost fence apply; no automatic paid retry. Further paid attempts need separate accounting and authorization.",
        segments=segments,
        primary_pair_only=dict(policies=["SPO_coordinate", "TSPO_uniform"], logical_calls=pair["counts"],
            not_satisfied_by_verified_inventory=primary_remainder["counts"],
            role="Budget visibility only; the proposed evaluation still includes all four frozen policies"),
        cache_materialization=materialization_summary,
        inputs=dict(proposal_sha256=read_json(PUBLIC / "E18_EXACT_CALL_BUDGET.json")["candidate_sha256"],
                    corrected_feature_inputs_sha256=sha256(PRIVATE / "PROPOSED2600_INPUTS_ONLY_v2.npz"),
                    frozen_choices_sha256=sha256(frozen_path)),
        encoder_cost=dict(question_encodings=2600, batch_size=8, batch_forward_calls=325,
                          mode="frozen non-generative MiniLM CPU feature extraction"),
        learned_predictor_updates=0, new_llm_generation_calls=0,
        fresh_gold_or_reward_reads=0, sealed_answer_reads=0,
        protected_input_fingerprint_pass="200 input-only records processed once under explicit user exception; no display/persistence/embedding",
        source_sha256=sha256(__file__), wall_s=time.perf_counter()-started)
    write_json(PUBLIC / "E18_FINAL_CALL_BUDGET.json", result)
    print("E18_FINAL_CACHE_BUDGET_RECONCILED", len(member_new), len(aggregate_new), flush=True)


if __name__ == "__main__":
    main()
