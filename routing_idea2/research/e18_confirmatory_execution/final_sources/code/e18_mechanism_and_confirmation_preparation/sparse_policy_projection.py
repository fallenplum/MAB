"""Pure sparse-score join for frozen E18 policy actions.

This module has no data reader, gold accessor, job factory or paid runner.
Callers must verify referenced scoring/execution receipts before supplying rows.
The kernel checks keys and protocol fields, then projects executed scores only.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re

import numpy as np


OPERATOR_FP="9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
SCORER_VERSION="safe_exact_v2_0_terminal_final_heading_fix"
GENERATION_SEEDS=(42,137,2024)
TEAMS=tuple(combinations(range(6),3))
FIELDS={"question_id","canonical_id","generation_seed","aggregation_seed","combination_index","team",
        "aggregator_fingerprint","scorer_version","score","aggregation_record_sha256","scoring_receipt_sha256"}


def _hash(value):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def project_frozen_policy_scores(policy_ids, policy_actions, question_ids, canonical_hashes, executed_scores,
                                generation_seeds=GENERATION_SEEDS):
    """Return rewards[policy, generation, original_question] from sparse rows.

    Duplicate execution keys are rejected. Same-cohort unselected teams may be
    present and are counted separately. A missing selected score always raises.
    Syntactic SHA checks do not verify referenced bytes or grant reward access.
    """
    actions=np.asarray(policy_actions)
    ids=list(question_ids)
    canonical=list(canonical_hashes)
    policies=list(policy_ids)
    if actions.ndim!=2 or actions.dtype.kind not in "iu" or actions.dtype.kind=="b":
        raise ValueError("Integer policy by question actions are required")
    if actions.shape!=(len(policies),len(ids)) or not policies or not ids:
        raise ValueError("Frozen policy and question axes differ")
    if len(set(policies))!=len(policies) or any(not isinstance(value,str) or not value for value in policies):
        raise ValueError("Unique named frozen policies are required")
    if len(canonical)!=len(ids) or len(set(ids))!=len(ids) or len(set(canonical))!=len(canonical):
        raise ValueError("Unique question and canonical identities are required")
    if any(not isinstance(value,str) or not value for value in ids) or not all(map(_hash,canonical)):
        raise ValueError("Malformed cohort identity")
    if np.any(actions<0) or np.any(actions>=20) or tuple(generation_seeds)!=GENERATION_SEEDS:
        raise ValueError("Frozen20-team and three-generation contract differs")
    by_id={qid:index for index,qid in enumerate(ids)}
    required={(index,seed,int(action)) for index,column in enumerate(actions.T)
              for action in set(map(int,column)) for seed in GENERATION_SEEDS}
    observed={}
    source_records={}
    for row in executed_scores:
        if not isinstance(row,dict) or set(row)!=FIELDS:
            raise ValueError("Sparse scored-record fields differ")
        qid=row["question_id"]
        if qid not in by_id or row["canonical_id"]!=canonical[by_id[qid]]:
            raise ValueError("Scored record does not match frozen cohort identity")
        if type(row["generation_seed"]) is not int or row["generation_seed"] not in GENERATION_SEEDS or row["aggregation_seed"]!=42:
            raise ValueError("Member-generation or aggregation seed differs")
        action=row["combination_index"]
        team=np.asarray(row["team"])
        if type(action) is not int or not 0<=action<20 or team.shape!=(3,) or team.dtype.kind not in "iu" or tuple(team)!=TEAMS[action]:
            raise ValueError("Physical team and combination column differ")
        if row["aggregator_fingerprint"]!=OPERATOR_FP or row["scorer_version"]!=SCORER_VERSION:
            raise ValueError("Scored record uses another frozen operator or scorer")
        if not _hash(row["aggregation_record_sha256"]) or not _hash(row["scoring_receipt_sha256"]):
            raise ValueError("Scoring and execution provenance references are required")
        score=row["score"]
        if type(score) not in (int,float) or not math.isfinite(score) or score not in (0,1):
            raise ValueError("Final aggregate score must be binary")
        key=(by_id[qid],row["generation_seed"],action)
        if key in observed:
            raise ValueError("Duplicate scored execution key")
        source=row["aggregation_record_sha256"]
        if source in source_records and source_records[source]!=key:
            raise ValueError("One aggregation record cannot supply distinct execution keys")
        source_records[source]=key
        observed[key]=float(score)
    missing=required-set(observed)
    if missing:
        raise ValueError("Missing "+str(len(missing))+" legally selected scored execution keys")
    rewards=np.asarray([[[observed[question,seed,int(actions[policy,question])] for question in range(len(ids))]
                         for seed in GENERATION_SEEDS] for policy in range(len(policies))],dtype=np.float64)
    return dict(status="SPARSE_FROZEN_POLICY_PROJECTION_COMPLETE",policy_ids=policies,
                generation_seeds=GENERATION_SEEDS,rewards=rewards,original_questions=len(ids),
                unique_required_scored_keys=len(required),available_scored_keys=len(observed),
                unused_same_cohort_scored_keys=len(set(observed)-required),
                question_cluster_unit="one original question retaining its three generation rewards",
                paid_runner_status="NOT_IMPLEMENTED",gold_reader_status="NOT_IMPLEMENTED",
                authorizes_execution=False,authorizes_reward_access=False,
                referenced_bytes_verified_by_this_kernel=False)


def qualify():
    ids=["synthetic-q"+str(i) for i in range(6)]
    canonical=[hashlib.sha256(value.encode()).hexdigest() for value in ids]
    policies=["Ridge","SPO_coordinate","TSPO_uniform","GlobalFixed"]
    actions=np.array([[0,1,2,3,4,5],[1,1,3,3,5,5],[0,1,2,3,4,5],[19,19,19,19,19,19]])
    required=sorted({(i,seed,int(action)) for i,column in enumerate(actions.T)
                     for action in column for seed in GENERATION_SEEDS})
    rows=[dict(question_id=ids[i],canonical_id=canonical[i],generation_seed=seed,aggregation_seed=42,
               combination_index=action,team=list(TEAMS[action]),aggregator_fingerprint=OPERATOR_FP,
               scorer_version=SCORER_VERSION,score=float((i+GENERATION_SEEDS.index(seed)+action)%2),
               aggregation_record_sha256=hashlib.sha256(str((i,seed,action,"execution")).encode()).hexdigest(),
               scoring_receipt_sha256=hashlib.sha256(str((i,seed,action,"scoring")).encode()).hexdigest())
          for i,seed,action in required]
    checks=[]
    output=project_frozen_policy_scores(policies,actions,ids,canonical,rows)
    expected=np.array([[[float((i+s+int(action))%2) for i,action in enumerate(column)]
                        for s in range(3)] for column in actions])
    if not np.array_equal(output["rewards"],expected):raise AssertionError("Sparse gather differs")
    checks.append("exact sparse projection in policy/generation/question axes")
    if not len(rows)<6*3*20:raise AssertionError("Synthetic qualification should not supply full20 matrix")
    checks.append("complete20-combination reward matrix is unnecessary")
    if not np.array_equal(output["rewards"][0],output["rewards"][2]):raise AssertionError("Shared chosen action reuse differs")
    checks.append("policy-shared actions reuse one scored key")
    if output["original_questions"]!=6 or output["rewards"].shape!=(4,3,6):raise AssertionError("Question clustering differs")
    checks.append("three generations remain within six original questions")

    def rejected(name,test_rows=rows,test_actions=actions,test_ids=ids,test_canonical=canonical,test_policies=policies,test_seeds=GENERATION_SEEDS):
        try:
            project_frozen_policy_scores(test_policies,test_actions,test_ids,test_canonical,test_rows,test_seeds)
        except ValueError:
            checks.append(name)
            return
        raise AssertionError("Expected fail-closed "+name)
    rejected("missing selected score rejected",rows[:-1])
    rejected("duplicate selected key rejected",rows+[rows[0]])
    mismatched_source=[dict(row) for row in rows]
    mismatched_source[1]["aggregation_record_sha256"]=mismatched_source[0]["aggregation_record_sha256"]
    rejected("same physical aggregation record cannot supply different keys",mismatched_source)
    for field,value,name in [
        ("aggregator_fingerprint","0"*64,"another operator rejected"),
        ("scorer_version","other-scorer","another scorer rejected"),
        ("generation_seed",43,"another member generation rejected"),
        ("aggregation_seed",137,"member seed cannot replace aggregation seed42"),
        ("question_id","outside-cohort","foreign question rejected"),
        ("canonical_id","0"*64,"mismatched canonical identity rejected"),
        ("team",[0,1,5],"team-column mismatch rejected"),
        ("score",.5,"nonbinary score rejected"),
        ("scoring_receipt_sha256","bad","malformed provenance rejected")]:
        changed=[dict(row) for row in rows]
        changed[0][field]=value
        rejected(name,changed)
    bad_actions=actions.copy();bad_actions[0,0]=20
    rejected("out-of-range policy action rejected",test_actions=bad_actions)
    bad_ids=ids.copy();bad_ids[0]=bad_ids[1]
    rejected("duplicate cohort qid rejected",test_ids=bad_ids)
    bad_canonical=canonical.copy();bad_canonical[0]=bad_canonical[1]
    rejected("duplicate canonical identity rejected",test_canonical=bad_canonical)
    rejected("reordered frozen generation axis rejected",test_seeds=(137,42,2024))
    rejected("duplicate policy identity rejected",test_policies=["Ridge"]*4)
    extra=dict(rows[0]);extra.update(combination_index=18,team=list(TEAMS[18]),
                                   aggregation_record_sha256=hashlib.sha256(b"extra-unselected-execution").hexdigest())
    with_extra=project_frozen_policy_scores(policies,actions,ids,canonical,rows+[extra])
    if with_extra["unused_same_cohort_scored_keys"]!=1 or not np.array_equal(with_extra["rewards"],expected):
        raise AssertionError("Unselected legal score projection differs")
    checks.append("same-cohort unselected output counted without altering service scores")
    return dict(status="PASS",checks=len(checks),check_names=checks,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                synthetic_original_questions=6,synthetic_policies=4,synthetic_score_rows=len(rows),
                full20_matrix_rows_not_required=6*3*20,
                paid_runner_status="NOT_IMPLEMENTED",gold_reader_status="NOT_IMPLEMENTED",
                actual_scoring_adapter_status="NOT_IMPLEMENTED",synthetic_score_join_kernel_status="QUALIFIED",
                physical_llm_calls=0,real_gold_reads=0,fresh_output_reads=0,sealed_payload_reads=0,
                confidence_intervals=0,bootstrap_runs=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--qualify",action="store_true",required=True)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("SPARSE_RESULT_PROJECTION_QUALIFICATION.json"))
    args=parser.parse_args()
    result=qualify()
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"],checks=result["checks"],paid_runner_status=result["paid_runner_status"])))


if __name__=="__main__":
    main()
