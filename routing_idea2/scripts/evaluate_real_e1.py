"""Assemble E1 strictly from frozen real member/aggregate/control execution stores.

No model is called here and no missing score is imputed. E1 is exactly 100
selection-calibration plus 200 diagnostic-evaluation questions, both tasks.
Normal service costs (3 members + aggregator, including known failed attempts)
are distinct from the sum of UNIQUE physical diagnostic calls. Call counts,
prompt/completion tokens and summed per-call wall seconds are reported separately.

Default root layout follows preparation.py/e1_controls.py. CLI accepts explicit
paths. Missing outputs, controls, E0 qualification or unknown failed-call usage
produce PARTIAL; corrupted provenance or invalid gold mapping raises an error.
Synthetic test use must set software_test_only=True and cannot emit E1 PASS.
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
from collections import defaultdict, Counter
from dataclasses import replace
import argparse
import hashlib
import json
import math
import sys
import unicodedata
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from budget_router.inference import ModelSpec, configuration_fingerprint
from budget_router.scoring import score_answer, SCORER_VERSION
from budget_router.diagnostics import evaluate_e1
from budget_router.member_execution import MEMBER_PROMPT_VERSION
from budget_router.execution import AGGREGATION_PROMPT_VERSION, TOKEN_POLICY_VERSION

E1_SPLITS={"e1_selection_calibration","e1_diagnostic_evaluation"}

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1048576),b""): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def text_sha(text): return hashlib.sha256(text.encode("utf8")).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def _quantity(value):
    return isinstance(value,int) and not isinstance(value,bool) and value>=0

def _sum_cost(costs):
    costs=list(costs)
    return {"calls":sum(c["calls"] for c in costs),
            "prompt_tokens":sum(c["prompt_tokens"] for c in costs),
            "completion_tokens":sum(c["completion_tokens"] for c in costs),
            "wall_time_s":sum(c["wall_time_s"] for c in costs),
            "known":all(c["known"] for c in costs),
            "unknown_usage_calls":sum(c["unknown_usage_calls"] for c in costs),
            "unknown_wall_time_calls":sum(c["unknown_wall_time_calls"] for c in costs)}

def _tokens(cost): return cost["prompt_tokens"]+cost["completion_tokens"] if cost["known"] else None

def _read_stores(paths,record_map,missing):
    """Read LF JSONL, preserving legal Unicode line separators inside text."""
    calls={}; successes={}; grouped=defaultdict(list); assets=[]
    ignored=0
    for path in dict.fromkeys(Path(p).resolve() for p in paths):
        if not path.is_file():
            missing.add("missing_execution_store:"+path.name); continue
        assets.append({"name":path.name,"sha256":file_sha(path)})
        with path.open(encoding="utf-8-sig") as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip(): continue
                row=json.loads(line)
                checksum=row.get("record_sha256")
                if checksum!=digest({k:v for k,v in row.items() if k!="record_sha256"}):
                    raise ValueError(f"Execution checksum mismatch: {path.name}:{line_number}")
                qid=row.get("question_id")
                if qid not in record_map:
                    ignored+=1; continue
                expected=record_map[qid]
                if any(row.get(key)!=expected.get(key) for key in ("task","subtask","split")):
                    raise ValueError("Execution task/subtask/split differs from locked E1 record")
                attempt=row.get("attempt_id")
                if not attempt or not row.get("job_id"): raise ValueError("Unique attempt_id/job_id required")
                if attempt in calls:
                    if calls[attempt]["record"]["record_sha256"]!=checksum:
                        raise ValueError("Conflicting records for one physical call")
                    continue
                if row.get("status") not in {"success","failure"}: raise ValueError("Unknown execution status")
                usage=row.get("result",{}) if row["status"]=="success" else row.get("actual_usage")
                known_usage=isinstance(usage,dict) and all(_quantity(usage.get(k)) for k in ("prompt_tokens","completion_tokens"))
                wall=row.get("result",{}).get("wall_time_s") if row["status"]=="success" else row.get("failure_wall_time_s")
                known_wall=isinstance(wall,(int,float)) and not isinstance(wall,bool) and math.isfinite(wall) and wall>=0
                known=known_usage and known_wall and row.get("cost_status")=="recorded_actual_usage"
                if not known: missing.add("unreconciled_actual_call_costs")
                cost={"calls":1,"prompt_tokens":usage["prompt_tokens"] if known_usage else 0,
                      "completion_tokens":usage["completion_tokens"] if known_usage else 0,
                      "wall_time_s":float(wall) if known_wall else 0.,"known":known,
                      "unknown_usage_calls":int(not known_usage),"unknown_wall_time_calls":int(not known_wall)}
                # Zeroes above are KNOWN-SUBTOTAL accumulators, never imputed total costs.
                calls[attempt]={"record":row,"cost":cost}
                grouped[row["job_id"]].append(cost)
                if row["status"]=="success":
                    if row["job_id"] in successes: raise ValueError("Duplicate successful executions need reconciliation")
                    successes[row["job_id"]]=row
    return successes,{job:_sum_cost(value) for job,value in grouped.items()},calls,assets,ignored

def _validate_result(row,spec):
    if row.get("model_fingerprint")!=configuration_fingerprint(spec):
        raise ValueError("Output fingerprint differs from frozen generation configuration")
    value=row.get("result",{})
    if value.get("model_id")!=spec.model_id or value.get("model_revision")!=spec.revision or value.get("backend")!=spec.backend or value.get("model_identity_verified") is not True:
        raise ValueError("Actual served model identity is not verified against frozen spec")
    if value.get("seed")!=row.get("seed") or not isinstance(value.get("text"),str):
        raise ValueError("Actual output seed/text mismatch")
    if not all(_quantity(value.get(k)) for k in ("prompt_tokens","completion_tokens")):
        raise ValueError("Successful output lacks actual billed token usage")
    if value["prompt_tokens"]>spec.max_input_tokens or value["completion_tokens"]>spec.max_new_tokens:
        raise ValueError("Successful output exceeded locked generation cap")

def _score(row,question):
    return score_answer(question["task"],row["result"]["text"],question["reference"],
        subtask=question.get("subtask"),choices=question.get("choices")).score

def _jobids(value):
    result=set()
    if isinstance(value,dict):
        for key,child in value.items():
            if key=="job_id" and isinstance(child,str): result.add(child)
            else: result.update(_jobids(child))
    elif isinstance(value,list):
        for child in value: result.update(_jobids(child))
    return result

def _validate_aggregate_sources(row,question,member_rows):
    hashes=row.get("source_hashes",{})
    if hashes.get("question_sha256")!=text_sha(question["question"]):
        raise ValueError("Aggregator source question hash mismatch")
    expected={alias:text_sha(member["result"]["text"]) for alias,member in member_rows.items()}
    if hashes.get("member_answers_sha256")!=expected:
        raise ValueError("Aggregator source answer hashes do not match real member outputs")
    if member_rows and not {r["job_id"] for r in member_rows.values()}<=_jobids(row.get("member_provenance",{})):
        raise ValueError("Aggregator provenance does not bind real member job IDs")
    if row.get("prompt_version")!=AGGREGATION_PROMPT_VERSION:
        raise ValueError("Aggregate prompt revision differs from locked execution protocol")
    policy=row.get("candidate_token_policy",{})
    if policy.get("version")!=TOKEN_POLICY_VERSION:
        raise ValueError("Candidate token-policy revision changed")
    if (policy.get("mode"),policy.get("source_token_cap"),policy.get("head_tokens"),policy.get("tail_tokens"))!=("tokenizer_head_tail",512,128,384):
        raise ValueError("Formal aggregator output lacks locked tokenizer head128/tail384 policy")

def _load_split_and_questions(root,manifest_path,references):
    manifest=read_json(manifest_path); records=manifest.get("records")
    fingerprints=manifest.get("canonical_question_fingerprints",{})
    if not isinstance(records,list): raise ValueError("Split manifest requires records list")
    ids=[r["question_id"] for r in records]
    if len(set(ids))!=len(ids) or set(fingerprints)!=set(ids) or len(set(fingerprints.values()))!=len(ids):
        raise ValueError("Global question ID/canonical fingerprint leakage")
    selected=[r for r in records if r["split"] in E1_SPLITS]
    counts=Counter((r["split"],r["task"]) for r in selected)
    if len(selected)!=300 or any(counts[(split,task)]!=count for split,count in (("e1_selection_calibration",50),("e1_diagnostic_evaluation",100)) for task in ("gsm8k","bbh")):
        raise ValueError("Frozen E1 requires 50 calibration+100 evaluation questions per task")
    record_map={r["question_id"]:r for r in selected}; questions={}
    for path in references:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip(): continue
                row=json.loads(line); qid=row.get("question_id")
                if qid not in record_map: continue
                if qid in questions: raise ValueError("Duplicate original reference record")
                locked=record_map[qid]
                if any(row.get(k)!=locked.get(k) for k in ("task","subtask","source_split")):
                    raise ValueError("Reference/source split differs from frozen manifest")
                canonical=" ".join(unicodedata.normalize("NFKC",row["question"]).split())
                if text_sha(canonical)!=fingerprints[qid]: raise ValueError("Canonical question source changed")
                # Validate gold independently, without generated text execution.
                score_answer(row["task"],row["reference"],row["reference"],subtask=row.get("subtask"),choices=row.get("choices"))
                questions[qid]=row
    if set(questions)!=set(record_map): raise ValueError("Missing E1 original questions/references")
    return manifest,selected,record_map,questions

def assemble_e1(root,*,split_manifest=None,pool_config=None,aggregator_config=None,
                member_stores=None,aggregate_stores=None,control_stores=None,references=None,
                e0_lock=None,selection_receipt=None,seeds=(42,),bootstrap_replicates=2000,
                software_test_only=False):
    root=Path(root).resolve(); missing=set()
    split_manifest=Path(split_manifest or root/"data/derived/split_manifest.json")
    pool_config=Path(pool_config or root/"configs/pool_v1.json")
    aggregator_config=Path(aggregator_config or root/"configs/aggregator_v1.json")
    e0_lock=Path(e0_lock or root/"protocol/E0_LOCK.json")
    selection_receipt=Path(selection_receipt or root/"artifacts/audit/e1_strong_model_selection.json")
    references=[Path(p) for p in (references or [root/"data/derived/available_records.jsonl",root/"data/derived/gsm8k_questions.jsonl"])]
    manifest,records,record_map,questions=_load_split_and_questions(root,split_manifest,references)
    pool=read_json(pool_config)["models"]
    if len(pool)!=6 or [m["index"] for m in pool]!=list(range(6)) or len({m["alias"] for m in pool})!=6:
        raise ValueError("Exactly six frozen pool indices/aliases required")
    aliases=[m["alias"] for m in pool]
    specs={}
    model_config_hashes={}
    for entry in pool:
        config=root/entry["config"]; spec=ModelSpec(**read_json(config)); spec.validate()
        if (spec.model_id,spec.revision)!=(entry["model_id"],entry["revision"]) or entry.get("fingerprint")!=configuration_fingerprint(spec):
            raise ValueError("Pool entry and pinned model configuration mismatch")
        specs[entry["alias"]]=spec; model_config_hashes[entry["alias"]]=file_sha(config)
    agg_spec=ModelSpec(**read_json(aggregator_config)); agg_spec.validate()
    agg_fingerprint=configuration_fingerprint(agg_spec)
    member_stores=member_stores or [root/"results/raw/e1"/f"members_{a.lower()}.jsonl" for a in aliases]
    aggregate_stores=aggregate_stores or [root/"results/raw/e1/aggregations.jsonl"]
    control_stores=control_stores or [root/"results/raw/e1/aggregator_alone.jsonl",root/"results/raw/e1/strong_samples.jsonl",root/"results/raw/e1/strong_aggregations.jsonl"]
    success,job_cost,calls,assets,ignored=_read_stores([*member_stores,*aggregate_stores,*control_stores],record_map,missing)
    member_index={}; aggregate_index={}; alone_index={}; strong_index={}
    for row in success.values():
        qid=row["question_id"]; question=questions[qid]
        if "model_alias" in row:
            alias=row["model_alias"]
            if alias not in specs: raise ValueError("Member output alias outside frozen pool")
            if row.get("prompt_version")!=MEMBER_PROMPT_VERSION:
                raise ValueError("Member prompt revision differs from locked execution protocol")
            sampling=row.get("member_mode")=="strong_model_resample"
            if row.get("sampling_control") is not sampling:
                raise ValueError("Member sampling-control role flag mismatch")
            spec=specs[alias]  # Control differs in role/independent seed, never hidden model parameters.
            _validate_result(row,spec)
            if row.get("source_question_sha256")!=text_sha(question["question"]): raise ValueError("Member question hash mismatch")
            if not sampling and (row.get("member_mode")!="independent_sampled" or row.get("sample_index")!=0):
                raise ValueError("Ordinary pool answer must be independent_sampled sample zero")
            if not sampling:
                key=(qid,row["seed"],alias)
                if key in member_index: raise ValueError("Ambiguous ordinary member output")
                member_index[key]=row
        else:
            _validate_result(row,agg_spec)
            control=row.get("member_provenance",{}).get("control")
            target=alone_index if control=="aggregator_alone" else strong_index if control=="strong_model_resample" else aggregate_index
            members=tuple(row.get("members",()))
            if control=="aggregator_alone" and members: raise ValueError("Aggregator-alone control contains member answers")
            if control is None and (len(members)!=3 or set(members)-set(aliases)): raise ValueError("Ordinary aggregate must use three pool members")
            key=(qid,row["seed"]) if control else (qid,row["seed"],tuple(sorted(members)))
            if key in target: raise ValueError("Ambiguous repeated aggregate/control outputs")
            target[key]=row
    combos=tuple(combinations(range(6),3)); combo_alias=[tuple(sorted(aliases[i] for i in c)) for c in combos]
    cal=[r for r in records if r["split"]=="e1_selection_calibration"]
    evaluation=[r for r in records if r["split"]=="e1_diagnostic_evaluation"]
    ordered=cal+evaluation; total=len(ordered); ns=len(seeds)
    member_scores=np.full((ns,total,6),np.nan); aggregate_scores=np.full((ns,total,20),np.nan)
    logical_tokens=np.full_like(aggregate_scores,np.nan); logical_calls=np.full_like(aggregate_scores,np.nan)
    logical_seconds=np.full_like(aggregate_scores,np.nan)
    member_tokens=np.full_like(member_scores,np.nan)
    source_combo_rows={}
    for si,seed in enumerate(seeds):
        for qi,record in enumerate(ordered):
            qid=record["question_id"]; question=questions[qid]
            for mi,alias in enumerate(aliases):
                member=member_index.get((qid,seed,alias))
                if member is None: continue
                member_scores[si,qi,mi]=_score(member,question)
                cost=job_cost[member["job_id"]]
                if cost["known"]: member_tokens[si,qi,mi]=_tokens(cost)
            for ci,names in enumerate(combo_alias):
                row=aggregate_index.get((qid,seed,names))
                members={a:member_index.get((qid,seed,a)) for a in names}
                if row is None or any(v is None for v in members.values()): continue
                _validate_aggregate_sources(row,question,members)
                aggregate_scores[si,qi,ci]=_score(row,question)
                cost=_sum_cost([job_cost[row["job_id"]],*[job_cost[m["job_id"]] for m in members.values()]])
                if cost["known"]:
                    logical_tokens[si,qi,ci]=_tokens(cost); logical_calls[si,qi,ci]=cost["calls"]; logical_seconds[si,qi,ci]=cost["wall_time_s"]
                source_combo_rows[(si,qi,ci)]=row
    missing_members=int(np.isnan(member_scores).sum()); missing_aggregates=int(np.isnan(aggregate_scores).sum())
    if missing_members: missing.add("incomplete_300x6_real_member_outputs")
    if missing_aggregates: missing.add("incomplete_300x20_real_aggregations")
    receipt=read_json(selection_receipt) if selection_receipt.is_file() else None
    receipt_sha=file_sha(selection_receipt) if receipt else None
    if receipt is None: missing.add("missing_calibration_strong_model_selection_receipt")
    strongest=[]
    if not np.isnan(member_scores[:,:100]).any():
        for si in range(ns):
            macro=np.mean([member_scores[si,:100][np.array([r["task"]==task for r in cal])].mean(axis=0) for task in ("gsm8k","bbh")],axis=0)
            strongest.append(int(np.argmax(macro)))
        if receipt:
            selected=set(receipt.get("selection_question_ids",[]))
            if receipt.get("status")!="SELECTED_ON_E1_CALIBRATION_ONLY" or selected!={r["question_id"] for r in cal} or len(receipt.get("selection_question_ids",[]))!=100:
                raise ValueError("Strong model was not selected on exactly E1 calibration")
            if receipt.get("split_manifest_sha256")!=file_sha(split_manifest) or receipt.get("pool_config_sha256")!=file_sha(pool_config):
                raise ValueError("Strong-model selection belongs to different split/pool")
            if any(receipt.get("model_index")!=i or receipt.get("model_alias")!=aliases[i] for i in strongest):
                raise ValueError("Strong-model choice differs from calibration-only recomputation")
            for source in receipt.get("source_member_outputs",[]):
                if file_sha(root/source["path"])!=source["sha256"]: raise ValueError("Strong selection member source changed")
    alone=np.full((ns,200),np.nan); strong=np.full((ns,200),np.nan)
    alone_cost=np.full_like(alone,np.nan); strong_cost=np.full_like(strong,np.nan)
    for si,seed in enumerate(seeds):
        for qi,record in enumerate(evaluation):
            qid=record["question_id"]; question=questions[qid]
            row=alone_index.get((qid,seed))
            if row:
                _validate_aggregate_sources(row,question,{})
                alone[si,qi]=_score(row,question)
                if job_cost[row["job_id"]]["known"]: alone_cost[si,qi]=_tokens(job_cost[row["job_id"]])
            row=strong_index.get((qid,seed))
            if not row or not strongest or receipt is None: continue
            alias=aliases[strongest[si]]
            expected_names={f"{alias}::sample{i}" for i in range(3)}
            if set(row.get("members",[]))!=expected_names or row.get("member_provenance",{}).get("selection_receipt_sha256")!=receipt_sha:
                raise ValueError("Strong aggregation lacks the calibration-selected independent samples")
            samples={}
            provenance=row["member_provenance"].get("samples",{})
            for name in sorted(expected_names):
                job=provenance.get(name,{}).get("job_id"); sample=success.get(job)
                if sample is None: continue
                index=int(name.rsplit("sample",1)[1])
                if sample.get("question_id")!=qid or sample.get("model_alias")!=alias or sample.get("sample_index")!=index or sample.get("member_mode")!="strong_model_resample":
                    raise ValueError("Independent strong sample identity mismatch")
                if provenance[name].get("record_sha256")!=sample["record_sha256"]: raise ValueError("Sample record hash mismatch")
                samples[name]=sample
            if len(samples)!=3: continue
            if len({v["seed"] for v in samples.values()})!=3 or seed in {v["seed"] for v in samples.values()}: raise ValueError("Strong samples require three fresh independent generation seeds")
            _validate_aggregate_sources(row,question,samples)
            strong[si,qi]=_score(row,question)
            cost=_sum_cost([job_cost[row["job_id"]],*[job_cost[v["job_id"]] for v in samples.values()]])
            if cost["known"]: strong_cost[si,qi]=_tokens(cost)
    if np.isnan(alone).any(): missing.add("incomplete_real_aggregator_alone_control")
    if np.isnan(strong).any(): missing.add("incomplete_real_strong_model_sampling_control")
    cost_totals=_sum_cost([entry["cost"] for entry in calls.values()])
    categorized=defaultdict(list)
    for entry in calls.values():
        row=entry["record"]
        if "model_alias" in row:
            category="strong_samples" if row.get("member_mode")=="strong_model_resample" else "ordinary_members"
            expected_fp=configuration_fingerprint(specs[row["model_alias"]]) if row.get("model_alias") in specs else None
        else:
            control=row.get("member_provenance",{}).get("control")
            category="aggregator_alone" if control=="aggregator_alone" else "strong_aggregations" if control=="strong_model_resample" else "ordinary_aggregations"
            expected_fp=agg_fingerprint
        if row.get("model_fingerprint")!=expected_fp: missing.add("billed_calls_outside_frozen_generation_configuration")
        categorized[category].append(entry["cost"])
    by_kind={}
    for category,costs in categorized.items():
        item=_sum_cost(costs)
        by_kind[category]={"unique_calls":item["calls"],"known_prompt_tokens":item["prompt_tokens"],
            "known_completion_tokens":item["completion_tokens"],"known_call_wall_time_sum_s":item["wall_time_s"],
            "total_tokens":_tokens(item),"all_costs_reconciled":item["known"]}
    lock=read_json(e0_lock) if e0_lock.is_file() else {}
    if lock.get("status") not in {"PASS","E0_PASS","E0_PASS_AND_LOCK","LOCKED"}: missing.add("e0_pass_and_lock_missing")
    if lock:
        for key,value in (("pool_config_sha256",file_sha(pool_config)),("split_manifest_sha256",file_sha(split_manifest)),("aggregator_fingerprint",agg_fingerprint)):
            if lock.get(key)!=value: missing.add("e0_lock_binding:"+key)
    qualification=[]
    for source in lock.get("qualification_receipts",[]):
        path=root/source["path"]
        if not path.is_file() or file_sha(path)!=source.get("sha256"): raise ValueError("Qualification receipt source changed")
        value=read_json(path)
        if value.get("status")!="PASS": missing.add("qualification_receipt_not_pass")
        qualification.append({"sha256":source["sha256"],"model_fingerprint":value.get("model_fingerprint")})
    required_fingerprints={configuration_fingerprint(spec) for spec in specs.values()}|{agg_fingerprint}
    if not required_fingerprints<={r["model_fingerprint"] for r in qualification}: missing.add("qualification_receipts_not_complete")
    scorer_path=ROOT/"src/budget_router/scoring.py"
    evidence={"e0_lock_sha256":file_sha(e0_lock) if e0_lock.is_file() else None,
        "cache_revision":lock.get("cache_revision") or "generated_member_artifacts:"+digest([r for r in assets if r["name"].startswith("members_")]),
        "aggregator_revision":agg_spec.revision,"aggregator_fingerprint":agg_fingerprint,
        "aggregator_config_sha256":file_sha(aggregator_config),"pool_config_sha256":file_sha(pool_config),
        "model_config_sha256":model_config_hashes,"qualification_receipts":qualification,
        "raw_aggregate_artifact_sha256":digest([r for r in assets if "aggregation" in r["name"]]),
        "raw_baseline_artifact_sha256":digest([r for r in assets if r["name"] in {"aggregator_alone.jsonl","strong_samples.jsonl","strong_aggregations.jsonl"}]),
        "raw_output_sources":assets,"reference_sources":[file_sha(p) for p in references],
        "split_manifest_sha256":file_sha(split_manifest),"scorer_version":SCORER_VERSION,"scorer_source_sha256":file_sha(scorer_path),
        "cost_unit":"logical_total_input_output_tokens; calls and summed per-call seconds separately reported",
        "real_frozen_llm_aggregation":not software_test_only,"all_combinations_executed":missing_aggregates==0,
        "budget_matched_controls":not np.isnan(alone_cost).any() and not np.isnan(strong_cost).any(),
        "e1_isolated_from_online_and_pca":True,"software_test_only":bool(software_test_only)}
    partial={"stage":"E1","status":"PARTIAL","research_complete":False,"missing_requirements":sorted(missing),
        "coverage":{"required_questions":300,"required_members":ns*1800,"required_aggregates":ns*6000,
            "missing_members":missing_members,"missing_aggregates":missing_aggregates,
            "missing_aggregator_alone":int(np.isnan(alone).sum()),"missing_strong_resample":int(np.isnan(strong).sum())},
        "physical_diagnostic_cost":{"unique_calls":cost_totals["calls"],"known_prompt_tokens":cost_totals["prompt_tokens"],
            "known_completion_tokens":cost_totals["completion_tokens"],"known_call_wall_time_sum_s":cost_totals["wall_time_s"],
            "unknown_usage_calls":cost_totals["unknown_usage_calls"],"unknown_wall_time_calls":cost_totals["unknown_wall_time_calls"],
            "total_tokens":_tokens(cost_totals),"all_costs_reconciled":cost_totals["known"],
            "ignored_non_E1_store_rows":ignored,"by_kind":by_kind},"evidence":evidence}
    if missing_members or missing_aggregates:
        if software_test_only: partial["status"]="SOFTWARE_TEST_ONLY"; partial["diagnostic_status"]="PARTIAL"
        return partial
    best_single_cost=np.stack([member_tokens[si,100:,strongest[si]] for si in range(ns)])
    baseline_costs={}
    for name,value in (("aggregator_alone",alone_cost),("strong_model_resample",strong_cost),("strongest_single_calibration",best_single_cost)):
        if np.isfinite(value).all(): baseline_costs[name]=value
    result=evaluate_e1(aggregate_scores=aggregate_scores[:,100:],member_scores=member_scores[:,100:],combinations=combos,
        calibration_aggregate_scores=aggregate_scores[:,:100],calibration_member_scores=member_scores[:,:100],
        question_ids=[r["question_id"] for r in evaluation],calibration_question_ids=[r["question_id"] for r in cal],
        task_labels=[r["task"] for r in evaluation],calibration_task_labels=[r["task"] for r in cal],experiment_seeds=seeds,
        aggregator_alone_scores=alone if np.isfinite(alone).all() else None,
        strong_model_resample_scores=strong if np.isfinite(strong).all() else None,
        execution_costs=logical_tokens[:,100:] if np.isfinite(logical_tokens).all() else None,baseline_costs=baseline_costs,
        diagnostic_total_execution_cost=_tokens(cost_totals),evidence=evidence,
        question_fingerprints=[manifest["canonical_question_fingerprints"][r["question_id"]] for r in evaluation],
        calibration_question_fingerprints=[manifest["canonical_question_fingerprints"][r["question_id"]] for r in cal],
        bootstrap_replicates=bootstrap_replicates)
    result["physical_diagnostic_cost"]=partial["physical_diagnostic_cost"]
    result["logical_standalone_service_cost"]={"unit":"total_input_output_tokens","sum_tokens":float(logical_tokens.sum()) if np.isfinite(logical_tokens).all() else None,
        "sum_calls":float(logical_calls.sum()) if np.isfinite(logical_calls).all() else None,
        "sum_call_wall_time_s":float(logical_seconds.sum()) if np.isfinite(logical_seconds).all() else None,
        "warning":"sum over all combinations duplicates shared member generation; not physical diagnostic total"}
    result["coverage"]=partial["coverage"]; result["research_complete"]=False
    result["missing_requirements"]=sorted(set(result.get("missing_requirements",[]))|missing)
    if result["missing_requirements"]: result["status"]="PARTIAL"
    if software_test_only: result["diagnostic_status"]=result["status"]; result["status"]="SOFTWARE_TEST_ONLY"
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",default=str(ROOT))
    parser.add_argument("--split-manifest"); parser.add_argument("--pool-config"); parser.add_argument("--aggregator-config")
    parser.add_argument("--member-store",action="append"); parser.add_argument("--aggregate-store",action="append")
    parser.add_argument("--control-store",action="append"); parser.add_argument("--references",action="append")
    parser.add_argument("--e0-lock"); parser.add_argument("--selection-receipt")
    parser.add_argument("--seeds",nargs="+",type=int,default=[42])
    parser.add_argument("--bootstrap-replicates",type=int,default=2000)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    result=assemble_e1(args.root,split_manifest=args.split_manifest,pool_config=args.pool_config,aggregator_config=args.aggregator_config,
        member_stores=args.member_store,aggregate_stores=args.aggregate_store,control_stores=args.control_store,references=args.references,
        e0_lock=args.e0_lock,selection_receipt=args.selection_receipt,seeds=tuple(args.seeds),bootstrap_replicates=args.bootstrap_replicates)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError("Refusing to overwrite an existing E1 scientific receipt")
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({"status":result["status"],"stage":"E1","research_complete":False,"output":str(output)},ensure_ascii=False))
    return 0 if result["status"] in {"PASS","STOP_E1"} else 2

if __name__=="__main__":
    raise SystemExit(main())

