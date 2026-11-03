"""Synthetic execution-store assembler tests, never E1 scientific evidence."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pytest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"src"))
from budget_router.inference import ModelSpec,configuration_fingerprint
from budget_router.member_execution import MEMBER_PROMPT_VERSION
from budget_router.execution import AGGREGATION_PROMPT_VERSION,TOKEN_POLICY_VERSION
spec=importlib.util.spec_from_file_location("real_e1_assembly_under_test",PROJECT/"scripts/evaluate_real_e1.py")
assembly=importlib.util.module_from_spec(spec); spec.loader.exec_module(assembly)

def sha(text): return hashlib.sha256(text.encode()).hexdigest()
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf8")

def signed(row):
    row=dict(row); row["record_sha256"]=assembly.digest(row)
    return row

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf8")

def result(spec,seed,text):
    return {"text":text,"prompt_tokens":2,"completion_tokens":1,"wall_time_s":.1,
            "model_id":spec.model_id,"model_revision":spec.revision,"seed":seed,
            "backend":spec.backend,"model_identity_verified":True,"completion_reached_cap":False,"finish_reason":"stop"}

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from itertools import combinations
    root=tmp_path_factory.mktemp("synthetic_e1_only")
    specs=[]; pool=[]
    for i in range(6):
        model=ModelSpec(f"software/model{i}",str(i+1)*40,model_path=f"unused/model{i}",
                        quantization="nf4",max_input_tokens=64,max_new_tokens=2048,temperature=.7,top_p=.95)
        specs.append(model)
        path=root/f"configs/member{i}.json"; write_json(path,asdict(model))
        pool.append({"index":i,"alias":f"m{i}","config":f"configs/member{i}.json",
                     "model_id":model.model_id,"revision":model.revision,"fingerprint":configuration_fingerprint(model)})
    write_json(root/"configs/pool_v1.json",{"models":pool,"software_test_only":True})
    aggregator=ModelSpec("software/aggregator","a"*40,model_path="unused/aggregator",
                        quantization="nf4",max_input_tokens=64,max_new_tokens=16)
    write_json(root/"configs/aggregator_v1.json",asdict(aggregator))
    records=[]; questions=[]; fingerprints={}
    for split,count in (("e1_selection_calibration",50),("e1_diagnostic_evaluation",100)):
        for task in ("gsm8k","bbh"):
            for i in range(count):
                qid=f"{split}:{task}:{i}"
                question="Synthetic question "+qid+("\u2028 legal separator" if i==0 else "")
                reference="42" if task=="gsm8k" else "yes"
                row={"question_id":qid,"task":task,"subtask":None if task=="gsm8k" else "causal_judgement",
                     "split":split,"source_split":"unit_test"}
                records.append(row)
                questions.append({**row,"question":question,"reference":reference})
                fingerprints[qid]=sha(" ".join(question.split()))
    manifest=root/"data/derived/split_manifest.json"
    write_json(manifest,{"records":records,"canonical_question_fingerprints":fingerprints,"software_test_only":True})
    references=root/"data/derived/available_records.jsonl"
    write_rows(references,questions)
    write_rows(root/"data/derived/gsm8k_questions.jsonl",[])
    mapping={q["question_id"]:q for q in questions}
    members={}; all_member_rows={a:[] for a in [m["alias"] for m in pool]}
    for row in records:
        q=mapping[row["question_id"]]
        for i,model in enumerate(specs):
            alias=f"m{i}"; correct=row["split"]=="e1_selection_calibration" and i==0
            text="Final answer: "+(q["reference"] if correct else "0" if row["task"]=="gsm8k" else "no")
            data={"job_id":sha(f"member:{q['question_id']}:{alias}"),"attempt_id":sha(f"attempt:member:{q['question_id']}:{alias}"),
                **row,"model_alias":alias,"sample_index":0,"seed":42,"member_fingerprint_unused":None,
                "model_fingerprint":configuration_fingerprint(model),"member_mode":"independent_sampled","sampling_control":False,"prompt_version":MEMBER_PROMPT_VERSION,
                "source_question_sha256":sha(q["question"]),"status":"success","cost_status":"recorded_actual_usage",
                "result":result(model,42,text)}
            data.pop("member_fingerprint_unused")
            data=signed(data)
            members[(q["question_id"],alias)]=data; all_member_rows[alias].append(data)
    for alias,rows in all_member_rows.items(): write_rows(root/f"results/raw/e1/members_{alias}.jsonl",rows)
    selected=root/"artifacts/audit/e1_strong_model_selection.json"
    write_json(selected,{"status":"SELECTED_ON_E1_CALIBRATION_ONLY","model_alias":"m0","model_index":0,
        "selection_question_ids":[r["question_id"] for r in records if r["split"]=="e1_selection_calibration"],
        "split_manifest_sha256":assembly.file_sha(manifest),"pool_config_sha256":assembly.file_sha(root/"configs/pool_v1.json"),
        "source_member_outputs":[{"path":f"results/raw/e1/members_{a}.jsonl","sha256":assembly.file_sha(root/f"results/raw/e1/members_{a}.jsonl")} for a in all_member_rows]})
    selected_sha=assembly.file_sha(selected)
    ordinary=[]; alone=[]; sampled=[]; sampled_aggregates=[]
    def aggregate(row,names,source_members,correct,kind=None):
        q=mapping[row["question_id"]]
        provenance={a:{"job_id":m["job_id"],"record_sha256":m["record_sha256"]} for a,m in source_members.items()}
        if kind=="aggregator_alone": provenance={"control":kind}
        if kind=="strong_model_resample":
            provenance={"control":kind,"selection_receipt_sha256":selected_sha,"samples":provenance}
        identifier=sha(f"aggregate:{row['question_id']}:{names}:{kind}")
        text="Reasoning\u2028Final answer: "+(q["reference"] if correct else "0" if row["task"]=="gsm8k" else "no")
        value=result(aggregator,42,text); value["prompt_tokens"]=4
        return signed({"job_id":identifier,"attempt_id":sha("attempt:"+identifier),**row,"seed":42,
            "members":sorted(names),"model_fingerprint":configuration_fingerprint(aggregator),
            "member_provenance":provenance,"prompt_version":AGGREGATION_PROMPT_VERSION,"status":"success","cost_status":"recorded_actual_usage",
            "source_hashes":{"question_sha256":sha(q["question"]),
                "member_answers_sha256":{a:sha(m["result"]["text"]) for a,m in source_members.items()}},
            "candidate_token_policy":{"version":TOKEN_POLICY_VERSION,"mode":"tokenizer_head_tail","source_token_cap":512,"head_tokens":128,"tail_tokens":384},
            "result":value})
    for qi,row in enumerate(records):
        for ci,combo in enumerate(combinations(range(6),3)):
            names=[f"m{i}" for i in combo]
            correct=(ci==0 if row["split"]=="e1_selection_calibration" else ci==1)
            ordinary.append(aggregate(row,names,{a:members[(row["question_id"],a)] for a in names},correct))
        if row["split"]!="e1_diagnostic_evaluation": continue
        alone.append(aggregate(row,[],{},False,"aggregator_alone"))
        three={}
        model=replace(specs[0],temperature=.7,top_p=.95)
        for sample in range(3):
            seed=100000+qi*3+sample
            identifier=sha(f"sample:{row['question_id']}:{sample}")
            q=mapping[row["question_id"]]
            text="Final answer: "+("0" if row["task"]=="gsm8k" else "no")
            value=signed({"job_id":identifier,"attempt_id":sha("attempt:"+identifier),**row,
                "model_alias":"m0","sample_index":sample,"seed":seed,"model_fingerprint":configuration_fingerprint(model),
                "member_mode":"strong_model_resample","sampling_control":True,"prompt_version":MEMBER_PROMPT_VERSION,"source_question_sha256":sha(q["question"]),
                "status":"success","cost_status":"recorded_actual_usage","result":result(model,seed,text)})
            sampled.append(value); three[f"m0::sample{sample}"]=value
        sampled_aggregates.append(aggregate(row,list(three),three,False,"strong_model_resample"))
    write_rows(root/"results/raw/e1/aggregations.jsonl",ordinary)
    write_rows(root/"results/raw/e1/aggregator_alone.jsonl",alone)
    write_rows(root/"results/raw/e1/strong_samples.jsonl",sampled)
    write_rows(root/"results/raw/e1/strong_aggregations.jsonl",sampled_aggregates)
    receipts=[]
    for i,model in enumerate(specs+[aggregator]):
        path=root/f"artifacts/audit/qualification{i}.json"
        write_json(path,{"status":"PASS","model_fingerprint":configuration_fingerprint(model),"software_test_only":True})
        receipts.append({"path":path.relative_to(root).as_posix(),"sha256":assembly.file_sha(path)})
    write_json(root/"protocol/E0_LOCK.json",{"status":"PASS","pool_config_sha256":assembly.file_sha(root/"configs/pool_v1.json"),
        "split_manifest_sha256":assembly.file_sha(manifest),"aggregator_fingerprint":configuration_fingerprint(aggregator),
        "qualification_receipts":receipts,"software_test_only":True})
    return root

def evaluate(root,**kwargs):
    return assembly.assemble_e1(root,software_test_only=True,bootstrap_replicates=100,**kwargs)

def test_complete_real_schema_assembly_counts_unique_calls_and_separate_logical_costs(corpus):
    report=evaluate(corpus)
    assert report["status"]=="SOFTWARE_TEST_ONLY" and report["research_complete"] is False
    assert report["coverage"]["missing_members"]==0
    assert report["coverage"]["missing_aggregates"]==0
    assert report["coverage"]["missing_aggregator_alone"]==0
    assert report["coverage"]["missing_strong_resample"]==0
    assert report["physical_diagnostic_cost"]["unique_calls"]==8800
    assert report["physical_diagnostic_cost"]["by_kind"]["ordinary_members"]["unique_calls"]==1800
    assert report["physical_diagnostic_cost"]["by_kind"]["ordinary_aggregations"]["unique_calls"]==6000
    assert report["physical_diagnostic_cost"]["total_tokens"]==39200
    assert report["physical_diagnostic_cost"]["known_call_wall_time_sum_s"]==pytest.approx(880.)
    assert report["logical_standalone_service_cost"]["sum_tokens"]==84000
    assert report["logical_standalone_service_cost"]["sum_calls"]==24000
    assert report["selected_on_calibration"]["best_fixed_indices"]==[0]
    assert report["selected_on_calibration"]["strongest_single_indices"]==[0]
    assert report["metrics"]["best_fixed_calibration"]["mean"]==0
    assert report["metrics"]["diagnostic_oracle_upper_reference"]["mean"]==1
    assert report["evidence"]["aggregator_fingerprint"]
    assert "Synthetic question" not in json.dumps(report)

def test_missing_combination_is_partial_and_never_filled_with_zero(corpus,tmp_path):
    source=corpus/"results/raw/e1/aggregations.jsonl"
    with source.open(encoding="utf8") as stream: lines=list(stream)
    missing=tmp_path/"one_missing.jsonl"; missing.write_text("".join(lines[1:]),encoding="utf8")
    report=evaluate(corpus,aggregate_stores=[missing])
    assert report["diagnostic_status"]=="PARTIAL"
    assert report["coverage"]["missing_aggregates"]==1
    assert "metrics" not in report

def test_missing_real_controls_are_absent_not_zero_baselines(corpus,tmp_path):
    report=evaluate(corpus,control_stores=[tmp_path/"not_generated.jsonl"])
    assert report["diagnostic_status"]=="PARTIAL"
    assert report["coverage"]["missing_aggregator_alone"]==200
    assert report["coverage"]["missing_strong_resample"]==200
    assert "aggregator_alone" not in report["metrics"]
    assert "strong_model_resample" not in report["metrics"]

def test_source_answer_tampering_is_rejected_even_with_valid_record_checksum(corpus,tmp_path):
    source=corpus/"results/raw/e1/aggregations.jsonl"
    with source.open(encoding="utf8") as stream: rows=[json.loads(line) for line in stream]
    row=rows[0]; row.pop("record_sha256")
    first=next(iter(row["source_hashes"]["member_answers_sha256"]))
    row["source_hashes"]["member_answers_sha256"][first]="0"*64
    rows[0]=signed(row)
    tampered=tmp_path/"tampered.jsonl"; write_rows(tampered,rows)
    with pytest.raises(ValueError,match="answer hashes"):
        evaluate(corpus,aggregate_stores=[tampered])

def test_duplicate_cached_physical_call_is_counted_once(corpus,tmp_path):
    source=corpus/"results/raw/e1/aggregations.jsonl"
    with source.open(encoding="utf8") as stream: first=next(stream)
    duplicate=tmp_path/"duplicate_store.jsonl"; duplicate.write_text(first,encoding="utf8")
    report=evaluate(corpus,aggregate_stores=[source,duplicate])
    assert report["physical_diagnostic_cost"]["unique_calls"]==8800
    assert report["physical_diagnostic_cost"]["total_tokens"]==39200

def test_unknown_failed_attempt_cost_is_partial_not_imputed_zero(corpus,tmp_path):
    source=corpus/"results/raw/e1/aggregations.jsonl"
    with source.open(encoding="utf8") as stream: first=json.loads(next(stream))
    failure={k:v for k,v in first.items() if k not in {"record_sha256","result"}}
    failure.update(status="failure",attempt_id=sha("unknown-failed-attempt"),cost_status="unknown_must_reconcile",
                   actual_usage=None,failure_wall_time_s=None)
    failed=tmp_path/"unknown_failure.jsonl"; write_rows(failed,[signed(failure)])
    report=evaluate(corpus,aggregate_stores=[source,failed])
    assert report["diagnostic_status"]=="PARTIAL"
    assert report["physical_diagnostic_cost"]["unknown_usage_calls"]==1
    assert report["physical_diagnostic_cost"]["total_tokens"] is None
    assert report["physical_diagnostic_cost"]["unique_calls"]==8801
    assert "unreconciled_actual_call_costs" in report["missing_requirements"]

def test_evaluation_questions_cannot_select_strong_model(corpus,tmp_path):
    receipt=assembly.read_json(corpus/"artifacts/audit/e1_strong_model_selection.json")
    receipt["selection_question_ids"][0]="e1_diagnostic_evaluation:gsm8k:0"
    wrong=tmp_path/"bad_selection.json"; write_json(wrong,receipt)
    with pytest.raises(ValueError,match="exactly E1 calibration"):
        evaluate(corpus,selection_receipt=wrong)

def test_split_pass_is_not_full_e0_pass(corpus,tmp_path):
    report=evaluate(corpus,e0_lock=tmp_path/"no_E0_LOCK.json")
    assert report["diagnostic_status"]=="PARTIAL"
    assert "e0_pass_and_lock_missing" in report["missing_requirements"]

