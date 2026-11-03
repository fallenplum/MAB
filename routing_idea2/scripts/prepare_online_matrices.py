"""Assemble private online matrices from complete real stage jobs/output stores.

No inference, embedding generation, PCA fitting, E1-label reuse, or E2 execution
occurs here. Frozen 200/600-question stages need six real members and twenty real
aggregates per question. Missing or invalid assets produce PARTIAL, never zeros.
Only a production manifest accepted by MatrixEnvironment.from_npz is published
as READY. Synthetic tests have a separate explicit SOFTWARE_TEST_ONLY status.
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
from collections import Counter
import argparse
import hashlib
import json
import os
import sys
import unicodedata
import uuid
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/"src"))
from budget_router.inference import ModelSpec,configuration_fingerprint
from budget_router.member_execution import make_member_job,validate_member_job
from budget_router.execution import validate_job,AGGREGATION_PROMPT_VERSION,TOKEN_POLICY_VERSION,SYSTEM_PROMPT
from budget_router.scoring import score_answer,SCORER_VERSION
import budget_router.scoring as scoring_module
from budget_router.simulation import MatrixEnvironment,_validate_locked_split,_validate_stage_bindings,_validate_pair
from scripts.evaluate_real_e1 import file_sha,digest,_read_stores,_validate_result,_validate_aggregate_sources,_sum_cost

STAGES={"online_calibration":200,"online_stream":600}

class AssetError(ValueError):
    pass

def _json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def _text_sha(value): return hashlib.sha256(value.encode("utf8")).hexdigest()

def _reference(root,record,label):
    if not isinstance(record,dict) or not record.get("path") or not record.get("sha256"):
        raise AssetError("missing_hashed_reference:"+label)
    path=Path(record["path"])
    if not path.is_absolute(): path=root/path
    path=path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or file_sha(path)!=record["sha256"]:
        raise AssetError("changed_or_missing_reference:"+label)
    return path

def _ref(path):
    path=Path(path).resolve()
    return {"path":path.as_posix(),"sha256":file_sha(path)}

def _read_jsonl(path):
    # Do not use splitlines(): Unicode line separators occur in legitimate text.
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip(): yield json.loads(line)

def _load_features(root,locked,config):
    audit_path=root/"artifacts/audit/feature_audit.json"
    if not audit_path.is_file(): raise AssetError("missing_feature_audit")
    audit=_json(audit_path)
    if (audit.get("status")!="REAL_FROZEN_FEATURES_READY_NOT_MODEL_COMBINATION_RESULTS"
        or audit.get("fit_split")!="online_calibration"
        or audit.get("e1_questions_labels_embeddings_used") is not False
        or audit.get("sealed_test_questions_encoded") is not False):
        raise AssetError("invalid_or_leaking_feature_audit")
    split_sha=file_sha(root/"data/derived/split_manifest.json")
    if audit.get("split_manifest_sha256")!=split_sha: raise AssetError("feature_split_manifest_changed")
    feature_path=_reference(root,audit.get("feature_artifact"),"features")
    pca_path=_reference(root,audit.get("pca_artifact"),"pca")
    encoder_path=root/"artifacts/audit/encoder_model_download.json"
    if not encoder_path.is_file() or file_sha(encoder_path)!=audit.get("encoder_source_receipt_sha256"):
        raise AssetError("encoder_source_receipt_changed")
    encoder=_json(encoder_path)
    frozen=config.get("features",{})
    if (audit.get("encoder_id"),audit.get("encoder_revision"))!=(frozen.get("encoder_id"),frozen.get("encoder_revision")):
        raise AssetError("feature_encoder_differs_from_locked_config")
    if (encoder.get("repo_id"),encoder.get("revision"))!=(audit["encoder_id"],audit["encoder_revision"]):
        raise AssetError("feature_encoder_source_identity_mismatch")
    if not encoder.get("files"): raise AssetError("encoder_source_files_missing")
    for source in encoder["files"]: _reference(root,source,"encoder_file")
    if audit.get("pooling")!=frozen.get("pooling") or audit.get("encoder_max_tokens")!=256:
        raise AssetError("frozen_encoder_pooling_or_input_protocol_changed")
    with np.load(feature_path,allow_pickle=False) as archive:
        if set(archive.files)!={"question_ids","task_labels","splits","phi","embeddings"}:
            raise AssetError("unexpected_label_or_feature_array_schema")
        feature={key:archive[key] for key in archive.files}
    with np.load(pca_path,allow_pickle=False) as archive:
        if set(archive.files)!={"components","mean","fit_question_ids","explained_variance"}:
            raise AssetError("invalid_frozen_pca_schema")
        pca={key:archive[key] for key in archive.files}
    records=locked["records"]
    selected=[r for r in records if r["split"] in STAGES]
    expected={r["question_id"]:r for r in selected}
    ids=feature["question_ids"]
    if ids.ndim!=1 or ids.dtype.kind not in "US" or len(ids)!=800 or len(set(ids.tolist()))!=800 or set(ids.tolist())!=set(expected):
        raise AssetError("features_must_cover_exactly_800_non_E1_nonsealed_questions")
    if feature["task_labels"].shape!=(800,) or feature["splits"].shape!=(800,):
        raise AssetError("feature_metadata_shape_mismatch")
    for i,qid in enumerate(ids.tolist()):
        if (str(feature["task_labels"][i]),str(feature["splits"][i]))!=(expected[qid]["task"],expected[qid]["split"]):
            raise AssetError("feature_question_task_or_stage_mismatch")
    fit=pca["fit_question_ids"]
    fit_ids=fit.tolist()
    actual_fit=[r["question_id"] for r in selected if r["split"]=="online_calibration"]
    if (fit.ndim!=1 or fit.dtype.kind not in "US" or len(fit_ids)!=200 or len(set(fit_ids))!=200
        or set(fit_ids)!=set(actual_fit) or audit.get("fit_question_ids")!=fit_ids
        or audit.get("fit_questions")!=200 or audit.get("questions")!=800):
        raise AssetError("PCA_fit_ids_must_be_exactly_200_online_calibration_questions")
    phi=np.asarray(feature["phi"],dtype=float); embeddings=np.asarray(feature["embeddings"],dtype=float)
    components=np.asarray(pca["components"],dtype=float); mean=np.asarray(pca["mean"],dtype=float)
    variance=np.asarray(pca["explained_variance"],dtype=float)
    if (phi.shape!=(800,17) or embeddings.ndim!=2 or embeddings.shape[0]!=800
        or components.shape!=(16,embeddings.shape[1]) or mean.shape!=(embeddings.shape[1],)
        or variance.shape!=(16,) or (variance<0).any()
        or audit.get("pca_dimensions")!=16 or audit.get("feature_dimensions")!=17):
        raise AssetError("frozen_feature_or_PCA_dimensions_changed")
    if any(not np.isfinite(value).all() for value in (phi,embeddings,components,mean,variance)):
        raise AssetError("nonfinite_frozen_feature_asset")
    if not np.allclose(np.linalg.norm(embeddings,axis=1),1.,atol=1e-5,rtol=1e-5):
        raise AssetError("encoder_embeddings_do_not_follow_frozen_L2_pooling")
    if not np.allclose(components@components.T,np.eye(16),atol=1e-6,rtol=1e-6):
        raise AssetError("frozen_PCA_components_are_not_orthonormal")
    fit_mask=np.array([qid in set(fit_ids) for qid in ids.tolist()])
    if not np.allclose(mean,embeddings[fit_mask].mean(axis=0),atol=1e-6,rtol=1e-6):
        raise AssetError("PCA_mean_is_not_training_only")
    transformed=np.column_stack((np.ones(800),(embeddings-mean)@components.T))
    if not np.allclose(phi,transformed,atol=1e-6,rtol=1e-6):
        raise AssetError("phi_does_not_match_saved_PCA_projection")
    if not np.allclose(phi[fit_mask,1:].mean(axis=0),0.,atol=1e-6):
        raise AssetError("PCA_training_features_not_centered")
    return {"audit":audit,"audit_path":audit_path,"feature_path":feature_path,"pca_path":pca_path,
            "phi":{qid:phi[i].copy() for i,qid in enumerate(ids.tolist())}}

def _load_common(root):
    required=[root/"configs/experiment_v1.json",root/"configs/pool_v1.json",root/"configs/aggregator_v1.json",
              root/"data/derived/split_manifest.json",root/"protocol/E0_LOCK.json",root/"results/e1/e1_diagnostics.json"]
    absent=[str(p.relative_to(root)) for p in required if not p.is_file()]
    if absent: raise AssetError("missing_common_assets:"+",".join(absent))
    config=_json(required[0]); pool=_json(required[1])["models"]; agg=ModelSpec(**_json(required[2])); agg.validate()
    locked=_json(required[3]); e0=_json(required[4]); e1=_json(required[5])
    if config.get("n_models")!=6 or config.get("k")!=3: raise AssetError("N6_K3_protocol_changed")
    if e0.get("status") not in {"PASS","E0_PASS","E0_PASS_AND_LOCK","LOCKED"} or e1.get("status")!="PASS":
        raise AssetError("E0_and_real_E1_PASS_are_required")
    records=locked.get("records")
    canonical=locked.get("canonical_question_fingerprints",{})
    if not isinstance(records,list): raise AssetError("locked_records_list_missing")
    identifiers=[r["question_id"] for r in records]
    if len(set(identifiers))!=len(identifiers) or set(identifiers)!=set(canonical) or len(set(canonical.values()))!=len(records):
        raise AssetError("global_question_or_canonical_split_leakage")
    for stage,count in STAGES.items():
        assigned=[r for r in records if r["split"]==stage]
        if len(assigned)!=count or Counter(r["task"] for r in assigned)!={"gsm8k":count//2,"bbh":count//2}:
            raise AssetError("full_scale_task_stage_counts_changed:"+stage)
    if len(pool)!=6 or [m.get("index") for m in pool]!=list(range(6)) or len({m["alias"] for m in pool})!=6:
        raise AssetError("frozen_six_model_order_changed")
    specs={}
    for model in pool:
        path=root/model["config"]; spec=ModelSpec(**_json(path)); spec.validate()
        if (spec.model_id,spec.revision,configuration_fingerprint(spec))!=(model["model_id"],model["revision"],model["fingerprint"]):
            raise AssetError("pool_model_full_fingerprint_changed")
        specs[model["alias"]]=spec
    features=_load_features(root,locked,config)
    # Hash all pinned split-source assets and receipts, without borrowing labels.
    for source in locked.get("source_files",[]):
        _reference(root,source,"locked_split_source")
        if source.get("receipt_path"):
            _reference(root,{"path":source["receipt_path"],"sha256":source["receipt_sha256"]},"locked_source_receipt")
    selected={r["question_id"]:r for r in records if r["split"] in STAGES}
    questions={}
    for name in ("available_records.jsonl","gsm8k_questions.jsonl"):
        path=root/"data/derived"/name
        if not path.is_file(): raise AssetError("missing_original_reference_source:"+name)
        for row in _read_jsonl(path):
            qid=row.get("question_id")
            if qid not in selected: continue
            if qid in questions: raise AssetError("duplicate_original_question_source")
            assigned=selected[qid]
            if any(row.get(k)!=assigned.get(k) for k in ("task","subtask","source_split")):
                raise AssetError("original_reference_task_or_source_split_changed")
            normalized=" ".join(unicodedata.normalize("NFKC",row["question"]).split())
            if _text_sha(normalized)!=canonical[qid]: raise AssetError("original_question_fingerprint_changed")
            score_answer(row["task"],row["reference"],row["reference"],subtask=row.get("subtask"),choices=row.get("choices"))
            questions[qid]=row
    if set(questions)!=set(selected): raise AssetError("missing_online_original_question_references")
    budget=config["budget"]
    if (config.get("feedback",{}).get("member_feedback_available_when_same_verifier_revealed") is not True
        or config["feedback"].get("mode")!="verifiable_reference_revealed_at_authorized_event"):
        raise AssetError("verifiable_feedback_permission_not_locked")
    return {"root":root,"config":config,"pool":pool,"specs":specs,"aggregator":agg,"split":locked,
        "e0":e0,"e1":e1,"features":features,"questions":questions,"budget":budget}

def _jobs(common,stage,missing):
    root=common["root"]; records=[r for r in common["split"]["records"] if r["split"]==stage]
    assigned={r["question_id"]:r for r in records}
    jobs={}; source_paths=[]; member_map={}; aggregate_map={}
    for entry in common["pool"]:
        path=root/"data/jobs"/stage/f"members_{entry['alias'].lower()}.jsonl"
        if not path.is_file(): missing.add("missing_member_job_file:"+entry["alias"]); continue
        source_paths.append(path)
        spec=common["specs"][entry["alias"]]
        for job in _read_jsonl(path):
            qid=job.get("question_id")
            if qid not in assigned or job.get("split")!=stage: raise AssetError("non_stage_E1_or_sealed_member_job")
            q=common["questions"][qid]
            expected=make_member_job(question_id=qid,task=q["task"],subtask=q["subtask"],question=q["question"],
                model_alias=entry["alias"],seed=42,model_spec=spec,split=stage,sample_index=0,sampling_control=False)
            validate_member_job(job,spec)
            if job!=expected: raise AssetError("member_job_differs_from_latest_protocol_or_original_question")
            key=(qid,entry["alias"])
            if key in member_map or job["job_id"] in jobs: raise AssetError("duplicate_member_job")
            member_map[key]=job; jobs[job["job_id"]]=job
    path=root/"data/jobs"/stage/"aggregations.jsonl"
    if not path.is_file(): missing.add("missing_real_aggregate_jobs")
    else:
        source_paths.append(path)
        aliases={m["alias"] for m in common["pool"]}
        for job in _read_jsonl(path):
            validate_job(job,common["aggregator"],require_token_policy=True)
            qid=job["question_id"]
            if qid not in assigned or job["split"]!=stage or job["seed"]!=42:
                raise AssetError("non_stage_E1_or_sealed_aggregate_job")
            q=common["questions"][qid]
            if job["source_hashes"]["question_sha256"]!=_text_sha(q["question"]):
                raise AssetError("aggregate_job_question_changed")
            if job["task"]!=q["task"] or job["subtask"]!=q.get("subtask") or len(job["members"])!=3 or not set(job["members"])<=aliases or job["member_provenance"].get("control"):
                raise AssetError("aggregate_job_is_not_one_real_three_member_team")
            key=(qid,tuple(job["members"]))
            if key in aggregate_map or job["job_id"] in jobs: raise AssetError("duplicate_real_aggregate_job")
            aggregate_map[key]=job; jobs[job["job_id"]]=job
    if len(member_map)!=len(records)*6: missing.add("incomplete_member_jobs")
    if len(aggregate_map)!=len(records)*20: missing.add("incomplete_real_aggregate_jobs")
    return records,assigned,jobs,member_map,aggregate_map,source_paths

def _stage(common,stage):
    missing=set(); records,assigned,jobs,member_jobs,aggregate_jobs,job_paths=_jobs(common,stage,missing)
    root=common["root"]
    member_paths=[root/"results/raw"/stage/f"members_{m['alias'].lower()}.jsonl" for m in common["pool"]]
    aggregate_path=root/"results/raw"/stage/"aggregations.jsonl"
    success,costs,calls,assets,ignored=_read_stores([*member_paths,aggregate_path],assigned,missing)
    if ignored: missing.add("execution_store_contains_E1_sealed_or_other_stage_rows")
    budget=common["budget"]
    member_call_cap=budget.get("member_call_cap",1); aggregate_call_cap=budget.get("aggregator_call_cap",1)
    member_prompt_cap=budget["input_cap_tokens"]; member_completion_cap=budget["member_max_output_tokens"]
    aggregate_prompt_cap=budget["aggregator_input_cap_tokens"]; aggregate_completion_cap=budget["aggregator_max_output_tokens"]
    # Preserve every billed attempt in the audit even when refusing over-cap cells.
    for entry in calls.values():
        row=entry["record"]; job=jobs.get(row["job_id"])
        if job is None: raise AssetError("output_call_has_no_matching_locked_job")
        shared=("question_id","task","subtask","split","seed","model_fingerprint","prompt_version")
        extra=("model_alias","sample_index","member_mode","sampling_control","source_question_sha256") if "model_alias" in job else ("members","source_hashes","member_provenance","candidate_token_policy","candidate_statistics")
        if any(row.get(key)!=job.get(key) for key in shared+extra):
            raise AssetError("actual_output_metadata_differs_from_its_exact_job")
    for job_id,cost in costs.items():
        job=jobs[job_id]; member="model_alias" in job
        call_cap=member_call_cap if member else aggregate_call_cap
        pcap=member_prompt_cap if member else aggregate_prompt_cap
        ccap=member_completion_cap if member else aggregate_completion_cap
        if cost["calls"]>call_cap: missing.add("actual_retry_calls_exceed_locked_cell_cap")
        if cost["prompt_tokens"]>cost["calls"]*pcap or cost["completion_tokens"]>cost["calls"]*ccap:
            missing.add("actual_tokens_exceed_locked_cell_cap")
    n=len(records); arrays={"question_ids":np.array([r["question_id"] for r in records]),
        "task_labels":np.array([r["task"] for r in records]),
        "phi":np.stack([common["features"]["phi"][r["question_id"]] for r in records])}
    values={key:np.full((n,width),np.nan) for width,keys in ((6,("member_scores","member_prompt_tokens","member_completion_tokens","member_execution_counts","member_wall_time_s")),
        (20,("aggregate_scores","aggregate_prompt_tokens","aggregate_completion_tokens","aggregate_execution_counts","aggregate_wall_time_s"))) for key in keys}
    combos=tuple(combinations(range(6),3)); aliases=[m["alias"] for m in common["pool"]]
    for qi,record in enumerate(records):
        qid=record["question_id"]; question=common["questions"][qid]; found={}
        for mi,alias in enumerate(aliases):
            job=member_jobs.get((qid,alias)); row=success.get(job["job_id"]) if job else None
            if row is None: continue
            _validate_result(row,common["specs"][alias])
            if row.get("sampling_control") is not False or row.get("member_mode")!="independent_sampled":
                raise AssetError("ordinary_member_output_has_wrong_sampling_role")
            found[alias]=row
            values["member_scores"][qi,mi]=score_answer(question["task"],row["result"]["text"],question["reference"],subtask=question.get("subtask"),choices=question.get("choices")).score
            cost=costs[row["job_id"]]
            if cost["known"]:
                for key,field in (("member_prompt_tokens","prompt_tokens"),("member_completion_tokens","completion_tokens"),("member_execution_counts","calls"),("member_wall_time_s","wall_time_s")):
                    values[key][qi,mi]=cost[field]
        for ci,combo in enumerate(combos):
            names=tuple(sorted(aliases[i] for i in combo))
            job=aggregate_jobs.get((qid,names)); row=success.get(job["job_id"]) if job else None
            if row is None or any(alias not in found for alias in names): continue
            _validate_result(row,common["aggregator"])
            _validate_aggregate_sources(row,question,{a:found[a] for a in names})
            values["aggregate_scores"][qi,ci]=score_answer(question["task"],row["result"]["text"],question["reference"],subtask=question.get("subtask"),choices=question.get("choices")).score
            cost=costs[row["job_id"]]
            if cost["known"]:
                for key,field in (("aggregate_prompt_tokens","prompt_tokens"),("aggregate_completion_tokens","completion_tokens"),("aggregate_execution_counts","calls"),("aggregate_wall_time_s","wall_time_s")):
                    values[key][qi,ci]=cost[field]
    missing_members=int(np.isnan(values["member_scores"]).sum()); missing_aggregates=int(np.isnan(values["aggregate_scores"]).sum())
    if missing_members: missing.add("incomplete_real_member_outputs")
    if missing_aggregates: missing.add("incomplete_real_aggregate_outputs")
    if any(not np.isfinite(v).all() for v in values.values()): missing.add("missing_actual_scores_or_costs_never_imputed")
    actual=_sum_cost(entry["cost"] for entry in calls.values())
    report={"stage":stage,"status":"PARTIAL" if missing else "VALIDATED","questions":n,
        "required_member_cells":n*6,"required_aggregate_cells":n*20,"missing_member_cells":missing_members,
        "missing_aggregate_cells":missing_aggregates,"missing_requirements":sorted(missing),
        "actual_generation_cost":{"unique_calls":actual["calls"],"known_prompt_tokens":actual["prompt_tokens"],
            "known_completion_tokens":actual["completion_tokens"],"known_call_wall_time_sum_s":actual["wall_time_s"],
            "unknown_usage_calls":actual["unknown_usage_calls"],"unknown_wall_time_calls":actual["unknown_wall_time_calls"],
            "total_tokens":actual["prompt_tokens"]+actual["completion_tokens"] if actual["known"] else None},
        "e2_executed":False,"research_complete":False}
    if missing: return report,None,None
    arrays.update({k:v if k.endswith(("_scores","_wall_time_s")) else v.astype(np.int64) for k,v in values.items()})
    feat=common["features"]; audit=feat["audit"]; agg=common["aggregator"]
    sources=[{"kind":"member_outputs",**_ref(p)} for p in member_paths]
    sources += [{"kind":"aggregate_outputs",**_ref(aggregate_path)},
        {"kind":"scorer_source",**_ref(Path(scoring_module.__file__))},
        {"kind":"feature_audit",**_ref(feat["audit_path"])},
        {"kind":"frozen_features",**_ref(feat["feature_path"])}]
    sources += [{"kind":"immutable_generation_jobs",**_ref(p)} for p in job_paths]
    manifest={"schema_version":1,"split":stage,"matrix_sha256":None,"combinations":[list(c) for c in combos],
        "model_pool":[{"model_id":common["specs"][alias].model_id,"revision":common["specs"][alias].revision} for alias in aliases],
        "cache_revision":"local_frozen_generation_v1",
        "aggregator":{"model_id":agg.model_id,"revision":agg.revision,"fingerprint":configuration_fingerprint(agg),
            "prompt_sha256":digest({"system":SYSTEM_PROMPT,"prompt_version":AGGREGATION_PROMPT_VERSION,"token_policy":TOKEN_POLICY_VERSION})},
        "scorer":{"version":SCORER_VERSION,"no_model_calls":True},
        "e0_lock":_ref(root/"protocol/E0_LOCK.json"),"e1_gate":_ref(root/"results/e1/e1_diagnostics.json"),
        "split_manifest":_ref(root/"data/derived/split_manifest.json"),"pool_config":_ref(root/"configs/pool_v1.json"),
        "pca_artifact":_ref(feat["pca_path"]),
        "pca":{"fit_split":"online_calibration","fit_question_ids":audit["fit_question_ids"],
            "encoder_id":audit["encoder_id"],"encoder_revision":audit["encoder_revision"]},
        "question_fingerprints":[common["split"]["canonical_question_fingerprints"][r["question_id"]] for r in records],
        "feedback_permissions":{"mode":"verifiable","generated_member_scores_on_reference_reveal":True},
        "sources":sources,"single_call_per_cell":member_call_cap==1 and aggregate_call_cap==1,
        "all_combinations_executed":True,"real_frozen_llm_aggregation":True,"aggregate_scores_from_executed_aggregator":True,
        "e1_isolated":True,"sealed_test_isolated":True,"failure_costs_included":True,
        "actual_generation_cost":report["actual_generation_cost"],
        "cost_scope":{"hard_budget_units":["tokens","calls","labels"],
            "wall_time_arrays":"sum of all known per-cell execution attempts, including failures",
            "controller_cpu_wall_time_included":False,
            "resource_claim_boundary":"Engineering token/call/label budgets alone do not establish complete-resource savings; E3 must measure controller CPU tradeoffs"},
        "e2_executed":False,"research_complete":False}
    _validate_stage_bindings(manifest,common["e0"],common["e1"])
    _validate_locked_split(common["split"],arrays["question_ids"].tolist(),manifest["question_fingerprints"],stage,manifest["pca"])
    return report,arrays,manifest

def _immutable_publish(source,destination):
    if destination.exists():
        if file_sha(source)!=file_sha(destination): raise AssetError("immutable_output_conflict:"+destination.name)
        source.unlink()
    else: os.replace(source,destination)

def assemble_online_matrices(root,stage,output_dir,*,software_test_only=False):
    """Validate all requested stages before publishing; never execute E2."""
    root=Path(root).resolve(); output=Path(output_dir).resolve()
    stages=tuple(STAGES) if stage=="both" else (stage,)
    if any(s not in STAGES for s in stages): raise ValueError("Only online_calibration/online_stream are allowed, never E1/sealed")
    output.mkdir(parents=True,exist_ok=True)
    result={"status":"PARTIAL","stages":{},"missing_requirements":[],"e2_executed":False,"research_complete":False,
            "software_test_only":bool(software_test_only)}
    staged=[]
    try:
        common=_load_common(root)
        prepared={}
        for name in stages:
            report,arrays,manifest=_stage(common,name)
            result["stages"][name]=report
            if arrays is not None: prepared[name]=(arrays,manifest)
        if len(prepared)!=len(stages):
            result["missing_requirements"]=["some_requested_stages_are_incomplete"]
        else:
            environments={}
            token=uuid.uuid4().hex
            for name in stages:
                arrays,manifest=prepared[name]
                matrix_tmp=output/f".{name}.{token}.npz"
                manifest_tmp=output/f".{name}.{token}.manifest.json"
                np.savez_compressed(matrix_tmp,**arrays)
                manifest["matrix_sha256"]=file_sha(matrix_tmp)
                if software_test_only:
                    manifest["software_test_only"]=True
                    manifest["real_frozen_llm_aggregation"]=False
                manifest_tmp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
                staged.extend((matrix_tmp,manifest_tmp))
                if software_test_only:
                    environments[name]=MatrixEnvironment.for_software_test(arrays,split=name)
                else:
                    environments[name]=MatrixEnvironment.from_npz(matrix_tmp,manifest_tmp,expected_split=name)
            if len(stages)==2:
                _validate_pair(environments["online_stream"],environments["online_calibration"],"none")
            for name in stages:
                matrix=output/f"{name}.npz"; manifest=output/f"{name}.manifest.json"
                arrays,metadata=prepared[name]
                matrix_tmp=next(p for p in staged if p.name==f".{name}.{token}.npz")
                manifest_tmp=next(p for p in staged if p.name==f".{name}.{token}.manifest.json")
                _immutable_publish(matrix_tmp,matrix); _immutable_publish(manifest_tmp,manifest)
                result["stages"][name].update(status="SOFTWARE_TEST_ONLY" if software_test_only else "READY",
                    loader_verified=not software_test_only,software_contract_verified=bool(software_test_only),
                    matrix=_ref(matrix),manifest=_ref(manifest))
            result["status"]="SOFTWARE_TEST_ONLY" if software_test_only else "REAL_MATRICES_READY" if stage=="both" else "READY_SINGLE_STAGE"
    except (ValueError,KeyError,FileNotFoundError) as error:
        result["missing_requirements"].append(str(error))
    finally:
        for path in staged:
            if path.exists(): path.unlink()
    receipt=output/"matrix_preparation_receipt.json"
    # The receipt is the final commit marker; matrices themselves are immutable.
    temporary=output/(".matrix_receipt."+uuid.uuid4().hex+".json")
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    os.replace(temporary,receipt)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",default=str(PROJECT))
    parser.add_argument("--stage",choices=tuple(STAGES)+("both",),default="both")
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args(argv)
    result=assemble_online_matrices(args.root,args.stage,args.output_dir)
    print(json.dumps({"status":result["status"],"stages":{k:v["status"] for k,v in result["stages"].items()},
        "missing_requirements":result["missing_requirements"],"e2_executed":False,"research_complete":False},ensure_ascii=False))
    return 0 if result["status"] in {"REAL_MATRICES_READY","READY_SINGLE_STAGE"} else 2

if __name__=="__main__": raise SystemExit(main())

