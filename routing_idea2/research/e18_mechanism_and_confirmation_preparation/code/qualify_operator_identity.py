"""Reconstruct the original typed V2 operator using metadata and file hashes.

No experiment outcome archive, generated answer, sealed payload or model weight
is opened. The independent statistical check uses synthetic reward arrays.
Outputs are append-only records in this new E18 source directory by default.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


EXPECTED_OPERATOR_FP = "9ef6c02bc9fb7f21f9ef5a2f0d39edfee80e7162c8d6edec82ee32c8cecb440e"
EXPECTED_MODEL_FP = "32aed6079298ebcf8eee99add17dd5c4a6828a504f13dd5d9236b234776861d3"
EXPECTED_LOCK_SHA = "66d77c2ab5eef0c6a360cfe2d8df01429369927c96fe3ae7b6a63697f4fc09b0"
SOURCE = Path(__file__).resolve()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_digest(value):
    encoded=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def differing_fields(actual, expected, prefix=""):
    """Report only field paths, keeping full source metadata in its owner file."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences=[]
        for name in sorted(set(actual)|set(expected)):
            path=prefix+"/"+name
            if name not in actual or name not in expected:
                differences.append(path)
            else:
                differences.extend(differing_fields(actual[name],expected[name],path))
        return differences
    return [] if actual==expected else [prefix]


def qualify_operator(assets):
    cpu,wall=time.process_time(),time.perf_counter()
    sys.path[:0]=[str(assets/"src"),str(assets/"scripts/v2")]
    from budget_router.inference import ModelSpec, configuration_fingerprint
    from budget_router_v2.scoring import SCORER_VERSION, BBH_TASKS
    from stage1_review_common import operator_identity
    from typed_operator_contract import validate_typed_freeze_header
    import typed_aggregation_execution as typed

    lock_path=assets/"results/decision_feedback_v2/s1_typed_aggcal_generation42_9ef6c02bc9fb/OPERATOR_FREEZE_LOCK_v1.json"
    require(sha256(lock_path)==EXPECTED_LOCK_SHA,"Original active operator lock changed")
    freeze=read_json(lock_path)
    frozen_identity=validate_typed_freeze_header(freeze)
    candidate_path=assets/"configs/decision_feedback_v2/aggregator_qwen3_8b_candidate_v1.json"
    candidate=ModelSpec(**read_json(candidate_path))
    base=assets/"artifacts/private/routing_redesign_20261104/stage1_run_20261104"
    original_logical=ModelSpec(**read_json(base/"logical_config.json"))
    model_fp=configuration_fingerprint(candidate)
    require(model_fp==EXPECTED_MODEL_FP==freeze["logical_model_config_fingerprint"],"Logical model profile differs")
    require(asdict(candidate)==asdict(original_logical),"Active candidate and original logical model differ")
    without_path=asdict(candidate)
    without_path.pop("model_path")
    require(without_path==frozen_identity["model_config_without_loading_path"],"Path-independent model config differs")

    # This historical base factory has the original untyped prompt. The exact
    # typed extension follows prepare_typed_aggregator_candidate.py129-138.
    identity=operator_identity(assets,base)
    untyped_fp=canonical_digest(identity)
    identity["schema"]="cross_machine_typed_aggregator_candidate_identity_v1"
    identity["prompt"]["base_version"]=identity["prompt"]["version"]
    identity["prompt"]["version"]=typed.PROMPT_VERSION
    identity["prompt"]["contract_version"]=typed.CONTRACT_VERSION
    contracts={"gsm8k":typed.output_contract("gsm8k",None)}
    contracts.update({"bbh/"+subtask:typed.output_contract("bbh",subtask) for subtask in sorted(BBH_TASKS)})
    require(len(contracts)==28,"All GSM8K and27 BBH contracts are required")
    identity["prompt"]["contracts_by_task_subtask"]=contracts
    identity["prompt"]["contracts_sha256"]=canonical_digest(contracts)
    adapter=assets/"scripts/v2/typed_aggregation_execution.py"
    scorer=assets/"src/budget_router_v2/scoring.py"
    identity["prompt"]["typed_adapter_source_sha256"]=sha256(adapter)
    identity["scorer"]={"version":SCORER_VERSION,"source_sha256":sha256(scorer)}
    differences=differing_fields(identity,frozen_identity)
    require(not differences,"Current typed operator differs at "+str(differences))
    require(canonical_digest(identity)==EXPECTED_OPERATOR_FP==freeze["aggregator_fingerprint"],"Complete operator fingerprint differs")

    original_sources=[]
    for relative,expected in frozen_identity["original_source_sha256"].items():
        actual=sha256(assets/relative)
        require(actual==expected,"Original scientific source changed "+relative)
        original_sources.append(dict(path=relative,sha256=actual))
    require(len(original_sources)==76,"Original76-source closure differs")
    tokenizer_files=[]
    directory=Path(candidate.model_path)
    for name,expected in frozen_identity["tokenizer"]["files_sha256"].items():
        actual=sha256(directory/name)
        require(actual==expected,"Original tokenizer changed "+name)
        tokenizer_files.append(dict(name=name,sha256=actual))
    require(canonical_digest(contracts)==frozen_identity["prompt"]["contracts_sha256"],"Typed contract digest differs")

    roster=[]
    pool_path=assets/"configs/pool_v1.json"
    for member in read_json(pool_path)["models"]:
        path=assets/member["config"]
        spec=ModelSpec(**read_json(path))
        require(configuration_fingerprint(spec)==member["fingerprint"],"Member profile differs")
        require(spec.temperature==.7 and spec.top_p==.95 and spec.max_new_tokens==2048 and spec.enable_thinking is False,
                "Original sampled member protocol differs")
        roster.append(dict(index=member["index"],config=member["config"],config_sha256=sha256(path),
                           model_fingerprint=member["fingerprint"],temperature=spec.temperature,top_p=spec.top_p,
                           maximum_total_tokens=spec.max_input_tokens+spec.max_new_tokens))
    require([row["index"] for row in roster]==list(range(6)),"Original ordered six-member roster required")
    dependencies=["src/budget_router/inference.py","src/budget_router/execution.py",
                  "src/budget_router_v2/scoring.py","scripts/v2/stage1_review_common.py",
                  "scripts/v2/prepare_typed_aggregator_candidate.py","scripts/v2/typed_aggregation_execution.py",
                  "scripts/v2/typed_operator_contract.py"]
    return dict(status="PASS",scope="metadata and file-hash qualification only",
                source_sha256=sha256(SOURCE),operator_lock_sha256=sha256(lock_path),
                candidate_config_sha256=sha256(candidate_path),base_model_fingerprint=model_fp,
                historical_untyped_operator_fingerprint=untyped_fp,
                active_typed_operator_fingerprint=canonical_digest(identity),
                identity_schema=identity["schema"],exact_fieldwise_match=True,different_field_paths=differences,
                typed_contracts=len(contracts),typed_contracts_sha256=canonical_digest(contracts),
                prompt_version=typed.PROMPT_VERSION,contract_version=typed.CONTRACT_VERSION,
                active_adapter_sha256=sha256(adapter),active_scorer=identity["scorer"],
                original_scientific_sources_verified=76,original_sources=original_sources,
                tokenizer_files=tokenizer_files,member_profiles=roster,
                aggregate_maximum_total_tokens=candidate.max_input_tokens+candidate.max_new_tokens,
                dependency_sources=[dict(path=name,sha256=sha256(assets/name)) for name in dependencies],
                guard_root_cause="Base ModelSpec fingerprint was compared to complete typed operator fingerprint; both identities are preserved and separately verified",
                numerical_model_runs=0,physical_llm_calls=0,model_weights_opened=0,
                real_outcome_arrays_opened=0,fresh_reward_reads=0,sealed_payload_reads=0,e17_writes=0,
                cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)


def qualify_statistics(code):
    """Independent formula reconstruction with80 GSM8K and40 BBH examples."""
    import numpy as np
    from scipy import stats
    cpu,wall=time.process_time(),time.perf_counter()
    source=code/"power_and_statistics.py"
    module_spec=importlib.util.spec_from_file_location("e18_statistical_audit_target",source)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rng=np.random.default_rng(1816)
    labels=np.array(["gsm8k"]*80+["bbh"]*40)
    candidate=rng.integers(0,2,size=(3,120))
    baseline=rng.integers(0,2,size=(3,120))
    actual=module.paired_question_test(candidate,baseline,labels)
    differences=(candidate-baseline).mean(axis=0)
    groups=[differences[labels==task] for task in ("gsm8k","bbh")]
    expected_estimate=sum(values.mean()/2 for values in groups)
    components=[values.var(ddof=1)/(4*len(values)) for values in groups]
    variance=sum(components)
    expected_df=variance**2/sum(component**2/(len(values)-1) for component,values in zip(components,groups))
    statistic=expected_estimate/math.sqrt(variance)
    expected_p=2*stats.t.sf(abs(statistic),expected_df)
    errors=dict(estimate=abs(actual["estimate"]-expected_estimate),
                standard_error=abs(actual["standard_error"]-math.sqrt(variance)),
                satterthwaite_degrees=abs(actual["degrees_of_freedom"]-expected_df),
                two_sided_p=abs(actual["p_value"]-expected_p))
    require(max(errors.values())<1e-12,"Independent paired statistics reconstruction failed")
    require(actual["original_questions"]==120 and actual["generation_records"]==360,"Generation repeats became independent questions")
    qualification=module.qualify()
    require(qualification["status"]=="PASS","Target numerical qualification failed")
    return dict(status="PASS",source_sha256=sha256(SOURCE),statistical_source_sha256=sha256(source),
                synthetic_seed=1816,original_questions=120,generation_records=360,
                task_counts=dict(gsm8k=80,bbh=40),task_weights=dict(gsm8k=.5,bbh=.5),
                independent_formula_errors=errors,maximum_error=max(errors.values()),target_qualification=qualification,
                formulas=dict(estimate="0.5*mean_G+0.5*mean_B",variance="0.25*s_G^2/n_G+0.25*s_B^2/n_B",
                              degrees="variance^2/sum(component_t^2/(n_t-1))",p_value="2*t_survival(abs(estimate/SE),df)"),
                evidence_scope="Synthetic implementation qualification only; no real-data interval reported",
                real_confidence_intervals=0,real_reward_arrays_opened=0,fresh_reward_reads=0,sealed_payload_reads=0,
                e17_writes=0,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--asset-root",type=Path,default=Path(os.environ.get("MAB_ASSET_ROOT",r"E:\语言模型组合路由")))
    parser.add_argument("--output-dir",type=Path,default=SOURCE.parent)
    args=parser.parse_args()
    outputs={"OPERATOR_IDENTITY_QUALIFICATION.json":None,"STATISTICAL_INDEPENDENT_QUALIFICATION.json":None}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    require(not any((args.output_dir/name).exists() for name in outputs),"Append-only qualification output already exists")
    # Load the sibling statistical module through its unchanged normal helpers.
    sys.path.insert(0,str(SOURCE.parent))
    outputs["OPERATOR_IDENTITY_QUALIFICATION.json"]=qualify_operator(args.asset_root)
    outputs["STATISTICAL_INDEPENDENT_QUALIFICATION.json"]=qualify_statistics(SOURCE.parent)
    for name,result in outputs.items():
        with (args.output_dir/name).open("x",encoding="utf-8") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    print(json.dumps({name:result["status"] for name,result in outputs.items()},ensure_ascii=False))


if __name__=="__main__":
    main()
