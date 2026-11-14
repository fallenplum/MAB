"""Reconcile measured E10 prefill and head costs with cached service accounting."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np

ASSETS=Path(r'E:\语言模型组合路由')
ROOT=ASSETS/'research/e10_input_prefill_matching_audit'
sys.path.insert(0,str(ASSETS/'research/e8_decision_focused_combination_learning/public/qualification'))
from e8_assets import load_historical
from itertools import combinations
TEAMS=tuple(combinations(range(6),3))


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duration(record):
    return (datetime.fromisoformat(record['completed_utc'].replace('Z','+00:00'))-
            datetime.fromisoformat(record['started_utc'].replace('Z','+00:00'))).total_seconds()


def main():
    head=read(ROOT/'private/ALL_PREDICTIONS_FREEZE_RECEIPT.json')
    features=read(ROOT/'private/features/FEATURE_FREEZE_RECEIPT.json')
    qualification=read(ROOT/'public/PREFILL_FEATURE_QUALIFICATION.json')
    records=[read(path) for path in sorted((ROOT/'private/features').glob('*/*.json'))]
    assert len(records)==606
    assert sum(row['input_tokens'] for row in records)==features['formal_input_tokens']
    assert abs(sum(row['gpu_milliseconds'] for row in records)/1000-features['formal_gpu_seconds'])<1e-9
    groups={}
    for variant in ('full','match'):
        selected=[row for row in records if row['variant']==variant]
        groups[variant]=dict(question_events=len(selected),input_tokens=sum(row['input_tokens'] for row in selected),
            gpu_event_elapsed_seconds=sum(row['gpu_milliseconds'] for row in selected)/1000,
            latency_wall_seconds_sum=sum(row['selector_feature_wall_seconds'] for row in selected),
            cpu_seconds_sum=sum(row['cpu_seconds'] for row in selected),
            warm_latency_wall_quantiles=dict(zip(('min','p50','p95','max'),
                np.quantile([row['selector_feature_wall_seconds'] for row in selected],[0,.5,.95,1]).tolist())))
    original_history=read(ASSETS/'research/e9_consistent_cost_sensitive_learning/public/COST.json')['shared_historical_feedback']
    prefill_attempts=[read(path) for path in sorted((ROOT/'private/logs').glob('PREFILL_ATTEMPT_*.json'))]
    head_attempt=read(ROOT/'private/logs/HEAD_TRAINING_ATTEMPT_001.json')
    evaluation_attempt=read(ROOT/'private/logs/EVALUATION_ATTEMPT_001.json')
    data,source_refs,_=load_historical(ASSETS)
    old=np.load(ASSETS/'research/e9_consistent_cost_sensitive_learning/private/ALL_E9_PREDICTIONS_FROZEN.npz',allow_pickle=False)
    new=np.load(ROOT/'private/ALL_E10_PREDICTIONS_FROZEN.npz',allow_pickle=False)
    service={}
    for name in ('A','R_full','R_match','GlobalFixed','NS'):
        action=(old if name in ('A','GlobalFixed','NS') else new)['actions__'+name]
        item={}
        for stem in ('member','aggregate'):
            for suffix in ('execution_counts','prompt_tokens','completion_tokens','wall_time_s'):
                values=data['cost'][stem+'_'+suffix]
                total=sum(float(values[g,i,list(TEAMS[a]) if stem=='member' else [a]].sum())
                    for g in range(3) for i,a in enumerate(action))
                item[stem+'_'+suffix]=total if suffix=='wall_time_s' else int(total)
        item['total_tokens']=sum(item[stem+'_'+suffix] for stem in ('member','aggregate')
            for suffix in ('prompt_tokens','completion_tokens'))
        item['scope']='cached counterfactual service footprint for600questions x3generations, no new answer calls'
        service[name]=item
    replay=read(ROOT/'public/CPU_EMBEDDING_REPLAY.json')
    inference=read(ROOT/'public/INFERENCE_CPU_PROFILE.json')
    cost=dict(status='RECONCILED_MEASURED_PREFILL_AND_CACHED_COSTS',
        new_gold=0,new_answer_generations=0,new_aggregations=0,new_teachers=0,
        new_prefill_formal_question_events=606,new_prefill_qualification_question_events=10,
        new_prefill_formal_batches=606,new_prefill_qualification_batches=6,
        formal={key:value for key,value in features.items() if key!='per_question_receipt_sha256'},
        formal_groups=groups,qualification=qualification,
        prefill_process_attempts=prefill_attempts,
        prefill_process_wall_seconds_sum=sum(duration(item) for item in prefill_attempts),
        prefill_hardware_failures=sum(item['exit_code']!=0 for item in prefill_attempts),
        head_training=dict(actual_new_heads=len(head['heads']),main_full=5,matched_coverage=5,content_controls=15,
            cpu_seconds_sum=sum(row['cpu_seconds'] for row in head['heads']),
            solver_wall_seconds_sum=sum(row['wall_seconds'] for row in head['heads']),
            preprocessing_wall_seconds_sum=sum(row['preprocessing_wall_seconds'] for row in head['heads']),
            evaluation_head_batch_wall_seconds_sum=sum(row['evaluation_head_batch_wall_seconds'] for row in head['heads']),
            whole_process_wall_seconds=duration(head_attempt),hyperparameter_searches=0),
        scoring_whole_process_wall_seconds=duration(evaluation_attempt),
        common_historical_supervision=original_history,cached_service_costs=service,
        auxiliary_input_qualification=dict(original_MiniLM_questions=600,gpu_used=False,
            load_wall_seconds=replay['model_load_wall_seconds'],encoding_wall_seconds=replay['encode_wall_seconds']),
        frozen_head_cpu_inference_profile=inference,
        online_increment=dict(question_representation='one full-canonical-text Qwen3 prefill to block24 plus fold-frozen PCA/Ridge',
            measured_warm_full_mean_latency=groups['full']['latency_wall_seconds_sum']/600,
            measured_mean_input_tokens=groups['full']['input_tokens']/600,
            measured_head_cpu_mean_latency=inference['mean_latency_seconds'],
            additive_warm_prefill_and_head_mean_latency=groups['full']['latency_wall_seconds_sum']/600+inference['mean_latency_seconds'],
            cold_model_load_seconds=qualification['model_load_wall_seconds'],
            training_features_cacheable=True,three_generation_scores_reuse_one_question_feature=True,
            full_and_match_models_change_their_own_training_features=True,
            prefill_reuse_with_answer_generation_assumed=False),
        timing_limits=dict(gpu_time='CUDA event elapsed interval including host scheduling, not pure kernel-time sum',
            cached_source_wall='historical call wall proxy, hardware differs and not a current measured GPU benchmark',
            full_research_pipeline_cpu_seconds=None,full_research_pipeline_elapsed_wall_seconds=None,
            head_training_peak_ram_bytes=None,qualification_exclusive_cpu_seconds=None,
            reason='parallel literature/input/audit/tool operations lack one complete exclusive pipeline timer'),
        provenance=dict(protocol_sha256=sha(ROOT/'public/E10_PROTOCOL.json'),
            feature_receipt_sha256=sha(ROOT/'private/features/FEATURE_FREEZE_RECEIPT.json'),
            head_receipt_sha256=sha(ROOT/'private/ALL_PREDICTIONS_FREEZE_RECEIPT.json'),source_refs=source_refs,
            generator_sha256=sha(Path(__file__))))
    output=ROOT/'public/COST.json'
    assert not output.exists()
    output.write_text(json.dumps(cost,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',formal_prefill=606,formal_tokens=features['formal_input_tokens'],
        formal_heads=len(head['heads']),groups=groups,process_wall=cost['prefill_process_wall_seconds_sum']),ensure_ascii=False))


if __name__=='__main__':
    main()
