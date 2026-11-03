"""Create immutable pre-feature and full final releases from accepted evidence.

Selection uses only the independently predeclared method/seed/budget roster.
No quality values, sealed question text or final labels are consulted.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.frozen_evaluation import (SEEDS,FINAL_GROUP_PLAN_SHA256,FINAL_SOURCES,
    NEIGHBOR_FINAL_SOURCES,FINAL_GENERATION_SOURCES,release_conditions_from_plan,
    validate_preparation_release,validate_final_release)
from budget_router.stage_validation import (read_json,artifact_reference,verified_reference,
    canonical_digest,sha_file)

STAGE_CONDITIONS={
    'e2':'E2_iid_random_combo','e3':'E3_iid_random_combo',
    'e4':'E4_mandatory_shared_permissions','flex_reference':'E4_flex_controlled_reference',
    'e5_paid_prior':'paid_member_prior_then_combo',
}
RELEASE_SOURCES=tuple(dict.fromkeys((*FINAL_SOURCES,*NEIGHBOR_FINAL_SOURCES,*FINAL_GENERATION_SOURCES,
    'scripts/build_final_release.py','scripts/prepare_final_features.py','src/budget_router/encoding.py',
    'src/budget_router/e5_validation.py','src/budget_router/flex_training.py',
    'src/budget_router/flex_simulation.py','src/budget_router/flex_validation.py','scripts/run_flex_reference.py',
    'scripts/prepare_second_aggregator.py','scripts/download_second_aggregator.py')))


def select_policy_groups(plan,roster):
    """Pure metadata join; unused E5 shifts and duplicated upstream controls stay evidence."""
    if (plan.get('schema')!='frozen_final_policy_groups_plan_v1' or plan.get('seeds')!=list(SEEDS)
            or plan.get('final_questions')!=200):
        raise ValueError('Original five-seed final method plan required')
    expected=plan.get('expected_groups',[])
    if not expected or len({g.get('method') for g in expected})!=len(expected):
        raise ValueError('Predeclared final method groups are missing or duplicated')
    indexed={}
    for row in roster:
        key=tuple(row.get(k) for k in ('condition_id','strategy','seed','label_rate','feedback_delay'))
        if key in indexed:raise ValueError('Duplicated frozen evidence roster entry')
        indexed[key]=row
    result=[]
    for group in expected:
        source=group.get('source_stage')
        if source not in STAGE_CONDITIONS:raise ValueError('Unknown predeclared final source stage')
        entries=[]
        for seed in plan['seeds']:
            key=(STAGE_CONDITIONS[source],group['strategy'],seed,group['label_rate'],group['feedback_delay'])
            if key not in indexed:raise ValueError('Missing frozen checkpoint for predeclared final group: '+group['method']+' seed'+str(seed))
            row=indexed[key]
            if any(not isinstance(row.get(name),dict) or not row[name].get('path') or not row[name].get('sha256')
                    for name in ('policy_checkpoint','episode_receipt','grid_plan')):
                raise ValueError('Frozen checkpoint roster has incomplete artifact references')
            entries.append({'seed':seed,'checkpoint':deepcopy(row['policy_checkpoint']),
                'episode_receipt':deepcopy(row['episode_receipt']),'grid_plan':deepcopy(row['grid_plan'])})
        result.append({**deepcopy(group),'checkpoints':entries})
    return result


def _output_path(path):
    output=Path(path).resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Release output must stay inside the experiment directory')
    if output.exists():raise FileExistsError('An immutable release already exists: '+str(output))
    return output


def publish_validated_release(path,value,validator):
    """Validate a create-only candidate before atomically publishing accepted JSON."""
    output=_output_path(path);output.parent.mkdir(parents=True,exist_ok=True)
    candidate=output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.candidate')
    try:
        with candidate.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        validated=validator(candidate)
        del validated
        if output.exists():raise FileExistsError('Another worker published this immutable release')
        os.rename(candidate,output)
    finally:
        if candidate.exists():candidate.unlink()
    return artifact_reference(output)


def _locked_group_plan():
    path=ROOT/'protocol/final_policy_groups_v1.json'
    if sha_file(path)!=FINAL_GROUP_PLAN_SHA256:raise ValueError('Original independent final group plan changed')
    plan=read_json(path)
    if len(plan.get('expected_groups',[]))!=216:raise ValueError('All original 216 final policy groups are required')
    return path,plan


def preparation_value(e5_pre_final_gate,*,e4_full_gate=None,evaluation_seed=20261031):
    if isinstance(evaluation_seed,bool) or not isinstance(evaluation_seed,int) or evaluation_seed<0:
        raise ValueError('Evaluation seed must be an explicit nonnegative integer')
    from budget_router.e5_validation import authorize_e5_final_batch
    gate_path=Path(e5_pre_final_gate).resolve();unverified=read_json(gate_path)
    run_plan=read_json(verified_reference(unverified['run_plan']));inputs=run_plan['inputs']
    ticket=authorize_e5_final_batch(gate_path,inputs);gate=ticket.gate
    if e4_full_gate is not None and artifact_reference(e4_full_gate)!=gate['e4_full_gate']:
        raise ValueError('Requested E4 full gate differs from the accepted E5 parent')
    group_path,group_plan=_locked_group_plan()
    groups=select_policy_groups(group_plan,gate['frozen_policy_checkpoints'])
    if len(groups)!=216 or sum(len(g['checkpoints']) for g in groups)!=1080:
        raise ValueError('The entire frozen 216-group/five-seed final roster is required')
    assets=inputs['assets'];locked=inputs['locked_provenance']
    artifacts={'config':deepcopy(assets['config']),
        **{name:deepcopy(locked[name]) for name in ('split_manifest','pca_artifact','pool_config')},
        'encoder_source_receipt':artifact_reference(ROOT/'artifacts/audit/encoder_model_download.json'),
        'final_policy_plan':artifact_reference(group_path)}
    robust=read_json(verified_reference(gate['e5_plan']))
    configs={name:artifact_reference(ROOT/name) for name in sorted({c['aggregator_config'] for c in robust['final_frozen_policy_conditions']})}
    conditions=release_conditions_from_plan(robust,configs)
    online_manifest=read_json(verified_reference(inputs['assets']['online_manifest']));replay_sources={}
    for kind in ('frozen_features','feature_audit'):
        candidates=[r for r in online_manifest['sources'] if r.get('kind')==kind]
        if len(candidates)!=1:raise ValueError('Exact original frozen feature/audit source required for calibration replay')
        record=deepcopy(candidates[0]);source=Path(record['path'])
        if not source.is_absolute():source=Path(inputs['assets']['online_manifest']['path']).parent/source
        record['path']=str(source.resolve());replay_sources[kind]=artifact_reference(verified_reference(record))
    split=read_json(verified_reference(artifacts['split_manifest']))
    original={row['path']:row for row in split['source_files']};questions=[]
    for relative in ('data/derived/available_records.jsonl','data/derived/gsm8k_questions.jsonl'):
        ref=artifact_reference(ROOT/relative)
        if relative not in original or ref['sha256']!=original[relative]['sha256']:
            raise ValueError('Public question source differs from the original locked split')
        questions.append(ref)
    value={'schema':'pre_final_feature_release_v1','status':'PRE_FINAL_FEATURES_RELEASED','software_test_only':False,
        'inputs':deepcopy(inputs),'artifacts':artifacts,'question_sources':questions,
        'e4_full_gate':deepcopy(gate['e4_full_gate']),'e5_pre_final_gate':artifact_reference(gate_path),
        'algorithm_sources':deepcopy(inputs['algorithm_sources']),
        'final_sources':{name:artifact_reference(ROOT/name) for name in RELEASE_SOURCES},
        'policy_groups':groups,'evaluation_seed':evaluation_seed,
        'robustness_plan':deepcopy(gate['e5_plan']),'aggregator_configs':configs,'evaluation_conditions':conditions,
        'calibration_encoder_replay_sources':replay_sources,
        'pending_feedback_policy':'never_flush_or_update','final_label_use':'evaluation_only_after_decisions_frozen',
        'builder_read_sealed_question_text_or_labels':False,'research_complete':False,
        'selection_basis':'Every predeclared method, budget, delay and seed; receipt metadata only; no quality-based selection'}
    value['preparation_release_sha256']=canonical_digest(value)
    return value,{'e5':ticket}


def complete_value(preparation_release,features,feature_manifest):
    pre_path=Path(preparation_release).resolve();pre=read_json(pre_path)
    core={k:v for k,v in pre.items() if k!='preparation_release_sha256'}
    if (pre.get('schema')!='pre_final_feature_release_v1' or pre.get('status')!='PRE_FINAL_FEATURES_RELEASED'
            or pre.get('software_test_only') is not False or pre.get('preparation_release_sha256')!=canonical_digest(core)):
        raise ValueError('Immutable pre-feature preparation release is required')
    split=read_json(verified_reference(pre['artifacts']['split_manifest']))
    final=[r for r in split['records'] if r['split']=='sealed_test'];canonical=split['canonical_question_fingerprints']
    value={'schema':'frozen_final_release_v1','status':'FINAL_RELEASED','software_test_only':False,
        'artifacts':{**{name:deepcopy(pre['artifacts'][name]) for name in ('config','split_manifest','pca_artifact','pool_config')},
            'features':artifact_reference(features),'feature_manifest':artifact_reference(feature_manifest)},
        'preparation_release':artifact_reference(pre_path),'algorithm_sources':deepcopy(pre['algorithm_sources']),
        'final_sources':deepcopy(pre['final_sources']),
        'stage_evidence':{'E4':deepcopy(pre['e4_full_gate']),'E5_PRE_FINAL':deepcopy(pre['e5_pre_final_gate'])},
        'policy_groups':deepcopy(pre['policy_groups']),'evaluation_seed':pre['evaluation_seed'],
        'robustness_plan':deepcopy(pre['robustness_plan']),'aggregator_configs':deepcopy(pre['aggregator_configs']),
        'evaluation_conditions':deepcopy(pre['evaluation_conditions']),
        'calibration_encoder_replay_sources':deepcopy(pre['calibration_encoder_replay_sources']),
        'final_questions':[{'question_id':r['question_id'],'task':r['task'],'canonical_fingerprint':canonical[r['question_id']]} for r in final],
        'pending_feedback_policy':'never_flush_or_update','final_label_use':'evaluation_only_after_decisions_frozen',
        'builder_read_sealed_question_text_or_labels':False,'research_complete':False}
    value['release_sha256']=canonical_digest(value)
    return value


def build_release(*,kind,output,e5_pre_final_gate=None,e4_full_gate=None,evaluation_seed=20261031,
                  preparation_release=None,features=None,feature_manifest=None):
    output=_output_path(output)
    if kind=='preparation':
        if e5_pre_final_gate is None:raise ValueError('--e5-pre-final-gate is required for preparation')
        value,batches=preparation_value(e5_pre_final_gate,e4_full_gate=e4_full_gate,evaluation_seed=evaluation_seed)
        reference=publish_validated_release(output,value,
            lambda candidate:validate_preparation_release(candidate,_batch_authorizations=batches))
    elif kind=='complete':
        if preparation_release is None or features is None or feature_manifest is None:
            raise ValueError('--preparation-release, --features and --feature-manifest are required for complete')
        value=complete_value(preparation_release,features,feature_manifest)
        reference=publish_validated_release(output,value,validate_final_release)
    else:raise ValueError('Unknown final release kind')
    return {'status':value['status'],'artifact':reference,'groups':len(value['policy_groups']),
        'frozen_policy_checkpoints':sum(len(g['checkpoints']) for g in value['policy_groups']),
        'generation_conditions':len(value['evaluation_conditions']),'research_complete':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',required=True,choices=('preparation','complete'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--e5-pre-final-gate',type=Path)
    parser.add_argument('--e4-full-gate',type=Path)
    parser.add_argument('--evaluation-seed',type=int,default=20261031)
    parser.add_argument('--preparation-release',type=Path)
    parser.add_argument('--features',type=Path)
    parser.add_argument('--feature-manifest',type=Path)
    args=parser.parse_args(argv)
    result=build_release(**vars(args))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
