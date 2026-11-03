"""Validate final evidence, compute fixed comparisons, and build a safe delivery.

Actual delivery requires a complete verified acceptance or a verified E1 stop.
This script does not execute models, update policies, or publish to GitHub.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from budget_router.stage_validation import (read_json,read_jsonl,atomic_json,immutable_json,artifact_reference,
 verified_reference,canonical_digest,sha_file)
SEEDS=(42,137,2024,2025,2026)
CONDITIONS=('primary_generation42','generation137','generation2024','second_aggregator_generation42')
METRICS=('quality','executions','labels','tokens','training_controller_cpu_s','training_controller_wall_s','final_source_wall_s')


def finite(value):return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


def _group_contract(release):
 groups=release.get('policy_groups',[])
 if len(groups)!=216 or len({g.get('method') for g in groups})!=216:raise ValueError('All independently declared 216 groups are required')
 grouped=defaultdict(list)
 for g in groups:
  if sorted(e.get('seed') for e in g.get('checkpoints',[]))!=list(SEEDS):raise ValueError('Every final group requires five distinct frozen algorithm seeds')
  grouped[(g.get('label_rate'),g.get('feedback_delay'))].append(g)
 if set(grouped)!={(r,d) for r in (10,25,50,100) for d in (0,5,20)}:raise ValueError('Final rate/delay grid changed')
 for values in grouped.values():
  primary=[g for g in values if g.get('source_stage')=='e3' and g.get('strategy')=='decision']
  if len(values)!=18 or len(primary)!=1:raise ValueError('Each cell needs the fixed primary decision and all 17 controls')
 return {g['method']:g for g in groups},grouped


def _evaluation_arrays(evaluation,groups,expected_ids=None):
 if evaluation.get('schema')!='frozen_final_matrix_evaluation_v1' or evaluation.get('status')!='EVALUATED_FROZEN_FINAL_POLICIES' or evaluation.get('research_complete') is not False:
  raise ValueError('A complete frozen-final evaluation is required')
 summaries=evaluation.get('summaries',[]);expected={(m,s) for m in groups for s in SEEDS}
 if len(summaries)!=1080 or {(r.get('method'),r.get('seed')) for r in summaries}!=expected:raise ValueError('Final evaluation needs all216 groups times five seeds')
 qualities={};tasks={};summary_index={}
 for summary in summaries:
  key=(summary['method'],summary['seed']);rows=summary.get('evaluation_question_rows',[])
  if summary.get('n_questions')!=200 or len(rows)!=200 or len({r['question_id'] for r in rows})!=200:raise ValueError('Final evaluation requires exactly 200 unique questions per state')
  by_id={r['question_id']:r for r in rows}
  if expected_ids is None:expected_ids=sorted(by_id)
  if set(by_id)!=set(expected_ids):raise ValueError('Final methods/conditions must evaluate exactly the same heldout question set')
  if Counter(r.get('task') for r in rows)!={'gsm8k':100,'bbh':100}:raise ValueError('Final task balance changed')
  vector=[]
  for qid in expected_ids:
   row=by_id[qid];value=row.get('quality')
   if not finite(value) or not 0<=value<=1:raise ValueError('Nonfinite/out-of-range final quality')
   if qid in tasks and tasks[qid]!=row['task']:raise ValueError('Final question task changed')
   tasks[qid]=row['task'];vector.append(float(value))
  qualities[key]=np.array(vector);summary_index[key]=summary
 means={m:np.mean([qualities[(m,s)] for s in SEEDS],axis=0) for m in groups}
 return means,np.array([tasks[q] for q in expected_ids]),expected_ids,summary_index


def validate_final_evaluation_set(release,evaluations):
 groups,_=_group_contract(release)
 declared=release.get('evaluation_conditions',[])
 if len(declared)!=4 or {x.get('condition_id') for x in declared}!=set(CONDITIONS):raise ValueError('All four fixed final conditions are required')
 if len(evaluations)!=4 or {x.get('condition_id') for x in evaluations}!=set(CONDITIONS):raise ValueError('Missing or duplicated final condition')
 expected_ids=None;output={};metadata={x['condition_id']:x for x in declared}
 for evaluation in evaluations:
  cid=evaluation['condition_id'];condition=metadata[cid]
  if evaluation.get('generation_seed')!=condition.get('generation_seed'):raise ValueError('Wrong final generation seed/condition pairing')
  arrays,tasks,expected_ids,summaries=_evaluation_arrays(evaluation,groups,expected_ids)
  output[cid]={'quality':arrays,'tasks':tasks,'ids':expected_ids,'summaries':summaries,'evaluation':evaluation}
 return output


def _interval(difference,tasks,*,replicates=2000,seed=20261031):
 if difference is None:return {'mean':None,'interval':None,'question_support':0,'per_task_support':{},'status':'UNMEASURED_NOT_ZERO'}
 values=np.asarray(difference,dtype=float)
 if values.shape!=tasks.shape or not np.isfinite(values).all():raise ValueError('Invalid paired metric')
 rng=np.random.default_rng(seed);strata={t:values[tasks==t] for t in ('gsm8k','bbh')}
 if any(len(v)!=100 for v in strata.values()):raise ValueError('Paired final intervals require 100+100 questions')
 draws={t:rng.choice(v,(replicates,len(v)),replace=True).mean(axis=1) for t,v in strata.items()};macro=np.mean(list(draws.values()),axis=0)
 return {'mean':float(np.mean([v.mean() for v in strata.values()])),
  'interval':[float(x) for x in np.quantile(macro,[.025,.975])],'question_support':len(values),
  'per_task_support':{t:len(v) for t,v in strata.items()},
  'per_task':{t:{'mean':float(v.mean()),'interval':[float(x) for x in np.quantile(draws[t],[.025,.975])]} for t,v in strata.items()},
  'confidence':.95,'bootstrap_replicates':replicates,'status':'QUESTION_PAIRED_AFTER_FIVE_SEED_MEAN'}


def paired_statistics(release,evaluations,*,costs=None,replicates=2000,seed=20261031):
 """No selection from outcomes: 12 cells x17 fixed controls in every condition."""
 groups,cells=_group_contract(release);data=validate_final_evaluation_set(release,evaluations);comparisons=[]
 metrics={cid:{'quality':value['quality']} for cid,value in data.items()}
 if costs is not None:
  for cid,by_metric in costs.items():metrics[cid].update(by_metric)
 def compare(scope,values,tasks):
  result=[]
  for (rate,delay),cell in sorted(cells.items()):
   treated=next(g for g in cell if g['source_stage']=='e3' and g['strategy']=='decision')['method']
   for control in sorted(g['method'] for g in cell if g['method']!=treated):
    measured={}
    for metric in METRICS:
     by_method=values.get(metric,{})
     a,b=by_method.get(treated),by_method.get(control)
     measured[metric+'_difference']=_interval(None if a is None or b is None else a-b,tasks,replicates=replicates,seed=seed)
    result.append({'scope':scope,'method':treated,'reference':control,'label_rate':rate,'feedback_delay':delay,
     'algorithm_seeds':list(SEEDS),'metrics':measured,'all_17_controls_retained':True})
  return result
 for cid in CONDITIONS:comparisons.extend(compare(cid,metrics[cid],data[cid]['tasks']))
 generation_ids=CONDITIONS[:3];averaged={}
 for metric in METRICS:
  averaged[metric]={}
  for method in groups:
   arrays=[metrics[cid].get(metric,{}).get(method) for cid in generation_ids]
   averaged[metric][method]=None if any(a is None for a in arrays) else np.mean(arrays,axis=0)
 three=compare('paired_mean_of_generation42_137_2024',averaged,data[CONDITIONS[0]]['tasks'])
 sensitivity=[]
 for method,g in sorted(groups.items()):
  q=metrics['second_aggregator_generation42']['quality'][method]-metrics['primary_generation42']['quality'][method]
  sensitivity.append({'method':method,'label_rate':g['label_rate'],'feedback_delay':g['feedback_delay'],
   'scope':'Mistral minus primary aggregator, same base42 member answers and exactly the same frozen trained policy; no retraining',
   'quality_difference':_interval(q,data[CONDITIONS[0]]['tasks'],replicates=replicates,seed=seed)})
 return {'schema':'complete_final_paired_statistics_v1','condition_comparisons':comparisons,
  'three_generation_paired_mean_comparisons':three,'second_aggregator_sensitivity':sensitivity,
  'condition_comparison_count':len(comparisons),'generation_mean_comparison_count':len(three),
  'groups':216,'states_per_condition':1080,'conditions':list(CONDITIONS),'algorithm_seeds':list(SEEDS),
  'actual_generation_seeds':[42,137,2024],
  'statistics_scope':'Paired question units: first mean five algorithm-seed policies per question. Three generation caches retain separate conditions and their paired mean; they are not 600 independent questions. Intervals are conditional on these realized caches and are descriptive without multiplicity adjustment.',
  'cost_scope':'End-to-end logical model resources and training controller costs are amortized over the 200 final questions; source wall uses the selected final source cells. Physical shared caches are accounted once separately. Unmeasured fields remain null.',
  'no_winner_selected':True,'research_complete':False}


def _training_summary(entry):
 receipt_path=verified_reference(entry['episode_receipt']);receipt=read_json(receipt_path)
 source=receipt_path.parent/'summary.json';ref=receipt['artifacts']['summary.json']
 if sha_file(source)!=ref['sha256'] or source.stat().st_size!=ref['bytes']:raise ValueError('Training summary differs from its accepted frozen episode')
 return read_json(source)


def _decisions(release,decisions_directory):
 from budget_router.frozen_evaluation import anonymous_question_id
 directory=Path(decisions_directory);receipt=read_json(directory/'decision_receipt.json');path=directory/'decisions.jsonl'
 if receipt.get('records')!=216*5*200 or receipt.get('release_sha256')!=release['release_sha256'] or sha_file(path)!=receipt.get('decisions_sha256') or receipt.get('pending_feedback_consumed') is not False or receipt.get('parameter_updates')!=0:
  raise ValueError('Final frozen decision receipt incomplete or changed')
 records={}
 for row in read_jsonl(path):
  key=(row['method'],row['seed'],anonymous_question_id(row['question_id']))
  if key in records:raise ValueError('Duplicate frozen final decision')
  records[key]=(int(row['main_index']),tuple(row['main_members']))
 if len(records)!=216*5*200:raise ValueError('Incomplete final choices')
 return records


def _selected_resources(arrays,row,index,members):
 calls=int(np.sum(arrays['member_execution_counts'][row,list(members)])+arrays['aggregate_execution_counts'][row,index])
 tokens=int(np.sum(arrays['member_prompt_tokens'][row,list(members)]+arrays['member_completion_tokens'][row,list(members)])+arrays['aggregate_prompt_tokens'][row,index]+arrays['aggregate_completion_tokens'][row,index])
 wall=None if 'member_wall_time_s' not in arrays or 'aggregate_wall_time_s' not in arrays else float(np.sum(arrays['member_wall_time_s'][row,list(members)])+arrays['aggregate_wall_time_s'][row,index])
 return calls,tokens,wall


def actual_final_costs(release,evaluations,decisions_directory):
 from budget_router.frozen_evaluation import anonymous_question_id
 groups,_=_group_contract(release);data=validate_final_evaluation_set(release,evaluations);decisions=_decisions(release,decisions_directory)
 training={(g['method'],entry['seed']):_training_summary(entry) for g in groups.values() for entry in g['checkpoints']}
 costs={};csv_rows=[]
 for cid,item in data.items():
  evaluation=item['evaluation'];ids=item['ids'];tasks=item['tasks'];values=defaultdict(dict)
  with np.load(verified_reference(evaluation['matrix']),allow_pickle=False) as arrays:
   lookup={anonymous_question_id(q):i for i,q in enumerate(arrays['question_ids'].tolist())}
   if set(lookup)!=set(ids):raise ValueError('Final resource matrix question membership changed')
   for method,g in groups.items():
    by_seed=defaultdict(list)
    for stream_seed in SEEDS:
     summary=training[(method,stream_seed)];ledger=summary['budget']['spent'];service={k:[] for k in ('executions','tokens')};source=[]
     for qid in ids:
      row=lookup[qid];index,members=decisions[(method,stream_seed,qid)]
      calls,tokens,actual_wall=_selected_resources(arrays,row,index,members)
      service['executions'].append(calls);service['tokens'].append(tokens);source.append(actual_wall)
     for metric in ('executions','tokens'):by_seed[metric].append(np.array(service[metric])+ledger[metric]/200.)
     by_seed['labels'].append(np.full(200,ledger['labels']/200.))
     for label,key in (('training_controller_cpu_s','controller_process_time_s'),('training_controller_wall_s','controller_wall_time_s')):
      clock=summary.get(key)
      if not finite(clock) or clock<0:raise ValueError('Measured training controller time missing')
      by_seed[label].append(np.full(200,float(clock)/200.))
     by_seed['final_source_wall_s'].append(None if any(v is None for v in source) else np.array(source))
     evaluation_summary=item['summaries'][(method,stream_seed)];qualities={r['question_id']:r['quality'] for r in evaluation_summary['evaluation_question_rows']}
     for task in ('gsm8k','bbh'):
      locations=np.flatnonzero(tasks==task)
      csv_rows.append({'condition_id':cid,'generation_seed':evaluation['generation_seed'],'method':method,'strategy':g['strategy'],
       'algorithm_seed':stream_seed,'label_rate':g['label_rate'],'feedback_delay':g['feedback_delay'],'task':task,'question_support':100,
       'quality':float(np.mean([qualities[ids[i]] for i in locations])),
       'final_actual_executions':int(np.sum(np.array(service['executions'])[locations])),'final_actual_tokens':int(np.sum(np.array(service['tokens'])[locations])),
       'final_policy_feedback_labels':0,'training_total_executions':ledger['executions'],'training_total_tokens':ledger['tokens'],'training_total_labels':ledger['labels'],
       'training_total_controller_cpu_s':summary['controller_process_time_s'],'training_total_controller_wall_s':summary['controller_wall_time_s'],
       'training_model_fit_cpu_s':summary.get('runtime_measurements',{}).get('training_process_time_s'),
       'training_model_fit_wall_s':summary.get('runtime_measurements',{}).get('training_wall_time_s'),
       'training_initialization_controller_cpu_s':summary.get('runtime_measurements',{}).get('initialization_controller_process_time_s'),
       'training_initialization_controller_wall_s':summary.get('runtime_measurements',{}).get('initialization_controller_wall_time_s'),
       'final_source_wall_s':None if any(source[i] is None for i in locations) else float(sum(source[i] for i in locations)),
       'final_selection_controller_cpu_s':None,'final_selection_controller_wall_s':None,
       'unmeasured_final_selection_time_status':'NOT_RECORDED_NOT_FILLED_WITH_ZERO',
       'training_scope':'Entire shared frozen training episode; repeated by task/condition for context, never sum these columns as physical costs'})
    for metric,arrays_for_seed in by_seed.items():values[metric][method]=None if any(v is None for v in arrays_for_seed) else np.mean(arrays_for_seed,axis=0)
  costs[cid]=dict(values)
 return costs,csv_rows


def validate_final_grading_measurement(evaluation):
 work=evaluation.get('physical_offline_grading_work',{})
 if (work.get('status')!='ACTUAL_GRADING_WORK_VERIFIED' or work.get('completed_matrix_member_labels')!=1200
  or work.get('completed_matrix_aggregate_labels')!=4000 or work.get('unique_reference_questions')!=200
  or work.get('unfully_observed_attempts')!=0 or work.get('policy_feedback_grants')!=0):
  raise ValueError('Complete final private-grading actual work is required; unknown interrupted work is not zero')
 for key in ('process_time_s','wall_time_s'):
  if not finite(work.get(key)) or work[key]<0:raise ValueError('Final private-grading actual CPU/wall missing')
 return work


def compute_final_statistics(release_path,evaluation_references,decisions_directory,*,replicates=2000):
 release=read_json(release_path);evaluations=[read_json(verified_reference(ref)) for ref in evaluation_references]
 costs,rows=actual_final_costs(release,evaluations,decisions_directory)
 result=paired_statistics(release,evaluations,costs=costs,replicates=replicates)
 result.update(release=artifact_reference(release_path),evaluation_sources=evaluation_references,
  frozen_decision_receipt=artifact_reference(Path(decisions_directory)/'decision_receipt.json'),
  per_seed_task_actual_metrics=rows,analysis_source_sha256=sha_file(Path(__file__)))
 return result


def _e1_replay(root):
 from evaluate_real_e1 import assemble_e1
 return assemble_e1(root)


def validate_acceptance(root,path=None):
 """Read-only complete evidence verification; never trust a status string alone."""
 root=Path(root).resolve();path=Path(path).resolve() if path else root/'FINAL_ACCEPTANCE.json';acceptance=read_json(path)
 status=acceptance.get('status')
 if (acceptance.get('schema')!='final_experiment_acceptance_v1' or status not in {'FULL_REQUIRED_EXPERIMENTS_VERIFIED','LEGITIMATE_STOP_E1_VERIFIED'}
  or acceptance.get('workflow_execution_complete') is not True or acceptance.get('ready_for_github_sync') is not True
  or acceptance.get('master_source_sha256')!=sha_file(root/'scripts/run_full_experiment.py')):
  raise ValueError('Complete source-bound verified FINAL_ACCEPTANCE is required; no delivery before real acceptance')
 from run_local_e1 import verify_e0_after_child
 verify_e0_after_child()
 e1_path=verified_reference(acceptance['e1_result']) if 'e1_result' in acceptance else root/'results/e1/e1_diagnostics.json'
 e1=read_json(e1_path);replay=_e1_replay(root)
 if replay!=e1 or e1.get('missing_requirements') or e1.get('n_total_e1')!=300 or e1.get('physical_diagnostic_cost',{}).get('all_costs_reconciled') is not True or e1['physical_diagnostic_cost']['unique_calls']<8800:
  raise ValueError('Full E1 and its control/actual-cost evidence do not independently reproduce')
 if status=='LEGITIMATE_STOP_E1_VERIFIED':
  if acceptance.get('research_complete') is not False or acceptance.get('E2_E5')!='NOT_RUN_PREREGISTERED_STOP' or e1.get('status')!='STOP_E1' or read_json(verified_reference(acceptance['independent_replay']))!=replay:
   raise ValueError('Incomplete or falsely extended E1 stop')
  return {'acceptance':acceptance,'acceptance_ref':artifact_reference(path),'e1':e1,'root':root}
 if acceptance.get('research_complete') is not True or e1.get('status')!='PASS':raise ValueError('Full acceptance cannot bypass E1')
 from budget_router.frozen_evaluation import evaluate_final_matrix
 from budget_router.e5_validation import validate_e5_pre_final_gate
 release_path=verified_reference(acceptance['release']);release=read_json(release_path)
 e5_path=verified_reference(acceptance['stage_gates']['E5_PRE_FINAL']);e5_gate=read_json(e5_path)
 run_plan=read_json(verified_reference(e5_gate['run_plan']));validate_e5_pre_final_gate(e5_path,run_plan['inputs'])
 for reference in acceptance['stage_gates'].values():verified_reference(reference)
 if release.get('stage_evidence',{}).get('E4')!=acceptance['stage_gates'].get('E4') or release.get('stage_evidence',{}).get('E5_PRE_FINAL')!=acceptance['stage_gates'].get('E5_PRE_FINAL') or e5_gate.get('e4_full_gate')!=acceptance['stage_gates'].get('E4') or run_plan.get('upstream',{}).get('e2_gate')!=acceptance['stage_gates'].get('E2') or run_plan.get('upstream',{}).get('e3_gate')!=acceptance['stage_gates'].get('E3'):raise ValueError('Final acceptance mixes different valid-but-unbound parent gates')
 decision_receipt=verified_reference(acceptance['decisions']);decisions_directory=decision_receipt.parent
 evaluations=[]
 for reference in acceptance.get('final_evaluations',[]):
  result=read_json(verified_reference(reference))
  if evaluate_final_matrix(release_path,decisions_directory,verified_reference(result['matrix']),verified_reference(result['matrix_manifest']))!=result:
   raise ValueError('A final condition does not independently reproduce')
  validate_final_grading_measurement(result);evaluations.append(result)
 validate_final_evaluation_set(release,evaluations)
 stats_path=verified_reference(acceptance['final_paired_statistics']);stats=read_json(stats_path)
 expected=compute_final_statistics(release_path,acceptance['final_evaluations'],decisions_directory)
 if stats!=expected:raise ValueError('All fixed final paired comparisons/actual costs must reproduce exactly')
 for name,reference in acceptance.get('analysis_sources',{}).items():
  if verified_reference(reference)!=(root/name).resolve():raise ValueError('Final analysis source binding changed')
 if set(acceptance.get('analysis_sources',{}))!={'scripts/build_delivery.py','scripts/run_full_experiment.py'}:raise ValueError('Final analysis closure incomplete')
 physical=read_json(verified_reference(acceptance['physical_model_accounting']))
 from run_full_experiment import physical_model_accounting
 if physical_model_accounting(project_root=root,write_receipt=False)!=physical:raise ValueError('Complete fixed physical source inventory/attempt identities do not independently reproduce')
 if physical.get('unique_physical_attempts',0)<49256 or physical.get('totals',{}).get('calls',0)<49256:raise ValueError('Full physical actual model accounting incomplete')
 return {'acceptance':acceptance,'acceptance_ref':artifact_reference(path),'e1':e1,'release':release,'evaluations':evaluations,'statistics':stats,'physical':physical,'root':root}

CONFIG_FILES=('experiment_v1.json','pool_v1.json','aggregator_v1.json','aggregator_mistral_v03.json','download_transport_v1.json','e1_strong_sampling.json',
 'member_cogito-v1-preview-llama-8b.json','member_deepseek-r1-distill-qwen-7b.json','member_deephermes-3-llama-3-8b-preview.json',
 'member_granite-3.3-8b-instruct.json','member_qwen2.5-coder-7b-instruct.json','member_qwen3-8b.json')
STATIC_PROTOCOL=('EXECUTION_CONTRACT.md','LOCAL_MODEL_PROTOCOL.md','NEIGHBOR_REPRODUCTION_BOUNDARY.md',
 'source_manifest.json','neighbor_implementation_manifest.json','flex_training_adaptation_manifest.json',
 'e5_robustness_plan_v1.json','final_policy_groups_v1.json','Idea2_LLM_Routing_Research_Plan_Formatted.txt',
 'Idea2_LLM_Routing_Research_Plan.txt','ACL研究报告_有限评测预算下的模型组合路由.txt')
THIRD_PARTY=('third_party/ces_source_audit/SOURCE_MANIFEST.json','third_party/ces_source_audit/upstream/LICENSE',
 'third_party/ces_source_audit/upstream/src/bandits/linucb_combinatorial.py','third_party/ces_source_audit/upstream/src/embeddings/projectors.py')
FORBIDDEN_PARTS={'models','data','.cache','.venv','venv','tools','logs','private','__pycache__','.git','node_modules'}
ALLOWED_SUFFIX={'.py','.ps1','.json','.md','.txt','.csv','.toml','.sha256'}
OPERATION_ONLY_FILES={'scripts/recycle_at_settled_boundary.py','tests/test_recycle_at_settled_boundary.py',
 'scripts/operator_abort_e1_call.py','tests/test_operator_abort_e1_call.py',
 'src/budget_router/operator_snapshot_proof.py','tests/test_operator_snapshot_proof.py',
 'scripts/review_interrupted_call_recovery.py','tests/test_interrupted_generation_accounting_review.py'}


def physical_summary(record,root):
 """Actual attempts, with slow tails intact; never copy answers or question IDs."""
 from budget_router.execution import _digest
 from budget_router.inference import _validated_usage
 attempts={};walls=[];groups={};totals={'calls':0,'prompt_tokens':0,'completion_tokens':0,'model_call_wall_time_s':0.,'failed_calls':0}
 for reference in record.get('source_stores',[]):
  path=verified_reference(reference)
  if path.with_name(path.name+'.lock').exists():raise ValueError('An accepted model source is still locked')
  relative=path.relative_to(root).parts
  stage=relative[2] if len(relative)>2 and relative[:2]==('results','raw') else 'OTHER_BOUND_SOURCE'
  with path.open(encoding='utf8') as stream:
   for line in stream:
    if not line.strip():continue
    row=json.loads(line);checksum=row.get('record_sha256')
    if checksum!=_digest({k:v for k,v in row.items() if k!='record_sha256'}):raise ValueError('Physical source checksum mismatch')
    attempt=row.get('attempt_id')
    if not attempt or (attempt in attempts and attempts[attempt]!=checksum):raise ValueError('Physical attempt identity missing/conflicting')
    if attempt in attempts:continue
    attempts[attempt]=checksum
    success=row.get('status')=='success';result=row.get('result',{}) if success else row.get('actual_usage')
    usage=_validated_usage(result);wall=row.get('result',{}).get('wall_time_s') if success else row.get('failure_wall_time_s')
    if row.get('status') not in {'success','failure'} or row.get('cost_status')!='recorded_actual_usage' or usage is None or not finite(wall) or wall<0:raise ValueError('Unreconciled physical model cost')
    model=row.get('result',{}).get('model_id',row.get('model_alias','UNRECORDED_FAILED_MODEL'))
    key=(stage,str(model),row.get('seed'),row.get('task'))
    item=groups.setdefault(key,{'calls':0,'prompt_tokens':0,'completion_tokens':0,'wall_time_s':0.,'failed_calls':0,'max_call_wall_time_s':0.})
    for target in (item,totals):
     target['calls']+=1;target['prompt_tokens']+=usage['prompt_tokens'];target['completion_tokens']+=usage['completion_tokens'];target['failed_calls']+=int(not success)
    item['wall_time_s']+=wall;item['max_call_wall_time_s']=max(item['max_call_wall_time_s'],wall);totals['model_call_wall_time_s']+=wall;walls.append(wall)
 if record.get('unique_physical_attempts')!=len(attempts) or record.get('attempt_identity_sha256')!=canonical_digest(attempts):raise ValueError('Physical unique attempt count/digest changed')
 for key,value in totals.items():
  expected=record.get('totals',{}).get(key)
  if not finite(expected) or not math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-8):raise ValueError('Physical accepted totals no longer reproduce')
 ordered=sorted(walls);p95=None if not walls else float(np.quantile(ordered,.95))
 return {'totals':totals,'unique_physical_attempts':len(attempts),'max_call_wall_time_s':None if not walls else max(walls),
  'p95_call_wall_time_s':p95,'wall_outliers_removed':False,
  'by_source_model_generation_task':[{'stage':k[0],'model_id':k[1],'generation_seed':k[2],'task':k[3],**v} for k,v in sorted(groups.items(),key=lambda x:str(x[0]))],
  'scope':'Actual unique source attempts, including known failures and every slow wall duration; no truncation or substitution. Shared member caches count once, not once per policy.'}


def _public_e1(e1):
 keys=('status','n_total_e1','n_selection_calibration','n_questions','n_combinations','n_models','k','generation_cache_replicates',
  'experimental_seeds','metrics','paired_differences','coverage_and_aggregation','gate','costs','physical_diagnostic_cost','logical_standalone_service_cost')
 return {k:e1[k] for k in keys if k in e1}


def _hypotheses(context):
 e1=context['e1'];out={'H1':{'status':'OPPORTUNITY_SUPPORTED' if e1['status']=='PASS' else 'OPPORTUNITY_GATE_NOT_PASSED',
  'engineering_gate':e1.get('gate'),'scope':'E1 all-information oracle opportunity diagnosis, never deployable router accuracy or proof of learnability/cost benefit.'}}
 if context['acceptance']['status']=='LEGITIMATE_STOP_E1_VERIFIED':
  out.update(H2={'status':'NOT_RUN_PREREGISTERED_STOP'},H3={'status':'NOT_RUN_PREREGISTERED_STOP'});return out
 gates=context['acceptance']['stage_gates'];e2_gate=read_json(verified_reference(gates['E2']));e3_gate=read_json(verified_reference(gates['E3']))
 e2=read_json(verified_reference(e2_gate['ranking_diagnostics']));e3=read_json(verified_reference(e3_gate['mechanism_evaluation']))
 out['H2']={'status':e2['scientific_learnability_conclusion'],'interval':e2['scientific_learnability_interval'],
  'diagnostic_policy':e2['diagnostic_policy'],'scope':e2['scientific_scope'],'topk_semantics':e2['topk_semantics'],
  'rich_calibration_reference':e2.get('references',{}),'main_fixed_definition':e2['main_fixed_definition']}
 out['H3']={'status':e3['academic_conclusion'],'primary':e3['academic_primary_comparison'],
  'scope':'Original E3 preregistered rate25/delay0 primary quality/actual-priced-cost criterion; retain NEGATIVE/INCONCLUSIVE, do not replace it with a favorable final cell.'}
 return out


HISTORY_CONTROL_SOURCES=('scripts/resume_e1_aggregation.py','src/budget_router/recovery_runtime.py',
 'scripts/run_full_experiment.py','scripts/monitor_experiment.py','scripts/launch_e1_recovery.ps1')
HISTORY_SCORER_FILES=('src/budget_router/scoring.py','configs/experiment_v1.json','protocol/EXECUTION_CONTRACT.md')


def _history_sha(value):
 if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:raise ValueError('Invalid source-history digest')
 return value


def _history_ref(root,reference,expected=None):
 # Follow only explicit metadata/code refs, never jobs, raw JSONL, matrices or weights.
 if not isinstance(reference,dict) or set(reference)!={'path','sha256','bytes'}:raise ValueError('Incomplete source-history reference')
 path=Path(reference['path']);path=(path if path.is_absolute() else root/path).resolve()
 if not path.is_relative_to(root) or (expected is not None and path!=(root/expected).resolve()):raise ValueError('Source-history reference escaped its expected project path')
 if path.suffix.lower() not in {'.json','.py','.ps1','.original'}:raise ValueError('Source-history references must be metadata or code')
 if type(reference['bytes']) is not int or path.stat().st_size!=reference['bytes'] or sha_file(path)!=_history_sha(reference['sha256']):raise ValueError('Source-history metadata/code changed')
 return path


def _observer_io_patch(root):
 path=root/'artifacts/private/observer_share_delete_repair_20261101/change_manifest.json'
 if not path.exists():return None
 value=read_json(path)
 if (value.get('schema')!='observer_share_delete_io_patch_v1'
  or value.get('scope')!='MONITOR_READ_HANDLES_ONLY_NO_INFERENCE_OR_MATH_CHANGE'
  or value.get('frozen_sources_unchanged') is not True or value.get('frozen_source_count')!=17
  or value.get('scientific_completion') is not False):raise ValueError('Invalid observer IO repair scope')
 before=_history_ref(root,value['before_monitor'])
 after=_history_ref(root,value['after_monitor'],'scripts/monitor_experiment.py')
 e0=read_json(_history_ref(root,value['e0'],'protocol/E0_LOCK.json'))
 if len(e0.get('source_code',[]))!=17:raise ValueError('Frozen source inventory changed')
 for item in e0['source_code']:
  if sha_file(root/item['path'])!=_history_sha(item['sha256']):raise ValueError('Frozen source changed during observer IO repair')
 import ast
 def nodes(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 a,b=nodes(before),nodes(after)
 if {k for k in a if a[k]!=b.get(k)}!={'read_json','observe_inflight'} or set(b)-set(a)!={'open_shared_read','read_bytes_shared'}:raise ValueError('Observer IO repair changed unrelated functions')
 test=value['software_tests'];test_path=(root/'artifacts/audit/monitor_shared_delete_final_tests.xml').resolve()
 if Path(test['path']).resolve()!=test_path or test_path.stat().st_size!=test['bytes'] or sha_file(test_path)!=_history_sha(test['sha256']):raise ValueError('Observer IO test receipt changed')
 import xml.etree.ElementTree as ET
 tree=ET.parse(test_path).getroot()
 if len(list(tree.iter('testcase')))!=8 or list(tree.iter('failure')) or list(tree.iter('error')):raise ValueError('Observer IO software checks incomplete')
 return value


def _historical_control_ref(root,reference,name):
 if name!='scripts/monitor_experiment.py' or sha_file(root/name)==reference['sha256']:
  return _history_ref(root,reference,name)
 patch=_observer_io_patch(root)
 if (patch is None or Path(reference['path']).resolve()!=(root/name).resolve()
  or reference['sha256']!=patch['before_monitor']['sha256'] or reference['bytes']!=patch['before_monitor']['bytes']):raise ValueError('Historical monitor transition has no verified closure')
 return _history_ref(root,patch['before_monitor'])


def _observer_io_source_revisions(root):
 patch=_observer_io_patch(root)
 if patch is None:return []
 return [{'change_kind':'OBSERVER_READ_HANDLE_SHARING','path':'scripts/monitor_experiment.py',
  'before_sha256':patch['before_monitor']['sha256'],'after_sha256':patch['after_monitor']['sha256'],
  'frozen_source_count':17,'frozen_source_byte_identity':'UNCHANGED',
  'software_tests':8,'software_test_sha256':patch['software_tests']['sha256'],
  'scope':'Read-only observer permits writer lock unlink/rename. Open-destination replace may still need bounded retries. Historical source binding only; not research completion or proof of the earlier external actor.'}]


def _scorer_source_revisions(root):
 history=root/'artifacts/private/e0_reference_repair_history';records=[]
 for path in sorted(history.glob('*/applied_manifest.json')):
  applied=read_json(path)
  if applied.get('schema')!='pre_e1_scorer_bugfix_application_v1':continue
  candidate_path=path.parent/'change_manifest.json';candidate=read_json(candidate_path)
  cpu_path=root/'artifacts/audit/scorer_v1_1_cpu_qualification_review.json';cache_path=root/'artifacts/audit/scorer_v1_1_existing_member_cache_verification.json'
  cpu=read_json(cpu_path);cache=read_json(cache_path);files=applied.get('changed_files',[])
  if len(files)!=3 or {r.get('path') for r in files}!=set(HISTORY_SCORER_FILES):raise ValueError('Unexpected scorer-revision public file inventory')
  indexed={r['path']:r for r in files};scorer=indexed[HISTORY_SCORER_FILES[0]]
  for item in files:_history_sha(item['old_sha256']);_history_sha(item['new_sha256'])
  if (applied.get('scorer_version')!='safe_exact_v1_1' or applied.get('real_inference_executed') is not False
   or applied.get('old_qualification_receipts_archived') is not True or applied.get('scientific_completion') is not False
   or candidate.get('old_sha256')!=scorer['old_sha256'] or candidate.get('candidate_sha256')!=scorer['new_sha256']
   or candidate.get('candidate_version')!='safe_exact_v1_1'
   or sha_file(root/HISTORY_SCORER_FILES[0])!=scorer['new_sha256']
   or sha_file(root/HISTORY_SCORER_FILES[2])!=indexed[HISTORY_SCORER_FILES[2]]['new_sha256']):raise ValueError('Scorer-revision source/application bindings changed')
  if (cpu.get('schema')!='scorer_revision_cpu_qualification_review_v1' or cpu.get('status')!='PASS_CPU_REVIEW_NOT_NEW_GENERATION_NOT_E0'
   or cpu.get('scorer_version')!='safe_exact_v1_1' or cpu.get('scorer_source_sha256')!=scorer['new_sha256']
   or type(cpu.get('reviewed_qualification_success_calls')) is not int or cpu['reviewed_qualification_success_calls']!=52
   or type(cpu.get('new_model_calls')) is not int or cpu['new_model_calls']!=0
   or cpu.get('original_receipts_preserved') is not True or cpu.get('model_or_answer_regeneration_performed') is not False
   or cpu.get('scientific_completion') is not False):raise ValueError('Complete CPU-only re-review evidence required')
  receipts=cpu.get('receipts',[])
  if (len(receipts)!=7 or any(type(r.get('jobs')) is not int for r in receipts)
   or sorted(r['jobs'] for r in receipts)!=[4,8,8,8,8,8,8]
   or any(r.get('status')!='PASS' or r.get('generation_identity_usage_and_raw_hashes_unchanged') is not True for r in receipts)):
   raise ValueError('Original seven qualification receipts were not preserved and re-reviewed')
  if (cache.get('schema')!='pre_e1_repair_existing_member_cache_reuse_v1' or cache.get('status')!='ALL1800_ORIGINAL_MEMBER_CACHE_HITS_NOT_NEW_INFERENCE'
   or type(cache.get('total_cache_hits')) is not int or cache['total_cache_hits']!=1800
   or type(cache.get('new_model_calls')) is not int or cache['new_model_calls']!=0 or cache.get('scientific_completion') is not False
   or cache.get('runtime_memory_journal_unchanged') is not True or cache.get('preserved_raw_and_model_config_hashes_verified') is not True
   or cache.get('baseline_sha256')!=cpu.get('source_baseline_manifest_sha256')):raise ValueError('Original member-cache reuse evidence changed')
  members=cache.get('members',[])
  if len(members)!=6 or any(type(r.get('jobs')) is not int or type(r.get('cache_hits')) is not int or r['jobs']!=300 or r['cache_hits']!=300 for r in members):raise ValueError('All original 1800 member outputs must be accounted for')
  baseline=_history_sha(cache['baseline_sha256'])
  if not any(sha_file(v)==baseline for v in history.glob('*baseline*/change_manifest.json')):raise ValueError('Original scorer source baseline missing')
  records.append({'change_kind':'PRE_E1_SCORER_FORMAT_BUGFIX','phase':'BEFORE_E0_FREEZE_AND_FORMAL_E1_SCORING',
   'from_version':'safe_exact_v1','to_version':'safe_exact_v1_1',
   'changed_public_files':[{'path':name,'before_sha256':indexed[name]['old_sha256'],'after_sha256':indexed[name]['new_sha256']} for name in HISTORY_SCORER_FILES],
   'syntax_change':'BBH word_sorting accepts ASCII alphabetic compound tokens joined by internal ampersands.',
   'reward_mapping_may_change':True,'global_reward_byte_identity_claimed':False,
   'qualification_review':{'mode':'CPU_REVIEW_OF_EXISTING_OUTPUTS','receipts':7,'original_generation_calls_reused':52,'new_model_calls':0},
   'member_cache_reuse':{'models':6,'original_outputs_reused':1800,'new_model_calls':0},
   'metadata_sha256':{'application':sha_file(path),'candidate':sha_file(candidate_path),'baseline':baseline,'cpu_review':sha_file(cpu_path),'member_cache_review':sha_file(cache_path)},
   'scope':'Pre-E0 configuration hashes describe that revision, not the later frozen configuration. Existing generated texts were retained; no new generation is implied.'})
 for path in sorted(history.glob('nonsealed_*/change_manifest.json')):
  value=read_json(path)
  if value.get('schema')!='e0_reference_boundary_and_version_patch_v1':continue
  chain=value.get('revision_chain',[]);after=value.get('files_written_by_this_patch',{}).get('src/budget_router/protocol_lock.py')
  if (len(chain)!=3 or value.get('scorer_roundtrip_call_ast_unchanged') is not True
   or value.get('source_guard_inventory_and_transaction_functions_unchanged') is not True
   or value.get('actual_e0_lock_exists_at_software_completion') is not False
   or chain[-1].get('sha256')!=after or sha_file(root/'src/budget_router/protocol_lock.py')!=_history_sha(after)):
   raise ValueError('Nonsealed-reference boundary repair metadata changed')
  records.append({'change_kind':'E0_NONSEALED_REFERENCE_PREFLIGHT','phase':'BEFORE_E0_FREEZE_AND_FORMAL_E1_SCORING',
   'changed_public_files':[{'path':'src/budget_router/protocol_lock.py','before_sha256':_history_sha(chain[0]['sha256']),'after_sha256':after}],
   'preflight_scope':'Five nonsealed splits only; sealed references deferred to released final grading; shared eager source loading was not changed.',
   'scorer_roundtrip_call_unchanged':True,'config_scorer_version_match_required':True,'metadata_sha256':sha_file(path)})
 return records


def _control_source_revisions(root):
 records=[]
 for path in sorted((root/'artifacts/private').glob('e1_aggregation_recovery_*/controlplane_deployment_readback.json')):
  deployed=read_json(path)
  if deployed.get('schema')!='actual_orphan_recovery_full_controlplane_deployment_v1':continue
  adoption_path=path.parent/'actual_adoption_readback.json';adoption=read_json(adoption_path)
  if (deployed.get('status')!='ACTUAL_RECOVERY_MONITOR_MASTER_RUNNING_FOLLOWING_UNCHANGED_GPU_WORKER'
   or deployed.get('scientific_completion') is not False or deployed.get('original_e0_bytes_unchanged') is not True
   or deployed.get('original_aggregation_not_restarted') is not True or deployed.get('new_gpu_worker_launched_by_recovery') is not False
   or adoption.get('schema')!='actual_e1_recovery_adoption_readback_v1'
   or adoption.get('status')!='ACTUAL_RECOVERY_OWNER_AND_BOTH_LEASES_VERIFIED_ORIGINAL_WORKER_UNTOUCHED'
   or adoption.get('scientific_completion') is not False or adoption.get('both_controller_leases_match_actual_recovery_owner') is not True
   or adoption.get('authorization')!=deployed.get('authorization') or adoption.get('owner')!=deployed.get('owner')):
   raise ValueError('Complete unchanged-worker recovery deployment evidence required')
  auth=read_json(_history_ref(root,deployed['authorization']));owner=read_json(_history_ref(root,deployed['owner']))
  e0_path=_history_ref(root,auth['original_e0'],'protocol/E0_LOCK.json');e0=read_json(e0_path);e0_sha=sha_file(e0_path)
  if (auth.get('schema')!='e1_aggregation_recovery_authorization_v1' or auth.get('scope')!='scheduling_recovery_only_no_math_change'
   or auth.get('operational_overlay')!='run_local_e1.atomic_json_only' or auth.get('software_test_only') is not False
   or owner.get('authorization')!=deployed['authorization'] or owner.get('recovery_sources')!=auth.get('recovery_sources')
   or owner.get('original_handles_acquired_and_birth_verified') is not True
   or deployed.get('original_e0_sha256')!=e0_sha or adoption.get('E0_lock_sha256')!=e0_sha):raise ValueError('Original recovery/E0 lineage changed')
  frozen=e0.get('source_code',[])
  if (e0.get('status'),e0.get('stage'),e0.get('N'),e0.get('K'),e0.get('e1_questions'))!=('PASS','E0',6,3,300):raise ValueError('Original full-scale E0 metadata changed')
  if len(frozen)!=17 or len({r.get('path') for r in frozen})!=17:raise ValueError('Original seventeen-source E0 inventory missing')
  for source in frozen:
   name=source['path'];safe_package_relative(name)
   target=(root/name).resolve()
   if not name.endswith('.py') or not name.startswith(('scripts/','src/')) or not target.is_relative_to(root) or sha_file(target)!=_history_sha(source['sha256']):raise ValueError('An original E0 source changed')
  sources={}
  for name in HISTORY_CONTROL_SOURCES:
   reference=deployed['sources'][name];_historical_control_ref(root,reference,name);sources[name]=reference['sha256']
  recovery_names=[]
  for reference in auth['recovery_sources']:
   code=_history_ref(root,reference);name=code.relative_to(root).as_posix();recovery_names.append(name)
   if name not in HISTORY_CONTROL_SOURCES[:2] or sources[name]!=reference['sha256']:raise ValueError('Recovery runtime source closure changed')
  if len(recovery_names)!=2 or set(recovery_names)!=set(HISTORY_CONTROL_SOURCES[:2]):raise ValueError('Recovery helper/common source pair required')
  adapters=[]
  for candidate_path in (root/'artifacts/private').glob('recovery_owner_master_monitor_*/change_manifest.json'):
   value=read_json(candidate_path)
   if (value.get('schema')=='recovery_owner_master_monitor_staged_adapter_v1'
    and value.get('master_current',{}).get('sha256')==sources['scripts/run_full_experiment.py']
    and value.get('monitor_staged_candidate',{}).get('sha256')==sources['scripts/monitor_experiment.py']):adapters.append((candidate_path,value))
  if len(adapters)!=1:raise ValueError('A unique recorded master/monitor source transition is required')
  adapter_path,adapter=adapters[0];transitions=[]
  for name,before_key,after_key in (('scripts/run_full_experiment.py','master_original','master_current'),('scripts/monitor_experiment.py','monitor_original','monitor_staged_candidate')):
   _history_ref(root,adapter[before_key]);_history_ref(root,adapter[after_key])
   transitions.append({'path':name,'before_sha256':adapter[before_key]['sha256'],'after_sha256':sources[name],'transition_scope':'DEPLOYED_SOURCE_REPLACEMENT'})
  duals=[]
  for dual_path in (root/'artifacts/private/e1_recovery_lease_patch_history').glob('dual_*/change_manifest.json'):
   value=read_json(dual_path);revisions=value.get('helper_revisions',[])
   if value.get('schema')=='e1_recovery_dual_lease_patch_v1' and revisions and revisions[-1].get('artifact',{}).get('sha256')==sources[HISTORY_CONTROL_SOURCES[0]]:duals.append((dual_path,value))
  if len(duals)!=1:raise ValueError('A unique recorded dual-lease source revision is required')
  dual_path,dual=duals[0];revisions=dual['helper_revisions']
  if len(revisions)!=3:raise ValueError('Original helper candidate lineage incomplete')
  _history_ref(root,dual['original_e0'],'protocol/E0_LOCK.json');_history_ref(root,dual['common'],'src/budget_router/recovery_runtime.py')
  if dual.get('frozen_source_sha256')!={r['path']:r['sha256'] for r in frozen}:raise ValueError('Dual-lease repair changed a frozen E0 source')
  _history_ref(root,revisions[0]['artifact']);_history_ref(root,revisions[-1]['artifact'])
  transitions.append({'path':'scripts/resume_e1_aggregation.py','before_sha256':revisions[0]['artifact']['sha256'],'after_sha256':sources['scripts/resume_e1_aggregation.py'],'transition_scope':'PRE_DEPLOYMENT_CANDIDATE_TO_DEPLOYED_HELPER'})
  records.append({'change_kind':'INDEPENDENT_E1_AGGREGATION_CONTROL_RECOVERY','phase':'E0_LOCKED_E1_AGGREGATION_IN_PROGRESS',
   'original_e0_sha256':e0_sha,'frozen_source_count':17,'frozen_source_byte_identity':'UNCHANGED',
   'deployed_public_source_sha256':sources,'source_transitions':transitions,'exclusive_supervisor_lease_count':2,
   'operational_overlay':'run_local_e1.atomic_json_only','original_aggregation_batch_restarted':False,
   'new_gpu_worker_started_during_adoption':False,'scientific_completion':False,
   'metadata_sha256':{'deployment':sha_file(path),'adoption':sha_file(adoption_path),'master_monitor_revision':sha_file(adapter_path),'dual_lease_revision':sha_file(dual_path)},
   'scope':'Historical deployment and adoption only, not current health or research completion. Later preregistered jobs retain normal inference accounting. Runtime identities and permission artifacts are omitted.'})
 return records


def sanitized_source_history(root):
 records=[]
 for path in sorted((root/'artifacts/private/runtime_memory_recovery').glob('*/change_manifest.json'),key=lambda p:(read_json(p).get('created_at_unix',0),str(p))):
  source=read_json(path)
  if source.get('qualified_math_source_byte_identity')!='UNCHANGED_VERIFIED' or not source.get('after'):continue
  before=source.get('qualified_math_sources_before',{});after=source.get('qualified_math_sources_after',{})
  if before!=after:raise ValueError('Runtime memory patch changed a qualified mathematical source')
  records.append({'revision':len(records)+1,'metadata_sha256':sha_file(path),
   'old_scheduler_sha256':{name:value['sha256'] for name,value in source['previous'].items()},
   'new_scheduler_sha256':{name:value['sha256'] for name,value in source['after'].items()},
   'qualified_six_core_sha256':before,'qualified_core_byte_identity':'UNCHANGED',
   'pre_patch_qualification_outputs_reported':source.get('existing_qualification_outputs_reported_by_operator'),
   'pre_patch_formal_output_prefix_minimum_reported':source.get('existing_formal_outputs_minimum_reported_by_operator'),
   'change_kind':source.get('change_kind','SCHEDULER_IMPLEMENTATION_PATCH'),
   'old_runner_success_prefix':source.get('old_runner_success_prefix'),
   'scope':'Historical operator-observed prefix, not a new count inferred from final performance. Old outputs retain old runner version; scheduler change never relabels them as new-runner outputs.'})
 return {'schema':'sanitized_scheduler_source_history_v1','revisions':records,
  'scorer_revisions':_scorer_source_revisions(root),'control_revisions':_control_source_revisions(root),
  'observer_io_revisions':_observer_io_source_revisions(root),
  'scope':'Allocator six-core equality is local to each allocator revision. Scorer and control revisions are separate; neither global source-byte identity nor unchanged reward mappings is claimed. Slow model-call wall time remains in actual physical costs.'}


def housekeeping_summary(root):
 path=root/'artifacts/audit/runtime_memory_housekeeping.jsonl'
 if not path.exists():return {'status':'NO_HOUSEKEEPING_JOURNAL','cleanup_cpu_s':None,'cleanup_wall_s':None}
 from budget_router.execution import _digest
 statuses=Counter();phases=Counter();cpu=wall=0.
 for row in read_jsonl(path):
  if row.get('record_sha256')!=_digest({k:v for k,v in row.items() if k!='record_sha256'}):raise ValueError('Housekeeping audit checksum changed')
  for key in ('cleanup_process_time_s','cleanup_wall_time_s'):
   if not finite(row.get(key)) or row[key]<0:raise ValueError('Housekeeping duration unknown/invalid')
  statuses[row.get('status','UNKNOWN')]+=1;phases[row.get('phase','UNKNOWN')]+=1
  cpu+=row['cleanup_process_time_s'];wall+=row['cleanup_wall_time_s']
 return {'status':'MEASURED_ALLOCATOR_HOUSEKEEPING_ONLY','cleanup_cpu_s':cpu,'cleanup_wall_s':wall,'by_phase':dict(phases),'by_status':dict(statuses),
  'journal_sha256':sha_file(path),'scope':'Independent host cleanup time; journal IO is measured separately in batch summaries. Allocator counters do not explain all WDDM residency; no untracked usage is labelled cache.'}


def sanitize_json(value,root):
 if isinstance(value,dict):
  result={}
  for key,item in value.items():
   low=key.casefold()
   if any(token in low for token in ('password','secret_key','access_token','refresh_token','cookie')) or low in {'api_key','token','authorization','authorization_header','client_secret','private_key','secret_access_key','credentials','credential'} or low.endswith(('_api_key','_access_key','_private_key','_token')):
    if item not in (None,'',False):raise ValueError('Credential-bearing config/protocol cannot be published')
   if low in {'pid','process_create_time'} or low.endswith('_pid'):raise ValueError('Runtime PID metadata cannot be published')
   result[key]=sanitize_json(item,root)
  return result
 if isinstance(value,list):return [sanitize_json(x,root) for x in value]
 if isinstance(value,str):
  if re.match(r'^[A-Za-z]:[\\/]',value) or value.startswith('/'):
   normalized=value.replace('\\','/');prefix=str(root).replace('\\','/').rstrip('/')+'/'
   return normalized[len(prefix):] if normalized.casefold().startswith(prefix.casefold()) else '<LOCAL_PATH_REDACTED>'
  if re.search(r'https?://[^/\s]+@',value) or re.search(r'[?&](?:token|api_key|access_token)=',value,re.I):raise ValueError('Credential-bearing URL cannot be published')
 return value


def validate_static_json_for_publication(value,root):
 # Validate credentials/PIDs without changing any byte of an executed config.
 # model_path participates in configuration_fingerprint; rewriting it invalidates
 # the original pool/qualification provenance even when both SHA values are kept.
 sanitize_json(value,root)
 def check(item):
  if isinstance(item,dict):
   for child in item.values():check(child)
  elif isinstance(item,list):
   for child in item:check(child)
  elif isinstance(item,str) and (re.match(r'^[A-Za-z]:[\\/]',item) or item.startswith('/')):
   if not Path(item).resolve().is_relative_to(Path(root).resolve()):
    raise ValueError('Static JSON contains an external local path; preserve configuration identity or exclude this artifact')
 check(value)


def safe_package_relative(name):
 path=Path(name)
 if path.is_absolute() or '..' in path.parts or any(part.casefold() in FORBIDDEN_PARTS for part in path.parts):raise ValueError('Forbidden delivery path')
 if path.suffix.lower() not in ALLOWED_SUFFIX and path.name not in {'LICENSE','.gitignore'}:raise ValueError('Only explicitly whitelisted text/source artifacts may be delivered')
 return path


def source_whitelist(root):
 selected=[]
 for base,patterns in (('src/budget_router',('*.py',)),('scripts',('*.py','*.ps1')),('tests',('test_*.py',))):
  for pattern in patterns:
   for path in sorted((root/base).glob(pattern)):
    if path.name.startswith(('_private','debug_','temporary_','private_')) or (base=='scripts' and path.name.startswith('_')):continue
    relative=path.relative_to(root).as_posix()
    if relative in OPERATION_ONLY_FILES:continue
    safe_package_relative(relative)
    if not path.resolve().is_relative_to(root):raise ValueError('Source symlink escaped experiment root')
    selected.append(relative)
 selected.extend('configs/'+name for name in CONFIG_FILES if (root/'configs'/name).is_file())
 selected.extend('protocol/'+name for name in STATIC_PROTOCOL if (root/'protocol'/name).is_file())
 selected.extend(name for name in ('README.md','PROJECT_INDEX.md','pyproject.toml','.gitignore') if (root/name).is_file())
 source_manifest=read_json(root/THIRD_PARTY[0]);audited={row['path']:row for row in source_manifest.get('files',[])}
 if source_manifest.get('repository')!='JinkunXu/CES' or source_manifest.get('commit')!='9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7':raise ValueError('CES pinned source manifest differs')
 for relative in THIRD_PARTY:
  if not (root/relative).is_file():raise ValueError('Required MIT CES source/license provenance is missing')
  if '/upstream/' in relative:
   name=relative.split('/upstream/',1)[1]
   if audited.get(name,{}).get('sha256')!=sha_file(root/relative):raise ValueError('MIT CES code/license differs from pinned audit')
  selected.append(relative)
 for relative in selected:
  if not (root/relative).resolve().is_relative_to(root):raise ValueError('Whitelisted artifact symlink escaped experiment root')
 return sorted(set(selected))


def _test_receipts(root):
 receipts=[]
 for path in sorted((root/'artifacts/audit').glob('*.xml')):
  if not any(word in path.name for word in ('final','complete','software','runtime_memory_phases','boundary_recycle')):continue
  try:
   xml=ET.parse(path).getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.findall('testsuite'))
   counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
   if counts['tests'] and not counts['errors'] and not counts['failures']:
    receipts.append({'report':path.name,'sha256':sha_file(path),**counts,
     'scope':'Software tests only; no model qualification or research result implied. Raw XML/stdout/paths remain local.'})
  except (ValueError,ET.ParseError,OSError):continue
 return receipts


def csv_text(rows):
 if not rows:return ''
 fields=list(dict.fromkeys(k for row in rows for k in row));stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
 for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
 return stream.getvalue()


def _paired_csv(statistics):
 rows=[]
 for key in ('condition_comparisons','three_generation_paired_mean_comparisons'):
  for comparison in statistics[key]:
   for metric,values in comparison['metrics'].items():
    for task in ('task_macro','gsm8k','bbh'):
     entry=values if task=='task_macro' else values.get('per_task',{}).get(task,{})
     interval=entry.get('interval')
     rows.append({'condition_id':comparison['scope'],'method':comparison['method'],'reference':comparison['reference'],
      'label_rate':comparison['label_rate'],'feedback_delay':comparison['feedback_delay'],'task':task,'metric':metric,
      'mean_difference':entry.get('mean'),'ci_low':None if interval is None else interval[0],'ci_high':None if interval is None else interval[1],
      'question_support':values.get('question_support',0) if task=='task_macro' else values.get('per_task_support',{}).get(task,0),
      'algorithm_seeds':SEEDS,'status':values.get('status'),'confidence':values.get('confidence',.95)})
 return rows


def render_report(public):
 outcome=public['outcome'];hyp=public['hypotheses'];full=outcome=='FULL_REQUIRED_EXPERIMENTS_VERIFIED'
 lines=['语言模型组合路由：实际实验验收报告','', '验收状态：'+outcome,
  '完整执行不代表学术阳性；单元测试、资格调用与进程启动均不是科研完成证据。',
  'H1：'+hyp['H1']['status']+'。E1全信息oracle只说明诊断机会，不能作为可部署路由准确率。',
  'H2：'+hyp['H2']['status']+'；保留原E2冻结预测排序、ties和支持数的结论。',
  'H3：'+hyp['H3']['status']+'；保留原E3 rate25/delay0的预声明主判据，不用最终有利格子替换。','']
 if full:
  lines+=['最终评价覆盖216组方法/预算/延迟，每组5个冻结状态、每条件200题，四条件均独立核验。',
   '每个预算/延迟对decision与其他17种方法完整配对。先在题内平均5个算法seed，再按任务分层以题为单位bootstrap。',
   '生成42/137/2024三个实际缓存分别保留，并报告同200题的三条件配对均值；不能当600道独立题。',
   '第二聚合器使用同一base42成员回答与同一已冻结策略，仅作敏感性评价，不重新训练。',
   '原版CES/C2MAB-V/Flex权限不同；本实验是明确限制和计费的适配比较，不宣称完整原版系统复现。']
 else:lines+=['完整E1及全部真实控制/费用已重放，按项目书机会门合法停止；E2-E5与final均未运行，不提供虚构后续结果。']
 history=public.get('source_history',{})
 if any(r.get('change_kind')=='PRE_E1_SCORER_FORMAT_BUGFIX' for r in history.get('scorer_revisions',[])):lines+=['正式E1评分前修复safe_exact_v1→safe_exact_v1_1合法词内&解析；奖励映射可能改变，不宣称全实验数学/奖励字节不变。原52次资格输出只作CPU重审，1800成员输出沿用原缓存，不冒充新增生成。']
 if history.get('control_revisions'):lines+=['独立恢复控制层保留原E0锁及17源码；通过真实身份绑定接续原聚合worker、双排他租约及仅状态writer叠加，不把恢复部署当科研完成。']
 lines+=['','费用口径：真实独立模型attempt与共享缓存只计一次；逻辑策略服务和训练费用另列。',
  '所有已记录慢调用wall与已知失败费用保留，不截尾；源wall不是GPU kernel时间或端到端并行延迟。',
  '训练controller CPU/wall与已记录source wall单列。最终选队未独立计时的字段保持未知，不填0。',
  'allocator清理仅在完整初始化后/首个generate前及fresh成功fsync后执行；不改模型、RNG、seed或token caps。',
  '旧107+输出前缀保留旧runner源码谱系；allocator修订当时六资格核心字节不变，不伪称同一runner版本。',
  '本包不含数据、问题/答案/gold、模型或已学习router权重、缓存、日志、PID、凭据或私有诊断。',
  '配置与静态JSON协议逐字节保留，发布SHA与原SHA一致；模型路径参与冻结指纹，迁移后须重新生成指纹和资格凭据，不能沿用旧凭据。',
  'GitHub尚需授权连接器实际推送并核验remote，本构建器不会发布。']
 return '\n'.join(lines)+'\n'


def portable_text(text,root):
 # Transform exported prose only; static JSON configuration bytes are preserved.
 for variant in sorted({str(root),str(root).replace('\\','/'),str(root).replace('\\','\\\\')},key=len,reverse=True):
  text=re.sub(re.escape(variant),'.',text,flags=re.I)
 return text


def _comparison_labels(stage,row,default_condition='iid_random_combo'):
 condition=row.get('condition_id',default_condition);kind=row.get('comparison_kind')
 if stage=='E5' and kind=='within_shift_same_caps':
  strategy=row['strategy']
  if not strategy.startswith('decision_vs_') or not strategy[len('decision_vs_'):]:raise ValueError('Unknown within-shift contrast label')
  return {'method':'decision','reference':strategy[len('decision_vs_'):],'method_condition':condition,'reference_condition':condition}
 if stage=='E5' and kind=='same_method_vs_reused_iid_random_warmup':
  return {'method':row['strategy'],'reference':row['strategy'],'method_condition':condition,'reference_condition':'iid_random_combo'}
 return {'method':row.get('strategy'),'reference':row.get('reference',row.get('strategy')),'method_condition':condition,'reference_condition':condition}


def render_documents(root,public):
 full=public['outcome']=='FULL_REQUIRED_EXPERIMENTS_VERIFIED';status='全部规定实验已独立验收；科学结论按证据保留阳性、阴性或不确定。' if full else '完整E1已独立重放并按预声明机会门合法停止；E2-E5与final未执行。'
 block='## 当前状态\n\n'+status+'\n\n最终本地验收为 `FINAL_ACCEPTANCE.json`；阅读 `EXPERIMENT_REPORT.txt` 与匿名汇总。测试/资格不等于科研完成。GitHub推送仍须实际执行并核验remote，未因生成此文档而声明已上传。\n\n'
 readme=(root/'README.md').read_text(encoding='utf-8-sig')
 if '## 当前状态' in readme:
  start=readme.index('## 当前状态');next_section=readme.find('\n## ',start+3);readme=readme[:start]+block+(readme[next_section:] if next_section>=0 else '')
 else:readme=readme+'\n'+block
 readme=readme.replace('## 后台运行范围','## 历史后台执行范围')
 index=(root/'PROJECT_INDEX.md').read_text(encoding='utf-8-sig')
 first=index.find('\n\n');index=index[:first+2]+status+' 以最终验收和匿名报告为准；原始证据保留本地。\n\n'+index[first+2:]
 entries=[('scripts/run_full_experiment.py','独立完整master：严格父gate、合法STOP、孤儿child跟随与四条件最终验收；不把dry-run当实验'),
  ('scripts/build_delivery.py','完整验收后才生成TXT/匿名统计与白名单SHA包；不推送GitHub'),
  ('scripts/resume_e1_aggregation.py','需新本地授权和精确进程绑定的聚合恢复入口；等待原worker结算后接续原尾部，不重跑E0或当前批次'),
  ('src/budget_router/recovery_runtime.py','独立恢复授权/owner/结算验证、只读QUERY及SYNCH句柄与有界状态replace重试；不重试模型推理'),
  ('scripts/launch_e1_recovery.ps1','本地隐藏启动恢复入口并保留原环境；授权、owner、状态、租约及日志不随公开交付包发布'),
  ('src/budget_router/scoring.py','safe_exact_v1_1在正式E1评分前补合法词内&；奖励映射可能改变，源版本需分别记录'),
  ('src/budget_router/protocol_lock.py','E0仅预检非封存参考并核对评分版本；封存校验延至合法final阶段，共享物理解码边界另作说明'),
  ('src/budget_router/runtime_memory.py','只在初始化完成和fresh成功fsync后清理unused allocator缓存；真实CPU/wall独立审计，缓存行不触CUDA'),
  ('src/budget_router/e5_validation.py','完整720消融、上游证据复用、pre-final门与opaque冻结批次授权'),
  ('scripts/run_e5_ablations.py','预声明paid-prior与两向任务顺序变化；不重跑已完成IID控制')]
 additional=[(name,description) for name,description in entries if '`'+name+'` |' not in index]
 if additional:index+='\n## 最终交付与运行安全索引\n\n| 文件 | 作用 |\n|---|---|\n'+''.join('| `'+name+'` | '+description+' |\n' for name,description in additional)
 index=index.replace('当前只读准备，未复现实验','受控适配证据以最终验收为准，不宣称原版native系统复现')
 return {'README.md':portable_text(readme,root),'PROJECT_INDEX.md':portable_text(index,root)}


def _stage_tables(context):
 root=context['root'];gates=context['acceptance']['stage_gates'];episode_rows=[];paired=[];ranking=[]
 e2=read_json(verified_reference(gates['E2']));e3=read_json(verified_reference(gates['E3']));e4=read_json(verified_reference(gates['E4']));e5=read_json(verified_reference(gates['E5_PRE_FINAL']))
 mandatory=read_json(verified_reference(e4['mandatory_e4_gate']));flex=read_json(verified_reference(e4['flex_reference_gate']))
 inventories=[('E2','iid_random_combo',e2['episode_receipts']),('E3','iid_random_combo',e3['episode_receipts']),
  ('E4','mandatory_shared_permissions',mandatory['episode_receipts']),('E4','flex_controlled_reference',flex['episode_receipts'])]
 inventories.extend(('E5',group['condition_id'],group['episode_receipts']) for group in e5['condition_grids'])
 for stage,condition,refs in inventories:
  for reference in refs:
   path=verified_reference(reference);receipt=read_json(path);artifact=receipt['artifacts']['summary.json'];summary_path=path.parent/'summary.json'
   if sha_file(summary_path)!=artifact['sha256']:raise ValueError('Stage summary differs from its accepted episode')
   summary=read_json(summary_path);spent=summary['budget']['spent']
   for task,quality in summary['per_task_quality'].items():
    episode_rows.append({'stage':stage,'condition':condition,'method':summary['strategy'],'algorithm_seed':summary['seed'],
     'label_rate':summary['label_rate'],'feedback_delay':summary['feedback_delay'],'task':task,'question_support':300,'quality':quality,
     'episode_total_executions':spent['executions'],'episode_total_tokens':spent['tokens'],'episode_total_labels':spent['labels'],
     'episode_total_controller_cpu_s':summary['controller_process_time_s'],'episode_total_controller_wall_s':summary['controller_wall_time_s'],
     'episode_flex_training_cpu_s':summary.get('runtime_measurements',{}).get('training_process_time_s'),
     'episode_flex_training_wall_s':summary.get('runtime_measurements',{}).get('training_wall_time_s'),
     'episode_online_source_service_wall_s':summary.get('source_service_execution_wall_time_s'),
     'episode_online_source_audit_wall_s':summary.get('source_audit_execution_wall_time_s'),
     'scope':'Episode totals repeated as context for each task; do not sum repeated totals. Actual logical spending, not equal hard caps. Missing source time stays empty.'})
 e2_report=read_json(verified_reference(e2['ranking_diagnostics']))
 for item in e2_report['episode_summaries']:
  for scope in ('unseen_question_ranking','previously_unlabelled_combination_ranking'):
   for task,values in item[scope]['per_task'].items():
    ranking.append({**item['spec'],'task':task,'scope':scope,**values,
     'episode_ordered_pair_support':item[scope]['pair_support'],
     'episode_mean_value_difference_mae':item[scope]['question_mean_value_difference_mae'],
     'note':'True ties omitted, predicted ties half credit; TopK proxy is not calibrated aggregate probability. Fixed support is paid-observed combos only.'})
 for stage,report_ref,keys in (('E3',e3['mechanism_evaluation'],('paired_comparisons',)),
  ('E4',mandatory['evaluation'],('paired_comparisons',)),('E5',e5['evaluation'],('versus_iid_comparisons','within_shift_comparisons'))):
  report=read_json(verified_reference(report_ref))
  for key in keys:
   for row in report[key]:
    for metric,interval in row['metrics'].items():
     bounds=interval.get('interval')
     paired.append({'stage':stage,'condition':row.get('condition_id','iid_random_combo'),'comparison_kind':row.get('comparison_kind',key),
      **_comparison_labels(stage,row),
      'label_rate':row['label_rate'],'feedback_delay':row['feedback_delay'],'metric':metric,
      'mean_difference':interval.get('mean'),'ci_low':None if bounds is None else bounds[0],'ci_high':None if bounds is None else bounds[1],
      'question_support':interval.get('question_support',0),'per_task_support':interval.get('per_task_support',{}),
      'algorithm_seeds':row['stream_seeds'],'adequate_support':interval.get('adequate_support',False),
      'scientific_verdict':row.get('strict_quality_priced_cost_dominance',row.get('quality_priced_cost_dominance_descriptive'))})
 return {'stage_per_seed_task.csv':episode_rows,'e2_ranking_support.csv':ranking,'stage_paired_comparisons.csv':paired}


def build_delivery(root=ROOT,acceptance_path=None,output_directory=None):
 root=Path(root).resolve();context=validate_acceptance(root,acceptance_path);accepted=context['acceptance'];full=accepted['status']=='FULL_REQUIRED_EXPERIMENTS_VERIFIED'
 output=Path(output_directory).resolve() if output_directory else root/'artifacts/delivery/final_review_v1'
 if not output.is_relative_to(root):raise ValueError('Delivery must remain within the experiment root')
 if output.exists():raise FileExistsError('Create a new reviewed delivery directory; never overwrite a published evidence package')
 public={'schema':'anonymous_experiment_delivery_v1','outcome':accepted['status'],'acceptance_sha256':context['acceptance_ref']['sha256'],
  'research_complete':accepted['research_complete'],'hypotheses':_hypotheses(context),'e1':_public_e1(context['e1']),
  'source_history':sanitized_source_history(root),'housekeeping':housekeeping_summary(root),
  'scientific_scope':'Complete execution is not an automatically positive scientific claim; all original negative/inconclusive results remain.'}
 tables={}
 if full:
  tables.update(_stage_tables(context))
  stat=context['statistics'];public['final']={k:v for k,v in stat.items() if k not in {'release','evaluation_sources','frozen_decision_receipt'}}
  public['physical_model_costs']=physical_summary(context['physical'],root)
  grading_keys=('status','completed_matrix_member_labels','completed_matrix_aggregate_labels','unique_reference_questions',
   'physical_member_label_evaluations','physical_aggregate_label_evaluations','physical_member_label_evaluations_bounds',
   'physical_aggregate_label_evaluations_bounds','process_time_s','wall_time_s','known_process_time_lower_bound_s',
   'known_wall_time_lower_bound_s','unfully_observed_attempts','policy_feedback_grants','model_judge_calls','scope')
  grading=[{'condition_id':item['condition_id'],**{key:item['physical_offline_grading_work'].get(key) for key in grading_keys}} for item in context['evaluations']]
  public['offline_private_grading_work']=grading;tables['final_private_grading_costs.csv']=grading
  tables['final_per_seed_task.csv']=stat['per_seed_task_actual_metrics'];tables['final_paired_comparisons.csv']=_paired_csv(stat)
  tables['second_aggregator_sensitivity.csv']=[{'method':x['method'],'label_rate':x['label_rate'],'feedback_delay':x['feedback_delay'],
   'difference':x['quality_difference']['mean'],'ci_low':x['quality_difference']['interval'][0],'ci_high':x['quality_difference']['interval'][1],
   'question_support':x['quality_difference']['question_support'],'scope':x['scope']} for x in stat['second_aggregator_sensitivity']]
  tables['physical_model_costs.csv']=public['physical_model_costs']['by_source_model_generation_task']
  tables['frozen_checkpoint_hashes.csv']=[{'method':g['method'],'strategy':g['strategy'],'source_stage':g['source_stage'],
   'label_rate':g['label_rate'],'feedback_delay':g['feedback_delay'],'algorithm_seed':entry['seed'],
   'checkpoint_sha256':entry['checkpoint']['sha256'],'episode_receipt_sha256':entry['episode_receipt']['sha256'],'grid_plan_sha256':entry['grid_plan']['sha256']}
   for g in context['release']['policy_groups'] for entry in g['checkpoints']]
 e1_metrics=[]
 for kind in ('metrics','paired_differences'):
  for method,value in context['e1'].get(kind,{}).items():
   for task in ('task_macro','gsm8k','bbh'):
    e1_metrics.append({'stage':'E1','metric_kind':kind,'method':method,'task':task,
     'mean':value.get('mean') if task=='task_macro' else value.get('per_task',{}).get(task),
     'ci_low':value.get('low') if task=='task_macro' else None,'ci_high':value.get('high') if task=='task_macro' else None,
     'source_scope':'Oracle is a diagnostic upper reference, not deployed performance','actual_generation_seeds':context['e1'].get('experimental_seeds')})
   per_seed=value.get('per_experiment_seed',[]);seeds=context['e1'].get('experimental_seeds',[])
   if per_seed and len(per_seed)!=len(seeds):raise ValueError('E1 actual-generation seed metric support changed')
   for seed,score in zip(seeds,per_seed):
    e1_metrics.append({'stage':'E1','metric_kind':kind,'method':method,'task':'task_macro','actual_generation_seed':seed,
     'mean':score,'ci_low':None,'ci_high':None,'source_scope':'Original per-realized-generation seed macro; per-seed task scores were not recorded and are not reconstructed from evaluation labels'})
 tables['e1_metrics.csv']=e1_metrics
 tables['e1_actual_control_costs.csv']=[{'kind':key,**value} for key,value in context['e1']['physical_diagnostic_cost'].get('by_kind',{}).items()]
 documents=render_documents(root,public);report=render_report(public);selected=source_whitelist(root)
 temporary=output.with_name('.'+output.name+'.pending.'+uuid.uuid4().hex);temporary.mkdir(parents=True)
 package=temporary/'package';package.mkdir();manifest=[]
 try:
  for relative in selected:
   safe_package_relative(relative);source=root/relative;target=package/relative;target.parent.mkdir(parents=True,exist_ok=True)
   if not source.resolve().is_relative_to(root):raise ValueError('Whitelisted source escaped root before copying')
   original_sha=sha_file(source)
   if relative in documents:target.write_text(documents[relative],encoding='utf8')
   elif source.suffix.lower()=='.json':
    validate_static_json_for_publication(read_json(source),root)
    shutil.copyfile(source,target)
   elif source.suffix.lower() in {'.md','.txt'}:target.write_text(portable_text(source.read_text(encoding='utf-8-sig'),root),encoding='utf8')
   else:shutil.copyfile(source,target)
   if sha_file(source)!=original_sha:raise ValueError('A delivery source changed during copying')
   manifest.append({'path':relative,'sha256':sha_file(target),'bytes':target.stat().st_size,'original_source_sha256':original_sha,
    'sanitized_or_final_status_updated':sha_file(target)!=original_sha})
  outputs={'EXPERIMENT_REPORT.txt':report,'summary/anonymous_summary.json':json.dumps(public,ensure_ascii=False,allow_nan=False,indent=2)+'\n',
   'summary/test_receipts.json':json.dumps({'software_receipts':_test_receipts(root),'research_results':False},ensure_ascii=False,indent=2)+'\n',
   'summary/source_history.json':json.dumps(public['source_history'],ensure_ascii=False,indent=2)+'\n'}
  outputs.update({'summary/'+name:csv_text(rows) for name,rows in tables.items()})
  for relative,text in outputs.items():
   safe_package_relative(relative);target=package/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf8',newline='' if relative.endswith('.csv') else None)
   manifest.append({'path':relative,'sha256':sha_file(target),'bytes':target.stat().st_size,'generated_from_acceptance_sha256':context['acceptance_ref']['sha256']})
  package_manifest={'schema':'reviewed_delivery_whitelist_v1','acceptance_sha256':context['acceptance_ref']['sha256'],'outcome':accepted['status'],
   'files':sorted(manifest,key=lambda x:x['path']),'github_pushed':False,
   'operation_only_sources_kept_private':sorted(OPERATION_ONLY_FILES),
   'excluded':'Raw questions/answers/gold, all model/router learned weights, source data caches, venv/tools/logs/private/PID/credentials. Only the two audited MIT CES source files plus LICENSE and source manifest are included from third_party.'}
  atomic_json(temporary/'DELIVERY_MANIFEST.json',package_manifest)
  with zipfile.ZipFile(temporary/'MAB_reviewed_delivery.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
   for row in package_manifest['files']:archive.write(package/row['path'],row['path'])
   archive.write(temporary/'DELIVERY_MANIFEST.json','DELIVERY_MANIFEST.json')
  (temporary/'MAB_reviewed_delivery.zip.sha256').write_text(sha_file(temporary/'MAB_reviewed_delivery.zip')+'  MAB_reviewed_delivery.zip\n',encoding='ascii')
  if output.exists():raise FileExistsError('Another builder published the delivery directory')
  os.rename(temporary,output)
  history=root/'artifacts/private/delivery_history'/uuid.uuid4().hex;history.mkdir(parents=True)
  for relative,text in documents.items():
   shutil.copyfile(root/relative,history/relative);(root/relative).write_text(text,encoding='utf8')
  (root/'EXPERIMENT_REPORT.txt').write_text(report,encoding='utf8')
  return {'status':'REVIEWED_DELIVERY_BUILT_NOT_PUSHED','output':str(output),'files':len(manifest),'manifest':artifact_reference(output/'DELIVERY_MANIFEST.json'),
   'archive':artifact_reference(output/'MAB_reviewed_delivery.zip'),'github_pushed':False,'accepted_outcome':accepted['status']}
 except BaseException:
  # Retain the pending directory for review; never remove or rewrite raw data.
  raise


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
 parser.add_argument('--acceptance',type=Path);parser.add_argument('--output-dir',type=Path)
 args=parser.parse_args(argv);result=build_delivery(args.root,args.acceptance,args.output_dir)
 print(json.dumps(result,ensure_ascii=False,allow_nan=False));return result


if __name__=='__main__':main()
