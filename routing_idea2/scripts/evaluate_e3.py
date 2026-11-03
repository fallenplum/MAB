"""Independently accept a complete E3 grid and compare actual quality/cost.

No policy is initialized or updated. E2 controls are reused from the fully
revalidated E2 gate. Technical completeness never requires a positive effect.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.simulation import MatrixEnvironment
from budget_router.ranking_evaluation import stratified_question_interval,strict_quality_cost_dominance
from budget_router.stage_validation import (EpisodeStore,read_json,atomic_json,immutable_json,
 verify_complete_grid,validate_locked_grid,episode_specs,episode_name,input_identity,canonical_digest,
 artifact_reference,verified_reference,validate_e2_gate,validate_e3_gate,verified_grid_summary,e3_bindings)

PRIMARY_POLICY={'label_rate':25,'feedback_delay':0,'strategy':'E3/decision','reference':'E3/max_uncertainty_per_cost',
 'confidence':.95,'minimum_questions_per_task':20,
 'positive':'quality difference CI lower > 0 AND actual priced logical cost difference CI upper <= 0',
 'negative':'quality difference CI upper < 0 AND actual priced logical cost difference CI lower >= 0',
 'otherwise':'INCONCLUSIVE','provenance':'Engineering diagnostic agreed before E3 evaluation; not a teacher-specified rule',
 'scope':'One preregistered primary contrast/cell conditional on one pinned generation cache; other contrasts/cells are descriptive',
 'priced_resources':'Locked execution/label/token weights; actual controller CPU and source wall times are reported separately and are not free or included in this three-resource price'}
REFERENCES=('E3/none','E3/random','E3/periodic','E3/max_uncertainty','E3/max_uncertainty_per_cost','E3/no_cost',
 'E2/fixed_combination','E2/contextual_topk','E2/linucb','E2/thompson')
METRICS=('quality','weighted_model_feedback_cost','executions','labels','tokens','controller_cpu_s','controller_wall_s','source_execution_wall_time_s')


def _id_rows(environment):
 return {hashlib.sha256(('routing-question-v1:'+str(q)).encode()).hexdigest():i for i,q in enumerate(environment._arrays['question_ids'])}


def _cell_resource(environment,row,stem,indices):
 a=environment._arrays
 return {'executions':int(np.sum(a[stem+'_execution_counts'][row,indices])),
  'tokens':int(np.sum(a[stem+'_prompt_tokens'][row,indices])+np.sum(a[stem+'_completion_tokens'][row,indices]))}


def _cell_wall(environment,row,stem,indices):
 if not indices:return 0.
 value=environment._arrays.get(stem+'_wall_time_s')
 return None if value is None else float(np.sum(value[row,indices]))


def _sum_known(values):
 return None if any(v is None for v in values) else float(sum(values))


def project_episode(result,online,calibration):
 """Independently replay only frozen service/resource arithmetic, never a policy."""
 s=result['summary'];n=len(online);weights=s['settings']['cost_weights'];by_online=_id_rows(online);by_cal=_id_rows(calibration)
 warmup={'executions':0,'labels':0,'tokens':0};warmup_wall=[];seen_cal=set()
 for record in result['warmup']:
  qid=record['question_id'];index=record['combination_index']
  if qid not in by_cal or qid in seen_cal or not isinstance(index,int) or not 0<=index<20:
   raise ValueError('Warmup source is not a unique legal calibration observation')
  seen_cal.add(qid);row=by_cal[qid];members=list(calibration.mapper.subsets[index]);warmup['labels']+=1
  for stem,indices in (('member',members),('aggregate',[index])):
   actual=_cell_resource(calibration,row,stem,indices)
   for key,value in actual.items():warmup[key]+=value
   warmup_wall.append(_cell_wall(calibration,row,stem,indices))
 if warmup!=s['warmup_cost']:raise ValueError('Warmup logical costs differ from actual source cells')
 warmup_time=_sum_known(warmup_wall);totals=dict(warmup);rows=[];seen=set()
 runtime=s['runtime_measurements'];initial_cpu=runtime['initialization_controller_process_time_s'];initial_wall=runtime['initialization_controller_wall_time_s']
 for trace in result['traces']:
  qid=trace['question_id'];main=trace['main_index'];alternate=trace['alternate_index']
  if qid not in by_online or qid in seen or not isinstance(main,int) or not 0<=main<20:
   raise ValueError('Frozen trace does not identify a unique online service')
  seen.add(qid);row=by_online[qid];members=list(online.mapper.subsets[main]);truth=float(online._arrays['aggregate_scores'][row,main])
  if trace['quality']!=truth or trace['main_members']!=members or trace['task']!=str(online._arrays['task_labels'][row]):
   raise ValueError('Frozen service/quality differs from bound online matrix')
  service={'executions':0,'labels':0,'tokens':0};extra={'executions':0,'labels':trace['requested_combination_labels'],'tokens':0};source_times=[]
  for stem,indices in (('member',members),('aggregate',[main])):
   actual=_cell_resource(online,row,stem,indices)
   for key,value in actual.items():service[key]+=value
   source_times.append(_cell_wall(online,row,stem,indices))
  if alternate is not None:
   if not isinstance(alternate,int) or not 0<=alternate<20 or alternate==main:raise ValueError('Invalid frozen alternate')
   new_members=sorted(set(online.mapper.subsets[alternate])-set(members))
   for stem,indices in (('member',new_members),('aggregate',[alternate])):
    actual=_cell_resource(online,row,stem,indices)
    for key,value in actual.items():extra[key]+=value
    source_times.append(_cell_wall(online,row,stem,indices))
  if trace['service_cost']!=service or trace['purchase_cost']!=extra:raise ValueError('Frozen service/audit cost differs from actual source cells')
  step={key:service[key]+extra[key] for key in totals}
  for key,value in step.items():totals[key]+=value
  if totals!=trace['cumulative_cost']:raise ValueError('Cumulative cost does not equal actual warmup/service/audit resources')
  allocated={key:step[key]+warmup[key]/n for key in step}
  rows.append({'question_id':qid,'task':trace['task'],'quality':truth,**allocated,
   'weighted_model_feedback_cost':sum(w*allocated[k] for w,k in zip(weights,('executions','labels','tokens'))),
   'controller_cpu_s':trace['controller_process_time_s']+initial_cpu/n,
   'controller_wall_s':trace['controller_wall_time_s']+initial_wall/n,
   'source_execution_wall_time_s':_sum_known([_sum_known(source_times),None if warmup_time is None else warmup_time/n]),
   'main_index':main,'alternate_index':alternate,'action_kind':trace['action_kind'],
   'cumulative_cost':trace['cumulative_cost']})
 if seen!=set(by_online) or totals!=s['budget']['spent']:raise ValueError('Incomplete service or total resource ledger')
 if any(totals[k]>s['budget']['limits'][k] for k in totals):raise ValueError('Actual spending exceeds a hard resource limit')
 return {'summary':s,'traces':rows}


def verify_reused_none(e2,e3):
 """The shared baseline must reproduce frozen actions/resources, not clock values."""
 def indexed(results):
  return {(r['summary']['seed'],r['summary']['label_rate'],r['summary']['feedback_delay']):r for r in results if r['summary']['strategy']=='none'}
 left=indexed(e2);right=indexed(e3)
 if set(left)!=set(right):raise ValueError('E2/E3 shared-baseline cells differ')
 keys=('question_id','task','quality','main_index','alternate_index','action_kind','cumulative_cost')
 for cell in left:
  if [{k:r[k] for k in keys} for r in left[cell]['traces']]!=[{k:r[k] for k in keys} for r in right[cell]['traces']]:
   raise ValueError('E2/E3 shared baseline changed frozen decisions/resources')


def paired_comparisons(e3,e2,*,replicates=2000,seed=20261030,confidence=.95):
 grouped={}
 for stage,results in (('E3',e3),('E2',e2)):
  for result in results:
   s=result['summary'];key=(stage+'/'+s['strategy'],s['label_rate'],s['feedback_delay'])
   cell=grouped.setdefault(key,{})
   if s['seed'] in cell:raise ValueError('Duplicate method/stream seed')
   cell[s['seed']]=result
 comparisons=[]
 cells=sorted({(r['summary']['label_rate'],r['summary']['feedback_delay']) for r in e3})
 for rate,delay in cells:
  treated=grouped[('E3/decision',rate,delay)]
  for reference in REFERENCES:
   controls=grouped[(reference,rate,delay)]
   if set(treated)!=set(controls) or len(treated)!=5:raise ValueError('Paired comparison needs the same five complete stream seeds')
   differences=[]
   for stream_seed in sorted(treated):
    a={r['question_id']:r for r in treated[stream_seed]['traces']};b={r['question_id']:r for r in controls[stream_seed]['traces']}
    if set(a)!=set(b):raise ValueError('Paired methods do not cover the same entire question set')
    for qid,row in a.items():
     if row['task']!=b[qid]['task']:raise ValueError('Question task mismatch')
     difference={'question_id':qid,'task':row['task']}
     for metric in METRICS:
      av,bv=row[metric],b[qid][metric];difference[metric+'_difference']=None if av is None or bv is None else av-bv
     differences.append(difference)
   metrics={metric+'_difference':stratified_question_interval(differences,value_key=metric+'_difference',
    replicates=replicates,seed=seed,confidence=confidence,minimum_per_task=20) for metric in METRICS}
   comparisons.append({'strategy':'E3/decision','reference':reference,'label_rate':rate,'feedback_delay':delay,
    'stream_seeds':sorted(treated),'question_count':len(a),'metrics':metrics,
    'strict_quality_priced_cost_dominance':strict_quality_cost_dominance(metrics['quality_difference'],metrics['weighted_model_feedback_cost_difference']),
    'spending_units':'Actual resource/weighted cost per served question; common calibration warmup amortized across all N questions; clocks separate',
    'not_equal_actual_spending_by_assumption':True})
 return comparisons


def evaluate(directory,gate_path=None):
 directory=Path(directory).resolve();out=directory/'e3_diagnostics';attempt_id=uuid.uuid4().hex;cpu=time.process_time();wall=time.perf_counter()
 try:
  identity,_,receipts=verify_complete_grid(directory)
  config=read_json(identity['inputs']['assets']['config']['path']);validate_locked_grid(identity,config,'e3')
  a=identity['inputs']['assets']
  current=input_identity(ROOT,online_matrix=a['online_matrix']['path'],online_manifest=a['online_manifest']['path'],
   calibration_matrix=a['calibration_matrix']['path'],calibration_manifest=a['calibration_manifest']['path'],config_path=a['config']['path'])
  if current!=identity['inputs']:raise ValueError('Current sources/config/assets differ from E3 execution')
  policy_path=out/'diagnostic_policy.json';immutable_json(policy_path,PRIMARY_POLICY)
  e2_reference=identity.get('e2_gate')
  if not e2_reference:raise ValueError('E3 execution does not bind an E2 gate')
  e2_gate,_=validate_e2_gate(verified_reference(e2_reference),current)
  e2_directory=verified_reference(e2_gate['grid_plan']).parent;e2_identity=read_json(e2_directory/'grid_plan.json')
  online=MatrixEnvironment.from_npz(a['online_matrix']['path'],a['online_manifest']['path'],expected_split='online_stream')
  calibration=MatrixEnvironment.from_npz(a['calibration_matrix']['path'],a['calibration_manifest']['path'],expected_split='online_calibration')
  if online.software_test_only or calibration.software_test_only or len(online)!=600 or len(calibration)!=200:
   raise ValueError('Full-scale production matrices required; toy input forbidden')
  if set(online._arrays['task_labels'].tolist())!={'gsm8k','bbh'} or sorted(np.unique(online._arrays['task_labels'],return_counts=True)[1].tolist())!=[300,300]:
   raise ValueError('Question-level cost task macro requires the original balanced 300+300 online split')
  projected={};runtime=[]
  for stage,folder,plan in (('E2',e2_directory,e2_identity),('E3',directory,identity)):
   store=EpisodeStore(folder,plan,resume=True);projected[stage]=[]
   for spec in episode_specs(plan['grid']):
    result=store.load(spec);projected[stage].append(project_episode(result,online,calibration))
    s=result['summary'];runtime.append({'stage':stage,'spec':spec,'quality_task_macro':s['quality_task_macro'],
     'actual_spent':s['budget']['spent'],'hard_limits':s['budget']['limits'],'runtime_measurements':s['runtime_measurements'],
     'source_service_execution_wall_time_s':s.get('source_service_execution_wall_time_s'),
     'source_audit_execution_wall_time_s':s.get('source_audit_execution_wall_time_s')})
    print(json.dumps({'independently_verified':stage+'/'+episode_name(spec),'research_complete':False}),flush=True)
  verify_reused_none(projected['E2'],projected['E3'])
  stats=config.get('statistics',{});comparisons=paired_comparisons(projected['E3'],projected['E2'],
   replicates=stats.get('e1_bootstrap_replicates',2000),seed=stats.get('e1_bootstrap_seed',20261030),confidence=.95)
  primary=next(x for x in comparisons if all(x[k]==PRIMARY_POLICY[k] for k in ('reference','label_rate','feedback_delay')))
  if len(comparisons)!=120:raise ValueError('Missing original-grid paired comparisons')
  original_e2_report=read_json(verified_reference(e2_gate['ranking_diagnostics']))
  report={'schema':'e3_budget_mechanism_evaluation_v1','run_identity_sha256':identity['identity_sha256'],
   'episodes_evaluated':len(projected['E3']),'e2_controls_reused_without_rerunning':True,
   'physical_generation_replicates':1,'algorithm_stream_seeds':identity['grid']['seeds'],
   'physical_vs_logical_scope':'No new physical model inference occurs in these CPU simulations. Actual selected source-cell resources are charged logically in every policy stream; full shared-cache physical generation remains in the original provenance/cost receipts and is not double billed.',
   'e2_control_grid_plan':e2_gate['grid_plan'],'e2_control_episode_receipts':e2_gate['episode_receipts'],
   'e2_ranking_report':e2_gate['ranking_diagnostics'],'diagnostic_policy':artifact_reference(policy_path),
   'academic_conclusion':primary['strict_quality_priced_cost_dominance'],'academic_primary_comparison':primary,
   'quality_and_cost_comparisons_complete':True,'comparison_count':len(comparisons),'paired_comparisons':comparisons,
   'runtime_and_actual_resources_by_episode':runtime,
   'rich_calibration_reference_from_e2':original_e2_report.get('references',{}),
   'quality_cost_curves':'Every committed episode contains hash-bound cost_curve.csv with quality, cumulative actual executions/labels/tokens, priced cost, and measured controller CPU/wall.',
   'causal_scope':'Descriptive algorithm comparisons under the same hard resource caps. Actual spending can differ; no positive scientific claim follows solely from an equal limit or one favorable cell.',
   'clock_scope':'Controller CPU/wall and source model wall times are separate. Missing source times remain null. Quality/cost primary conclusion prices only the three locked resources; full CPU/wall impacts remain visible.',
   'warmup_allocation_scope':'For paired per-question spending/CPU bootstrap only, each method actual calibration warmup is amortized uniformly across its 600 served questions; total actual spending is also reported independently.',
   'technical_gate_independent_of_academic_positivity':True,'research_complete':False}
  report_path=out/'mechanism_report.json';immutable_json(report_path,report)
  summary=verified_grid_summary(directory,identity,receipts)
  gate={'schema':'e3_budget_mechanism_gate_v1','status':'E3_TECHNICAL_PASS','software_test_only':False,
   'scope':'technical_completeness_not_scientific_positivity','run_identity_sha256':identity['identity_sha256'],
   'input_identity_sha256':canonical_digest(current),'grid_plan':artifact_reference(directory/'grid_plan.json'),
   'grid_summary':summary,'episode_receipts':receipts,'e2_gate':e2_reference,
   'mechanism_evaluation':artifact_reference(report_path),'bindings':e3_bindings(identity,receipts,summary),
   'academic_conclusion':report['academic_conclusion'],'research_complete':False}
  gate_path=Path(gate_path).resolve() if gate_path else directory/'e3_gate.json'
  candidate=gate_path.with_name(gate_path.name+'.candidate.'+attempt_id);immutable_json(candidate,gate)
  validate_e3_gate(candidate,current);immutable_json(gate_path,gate);candidate.unlink()
  atomic_json(out/'evaluation_attempts'/(attempt_id+'.json'),{'status':'E3_TECHNICAL_PASS','gate':artifact_reference(gate_path),
   'process_time_s':time.process_time()-cpu,'wall_time_s':time.perf_counter()-wall,'research_complete':False})
  return gate
 except BaseException as error:
  atomic_json(out/'evaluation_attempts'/(attempt_id+'.json'),{'status':'PARTIAL_E3_NO_ACCEPTED_GATE','error':type(error).__name__+': '+str(error),
   'process_time_s':time.process_time()-cpu,'wall_time_s':time.perf_counter()-wall,'research_complete':False})
  raise


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-dir',required=True);p.add_argument('--output-gate')
 args=p.parse_args(argv);gate=evaluate(args.runs_dir,args.output_gate)
 print(json.dumps({'status':gate['status'],'academic_conclusion':gate['academic_conclusion'],'research_complete':False}),flush=True)


if __name__=='__main__':
 try:main()
 except (ValueError,FileNotFoundError,FileExistsError,KeyError) as error:
  print(json.dumps({'status':'PARTIAL_E3_NO_ACCEPTED_GATE','error':str(error),'research_complete':False}),file=sys.stderr)
  raise SystemExit(2)
