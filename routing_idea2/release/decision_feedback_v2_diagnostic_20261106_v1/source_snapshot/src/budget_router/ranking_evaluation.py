"""Offline diagnostics of already-frozen policy predictions.

This module never updates a learner. Hidden labels are evaluation targets only.
"""
from collections import defaultdict
import csv
import hashlib
import math
from pathlib import Path
import numpy as np


def _mean(values):
 values=[float(v) for v in values if v is not None]
 return None if not values else float(np.mean(values))


def ranking_metrics(predictions,targets,*,supported=None):
 """Tie-safe within-question ordering; missing predictions never become zeros."""
 if len(predictions)!=len(targets):raise ValueError('Prediction/target dimension mismatch')
 y=np.asarray(targets,dtype=float)
 if not np.isfinite(y).all():raise ValueError('Nonfinite evaluation target')
 indices=list(range(len(y))) if supported is None else list(supported)
 if len(indices)!=len(set(indices)) or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or i<0 or i>=len(y) for i in indices):
  raise ValueError('Invalid prediction support')
 valid=[]
 for index in indices:
  value=predictions[index]
  if value is None:continue
  if isinstance(value,bool) or not isinstance(value,(float,int,np.number)) or not math.isfinite(float(value)):
   raise ValueError('Invalid prediction')
  valid.append(int(index))
 pair_credit=[];gap_errors=[];variable_gap_errors=[]
 for p,i in enumerate(valid):
  for j in valid[p+1:]:
   truth=float(y[i]-y[j]);estimate=float(predictions[i]-predictions[j]);error=abs(estimate-truth);gap_errors.append(error)
   if truth==0:continue
   variable_gap_errors.append(error)
   pair_credit.append(.5 if estimate==0 else float((estimate>0)==(truth>0)))
 if valid:
  best_prediction=max(float(predictions[i]) for i in valid)
  ties=[i for i in valid if float(predictions[i])==best_prediction]
  selected=min(ties)
  top_regret=float(np.max(y)-y[selected])
  optimistic_regret=float(np.max(y)-max(float(y[i]) for i in ties))
  pessimistic_regret=float(np.max(y)-min(float(y[i]) for i in ties))
 else:selected=None;top_regret=optimistic_regret=pessimistic_regret=None
 return {'prediction_support':len(valid),'ordered_pair_support':len(pair_credit),'all_pair_support':len(gap_errors),
  'pairwise_accuracy':_mean(pair_credit),'value_difference_mae':_mean(gap_errors),
  'non_tied_value_difference_mae':_mean(variable_gap_errors),'predicted_top_index_lexicographic':selected,
  'top1_regret_lexicographic':top_regret,'top1_regret_best_prediction_tie':optimistic_regret,
  'top1_regret_worst_prediction_tie':pessimistic_regret,'all_targets_constant':bool(np.all(y==y[0]))}


def evaluate_episode_predictions(result,environment):
 if environment.split!='online_stream':raise ValueError('Ranking targets must be the isolated online split')
 ids=environment._arrays['question_ids'].tolist()
 hashed={hashlib.sha256(('routing-question-v1:'+str(q)).encode()).hexdigest():i for i,q in enumerate(ids)}
 if len(hashed)!=len(ids):raise ValueError('Duplicate question identities')
 rows=[];seen=set()
 for trace in result['traces']:
  qid=trace['question_id']
  if qid not in hashed or qid in seen:raise ValueError('Trace question missing, repeated or outside online split')
  seen.add(qid);row=hashed[qid]
  if trace.get('prediction_snapshot_precedes_current_execution_and_feedback') is not True:
   raise ValueError('Prediction snapshot is not prospective')
  if trace['task']!=str(environment._arrays['task_labels'][row]):raise ValueError('Trace task identity mismatch')
  prediction=trace['predicted_combination_scores_before_decision'];support=trace.get('prediction_supported_combination_indices',list(range(20)))
  trained=trace['arrived_combination_indices_before_decision']
  if len(trained)!=len(set(trained)) or any(not isinstance(i,int) or not 0<=i<20 for i in trained):raise ValueError('Invalid arrived-combination evidence')
  # Private arrays are read only here, after the episode has frozen every action.
  truth=environment._arrays['aggregate_scores'][row]
  full=ranking_metrics(prediction,truth,supported=support)
  unseen=ranking_metrics(prediction,truth,supported=[i for i in support if i not in set(trained)])
  main=trace['main_index'];members=list(environment.mapper.subsets[main])
  actual=float(truth[main]);covered=float(np.any(environment._arrays['member_scores'][row,members]>0))
  if actual!=trace['quality'] or members!=trace['main_members']:raise ValueError('Trace service reward/selection differs from bound matrix')
  proxy=trace['prediction_semantics'].startswith('mean_member_prediction')
  if proxy:
   for metric in (full,unseen):
    metric['ranking_proxy_difference_mae']=metric['value_difference_mae'];metric['value_difference_mae']=None
    metric['non_tied_ranking_proxy_difference_mae']=metric['non_tied_value_difference_mae'];metric['non_tied_value_difference_mae']=None
  rows.append({'question_id':qid,'task':trace['task'],'step':trace['step'],
   'prediction_semantics':trace['prediction_semantics'],'combination_predictions_are_aggregate_reward_model':not proxy and trace['prediction_semantics'].startswith('shared_second_order'),
   'unseen_question_ranking':full,'previously_unlabelled_combination_ranking':unseen,
   'difference_error_scope':'member-score ranking proxy discrepancy, not calibrated aggregate-value error' if proxy else 'model-predicted difference versus actual aggregate-score difference',
   'service_quality':actual,'service_oracle_regret':float(np.max(truth)-actual),
   'chosen_member_coverage':covered,'coverage_minus_aggregate_quality':covered-actual,
   'source':'full private matrix used by evaluator only; not revealed to policy'})
 if seen!=set(hashed):raise ValueError('Episode lacks complete online prediction coverage')
 return rows


def summarize_prediction_rows(rows):
 out={}
 for scope in ('unseen_question_ranking','previously_unlabelled_combination_ranking'):
  supported=[r for r in rows if r[scope]['pairwise_accuracy'] is not None]
  out[scope]={'questions':len(rows),'rankable_questions':len(supported),
   'pair_support':sum(r[scope]['ordered_pair_support'] for r in rows),
   'question_mean_pairwise_accuracy':_mean([r[scope]['pairwise_accuracy'] for r in supported]),
   'question_mean_value_difference_mae':_mean([r[scope]['value_difference_mae'] for r in rows]),
   'question_mean_top1_regret_lexicographic':_mean([r[scope]['top1_regret_lexicographic'] for r in rows]),
   'per_task':{task:{'rankable_questions':sum(r['task']==task for r in supported),
    'question_mean_pairwise_accuracy':_mean([r[scope]['pairwise_accuracy'] for r in supported if r['task']==task])}
    for task in sorted({r['task'] for r in rows})}}
 out['service_quality_task_macro']=float(np.mean([np.mean([r['service_quality'] for r in rows if r['task']==task]) for task in sorted({r['task'] for r in rows})]))
 out['coverage_minus_quality_mean']=_mean([r['coverage_minus_aggregate_quality'] for r in rows])
 return out


def stratified_question_interval(rows,*,value_key,replicates=2000,seed=20261030,confidence=.95,minimum_per_task=20):
 """Average repeated stream seeds within question before paired task bootstrap."""
 if replicates<2 or not 0<confidence<1:raise ValueError('Invalid bootstrap settings')
 grouped=defaultdict(list);tasks={}
 for row in rows:
  value=row.get(value_key)
  if value is None:continue
  if not math.isfinite(float(value)):raise ValueError('Nonfinite metric')
  q=row['question_id']
  if q in tasks and tasks[q]!=row['task']:raise ValueError('Question task changes across seeds')
  tasks[q]=row['task'];grouped[q].append(float(value))
 strata={task:np.array([np.mean(values) for q,values in grouped.items() if tasks[q]==task]) for task in sorted(set(tasks.values()))}
 if not strata:return {'mean':None,'interval':None,'question_support':0,'per_task_support':{},'adequate_support':False}
 rng=np.random.default_rng(seed);draws=np.mean([rng.choice(v,(replicates,len(v)),replace=True).mean(axis=1) for v in strata.values()],axis=0)
 alpha=(1-confidence)/2
 return {'mean':float(np.mean([v.mean() for v in strata.values()])),
  'interval':[float(x) for x in np.quantile(draws,[alpha,1-alpha])],
  'question_support':len(grouped),'per_task_support':{task:len(v) for task,v in strata.items()},
  'adequate_support':len(strata)==2 and all(len(v)>=minimum_per_task for v in strata.values()),
  'confidence':confidence,'bootstrap_replicates':replicates,
  'scope':'conditional on one pinned generation cache; question unit; stream seeds averaged within question; stratified task macro'}


def write_cost_curve(result,path):
 """Resource-vector curve; weighted model/feedback cost does not price CPU."""
 path=Path(path);weights=result['summary']['settings']['cost_weights'];measure=result['summary'].get('runtime_measurements',{})
 cpu=measure.get('initialization_controller_process_time_s',0.);wall=measure.get('initialization_controller_wall_time_s',0.);qualities=[];tasks=defaultdict(list)
 fields=('served_questions','question_id','task','quality','prefix_quality_micro','prefix_quality_task_macro','logical_executions','logical_labels','logical_tokens','weighted_model_feedback_cost','controller_cpu_s','controller_wall_s','source_service_wall_s','source_audit_wall_s')
 with path.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for trace in result['traces']:
   cpu+=trace['controller_process_time_s'];wall+=trace['controller_wall_time_s'];qualities.append(trace['quality']);tasks[trace['task']].append(trace['quality']);cost=trace['cumulative_cost']
   writer.writerow({'served_questions':trace['step']+1,'question_id':trace['question_id'],'task':trace['task'],'quality':trace['quality'],
    'prefix_quality_micro':float(np.mean(qualities)),'prefix_quality_task_macro':float(np.mean([np.mean(v) for v in tasks.values()])),
    'logical_executions':cost['executions'],'logical_labels':cost['labels'],'logical_tokens':cost['tokens'],
    'weighted_model_feedback_cost':sum(w*cost[k] for w,k in zip(weights,('executions','labels','tokens'))),
    'controller_cpu_s':cpu,'controller_wall_s':wall,'source_service_wall_s':trace.get('source_service_execution_wall_time_s'),
    'source_audit_wall_s':trace.get('source_audit_execution_wall_time_s')})


def _actual_resource(environment,rows,combination_indices=None,*,labels=0,all_members=False):
 """Counts source attempts once per generated member/aggregate cell, never 20x members."""
 a=environment._arrays;total={'executions':0,'labels':int(labels),'tokens':0};wall=0.;complete_wall=True
 for offset,row in enumerate(rows):
  indices=list(range(20)) if combination_indices is None else [int(combination_indices[offset])]
  members=list(range(6)) if all_members else list(environment.mapper.subsets[indices[0]])
  for stem,cells in (('member',members),('aggregate',indices)):
   count=a.get(stem+'_execution_counts')
   if count is None:raise ValueError('Actual source execution counts are absent')
   total['executions']+=int(np.sum(count[row,cells]));total['tokens']+=int(np.sum(a[stem+'_prompt_tokens'][row,cells])+np.sum(a[stem+'_completion_tokens'][row,cells]))
   measured=a.get(stem+'_wall_time_s')
   if measured is None:complete_wall=False
   else:wall+=float(np.sum(measured[row,cells]))
 return {'resources':total,'source_execution_wall_time_s':wall if complete_wall else None,
  'wall_time_status':'MEASURED' if complete_wall else 'MISSING_NOT_FILLED_WITH_ZERO',
  'scope':'Sum of unique source cells/known retries. This is the resource equivalent of training/serving, not an extra physical bill on top of already generated project matrices.'}


def rich_fixed_references(calibration,online,*,main_feedback_natural=False):
 """High-information external controls; never return initialization labels/state."""
 if calibration.split!='online_calibration' or online.split!='online_stream':raise ValueError('Fixed references require isolated calibration and online splits')
 cal=calibration._arrays;on=online._arrays
 tasks=sorted(set(cal['task_labels'].tolist()))
 means=np.mean([cal['aggregate_scores'][cal['task_labels']==task].mean(axis=0) for task in tasks],axis=0)
 chosen=int(np.argmax(means));n=len(online)
 q={task:float(on['aggregate_scores'][on['task_labels']==task,chosen].mean()) for task in sorted(set(on['task_labels'].tolist()))}
 offline=_actual_resource(calibration,list(range(len(calibration))),labels=len(calibration)*20,all_members=True)
 serving=_actual_resource(online,list(range(n)),[chosen]*n,labels=n if main_feedback_natural else 0)
 hindsight_means=np.mean([on['aggregate_scores'][on['task_labels']==task].mean(axis=0) for task in sorted(set(on['task_labels'].tolist()))],axis=0)
 hindsight=int(np.argmax(hindsight_means))
 return {'rich_calibration_best_fixed_reference':{'status':'REFERENCE_MEASURED','combination_index':chosen,
  'members':list(online.mapper.subsets[chosen]),'calibration_task_macro_quality':float(means[chosen]),
  'online_quality_task_macro':float(np.mean(list(q.values()))),'online_per_task_quality':q,
  'additional_calibration_final_scores_accessed':len(calibration)*20,'online_optional_feedback_purchased':0,
  'offline_training_resource_equivalent':offline,'online_service_resource_equivalent':serving,
  'same_budget_primary_strategy':False,'labels_transferred_to_primary_learner':False,
  'interpretation':'Rich full-calibration static reference with extra 4000 labels; not the shared small paid-warmup fixed primary baseline. Precomputed physical generation is reported separately and must not be double billed.'},
  'online_hindsight_best_fixed_upper_reference':{'combination_index':hindsight,
   'quality_task_macro':float(hindsight_means[hindsight]),'selection_evaluation_labels_accessed':len(online)*20,
   'deployable':False,'labels_transferred_to_primary_learner':False,
   'interpretation':'Evaluation-label-selected fixed upper reference, never an online policy or same-budget result.'}}


def strict_quality_cost_dominance(quality_interval,cost_interval):
 """Single declared contrast: strictly better quality with no higher priced cost.

 This is an engineering diagnostic conditional on a pinned generation cache.
 CPU and missing source wall times are reported separately, never priced as zero.
 """
 if not quality_interval.get('adequate_support') or not cost_interval.get('adequate_support'):
  return 'INCONCLUSIVE'
 q=quality_interval.get('interval');c=cost_interval.get('interval')
 if q is None or c is None:return 'INCONCLUSIVE'
 if len(q)!=2 or len(c)!=2 or not np.isfinite(q+c).all() or q[0]>q[1] or c[0]>c[1]:
  raise ValueError('Invalid quality/cost uncertainty interval')
 if q[0]>0 and c[1]<=0:return 'POSITIVE'
 if q[1]<0 and c[0]>=0:return 'NEGATIVE'
 return 'INCONCLUSIVE'
