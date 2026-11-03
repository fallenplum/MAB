"""Independently evaluate a complete E2 grid and issue a strictly bound gate.

No policy is initialized or updated here. Frozen predecision trace predictions
are evaluated against isolated online matrices. Technical completeness and
scientific learnability conclusions have separate status fields.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.simulation import MatrixEnvironment
from budget_router.ranking_evaluation import (evaluate_episode_predictions,summarize_prediction_rows,
 stratified_question_interval,rich_fixed_references)
from budget_router.stage_validation import (EpisodeStore,read_json,atomic_json,immutable_json,
 verify_complete_grid,validate_protocol_config,validate_locked_grid,expected_grid,settings_from_config,episode_specs,
 episode_name,input_identity,canonical_digest,artifact_reference,e2_bindings,validate_e2_gate)


def learnability_conclusion(interval):
 if not interval.get('adequate_support') or interval.get('interval') is None:return 'INCONCLUSIVE'
 low,high=interval['interval']
 if low>.5:return 'SUPPORTED'
 if high<=.5:return 'NOT_SUPPORTED'
 return 'INCONCLUSIVE'


def evaluate(directory,gate_path=None):
 directory=Path(directory).resolve();diagnostics_dir=directory/'e2_diagnostics'
 attempt_start=time.perf_counter();cpu_start=time.process_time();attempt_id=uuid.uuid4().hex
 try:
  identity,compact,receipts=verify_complete_grid(directory)
  config=read_json(identity['inputs']['assets']['config']['path']);validate_protocol_config(config)
  validate_locked_grid(identity,config,'e2')
  a=identity['inputs']['assets']
  current=input_identity(ROOT,online_matrix=a['online_matrix']['path'],online_manifest=a['online_manifest']['path'],
   calibration_matrix=a['calibration_matrix']['path'],calibration_manifest=a['calibration_manifest']['path'],config_path=a['config']['path'])
  if current!=identity['inputs']:raise ValueError('Current sources/config/assets differ from recorded E2 execution')
  online=MatrixEnvironment.from_npz(a['online_matrix']['path'],a['online_manifest']['path'],expected_split='online_stream')
  calibration=MatrixEnvironment.from_npz(a['calibration_matrix']['path'],a['calibration_manifest']['path'],expected_split='online_calibration')
  if online.software_test_only or calibration.software_test_only:raise ValueError('Production gate rejects toy matrices')
  if len(online)!=600 or len(calibration)!=200:raise ValueError('Acceptance cannot shrink the original protocol')
  store=EpisodeStore(directory,identity,resume=True);episode_refs=[];episode_summaries=[];primary=[];coverage=[]
  primary_policy={'strategy':'none','label_rate':25,'feedback_delay':0,
   'metric':'question_mean_pairwise_accuracy_over_supported_combinations_on_unseen_online_questions',
   'null_accuracy':.5,'confidence':.95,'minimum_rankable_questions_per_task':20,
   'provenance':'Engineering diagnostic declared before E2 evaluation; not a teacher-specified threshold or a technical gate requirement',
   'scope':'Shared second-order paid-warmup/main-feedback ridge only; conditional on this pinned generation cache'}
  for spec in episode_specs(identity['grid']):
   result=store.load(spec);rows=evaluate_episode_predictions(result,online)
   summary=summarize_prediction_rows(rows)
   diagnostic={'schema':'episode_ranking_diagnostics_v1','spec':spec,
    'run_identity_sha256':identity['identity_sha256'],'summary':summary,'rows':rows,
    'label_access_scope':'Evaluator-only labels; no learner construction, fit, update, or delayed-feedback flush'}
   path=diagnostics_dir/'episodes'/(episode_name(spec)+'.json');immutable_json(path,diagnostic)
   episode_refs.append(artifact_reference(path));episode_summaries.append({'spec':spec,**summary})
   if all(spec[k]==primary_policy[k] for k in ('strategy','label_rate','feedback_delay')):
    for row in rows:
     primary.append({'question_id':row['question_id'],'task':row['task'],'pairwise_accuracy':row['unseen_question_ranking']['pairwise_accuracy']})
     coverage.append({'question_id':row['question_id'],'task':row['task'],'coverage_minus_quality':row['coverage_minus_aggregate_quality']})
   print(json.dumps({'evaluated_episode':episode_name(spec),'rankable_questions':summary['unseen_question_ranking']['rankable_questions'],'research_complete':False}),flush=True)
  stats=config.get('statistics',{});bootstrap=stats.get('e1_bootstrap_replicates',2000);seed=stats.get('e1_bootstrap_seed',20261030)
  interval=stratified_question_interval(primary,value_key='pairwise_accuracy',replicates=bootstrap,seed=seed,confidence=.95,minimum_per_task=20)
  coverage_interval=stratified_question_interval(coverage,value_key='coverage_minus_quality',replicates=bootstrap,seed=seed,confidence=.95,minimum_per_task=20)
  try:references=rich_fixed_references(calibration,online,main_feedback_natural=identity['settings']['main_feedback_natural'])
  except (ValueError,KeyError) as error:
   references={'rich_calibration_best_fixed_reference':{'status':'INCOMPLETE','error':str(error),
    'technical_gate_blocking':False,'cost_missing_values_are_not_zero':True}}
  report={'schema':'e2_ranking_diagnostics_v1','run_identity_sha256':identity['identity_sha256'],
   'episodes_evaluated':len(compact),'episode_diagnostics':episode_refs,'episode_summaries':episode_summaries,
   'diagnostic_policy':primary_policy,'scientific_learnability_conclusion':learnability_conclusion(interval),
   'scientific_learnability_interval':interval,'chosen_coverage_minus_actual_aggregation_interval':coverage_interval,
   'scientific_scope':'Pairwise ordering support under the declared predictor/budget/cache; not causal complementarity, model generation variance, or proof of all later research claims',
   'technical_gate_independent_of_scientific_positivity':True,
   'main_fixed_definition':'budgeted_warmup_fixed; observed common paid warmup only; zero voluntary online monitoring',
   'cost_interpretation':'Main five baselines share hard limits; compare actual executions, labels, tokens and CPU/wall separately. Rich full-calibration fixed is an additional-information reference, excluded from same-budget primary methods.',
   'ranking_ties':'True aggregate-score ties excluded from ordered-pair accuracy; prediction ties get 0.5; top-1 ties report lexicographic and best/worst tied-prediction regret.',
   'topk_semantics':'Mean member scores provide a ranking proxy, never calibrated aggregation probabilities or aggregate-value error.',
   'references':references,'research_complete':False}
  report_path=diagnostics_dir/'ranking_report.json';immutable_json(report_path,report)
  gate={'schema':'e2_reliable_baselines_gate_v1','status':'E2_PASS','software_test_only':False,
   'scope':'technical_completeness_not_scientific_positivity',
   'run_identity_sha256':identity['identity_sha256'],'input_identity_sha256':canonical_digest(current),
   'grid_plan':artifact_reference(directory/'grid_plan.json'),'episode_receipts':receipts,
   'ranking_diagnostics':artifact_reference(report_path),'bindings':e2_bindings(identity),
   'scientific_learnability_conclusion':report['scientific_learnability_conclusion'],'research_complete':False}
  gate_path=Path(gate_path).resolve() if gate_path else directory/'e2_gate.json'
  candidate=gate_path.with_name(gate_path.name+'.candidate.'+attempt_id)
  immutable_json(candidate,gate)
  validate_e2_gate(candidate,current)
  immutable_json(gate_path,gate)
  candidate.unlink()
  atomic_json(diagnostics_dir/'evaluation_attempts'/(attempt_id+'.json'),{'status':'TECHNICAL_GATE_ACCEPTED',
   'gate':artifact_reference(gate_path),'process_time_s':time.process_time()-cpu_start,'wall_time_s':time.perf_counter()-attempt_start,'research_complete':False})
  return gate
 except BaseException as error:
  atomic_json(diagnostics_dir/'evaluation_attempts'/(attempt_id+'.json'),{'status':'PARTIAL_E2_NO_ACCEPTED_GATE',
   'error':type(error).__name__+': '+str(error),'process_time_s':time.process_time()-cpu_start,
   'wall_time_s':time.perf_counter()-attempt_start,'research_complete':False})
  raise


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runs-dir',required=True);p.add_argument('--output-gate')
 args=p.parse_args(argv);gate=evaluate(args.runs_dir,args.output_gate)
 print(json.dumps({'status':gate['status'],'scientific_learnability_conclusion':gate['scientific_learnability_conclusion'],'research_complete':False}),flush=True)


if __name__=='__main__':
 try:main()
 except (ValueError,FileNotFoundError,FileExistsError,KeyError) as error:
  print(json.dumps({'status':'PARTIAL_E2_NO_ACCEPTED_GATE','error':str(error),'research_complete':False}),file=sys.stderr)
  raise SystemExit(2)
