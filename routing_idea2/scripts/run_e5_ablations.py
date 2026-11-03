"""Run only the predeclared E5 ablations after a complete E4/Flex gate.

Three independent immutable condition grids contain 120 paid-prior and 2x300
full-question task-shift episodes. Original IID controls are reused, not rerun.
The output is a pre-final technical gate; sealed-final evaluation remains separate.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from budget_router.simulation import run_simulation,save_simulation,validate_strategy_requirements
from budget_router.ranking_evaluation import write_cost_curve
from budget_router.stage_validation import (EpisodeStore,read_json,atomic_json,immutable_json,input_identity,
 settings_from_config,episode_specs,episode_name,artifact_reference,authorize_e2_pair)
from budget_router.e5_validation import (make_e5_run_plan,verify_run_plan,make_condition_identity,
 project_e5_episode,load_environments,evaluate_bundle,CONDITIONS)


def parser():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('online-matrix','online-manifest','calibration-matrix','calibration-manifest','e4-full-gate','output-dir'):
  p.add_argument('--'+name,required=True)
 p.add_argument('--config',default=str(ROOT/'configs/experiment_v1.json'))
 p.add_argument('--e5-plan',default=str(ROOT/'protocol/e5_robustness_plan_v1.json'))
 p.add_argument('--resume',action='store_true');p.add_argument('--evaluate-only',action='store_true')
 p.add_argument('--output-gate')
 return p


def main(argv=None):
 args=parser().parse_args(argv);inputs=input_identity(ROOT,online_matrix=args.online_matrix,online_manifest=args.online_manifest,
  calibration_matrix=args.calibration_matrix,calibration_manifest=args.calibration_manifest,config_path=args.config)
 plan=make_e5_run_plan(ROOT,inputs,args.e4_full_gate,args.e5_plan)
 output=Path(args.output_dir).resolve();plan_path=output/'ablation_plan.json'
 if output.exists():
  if not (args.resume or args.evaluate_only) or not plan_path.is_file() or read_json(plan_path)!=plan:
   raise ValueError('E5 resume requires the identical locked bundle; preserve existing evidence')
 else:
  if args.evaluate_only:raise ValueError('Evaluation-only requires an existing complete E5 bundle')
  output.mkdir(parents=True);atomic_json(plan_path,plan)
 if args.evaluate_only:return evaluate_bundle(output,args.output_gate)
 config=read_json(args.config);online,cal=load_environments(inputs)
 authorize_e2_pair(online,cal,plan['upstream']['e2_gate']['path'],inputs)
 completed=0
 try:
  for condition in CONDITIONS:
   cfg=plan['conditions'][condition];base=settings_from_config(config)
   settings=replace(base,warmup_mode=cfg['settings']['warmup_mode'],stream_order=cfg['settings']['stream_order'])
   for strategy in cfg['grid']['strategies']:validate_strategy_requirements(online,cal,strategy,settings)
   identity=make_condition_identity(plan,plan_path,condition)
   store=EpisodeStore(output/condition,identity,resume=args.resume);receipt_refs=[];resources=[]
   for spec in episode_specs(identity['grid']):
    verify_run_plan(plan,parent=False);result=store.load(spec);reused=result is not None
    if result is None:
     attempt=output/'attempts'/(condition+'.'+episode_name(spec)+'.'+uuid.uuid4().hex+'.json')
     record={'schema':'e5_simulation_attempt_v1','status':'STARTED','condition_id':condition,'spec':spec,
      'run_identity_sha256':plan['identity_sha256'],'started_at_unix':time.time(),'process_time_s':None,'wall_time_s':None,
      'scope':'CPU simulation and persistence only; no new physical model generation'}
     atomic_json(attempt,record);cpu=time.process_time();wall=time.perf_counter()
     try:
      result=run_simulation(online,cal,strategy=spec['strategy'],seed=spec['seed'],label_rate=spec['label_rate'],delay=spec['feedback_delay'],settings=settings)
      project_e5_episode(result,online,cal);verify_run_plan(plan,parent=False)
      def writer(value,folder):save_simulation(value,folder);write_cost_curve(value,folder/'cost_curve.csv')
      store.commit(result,spec,writer);record['status']='COMMITTED'
     except BaseException as error:
      record.update(status='INTERRUPTED_OR_FAILED',error=type(error).__name__+': '+str(error));raise
     finally:
      record.update(process_time_s=time.process_time()-cpu,wall_time_s=time.perf_counter()-wall);atomic_json(attempt,record)
    else:project_e5_episode(result,online,cal)
    receipt_refs.append(artifact_reference(output/condition/episode_name(spec)/'episode_receipt.json'))
    s=result['summary'];resources.append({'spec':spec,'actual_resources':s['budget']['spent'],'hard_limits':s['budget']['limits'],
     'warmup_cost':s['warmup_cost'],'paid_member_reference_events':s['paid_member_reference_events'],
     'prior_member_labels_used':s['prior_member_labels_used'],'runtime_measurements':s['runtime_measurements']})
    completed+=1
    atomic_json(output/'progress.json',{'status':'RUNNING_E5_ABLATIONS','completed':completed,'required':720,
     'condition_id':condition,'last_episode':episode_name(spec),'reused_last_episode':reused,'research_complete':False})
    print(json.dumps({'condition_id':condition,'validated_episode':episode_name(spec),'completed':completed,'required':720,'research_complete':False}),flush=True)
   summary={'schema':'e5_condition_grid_summary_v1','status':'COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY',
    'run_identity_sha256':identity['identity_sha256'],'condition_id':condition,'grid':identity['grid'],
    'runs':len(receipt_refs),'episode_receipts':receipt_refs,'actual_resources_and_clocks':resources,'research_complete':False}
   immutable_json(output/condition/'grid_summary.json',summary)
  gate=evaluate_bundle(output,args.output_gate)
  atomic_json(output/'progress.json',{'status':gate['status'],'completed':720,'required':720,'final_evaluation_status':'NOT_RUN','research_complete':False})
  return gate
 except BaseException as error:
  atomic_json(output/('PARTIAL.'+uuid.uuid4().hex+'.json'),{'status':'PARTIAL_E5_NO_NEW_ACCEPTED_GATE','completed':completed,
   'error':type(error).__name__+': '+str(error),'research_complete':False});raise


if __name__=='__main__':
 try:
  gate=main();print(json.dumps({'status':gate['status'],'final_evaluation_status':'NOT_RUN','research_complete':False}),flush=True)
 except (ValueError,FileNotFoundError,FileExistsError,KeyError) as error:
  print(json.dumps({'status':'BLOCKED_E5_PRECONDITIONS_OR_PARTIAL','error':str(error),'research_complete':False}),file=sys.stderr)
  raise SystemExit(2)
