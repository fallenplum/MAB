"""Execute the full-scale real-matrix E2/E3 grid with immutable episode receipts.

E2 defaults to five declared reliable baselines. E3 requires an independently
validated --e2-gate; original matrix manifests are never edited. --resume
revalidates all completed episodes and reruns only uncommitted CPU simulations.
No CLI accepts toy inputs. One pinned generation cache is not five LM replicates.
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
from budget_router.simulation import (MatrixEnvironment,run_simulation,save_simulation,
 paired_question_bootstrap,validate_strategy_requirements)
from budget_router.ranking_evaluation import write_cost_curve
from budget_router.stage_validation import (E2_STRATEGIES,E3_STRATEGIES,EpisodeStore,
 atomic_json,immutable_json,read_json,input_identity,make_run_identity,settings_from_config,
 validate_protocol_config,expected_grid,episode_specs,episode_name,compact_result,
 artifact_reference,authorize_e2_pair)


def parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--stage',choices=('e2','e3'),default='e2')
 for name in ('online-matrix','online-manifest','calibration-matrix','calibration-manifest','output-dir'):
  p.add_argument('--'+name,required=True)
 p.add_argument('--config',default=str(ROOT/'configs/experiment_v1.json'))
 p.add_argument('--resume',action='store_true')
 p.add_argument('--e2-gate')
 p.add_argument('--execution-limit',type=int);p.add_argument('--token-limit',type=int)
 p.add_argument('--cost-weights',nargs=3,type=float,metavar=('EXEC','LABEL','TOKEN'))
 p.add_argument('--strategies',nargs='+',choices=tuple(dict.fromkeys(E2_STRATEGIES+E3_STRATEGIES)))
 p.add_argument('--seeds',nargs='+',type=int)
 p.add_argument('--rates',nargs='+',type=int,choices=(10,25,50,100))
 p.add_argument('--delays',nargs='+',type=int,choices=(0,5,20))
 p.add_argument('--period',type=int,default=5)
 p.add_argument('--cost-prediction',choices=('history','caps'),default='history')
 return p


def main(argv=None):
 args=parser().parse_args(argv);config=read_json(args.config);validate_protocol_config(config)
 grid=expected_grid(config,args.stage)
 for key,value in (('strategies',args.strategies),('seeds',args.seeds),('rates',args.rates),('delays',args.delays)):
  if value is not None:grid[key]=value
 if len(grid['seeds'])!=5 or len(set(grid['seeds']))!=5:raise ValueError('Five distinct stream seeds required')
 for key in ('strategies','seeds','rates','delays'):
  if not grid[key] or len(grid[key])!=len(set(grid[key])):raise ValueError('Empty or duplicate grid '+key)
 allowed=E2_STRATEGIES if args.stage=='e2' else E3_STRATEGIES
 if not set(grid['strategies']).issubset(allowed):raise ValueError('Strategy does not belong to requested stage')
 if args.stage=='e3' and not args.e2_gate:raise ValueError('E3 requires explicit independently validated --e2-gate')
 if args.stage=='e2' and args.e2_gate:raise ValueError('E2 must not import an E2 gate')
 settings=settings_from_config(config,execution_limit=args.execution_limit,token_limit=args.token_limit,
  cost_weights=args.cost_weights,period=args.period,cost_prediction=args.cost_prediction)
 inputs=input_identity(ROOT,online_matrix=args.online_matrix,online_manifest=args.online_manifest,
  calibration_matrix=args.calibration_matrix,calibration_manifest=args.calibration_manifest,config_path=args.config)
 online=MatrixEnvironment.from_npz(args.online_matrix,args.online_manifest,expected_split='online_stream')
 calibration=MatrixEnvironment.from_npz(args.calibration_matrix,args.calibration_manifest,expected_split='online_calibration')
 if online.software_test_only or calibration.software_test_only:raise ValueError('Software/toy matrices are forbidden in the production CLI')
 if len(online)!=config['split']['online_total'] or len(calibration)!=config['split']['router_calibration_total']:
  raise ValueError('Matrix sizes differ from the locked original scale')
 if online.phi_dimension!=config['features']['pca_dimensions']+1 or calibration.phi_dimension!=online.phi_dimension:
  raise ValueError('Frozen PCA feature dimension mismatch')
 if args.stage=='e3':authorize_e2_pair(online,calibration,args.e2_gate,inputs)
 for strategy in grid['strategies']:validate_strategy_requirements(online,calibration,strategy,settings)
 identity=make_run_identity(inputs,grid,settings,args.e2_gate)
 output=Path(args.output_dir).resolve();store=EpisodeStore(output,identity,resume=args.resume)
 results=[];reused=0;computed=0;specs=episode_specs(grid)
 try:
  for spec in specs:
   result=store.load(spec);name=episode_name(spec)
   if result is not None:reused+=1
   else:
    attempt=output/'attempts'/(name+'.'+uuid.uuid4().hex+'.json')
    record={'schema':'simulation_execution_attempt_v1','run_identity_sha256':identity['identity_sha256'],
     'spec':spec,'status':'STARTED','started_at_unix':time.time(),
     'scope':'Actual local CPU simulation including validation/serialization; no new model inference',
     'process_time_s':None,'wall_time_s':None}
    atomic_json(attempt,record);cpu=time.process_time();wall=time.perf_counter()
    try:
     result=run_simulation(online,calibration,strategy=spec['strategy'],seed=spec['seed'],
      label_rate=spec['label_rate'],delay=spec['feedback_delay'],settings=settings)
     def writer(value,directory):
      save_simulation(value,directory);write_cost_curve(value,directory/'cost_curve.csv')
     store.commit(result,spec,writer);computed+=1
     record.update(status='COMMITTED')
    except BaseException as error:
     record.update(status='INTERRUPTED_OR_FAILED',error=type(error).__name__+': '+str(error));raise
    finally:
     record.update(process_time_s=time.process_time()-cpu,wall_time_s=time.perf_counter()-wall)
     atomic_json(attempt,record)
   results.append(compact_result(result))
   atomic_json(output/'progress.json',{'status':'RUNNING_GRID','run_identity_sha256':identity['identity_sha256'],
    'validated_complete_episodes':len(results),'required_episodes':len(specs),'reused_this_invocation':reused,
    'computed_this_invocation':computed,'last_episode':name,'research_complete':False})
   print(json.dumps({'validated_episode':name,'complete':len(results),'required':len(specs),
    'quality':result['summary']['quality_task_macro'],'research_complete':False}),flush=True)
  statistics=config.get('statistics',{})
  comparisons=paired_question_bootstrap(results,replicates=statistics.get('e1_bootstrap_replicates',2000),
   seed=statistics.get('e1_bootstrap_seed',20261030),confidence=statistics.get('confidence',.95)) if 'none' in grid['strategies'] and len(grid['strategies'])>1 else []
  receipt_refs=[artifact_reference(output/episode_name(spec)/'episode_receipt.json') for spec in specs]
  resource_rows=[{'spec':spec,'actual_logical_resources':r['summary']['budget']['spent'],
   'resource_limits':r['summary']['budget']['limits'],
   'controller_process_time_s':r['summary']['controller_process_time_s'],
   'controller_wall_time_s':r['summary']['controller_wall_time_s'],
   'runtime_measurements':r['summary']['runtime_measurements'],
   'source_service_execution_wall_time_s':r['summary'].get('source_service_execution_wall_time_s'),
   'source_audit_execution_wall_time_s':r['summary'].get('source_audit_execution_wall_time_s')}
   for spec,r in zip(specs,results)]
  report={'schema':'restricted_feedback_grid_summary_v1','status':'COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY',
   'run_identity_sha256':identity['identity_sha256'],'grid':grid,'runs':len(results),'episode_receipts':receipt_refs,
   'complete_locked_stage_grid':grid==expected_grid(config,args.stage) and identity['settings']==json.loads(json.dumps(asdict(settings_from_config(config)))),
   'paired_question_comparisons':comparisons,'resource_measurements_by_episode':resource_rows,
   'resource_interpretation':'Actual resource vectors are reported. Identical hard limits do not imply equal spending; CPU clocks are separate from priced model/feedback resources. Attempts include interrupted/repeated CPU work separately.',
   'physical_generation_replicates':1,'stream_seeds_are_not_new_generations':True,
   'stage_scope':'E2/E3 restricted-feedback simulations; later mandatory E4/E5 and sealed final evaluation are separate',
   'research_complete':False}
  immutable_json(output/'grid_summary.json',report)
  atomic_json(output/'progress.json',{'status':'GRID_COMPLETED_AWAITING_INDEPENDENT_STAGE_EVALUATION',
   'run_identity_sha256':identity['identity_sha256'],'validated_complete_episodes':len(results),'required_episodes':len(specs),'research_complete':False})
  return report
 except BaseException as error:
  atomic_json(output/('PARTIAL.'+uuid.uuid4().hex+'.json'),{'status':'PARTIAL_SIMULATION',
   'run_identity_sha256':identity['identity_sha256'],'completed_runs':len(results),'error':type(error).__name__+': '+str(error),'research_complete':False})
  raise


if __name__=='__main__':
 try:main()
 except (ValueError,FileNotFoundError,FileExistsError,KeyError) as error:
  print(json.dumps({'status':'BLOCKED_REAL_ONLINE_ASSETS_OR_PROTOCOL','error':str(error),'research_complete':False}),file=sys.stderr)
  raise SystemExit(2)
