"""Immutable episode storage and strictly bound E2 authorization.

These utilities validate measurement completeness, never scientific positivity.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import time
import uuid

E2_STRATEGIES = ('fixed_combination','contextual_topk','none','linucb','thompson')
E3_STRATEGIES = ('none','random','periodic','max_uncertainty','max_uncertainty_per_cost','no_cost','decision')
SOURCE_FILES = (
 'scripts/run_online.py','scripts/evaluate_e2.py','scripts/evaluate_e3.py',
 'src/budget_router/stage_validation.py','src/budget_router/ranking_evaluation.py','src/budget_router/__init__.py',
 'src/budget_router/simulation.py','src/budget_router/acquisition.py',
 'src/budget_router/learner.py','src/budget_router/features.py','src/budget_router/budget.py',
)
ARTIFACTS = ('summary.json','traces.jsonl','warmup_anonymous.json','policy_checkpoint.json','cost_curve.csv')
BINDING_FIELDS = ('online_matrix_sha256','online_manifest_sha256','calibration_matrix_sha256',
 'calibration_manifest_sha256','e0_lock_sha256','e1_gate_sha256','pca_sha256','pool_config_sha256',
 'config_sha256','required_grid_sha256','algorithm_sources_sha256')


def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def canonical_digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _constant(value):
 raise ValueError('Non-finite JSON constants are forbidden')


def read_json(path):
 return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=_constant)


def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
 try:
  with temporary.open('w',encoding='utf-8',newline='\n') as stream:
   json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  os.replace(temporary,path)
 finally:temporary.unlink(missing_ok=True)


def immutable_json(path,value):
 path=Path(path)
 if path.exists():
  if read_json(path)!=value:raise ValueError('Existing immutable JSON differs: '+str(path))
 else:atomic_json(path,value)


def artifact_reference(path):
 p=Path(path).resolve(strict=True)
 return {'path':str(p),'sha256':sha_file(p),'bytes':p.stat().st_size}


def verified_reference(record):
 if not isinstance(record,dict) or set(('path','sha256'))-set(record):raise ValueError('Missing artifact reference')
 p=Path(record['path']).resolve(strict=True)
 if not p.is_file() or sha_file(p)!=record['sha256'] or ('bytes' in record and p.stat().st_size!=record['bytes']):
  raise ValueError('Changed or missing artifact: '+str(p))
 return p


def source_references(root):
 return {name:artifact_reference(Path(root)/name) for name in SOURCE_FILES}


def validate_protocol_config(config):
 if config.get('n_models')!=6 or config.get('k')!=3 or config.get('split',{}).get('online_total')!=600 or config.get('split',{}).get('router_calibration_total')!=200 or config.get('split',{}).get('e1_development',{}).get('total')!=300:
  raise ValueError('The original full-scale 6/K3, 300 E1, 200 calibration and 600 online protocol is required')


def expected_grid(config,stage='e2'):
 seeds=config.get('statistics',{}).get('algorithm_stream_seeds')
 rates=config.get('feedback',{}).get('final_score_budget_per_100_questions')
 delays=config.get('feedback',{}).get('delays')
 if not isinstance(seeds,list) or len(seeds)!=5 or len(set(seeds))!=5:raise ValueError('Five distinct locked stream seeds required')
 if rates!=[10,25,50,100] or delays!=[0,5,20]:raise ValueError('The complete declared feedback/delay grid is required')
 strategies=E2_STRATEGIES if stage=='e2' else E3_STRATEGIES
 return {'stage':stage,'strategies':list(strategies),'seeds':seeds,'rates':rates,'delays':delays}


def episode_specs(grid):
 return [{'strategy':strategy,'seed':int(seed),'label_rate':int(rate),'feedback_delay':int(delay)}
  for seed in grid['seeds'] for rate in grid['rates'] for delay in grid['delays'] for strategy in grid['strategies']]


def episode_name(spec):
 return f"{spec['strategy']}_seed{spec['seed']}_rate{spec['label_rate']}_delay{spec['feedback_delay']}"


def settings_from_config(config,*,execution_limit=None,token_limit=None,cost_weights=None,period=5,cost_prediction='history'):
 from .simulation import CallCaps,SimulationSettings
 b=config.get('budget',{});limits=b.get('total_execution_budget',{});weights=cost_weights or b.get('cost_weights')
 if isinstance(weights,dict):weights=[weights[k] for k in ('executions','labels','tokens')]
 if not isinstance(limits,dict) or weights is None:raise ValueError('Explicit locked separate resource limits/weights required')
 f=config.get('feedback',{});reg=config.get('regression',{});acq=config.get('acquisition',{})
 return SimulationSettings(
  execution_limit=execution_limit if execution_limit is not None else limits['executions'],
  token_limit=token_limit if token_limit is not None else limits['tokens'],
  caps=CallCaps(member_prompt=b.get('input_cap_tokens',4096),member_completion=b.get('member_max_output_tokens',256),
   aggregate_prompt=b.get('aggregator_input_cap_tokens',b.get('input_cap_tokens',4096)),aggregate_completion=b.get('aggregator_max_output_tokens',256),
   member_calls=b.get('member_call_cap',1),aggregate_calls=b.get('aggregator_call_cap',1)),
  cost_weights=tuple(weights),warmup_max_labels=config.get('warmup',{}).get('max_labels',32),
  ridge=reg.get('ridge_lambda',1.),noise_variance=reg.get('working_noise_variance',1.),
  bootstrap_replicates=reg.get('bootstrap_replicates',200),bootstrap_min_labels=reg.get('bootstrap_min_arrived_labels',32),
  kappa=acq.get('competitor_kappa',2.),threshold=acq.get('minimum_information_per_cost',0.),period=period,
  main_feedback_natural=f.get('main_feedback_naturally_arrives',False),
  feedback_mode='verifiable' if f.get('member_feedback_available_when_same_verifier_revealed',False) else 'final_only',
  cost_prediction=cost_prediction)


def validate_locked_grid(identity,config,stage):
 validate_protocol_config(config)
 expected=json.loads(json.dumps(asdict(settings_from_config(config)),allow_nan=False))
 if identity['grid']!=expected_grid(config,stage) or identity['settings']!=expected or identity['settings'].get('warmup_mode')!='random_combo' or identity['settings'].get('stream_order')!='iid':
  raise ValueError(stage.upper()+' requires the complete locked baseline grid/settings, random_combo warmup and iid stream')


def input_identity(root,*,online_matrix,online_manifest,calibration_matrix,calibration_manifest,config_path):
 refs={name:artifact_reference(path) for name,path in (
  ('online_matrix',online_matrix),('online_manifest',online_manifest),('calibration_matrix',calibration_matrix),
  ('calibration_manifest',calibration_manifest),('config',config_path))}
 manifests={key:read_json(refs[key]['path']) for key in ('online_manifest','calibration_manifest')}
 locked={}
 for name in ('e0_lock','e1_gate','pca_artifact','pool_config','split_manifest'):
  records=[]
  for key,m in manifests.items():
   record=dict(m[name]);p=Path(record['path'])
   if not p.is_absolute():p=Path(refs[key]['path']).parent/p
   record['path']=str(p.resolve());verified_reference(record);records.append(record)
  if records[0]['sha256']!=records[1]['sha256']:raise ValueError('Cross-matrix provenance mismatch: '+name)
  locked[name]=records[0]
 return {'schema':'online_input_identity_v1','assets':refs,'locked_provenance':locked,'algorithm_sources':source_references(root)}


def verify_input_identity(identity):
 if identity.get('schema')!='online_input_identity_v1':raise ValueError('Unsupported input identity')
 for collection in ('assets','locked_provenance','algorithm_sources'):
  for ref in identity[collection].values():verified_reference(ref)


def make_run_identity(inputs,grid,settings,e2_gate=None):
 value={'schema':'restricted_feedback_grid_identity_v1','inputs':inputs,'grid':grid,'settings':json.loads(json.dumps(asdict(settings),allow_nan=False)),
  'e2_gate':None if e2_gate is None else artifact_reference(e2_gate),
  'one_pinned_generation_cache':True,'stream_seeds_are_not_new_generations':True}
 value['identity_sha256']=canonical_digest(value)
 return value


def validate_run_identity(identity):
 core={k:v for k,v in identity.items() if k!='identity_sha256'}
 if identity.get('identity_sha256')!=canonical_digest(core):raise ValueError('Run identity digest changed')
 verify_input_identity(identity['inputs'])
 if identity.get('e2_gate'):verified_reference(identity['e2_gate'])


class EpisodeStore:
 """Completed episode directories are committed atomically and never overwritten."""
 def __init__(self,directory,identity,*,resume=False):
  self.directory=Path(directory).resolve();self.identity=identity
  plan=self.directory/'grid_plan.json'
  if self.directory.exists():
   if not resume:raise FileExistsError('Use --resume for an existing result directory')
   if not plan.is_file() or read_json(plan)!=identity:raise ValueError('Resume run identity differs; preserve existing evidence')
  else:
   self.directory.mkdir(parents=True);atomic_json(plan,identity)
  validate_run_identity(identity)
  self.allowed={episode_name(v):v for v in episode_specs(identity['grid'])}

 def _inside(self,path):
  p=Path(path).resolve()
  if not p.is_relative_to(self.directory):raise ValueError('Episode path escaped output directory')
  return p

 def load(self,spec):
  name=episode_name(spec)
  if self.allowed.get(name)!=spec:raise ValueError('Episode not in immutable grid')
  folder=self._inside(self.directory/name)
  if not folder.exists():return None
  receipt=read_json(folder/'episode_receipt.json')
  if receipt.get('schema')!='completed_episode_v1' or receipt.get('spec')!=spec or receipt.get('run_identity_sha256')!=self.identity['identity_sha256']:
   raise ValueError('Episode receipt identity mismatch')
  if set(receipt.get('artifacts',{}))!=set(ARTIFACTS):raise ValueError('Incomplete episode artifact inventory')
  for artifact in ARTIFACTS:
   ref=receipt['artifacts'][artifact];path=self._inside(folder/artifact)
   if ref.get('path')!=artifact or ref.get('sha256')!=sha_file(path) or ref.get('bytes')!=path.stat().st_size:
    raise ValueError('Episode artifact changed: '+str(path))
  result={'summary':read_json(folder/'summary.json'),'warmup':read_json(folder/'warmup_anonymous.json'),
   'policy_checkpoint':read_json(folder/'policy_checkpoint.json'),'traces':read_jsonl(folder/'traces.jsonl')}
  validate_episode_result(result,spec,self.identity)
  return result

 def commit(self,result,spec,writer):
  name=episode_name(spec)
  if self.load(spec) is not None:raise FileExistsError('Completed episode cannot be overwritten')
  validate_episode_result(result,spec,self.identity)
  temporary=self._inside(self.directory/('.'+name+'.pending.'+uuid.uuid4().hex))
  started={'schema':'simulation_attempt_v1','spec':spec,'run_identity_sha256':self.identity['identity_sha256'],
   'started_at_unix':time.time(),'resource_scope':'CPU simulation only; no extra model generation or label acquisition'}
  try:
   writer(result,temporary)
   atomic_json(temporary/'attempt.json',started)
   artifacts={}
   for artifact in ARTIFACTS:
    p=temporary/artifact
    with p.open('r+b') as f:os.fsync(f.fileno())
    artifacts[artifact]={'path':artifact,'sha256':sha_file(p),'bytes':p.stat().st_size}
   receipt={'schema':'completed_episode_v1','spec':spec,'run_identity_sha256':self.identity['identity_sha256'],
    'artifacts':artifacts,'committed_at_unix':time.time(),'research_complete':False}
   atomic_json(temporary/'episode_receipt.json',receipt)
   destination=self._inside(self.directory/name)
   if destination.exists():raise FileExistsError('Another worker completed this episode')
   os.replace(temporary,destination)
  except Exception:
   # Preserve interrupted staging for audit. A resumed CPU episode starts anew.
   raise
  return self.load(spec)


def read_jsonl(path):
 records=[]
 with Path(path).open(encoding='utf-8-sig') as f:
  for index,line in enumerate(f,1):
   if not line.strip():continue
   try:records.append(json.loads(line,parse_constant=_constant))
   except ValueError as e:raise ValueError(f'Invalid committed JSONL at line {index}') from e
 return records


def validate_episode_result(result,spec,identity):
 summary=result['summary'];traces=result['traces'];checkpoint=result['policy_checkpoint']
 if any(summary.get(k)!=v for k,v in spec.items()):raise ValueError('Episode summary parameters differ')
 config=read_json(identity['inputs']['assets']['config']['path']);n=config['split']['online_total']
 if summary.get('status')!='COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY' or summary.get('all_questions_served') is not True or summary.get('n_questions')!=n:
  raise ValueError('Incomplete or software-only result cannot be committed as production')
 if len(traces)!=n or len({x['question_id'] for x in traces})!=n or [x.get('step') for x in traces]!=list(range(n)):
  raise ValueError('Episode does not cover each complete online question exactly once')
 if json.loads(json.dumps(summary.get('settings'),allow_nan=False))!=identity['settings']:raise ValueError('Episode settings differ')
 if checkpoint.get('schema')!='restricted_feedback_policy_checkpoint_v1' or checkpoint.get('software_test_only') is not False or checkpoint.get('next_tick')!=n or checkpoint.get('pending_feedback_flushed') is not False:
  raise ValueError('Missing or invalid frozen policy checkpoint')
 for key,value in spec.items():
  ck='feedback_delay' if key=='feedback_delay' else key
  if checkpoint.get(ck)!=value:raise ValueError('Checkpoint parameters differ')
 for split,asset,manifest in (('online','online_matrix','online_manifest'),('calibration','calibration_matrix','calibration_manifest')):
  expected=identity['inputs'];raw=read_json(expected['assets'][manifest]['path'])
  required={'matrix_sha256':expected['assets'][asset]['sha256'],
   'e0_lock_sha256':expected['locked_provenance']['e0_lock']['sha256'],
   'e1_gate_sha256':expected['locked_provenance']['e1_gate']['sha256'],
   'pca_sha256':expected['locked_provenance']['pca_artifact']['sha256']}
  for key in ('model_pool','cache_revision','aggregator','scorer'):
   if key in raw:required[key]=raw[key]
  if identity.get('e2_gate'):
   required.update(e2_gate_sha256=identity['e2_gate']['sha256'])
  for container in (summary,checkpoint):
   observed=container.get(split+'_provenance',{})
   if any(observed.get(k)!=v for k,v in required.items()):raise ValueError('Episode provenance differs from bound '+split+' assets')
 if checkpoint.get('frozen_as_of_tick')!=n-1 or checkpoint.get('settings')!=identity['settings']:
  raise ValueError('Checkpoint is not frozen at the exact final decision/settings')
 for state in [checkpoint.get('shared_ridge',{})]+(checkpoint.get('member_predictor') or {}).get('models',[]):
  for label in state.get('arrived_original_labels',[]):
   if label.get('available_at',n)>n-1:raise ValueError('Future feedback entered the frozen learner checkpoint')
 for broker_name in ('feedback','warmup_feedback'):
  broker=checkpoint.get(broker_name,{})
  if broker.get('pending_values_serialized') is not False:raise ValueError('Pending hidden feedback must not be serialized')
  for pending in broker.get('pending_paid_requests_without_values',[]):
   if {'value','values','score','features','member_scores'}.intersection(pending):raise ValueError('Unarrived hidden values in checkpoint')
 for trace in traces:
  if trace.get('service_frozen_before_audit') is not True or trace.get('prediction_snapshot_precedes_current_execution_and_feedback') is not True:
   raise ValueError('Decision/prediction was not frozen before current outcomes')
  scores=trace.get('predicted_combination_scores_before_decision')
  if not isinstance(scores,list) or len(scores)!=20:raise ValueError('Missing complete predecision predictions')
 for key in ('controller_process_time_s','controller_wall_time_s','offline_evaluation_process_time_s','offline_evaluation_wall_time_s'):
  value=summary.get(key)
  if isinstance(value,bool) or not isinstance(value,(int,float)) or value<0:raise ValueError('Missing measured controller/evaluator clock')
 # Reparse a canonical finite representation, rejecting NaN/Infinity anywhere.
 json.loads(json.dumps(result,allow_nan=False),parse_constant=_constant)


def compact_result(result):
 return {'summary':result['summary'],'traces':[{k:r[k] for k in ('question_id','task','quality')} for r in result['traces']]}


def verify_complete_grid(directory):
 directory=Path(directory).resolve();identity=read_json(directory/'grid_plan.json');validate_run_identity(identity)
 store=EpisodeStore(directory,identity,resume=True);receipts=[];results=[]
 for spec in episode_specs(identity['grid']):
  result=store.load(spec)
  if result is None:raise ValueError('Grid incomplete: '+episode_name(spec))
  receipts.append(artifact_reference(directory/episode_name(spec)/'episode_receipt.json'));results.append(compact_result(result))
 return identity,results,receipts


def e2_bindings(identity):
 i=identity['inputs'];a=i['assets'];p=i['locked_provenance']
 return {'online_matrix_sha256':a['online_matrix']['sha256'],'online_manifest_sha256':a['online_manifest']['sha256'],
  'calibration_matrix_sha256':a['calibration_matrix']['sha256'],'calibration_manifest_sha256':a['calibration_manifest']['sha256'],
  'e0_lock_sha256':p['e0_lock']['sha256'],'e1_gate_sha256':p['e1_gate']['sha256'],
  'pca_sha256':p['pca_artifact']['sha256'],'pool_config_sha256':p['pool_config']['sha256'],
  'config_sha256':a['config']['sha256'],'required_grid_sha256':canonical_digest(identity['grid']),
  'algorithm_sources_sha256':canonical_digest(i['algorithm_sources'])}


def validate_e2_gate(path,current_inputs):
 path=Path(path).resolve();gate=read_json(path)
 if gate.get('schema')!='e2_reliable_baselines_gate_v1' or gate.get('status')!='E2_PASS' or gate.get('software_test_only') is not False:
  raise ValueError('An independently accepted production E2 gate is required')
 if gate.get('scope')!='technical_completeness_not_scientific_positivity':raise ValueError('Unrecognized E2 gate scope')
 directory=verified_reference(gate['grid_plan']).parent
 identity,results,receipts=verify_complete_grid(directory)
 if identity['inputs']!=current_inputs or gate.get('input_identity_sha256')!=canonical_digest(current_inputs):
  raise ValueError('E2 gate belongs to different matrices/config/algorithms')
 config=read_json(current_inputs['assets']['config']['path'])
 validate_locked_grid(identity,config,'e2')
 if gate.get('episode_receipts')!=receipts or gate.get('run_identity_sha256')!=identity['identity_sha256']:
  raise ValueError('E2 gate episode inventory changed')
 diagnostics=verified_reference(gate['ranking_diagnostics']);report=read_json(diagnostics)
 if report.get('schema')!='e2_ranking_diagnostics_v1' or report.get('run_identity_sha256')!=identity['identity_sha256'] or report.get('episodes_evaluated')!=len(results):
  raise ValueError('E2 ranking diagnostics incomplete or from another run')
 episode_diagnostics=report.get('episode_diagnostics',[])
 if len(episode_diagnostics)!=len(results):raise ValueError('Missing per-episode ranking diagnostics')
 for spec,reference in zip(episode_specs(identity['grid']),episode_diagnostics):
  diagnostic=read_json(verified_reference(reference))
  if diagnostic.get('schema')!='episode_ranking_diagnostics_v1' or diagnostic.get('spec')!=spec or diagnostic.get('run_identity_sha256')!=identity['identity_sha256'] or len(diagnostic.get('rows',[]))!=config['split']['online_total']:
   raise ValueError('Per-episode ranking evidence is incomplete or misbound')
 bindings=e2_bindings(identity)
 if gate.get('bindings')!=bindings:raise ValueError('E2 binding digest mismatch')
 return gate,bindings


def authorize_e2_pair(online,calibration,gate_path,current_inputs):
 if online.software_test_only or calibration.software_test_only:raise ValueError('Production E2 authorization cannot authorize software-test matrices')
 gate,bindings=validate_e2_gate(gate_path,current_inputs);digest=sha_file(gate_path)
 actual={'online_matrix_sha256':online._matrix_file_sha256,'online_manifest_sha256':online._manifest_file_sha256,
  'calibration_matrix_sha256':calibration._matrix_file_sha256,'calibration_manifest_sha256':calibration._manifest_file_sha256}
 if any(bindings[k]!=v for k,v in actual.items()):raise ValueError('Loaded environment differs from E2 gate')
 for environment in (online,calibration):
  environment._manifest.update(e2_pass=True,e2_gate_sha256=digest,e2_gate_bindings=dict(bindings))
 return gate


def verified_grid_summary(directory,identity,receipts):
 path=Path(directory)/'grid_summary.json';report=read_json(path)
 if (report.get('schema')!='restricted_feedback_grid_summary_v1'
  or report.get('status')!='COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY'
  or report.get('run_identity_sha256')!=identity['identity_sha256']
  or report.get('grid')!=identity['grid'] or report.get('runs')!=len(receipts)
  or report.get('episode_receipts')!=receipts or report.get('complete_locked_stage_grid') is not True):
  raise ValueError('Complete original-grid summary is missing or changed')
 return artifact_reference(path)


def e3_bindings(identity,receipts,grid_summary):
 if not identity.get('e2_gate'):raise ValueError('E3 identity lacks its mandatory E2 gate')
 return {**e2_bindings(identity),'e2_gate_sha256':identity['e2_gate']['sha256'],
  'e3_grid_plan_sha256':sha_file(Path(grid_summary['path']).parent/'grid_plan.json'),
  'e3_grid_summary_sha256':grid_summary['sha256'],
  'e3_episode_receipts_sha256':canonical_digest(receipts)}


def validate_e3_gate(path,current_inputs):
 """Validate E2 and the complete E3 grid once before an E4 batch.

 Returns (gate, bindings). Callers may create a private typed batch authorization
 after this expensive verification, then cheaply recheck the gate/source hashes
 and actual environment identities per episode. This function never authorizes
 a toy environment or turns a positive effect into a technical requirement.
 """
 gate=read_json(path)
 if (gate.get('schema')!='e3_budget_mechanism_gate_v1' or gate.get('status')!='E3_TECHNICAL_PASS'
  or gate.get('software_test_only') is not False or gate.get('scope')!='technical_completeness_not_scientific_positivity'):
  raise ValueError('An independently accepted production E3 technical gate is required')
 directory=verified_reference(gate['grid_plan']).parent
 identity,results,receipts=verify_complete_grid(directory)
 if identity['inputs']!=current_inputs or gate.get('input_identity_sha256')!=canonical_digest(current_inputs):
  raise ValueError('E3 gate belongs to different matrices/config/algorithms')
 config=read_json(current_inputs['assets']['config']['path']);validate_locked_grid(identity,config,'e3')
 e2_reference=identity.get('e2_gate')
 if not e2_reference or gate.get('e2_gate')!=e2_reference:raise ValueError('E3 gate does not bind its original E2 gate')
 e2_gate,_=validate_e2_gate(verified_reference(e2_reference),current_inputs)
 if gate.get('run_identity_sha256')!=identity['identity_sha256'] or gate.get('episode_receipts')!=receipts:
  raise ValueError('E3 episode inventory changed')
 summary_reference=verified_grid_summary(directory,identity,receipts)
 if gate.get('grid_summary')!=summary_reference:raise ValueError('E3 grid summary reference mismatch')
 report=read_json(verified_reference(gate['mechanism_evaluation']))
 if (report.get('schema')!='e3_budget_mechanism_evaluation_v1'
  or report.get('run_identity_sha256')!=identity['identity_sha256'] or report.get('episodes_evaluated')!=len(results)
  or report.get('e2_control_grid_plan')!=e2_gate['grid_plan']
  or report.get('e2_control_episode_receipts')!=e2_gate['episode_receipts']
  or report.get('quality_and_cost_comparisons_complete') is not True
  or report.get('comparison_count')!=120 or len(report.get('paired_comparisons',[]))!=120
  or report.get('academic_conclusion') not in ('POSITIVE','NEGATIVE','INCONCLUSIVE')):
  raise ValueError('E3 independent mechanism/cost/control evidence incomplete or misbound')
 expected={(r,d,m) for r in identity['grid']['rates'] for d in identity['grid']['delays']
  for m in ('E3/none','E3/random','E3/periodic','E3/max_uncertainty','E3/max_uncertainty_per_cost','E3/no_cost',
   'E2/fixed_combination','E2/contextual_topk','E2/linucb','E2/thompson')}
 observed={(x.get('label_rate'),x.get('feedback_delay'),x.get('reference')) for x in report['paired_comparisons']}
 if observed!=expected:raise ValueError('E3 paired comparisons do not cover every required cell/reference')
 for comparison in report['paired_comparisons']:
  if comparison.get('strategy')!='E3/decision' or comparison.get('stream_seeds')!=sorted(identity['grid']['seeds']) or comparison.get('question_count')!=600:
   raise ValueError('E3 comparison lacks the five full paired streams')
  for metric in ('quality_difference','weighted_model_feedback_cost_difference','executions_difference','labels_difference','tokens_difference','controller_cpu_s_difference','controller_wall_s_difference'):
   interval=comparison.get('metrics',{}).get(metric,{})
   if interval.get('question_support')!=600 or interval.get('adequate_support') is not True or interval.get('per_task_support')!={'bbh':300,'gsm8k':300}:
    raise ValueError('E3 paired metric lacks full question/task support')
 policy=read_json(verified_reference(report['diagnostic_policy']))
 required_policy={'label_rate':25,'feedback_delay':0,'strategy':'E3/decision','reference':'E3/max_uncertainty_per_cost','confidence':.95}
 if any(policy.get(k)!=v for k,v in required_policy.items()):raise ValueError('E3 academic policy differs from its declared primary contrast')
 primary=next(x for x in report['paired_comparisons'] if all(x.get(k)==v for k,v in required_policy.items() if k!='confidence'))
 from .ranking_evaluation import strict_quality_cost_dominance
 conclusion=strict_quality_cost_dominance(primary['metrics']['quality_difference'],primary['metrics']['weighted_model_feedback_cost_difference'])
 if report.get('academic_primary_comparison')!=primary or report.get('academic_conclusion')!=conclusion or gate.get('academic_conclusion')!=conclusion:
  raise ValueError('E3 academic conclusion does not follow its frozen primary policy')
 bindings=e3_bindings(identity,receipts,summary_reference)
 if gate.get('bindings')!=bindings:raise ValueError('E3 binding digest mismatch')
 return gate,bindings
