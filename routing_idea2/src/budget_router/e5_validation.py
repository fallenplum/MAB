"""Independent, pre-final E5 ablation validation and paid-source cost replay.

This closure is separate from the frozen E2/E3 source set. It never reads sealed
final matrices or final features, and never treats positive results as a gate.
"""
from pathlib import Path
from dataclasses import asdict,replace
from collections import defaultdict
import hashlib
import json
import numpy as np
from .stage_validation import (EpisodeStore,read_json,read_jsonl,atomic_json,immutable_json,
 artifact_reference,verified_reference,canonical_digest,verify_input_identity,validate_run_identity,
 make_run_identity,settings_from_config,validate_protocol_config,episode_specs,episode_name,sha_file)
from .simulation import MatrixEnvironment
from .ranking_evaluation import stratified_question_interval,strict_quality_cost_dominance

PLAN_SHA256='34532fe24273c6732e99b6b7c8631d0275281011f5cbfdb141ad4d24425d5b41'
EXTRA_SOURCES=('scripts/run_e5_ablations.py','src/budget_router/e5_validation.py')
CONDITIONS=('paid_member_prior_then_combo','gsm8k_then_bbh','bbh_then_gsm8k')
SHIFT_METHODS=('none','random','max_uncertainty_per_cost','no_cost','decision')
METRICS=('quality','weighted_model_feedback_cost','executions','labels','tokens','controller_cpu_s','controller_wall_s','source_execution_wall_time_s')


def _full_gate(path,inputs):
 from .flex_validation import validate_e4_full_gate
 return validate_e4_full_gate(path,inputs)


def validate_plan(path,config):
 if sha_file(path)!=PLAN_SHA256:raise ValueError('E5 plan differs from the independently locked preregistration')
 p=read_json(path);validate_protocol_config(config)
 if (p.get('schema')!='e5_robustness_preregistration_v1' or p.get('research_complete') is not False
  or p.get('algorithm_stream_seeds')!=config['statistics']['algorithm_stream_seeds']
  or p.get('feedback_rates_per_100')!=[10,25,50,100] or p.get('online_feedback_delays')!=[0,5,20]
  or p.get('full_problem_scale')!={'N':6,'K':3,'combinations':20,'calibration':200,'online':600,'sealed_final':200}
  or p.get('warmup_ablation',{}).get('strategies')!=['none','decision']
  or p.get('task_shift',{}).get('orders')!=list(CONDITIONS[1:])
  or p.get('task_shift',{}).get('strategies')!=list(SHIFT_METHODS)):
  raise ValueError('Incomplete or altered full-scale E5 contract')
 return p


def condition_specifications(config):
 base=settings_from_config(config);out={}
 for condition in CONDITIONS:
  paid=condition=='paid_member_prior_then_combo'
  settings=replace(base,warmup_mode='paid_member_prior_then_combo' if paid else 'random_combo',stream_order='iid' if paid else condition)
  grid={'stage':'e5','strategies':['none','decision'] if paid else list(SHIFT_METHODS),
   'seeds':config['statistics']['algorithm_stream_seeds'],'rates':[10,25,50,100],'delays':[0,5,20]}
  out[condition]={'grid':grid,'settings':json.loads(json.dumps(asdict(settings)))}
 return out


def upstream_refs(full):
 mandatory=read_json(verified_reference(full['mandatory_e4_gate']))
 e3_reference=mandatory['e3_gate'];e3=read_json(verified_reference(e3_reference))
 return {'e3_gate':e3_reference,'e2_gate':e3['e2_gate'],
  'mandatory_e4_gate':full['mandatory_e4_gate'],'flex_reference_gate':full['flex_reference_gate']}


def make_e5_run_plan(root,inputs,e4_full_gate,plan_path):
 config=read_json(inputs['assets']['config']['path']);validate_plan(plan_path,config)
 full,_=_full_gate(e4_full_gate,inputs)
 value={'schema':'e5_ablations_run_identity_v1','inputs':inputs,'e5_plan':artifact_reference(plan_path),
  'e4_full_gate':artifact_reference(e4_full_gate),'upstream':upstream_refs(full),
  'extra_sources':{name:artifact_reference(Path(root)/name) for name in EXTRA_SOURCES},
  'conditions':condition_specifications(config),'sealed_final_read_or_encoded':False,'research_complete':False}
 value['identity_sha256']=canonical_digest(value);return value


def verify_run_plan(plan,*,parent=True):
 if plan.get('schema')!='e5_ablations_run_identity_v1' or plan.get('identity_sha256')!=canonical_digest({k:v for k,v in plan.items() if k!='identity_sha256'}):
  raise ValueError('E5 bundle identity changed')
 verify_input_identity(plan['inputs']);config=read_json(plan['inputs']['assets']['config']['path'])
 validate_plan(verified_reference(plan['e5_plan']),config)
 if set(plan.get('extra_sources',{}))!=set(EXTRA_SOURCES):raise ValueError('E5 source closure incomplete')
 for reference in plan['extra_sources'].values():verified_reference(reference)
 full_path=verified_reference(plan['e4_full_gate'])
 full=_full_gate(full_path,plan['inputs'])[0] if parent else read_json(full_path)
 if plan.get('upstream')!=upstream_refs(full) or plan.get('conditions')!=condition_specifications(config):
  raise ValueError('E5 prerequisite references/conditions differ from the locked full plan')
 if plan.get('sealed_final_read_or_encoded') is not False:raise ValueError('E5 ablations must precede any final feature release')
 return plan


def make_condition_identity(run_plan,run_plan_path,condition):
 cfg=run_plan['conditions'][condition];config=read_json(run_plan['inputs']['assets']['config']['path'])
 base=settings_from_config(config);settings=replace(base,warmup_mode=cfg['settings']['warmup_mode'],stream_order=cfg['settings']['stream_order'])
 identity=make_run_identity(run_plan['inputs'],cfg['grid'],settings,run_plan['upstream']['e2_gate']['path'])
 identity.update(e5_condition=condition,e5_plan=run_plan['e5_plan'],e4_full_gate=run_plan['e4_full_gate'],
  extra_sources=run_plan['extra_sources'],e5_run_plan=artifact_reference(run_plan_path))
 identity['identity_sha256']=canonical_digest({k:v for k,v in identity.items() if k!='identity_sha256'})
 return identity


def verify_condition_identity(identity,run_plan=None,*,parent=True):
 validate_run_identity(identity)
 if run_plan is None:run_plan=read_json(verified_reference(identity['e5_run_plan']))
 verify_run_plan(run_plan,parent=parent)
 condition=identity.get('e5_condition')
 if condition not in CONDITIONS or identity!=make_condition_identity(run_plan,identity['e5_run_plan']['path'],condition):
  raise ValueError('E5 condition settings/grid/source binding changed')
 return run_plan


def _indices(environment):
 return {hashlib.sha256(('routing-question-v1:'+str(q)).encode()).hexdigest():i for i,q in enumerate(environment._arrays['question_ids'])}


def _cost(environment,row,stem,cells):
 a=environment._arrays
 return {'executions':int(np.sum(a[stem+'_execution_counts'][row,cells])),
  'tokens':int(np.sum(a[stem+'_prompt_tokens'][row,cells])+np.sum(a[stem+'_completion_tokens'][row,cells]))}


def _wall(environment,row,stem,cells):
 if not cells:return 0.
 a=environment._arrays.get(stem+'_wall_time_s')
 return None if a is None else float(np.sum(a[row,cells]))


def _known(values):return None if any(x is None for x in values) else float(sum(values))


def project_e5_episode(result,online,calibration):
 """Audit paid member-only references without generating pseudo aggregate labels."""
 s=result['summary'];settings=s['settings'];n=len(online);cal_index=_indices(calibration);on_index=_indices(online)
 paid=settings['warmup_mode']=='paid_member_prior_then_combo';cap=min(32,n*s['label_rate']//100//5,len(calibration))
 target=cap//2 if paid else 0;records=result['warmup'];warm={'executions':0,'labels':0,'tokens':0};walls=[];references=[];seen=set()
 init=np.random.default_rng(np.random.SeedSequence([s['seed'],1001]));cal_order=init.permutation(len(calibration))[:cap]
 if len(records)!=cap:raise ValueError('E5 must execute the complete planned warmup; no silent budget fallback')
 for position,record in enumerate(records):
  qid=record['question_id'];row=cal_index.get(qid);index=record['combination_index'];member_only=position<target
  planned_combo=int(init.integers(20))
  if row!=int(cal_order[position]) or qid in seen or record.get('feedback_units')!=1:
   raise ValueError('E5 warmup changed fixed shuffle/reference schedule')
  seen.add(qid)
  if member_only:
   if index is not None or record.get('purchase_kind')!='paid_all_six_member_reference' or record.get('aggregate_labels')!=0 or record.get('generated_member_labels')!=6:
    raise ValueError('Member-only reference must pay for six real members and zero aggregate labels')
   cells=[('member',list(range(6)))];references.append(row)
  else:
   if index!=planned_combo or record.get('purchase_kind')!='paid_combo_final_label' or record.get('aggregate_labels')!=1:
    raise ValueError('Missing planned real aggregate warmup label')
   cells=[('member',list(calibration.mapper.subsets[index])),('aggregate',[index])]
  actual={'executions':0,'labels':1,'tokens':0}
  for stem,indices in cells:
   for key,value in _cost(calibration,row,stem,indices).items():actual[key]+=value
   walls.append(_wall(calibration,row,stem,indices))
  if record.get('actual_cost')!=actual:raise ValueError('Warmup event actual resources disagree with source cells')
  for key in warm:warm[key]+=actual[key]
 if warm!=s['warmup_cost'] or s.get('paid_member_reference_events')!=target or s.get('prior_member_labels_used')!=target*6 or s.get('warmup_arrived_combo_labels')!=cap-target:
  raise ValueError('E5 paid reference/auxiliary labels/actual warmup costs do not reconcile')
 state=result['policy_checkpoint']['shared_ridge'];prior=np.asarray(state['prior_mean']);expected=np.zeros((online.phi_dimension,online.mapper.basis_dim))
 if paid and target:
  x=calibration._arrays['phi'][references];variance=settings['noise_variance'];design=settings['ridge']*np.eye(online.phi_dimension)+x.T@x/variance
  for member in range(6):expected[:,1+member]=np.linalg.solve(design,x.T@calibration._arrays['member_scores'][references,member]/variance)/3.
  if s.get('explicit_member_prior_installed') is not True:raise ValueError('Paid member prior was not explicitly installed')
 if prior.shape!=(expected.size,) or not np.allclose(prior,expected.ravel(),rtol=1e-9,atol=1e-10):raise ValueError('Prior mean differs from only the paid contextual member evidence')
 member_qids={calibration.context(i).question_id for i in references}
 if any(label['question_id'] in member_qids for label in state['arrived_original_labels']):raise ValueError('A member-only reference became an aggregate pseudo-label')
 order=np.random.default_rng(s['seed']).permutation(n)
 if settings['stream_order']!='iid':
  tasks=('gsm8k','bbh') if settings['stream_order']=='gsm8k_then_bbh' else ('bbh','gsm8k')
  if set(online._arrays['task_labels'].tolist())!=set(tasks):raise ValueError('Task shift must retain both original tasks')
  order=[i for task in tasks for i in order if str(online._arrays['task_labels'][i])==task]
 if [r['question_id'] for r in result['traces']]!=[online.context(int(i)).question_id for i in order]:raise ValueError('Task order changed, dropped questions or lost fixed within-task permutation')
 total=dict(warm);warm_wall=_known(walls);runtime=s['runtime_measurements'];rows=[];weights=settings['cost_weights']
 for trace in result['traces']:
  row=on_index[trace['question_id']];main=trace['main_index'];alternate=trace['alternate_index']
  if isinstance(main,bool) or not isinstance(main,int) or not 0<=main<20 or (alternate is not None and (isinstance(alternate,bool) or not isinstance(alternate,int) or not 0<=alternate<20 or alternate==main)):raise ValueError('Invalid frozen service/alternate index')
  members=list(online.mapper.subsets[main]);truth=float(online._arrays['aggregate_scores'][row,main])
  if trace['quality']!=truth or trace['main_members']!=members or trace['task']!=str(online._arrays['task_labels'][row]):raise ValueError('Frozen service quality differs from its original matrix')
  service={'executions':0,'labels':0,'tokens':0};extra={'executions':0,'labels':trace['requested_combination_labels'],'tokens':0};source=[]
  for destination,cells in ((service,[('member',members),('aggregate',[main])]),
   (extra,[] if alternate is None else [('member',sorted(set(online.mapper.subsets[alternate])-set(members))),('aggregate',[alternate])])):
   for stem,indices in cells:
    for key,value in _cost(online,row,stem,indices).items():destination[key]+=value
    source.append(_wall(online,row,stem,indices))
  if service!=trace['service_cost'] or extra!=trace['purchase_cost']:raise ValueError('E5 service/audit actual resource mismatch')
  step={key:service[key]+extra[key] for key in total}
  for key in total:total[key]+=step[key]
  if total!=trace['cumulative_cost']:raise ValueError('E5 cumulative spending differs from actual paid sources')
  allocated={k:step[k]+warm[k]/n for k in total}
  rows.append({'question_id':trace['question_id'],'task':trace['task'],'quality':truth,**allocated,
   'weighted_model_feedback_cost':sum(w*allocated[k] for w,k in zip(weights,('executions','labels','tokens'))),
   'controller_cpu_s':trace['controller_process_time_s']+runtime['initialization_controller_process_time_s']/n,
   'controller_wall_s':trace['controller_wall_time_s']+runtime['initialization_controller_wall_time_s']/n,
   'source_execution_wall_time_s':_known([_known(source),None if warm_wall is None else warm_wall/n])})
 if total!=s['budget']['spent'] or any(total[k]>s['budget']['limits'][k] for k in total):raise ValueError('E5 actual total exceeds budget or fails to reconcile')
 return {'summary':s,'traces':rows}


def load_environments(inputs):
 a=inputs['assets'];online=MatrixEnvironment.from_npz(a['online_matrix']['path'],a['online_manifest']['path'],expected_split='online_stream')
 cal=MatrixEnvironment.from_npz(a['calibration_matrix']['path'],a['calibration_manifest']['path'],expected_split='online_calibration')
 if online.software_test_only or cal.software_test_only or len(online)!=600 or len(cal)!=200 or online.phi_dimension!=17 or cal.phi_dimension!=17:
  raise ValueError('E5 requires original real full-scale 600/200 and 17 frozen features')
 if dict(zip(*np.unique(online._arrays['task_labels'],return_counts=True)))!={'bbh':300,'gsm8k':300} or dict(zip(*np.unique(cal._arrays['task_labels'],return_counts=True)))!={'bbh':100,'gsm8k':100}:raise ValueError('E5 original task balance changed')
 return online,cal


def _frozen_entry(directory,condition,spec):
 directory=Path(directory).resolve();folder=directory/episode_name(spec);receipt=read_json(folder/'episode_receipt.json');c=receipt['artifacts']['policy_checkpoint.json']
 checkpoint={'path':str(folder/'policy_checkpoint.json'),'sha256':c['sha256'],'bytes':c['bytes']};verified_reference(checkpoint)
 return {'condition_id':condition,**spec,'grid_plan':artifact_reference(directory/'grid_plan.json'),
  'episode_receipt':artifact_reference(folder/'episode_receipt.json'),'policy_checkpoint':checkpoint}


def upstream_inventory(plan):
 inventories={};frozen=[]
 for name,key in (('E2_iid_random_combo','e2_gate'),('E3_iid_random_combo','e3_gate'),
  ('E4_mandatory_shared_permissions','mandatory_e4_gate'),('E4_flex_controlled_reference','flex_reference_gate')):
  gate=read_json(verified_reference(plan['upstream'][key]));directory=verified_reference(gate['grid_plan']).parent
  identity=read_json(directory/'grid_plan.json');specs=episode_specs(identity['grid'])
  receipts=[artifact_reference(directory/episode_name(spec)/'episode_receipt.json') for spec in specs]
  required_count={'E2_iid_random_combo':300,'E3_iid_random_combo':420,'E4_mandatory_shared_permissions':240,'E4_flex_controlled_reference':60}[name]
  if gate.get('episode_receipts')!=receipts or len(receipts)!=required_count:raise ValueError('Upstream reused checkpoint inventory changed')
  inventories[name]={'grid_plan':gate['grid_plan'],'episode_receipts':receipts}
  frozen.extend(_frozen_entry(directory,name,spec) for spec in specs)
 return inventories,frozen


def _comparison(treated,control,condition,method,kind,*,replicates,seed):
 a={r['summary']['seed']:r for r in treated};b={r['summary']['seed']:r for r in control}
 if set(a)!=set(b) or len(a)!=5:raise ValueError('E5 comparisons require five matching stream seeds')
 differences=[]
 for stream_seed in sorted(a):
  left={r['question_id']:r for r in a[stream_seed]['traces']};right={r['question_id']:r for r in b[stream_seed]['traces']}
  if set(left)!=set(right):raise ValueError('E5 comparison dropped original questions')
  for qid,row in left.items():
   if row['task']!=right[qid]['task']:raise ValueError('Paired task mismatch')
   d={'question_id':qid,'task':row['task']}
   for metric in METRICS:
    av,bv=row[metric],right[qid][metric];d[metric+'_difference']=None if av is None or bv is None else av-bv
   differences.append(d)
 metrics={m+'_difference':stratified_question_interval(differences,value_key=m+'_difference',replicates=replicates,seed=seed,confidence=.95,minimum_per_task=20) for m in METRICS}
 s=treated[0]['summary']
 return {'condition_id':condition,'strategy':method,'comparison_kind':kind,'label_rate':s['label_rate'],'feedback_delay':s['feedback_delay'],
  'stream_seeds':sorted(a),'question_count':len(left),'metrics':metrics,
  'quality_priced_cost_dominance_descriptive':strict_quality_cost_dominance(metrics['quality_difference'],metrics['weighted_model_feedback_cost_difference'])}


def all_comparisons(conditions,iid,config):
 def group(results):
  out=defaultdict(list)
  for result in results:
   s=result['summary'];out[(s['strategy'],s['label_rate'],s['feedback_delay'])].append(result)
  return out
 baseline=group(iid);versus_iid=[];within=[];stats=config['statistics'];kwargs={'replicates':stats.get('e1_bootstrap_replicates',2000),'seed':stats.get('e1_bootstrap_seed',20261030)}
 for condition,results in conditions.items():
  grouped=group(results)
  for (method,rate,delay),treated in grouped.items():
   versus_iid.append(_comparison(treated,baseline[(method,rate,delay)],condition,method,'same_method_vs_reused_iid_random_warmup',**kwargs))
   if condition!='paid_member_prior_then_combo' and method!='decision':
    within.append(_comparison(grouped[('decision',rate,delay)],treated,condition,'decision_vs_'+method,'within_shift_same_caps',**kwargs))
 return versus_iid,within


def verify_complete_bundle(directory,*,parent=True,project=True):
 directory=Path(directory).resolve();plan=read_json(directory/'ablation_plan.json');verify_run_plan(plan,parent=parent)
 env=load_environments(plan['inputs']) if project else None
 inventory=[];conditions={};frozen=[]
 for condition in CONDITIONS:
  folder=directory/condition;identity=read_json(folder/'grid_plan.json');verify_condition_identity(identity,plan,parent=False)
  store=EpisodeStore(folder,identity,resume=True);receipts=[];conditions[condition]=[]
  for spec in episode_specs(identity['grid']):
   result=store.load(spec)
   if result is None:raise ValueError('Incomplete E5 grid: '+condition+'/'+episode_name(spec))
   if project:conditions[condition].append(project_e5_episode(result,*env))
   receipts.append(artifact_reference(folder/episode_name(spec)/'episode_receipt.json'));frozen.append(_frozen_entry(folder,condition,spec))
  summary=read_json(folder/'grid_summary.json')
  if summary.get('schema')!='e5_condition_grid_summary_v1' or summary.get('status')!='COMPLETED_RESTRICTED_FEEDBACK_SIMULATION_ONLY' or summary.get('condition_id')!=condition or summary.get('grid')!=identity['grid'] or summary.get('run_identity_sha256')!=identity['identity_sha256'] or summary.get('episode_receipts')!=receipts or summary.get('runs')!=len(receipts):
   raise ValueError('E5 condition summary incomplete or changed')
  inventory.append({'condition_id':condition,'grid_plan':artifact_reference(folder/'grid_plan.json'),'grid_summary':artifact_reference(folder/'grid_summary.json'),'episode_receipts':receipts})
 upstream,previous=upstream_inventory(plan);frozen=previous+frozen
 return plan,inventory,upstream,frozen,conditions,env


def evaluate_bundle(directory,gate_path=None):
 directory=Path(directory).resolve();plan,inventory,upstream,frozen,conditions,env=verify_complete_bundle(directory,parent=True,project=True)
 online,cal=env;e3_dir=verified_reference(upstream['E3_iid_random_combo']['grid_plan']).parent;e3_identity=read_json(e3_dir/'grid_plan.json');store=EpisodeStore(e3_dir,e3_identity,resume=True)
 iid=[project_e5_episode(store.load(spec),online,cal) for spec in episode_specs(e3_identity['grid']) if spec['strategy'] in SHIFT_METHODS]
 config=read_json(plan['inputs']['assets']['config']['path']);versus,within=all_comparisons(conditions,iid,config)
 if len(versus)!=144 or len(within)!=96:raise ValueError('E5 paired ablation comparisons incomplete')
 e3_gate=read_json(verified_reference(plan['upstream']['e3_gate']));e3_report=read_json(verified_reference(e3_gate['mechanism_evaluation']))
 reused=[x for x in e3_report['paired_comparisons'] if x['reference'] in ('E3/max_uncertainty_per_cost','E3/max_uncertainty','E3/no_cost')]
 if len(reused)!=36:raise ValueError('Existing remove-direction/no-cost evidence incomplete')
 runtime=[{'condition_id':condition,'strategy':r['summary']['strategy'],'seed':r['summary']['seed'],'label_rate':r['summary']['label_rate'],
  'feedback_delay':r['summary']['feedback_delay'],'actual_resources':r['summary']['budget']['spent'],'hard_limits':r['summary']['budget']['limits'],
  'warmup_cost':r['summary']['warmup_cost'],'paid_member_reference_events':r['summary']['paid_member_reference_events'],
  'prior_member_labels_used':r['summary']['prior_member_labels_used'],'runtime_measurements':r['summary']['runtime_measurements'],
  'source_execution_wall_time_s':_known([row['source_execution_wall_time_s'] for row in r['traces']])} for condition,results in conditions.items() for r in results]
 report={'schema':'e5_ablations_evaluation_v1','run_identity_sha256':plan['identity_sha256'],'episodes_evaluated':720,
  'versus_iid_comparisons':versus,'within_shift_comparisons':within,'reused_main_mechanism_evidence':reused,
  'reused_e3_mechanism_report':e3_gate['mechanism_evaluation'],'actual_costs_independently_recomputed':True,'runtime_and_actual_costs':runtime,
  'scope':'Paid-prior changes purchased information under identical hard ceilings; task shift changes temporal task proportions only. Actual spending differs. Negative results remain in the report and do not block technical completeness.',
  'statistics_scope':'Paired question unit, seed means within question, task-stratified bootstrap. All ablation contrasts are descriptive; no best condition is selected for final evaluation.',
  'spending_units':'Comparisons use actual cost per served question, with each method paid calibration warmup amortized uniformly across its 600 questions. Full actual episode resource vectors are reported separately; identical hard ceilings do not imply identical spending.',
  'physical_vs_logical_scope':'CPU replay generates no new model answers. Selected pinned source cells are charged logically per policy; original shared-cache physical generation is accounted separately and must not be double billed.',
  'algorithm_stream_seeds_are_not_independent_model_generations':True,
  'source_wall_time_scope':'Actual selected source cells including paid six-member reference warmup. Missing times stay null; CPU and the three priced resources are distinct.',
  'physical_generation_replicates':1,'sealed_final_read_or_encoded':False,'final_evaluation_status':'NOT_RUN','research_complete':False}
 rp=directory/'e5_ablation_evaluation.json';immutable_json(rp,report)
 gate={'schema':'e5_ablations_pre_final_gate_v1','status':'E5_PRE_FINAL_TECHNICAL_PASS','software_test_only':False,
  'scope':'technical_completeness_not_scientific_positivity','run_plan':artifact_reference(directory/'ablation_plan.json'),
  'input_identity_sha256':canonical_digest(plan['inputs']),'e4_full_gate':plan['e4_full_gate'],'e5_plan':plan['e5_plan'],
  'extra_sources':plan['extra_sources'],'condition_grids':inventory,'reused_upstream_grids':upstream,
  'frozen_policy_checkpoints':frozen,'evaluation':artifact_reference(rp),'final_evaluation_status':'NOT_RUN',
  'sealed_final_read_or_encoded':False,'research_complete':False}
 gate['bindings']=gate_bindings(gate)
 path=Path(gate_path).resolve() if gate_path else directory/'e5_pre_final_gate.json'
 candidate=path.with_name(path.name+'.candidate');immutable_json(candidate,gate);validate_e5_pre_final_gate(candidate,plan['inputs']);immutable_json(path,gate);candidate.unlink()
 return gate


def gate_bindings(gate):
 return {'input_identity_sha256':gate['input_identity_sha256'],'e5_plan_sha256':gate['e5_plan']['sha256'],
  'e4_full_gate_sha256':gate['e4_full_gate']['sha256'],'e5_extra_sources_sha256':canonical_digest(gate['extra_sources']),
  'e5_run_plan_sha256':gate['run_plan']['sha256'],'condition_grid_inventory_sha256':canonical_digest(gate['condition_grids']),
  'reused_main_inventory_sha256':canonical_digest(gate['reused_upstream_grids']),
  'frozen_policy_checkpoints_sha256':canonical_digest(gate['frozen_policy_checkpoints']),'e5_evaluation_sha256':gate['evaluation']['sha256']}


def validate_e5_pre_final_gate(path,current_inputs):
 gate=read_json(path)
 if gate.get('schema')!='e5_ablations_pre_final_gate_v1' or gate.get('status')!='E5_PRE_FINAL_TECHNICAL_PASS' or gate.get('software_test_only') is not False or gate.get('scope')!='technical_completeness_not_scientific_positivity':
  raise ValueError('A complete artifact-bound E5 pre-final gate is required')
 directory=verified_reference(gate['run_plan']).parent
 declared_plan=read_json(directory/'ablation_plan.json')
 if declared_plan.get('inputs')!=current_inputs or gate.get('input_identity_sha256')!=canonical_digest(current_inputs):raise ValueError('E5 gate belongs to different original matrices/config/source')
 plan,inventory,upstream,frozen,_,_=verify_complete_bundle(directory,parent=True,project=True)
 if plan['inputs']!=current_inputs or gate.get('input_identity_sha256')!=canonical_digest(current_inputs):raise ValueError('E5 gate belongs to different original matrices/config/source')
 for key,value in (('e4_full_gate',plan['e4_full_gate']),('e5_plan',plan['e5_plan']),('extra_sources',plan['extra_sources']),
  ('condition_grids',inventory),('reused_upstream_grids',upstream),('frozen_policy_checkpoints',frozen)):
  if gate.get(key)!=value:raise ValueError('E5 gate inventory differs: '+key)
 report=read_json(verified_reference(gate['evaluation']))
 if (report.get('schema')!='e5_ablations_evaluation_v1' or report.get('run_identity_sha256')!=plan['identity_sha256'] or report.get('episodes_evaluated')!=720
  or len(report.get('versus_iid_comparisons',[]))!=144 or len(report.get('within_shift_comparisons',[]))!=96
  or len(report.get('reused_main_mechanism_evidence',[]))!=36 or report.get('actual_costs_independently_recomputed') is not True):
  raise ValueError('E5 ablation/cost evidence incomplete')
 expected={(c,m,r,d) for c,cfg in plan['conditions'].items() for m in cfg['grid']['strategies'] for r in (10,25,50,100) for d in (0,5,20)}
 observed={(x.get('condition_id'),x.get('strategy'),x.get('label_rate'),x.get('feedback_delay')) for x in report['versus_iid_comparisons']}
 expected_within={(c,'decision_vs_'+m,r,d) for c in CONDITIONS[1:] for m in SHIFT_METHODS if m!='decision' for r in (10,25,50,100) for d in (0,5,20)}
 observed_within={(x.get('condition_id'),x.get('strategy'),x.get('label_rate'),x.get('feedback_delay')) for x in report['within_shift_comparisons']}
 if observed!=expected or observed_within!=expected_within:raise ValueError('E5 comparison grid incomplete or duplicated')
 for comparison in report['versus_iid_comparisons']+report['within_shift_comparisons']:
  if comparison.get('stream_seeds')!=sorted(plan['conditions'][comparison['condition_id']]['grid']['seeds']) or comparison.get('question_count')!=600:raise ValueError('E5 paired stream support incomplete')
  for metric in METRICS[:-1]:
   interval=comparison.get('metrics',{}).get(metric+'_difference',{})
   if interval.get('question_support')!=600 or interval.get('adequate_support') is not True or interval.get('per_task_support')!={'bbh':300,'gsm8k':300}:raise ValueError('E5 metric lacks complete paired question/task support')
 for obj in (gate,report):
  if obj.get('sealed_final_read_or_encoded') is not False or obj.get('final_evaluation_status')!='NOT_RUN' or obj.get('research_complete') is not False:raise ValueError('Pre-final gate cannot claim final completion or depend on final data')
 runtime=report.get('runtime_and_actual_costs',[])
 runtime_expected={(c,m,s,r,d) for c,cfg in plan['conditions'].items() for m in cfg['grid']['strategies'] for s in cfg['grid']['seeds'] for r in (10,25,50,100) for d in (0,5,20)}
 runtime_observed={(x.get('condition_id'),x.get('strategy'),x.get('seed'),x.get('label_rate'),x.get('feedback_delay')) for x in runtime}
 if len(runtime)!=720 or runtime_observed!=runtime_expected:raise ValueError('E5 per-episode resources/clocks incomplete')
 for row in runtime:
  for key in ('executions','labels','tokens'):
   value=row.get('actual_resources',{}).get(key)
   if isinstance(value,bool) or not isinstance(value,int) or value<0 or value>row.get('hard_limits',{}).get(key,-1):raise ValueError('Invalid E5 actual resource statistics')
  if not isinstance(row.get('runtime_measurements'),dict) or 'source_execution_wall_time_s' not in row:raise ValueError('E5 measured controller/source time status missing')
  for clock in ('controller_process_time_s','controller_wall_time_s'):
   value=row['runtime_measurements'].get(clock)
   if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:raise ValueError('E5 controller clocks missing or invalid')
  source_time=row['source_execution_wall_time_s']
  if source_time is not None and (isinstance(source_time,bool) or not isinstance(source_time,(int,float)) or not np.isfinite(source_time) or source_time<0):raise ValueError('E5 source clock must be measured nonnegative or explicitly missing')
 e3_gate=read_json(verified_reference(plan['upstream']['e3_gate']));e3_reference=e3_gate['mechanism_evaluation'];e3_report=read_json(verified_reference(e3_reference))
 expected_reused=[x for x in e3_report['paired_comparisons'] if x['reference'] in ('E3/max_uncertainty_per_cost','E3/max_uncertainty','E3/no_cost')]
 if report.get('reused_e3_mechanism_report')!=e3_reference or report.get('reused_main_mechanism_evidence')!=expected_reused:raise ValueError('E5 remove-direction/no-cost evidence differs from original E3 receipts')
 if gate.get('bindings')!=gate_bindings(gate):raise ValueError('E5 gate binding digest changed')
 return gate,gate['bindings']


def validate_e5_episode(checkpoint_ref,episode_receipt_ref,grid_plan_ref,*,batch_authorization=None):
 plan_path=verified_reference(grid_plan_ref);identity=read_json(plan_path)
 if batch_authorization is None:verify_condition_identity(identity,parent=True);environments=None
 else:
  if not isinstance(batch_authorization,_E5FinalBatchAuthorization) or batch_authorization._token is not _E5_AUTH_TOKEN:raise ValueError('An opaque verified E5 final-batch authorization is required')
  batch_authorization.verify_unchanged(full=False)
  key=_episode_authorization_key(checkpoint_ref,episode_receipt_ref,grid_plan_ref)
  if key not in batch_authorization._allowed:raise ValueError('Checkpoint is outside the fully accepted E5 batch')
  run_plan=json.loads(batch_authorization._plan_json)
  verify_condition_identity(identity,run_plan,parent=False);environments=batch_authorization._environments
 receipt_path=verified_reference(episode_receipt_ref);receipt=read_json(receipt_path)
 if receipt_path!=(plan_path.parent/episode_name(receipt['spec'])/'episode_receipt.json').resolve():raise ValueError('E5 checkpoint episode outside its exact condition grid')
 store=EpisodeStore(plan_path.parent,identity,resume=True);result=store.load(receipt['spec'])
 if verified_reference(checkpoint_ref)!=(receipt_path.parent/'policy_checkpoint.json').resolve():raise ValueError('E5 frozen checkpoint differs from its paid episode')
 project_e5_episode(result,*(load_environments(identity['inputs']) if environments is None else environments))
 return result


_E5_AUTH_TOKEN=object()

def _episode_authorization_key(checkpoint,receipt,plan):
 return tuple((str(Path(r['path']).resolve()),r['sha256']) for r in (checkpoint,receipt,plan))


def _references(value):
 if isinstance(value,dict):
  if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):yield value
  else:
   for item in value.values():yield from _references(item)
 elif isinstance(value,(list,tuple)):
  for item in value:yield from _references(item)


class _E5FinalBatchAuthorization:
 __slots__=('_token','_plan_json','_allowed','_critical_refs','_all_refs','_environments','_gate_json')
 def __init__(self,token,plan,allowed,critical_refs,all_refs,environments,gate):
  if token is not _E5_AUTH_TOKEN:raise ValueError('E5 authorization must come from the strict batch factory')
  self._token=token;self._plan_json=json.dumps(plan,sort_keys=True,allow_nan=False);self._allowed=frozenset(allowed)
  self._critical_refs=tuple(json.dumps(r,sort_keys=True) for r in critical_refs)
  self._all_refs=tuple(json.dumps(r,sort_keys=True) for r in all_refs);self._environments=environments;self._gate_json=json.dumps(gate,sort_keys=True,allow_nan=False)
 @property
 def gate(self):return json.loads(self._gate_json)
 @property
 def bindings(self):return json.loads(self._gate_json)['bindings']
 def verify_unchanged(self,*,full=True):
  if self._token is not _E5_AUTH_TOKEN:raise ValueError('Invalid E5 batch authorization')
  plan=json.loads(self._plan_json);verify_input_identity(plan['inputs'])
  for text in self._critical_refs+(self._all_refs if full else ()):
   verified_reference(json.loads(text))
  return True


def authorize_e5_final_batch(path,current_inputs):
 """Verify the full parent/grid once, then issue a private in-memory batch ticket.

 Per-episode verification still checks the selected artifact hashes, all core
 and E5 sources, parent gate/plan identities, and paid source arithmetic. Call
 ticket.verify_unchanged() at batch end to rescan every frozen reference.
 """
 gate,_=validate_e5_pre_final_gate(path,current_inputs);plan=read_json(verified_reference(gate['run_plan']))
 critical=[artifact_reference(path),gate['run_plan'],gate['e4_full_gate'],gate['e5_plan']]
 critical.extend(_references(plan))
 for inventory in gate['reused_upstream_grids'].values():
  critical.extend(_references(read_json(verified_reference(inventory['grid_plan']))))
 for condition in gate['condition_grids']:critical.extend([condition['grid_plan'],condition['grid_summary']])
 def unique(values):return list({(r['path'],r['sha256']):r for r in values}.values())
 all_refs=list(_references(gate['frozen_policy_checkpoints']));allowed=[]
 for item in gate['frozen_policy_checkpoints']:
  if item['condition_id'] in CONDITIONS:allowed.append(_episode_authorization_key(item['policy_checkpoint'],item['episode_receipt'],item['grid_plan']))
 return _E5FinalBatchAuthorization(_E5_AUTH_TOKEN,plan,allowed,unique(critical),unique(all_refs),load_environments(current_inputs),gate)
