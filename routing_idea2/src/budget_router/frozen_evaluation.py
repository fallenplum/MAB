"""Frozen holdout evaluation: no training, feedback delivery or PCA fitting.
Production decisions require a hash-bound release. Final matrices are opened
only by an evaluator after the full immutable decision bundle is verified.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import uuid
import numpy as np
from .features import SubsetFeatureMap
from .stage_validation import (SOURCE_FILES, read_json, canonical_digest, sha_file,
    artifact_reference, verified_reference, validate_run_identity, EpisodeStore)

SEEDS = (42,137,2024,2025,2026)
FINAL_GROUP_PLAN_SHA256 = 'ae61f219cb6be15c15a8e648a773e6c529c0fc0dd1448a7ba30cd15836d6b3c6'
FINAL_SOURCES = ('src/budget_router/frozen_evaluation.py','scripts/evaluate_frozen_policies.py')
FINAL_GENERATION_SOURCES = ('src/budget_router/final_generation.py','scripts/prepare_final_jobs.py',
    'scripts/run_final_generation.py','scripts/prepare_final_matrices.py','src/budget_router/runtime_memory.py','src/budget_router/runtime_diagnostics.py')
NEIGHBOR_FINAL_SOURCES = ('src/budget_router/neighbor_validation.py','src/budget_router/neighbor_policies.py','src/budget_router/neighbor_simulation.py')
PENDING_KEYS = {'available_at','request_sequence','question_id','label_ids',
    'combination_indices','generated_member_indices'}
PROJECT = Path(__file__).resolve().parents[2]
PROVENANCE_KEYS = ('e0_lock_sha256','e1_gate_sha256','pca_sha256','model_pool',
    'cache_revision','aggregator','scorer')


def _int(value, minimum=0):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError('Expected an explicit bounded integer')
    return value


def _array(value,shape):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError('Invalid or nonfinite frozen parameter/feature shape')
    return a.copy()


def _digest(value):
    return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def anonymous_question_id(raw):
    return hashlib.sha256(('routing-question-v1:'+str(raw)).encode()).hexdigest()


def _compact_checkpoint_metadata(cp):
    keys=('schema','strategy','seed','label_rate','feedback_delay','next_tick','frozen_as_of_tick',
          'feature_map','settings','rng_states','online_provenance','calibration_provenance')
    value={key:deepcopy(cp[key]) for key in keys if key in cp}
    value['budget']={key:deepcopy(cp.get('budget',{}).get(key,{})) for key in ('spent','limits')}
    value['full_ledger_location']='Exact hash-bound original checkpoint; not retained in policy memory'
    return value


def _ridge_state(state,dimension,tick):
    if state.get('feature_dim')!=dimension:raise ValueError('Frozen ridge dimension mismatch')
    ridge=float(state['ridge']);noise=float(state['noise_variance'])
    if not np.isfinite([ridge,noise]).all() or min(ridge,noise)<=0:
        raise ValueError('Invalid frozen ridge precision/noise')
    design=_array(state['design'],(dimension,dimension))
    inverse=_array(state['design_inverse'],(dimension,dimension))
    theta=_array(state['theta'],(dimension,));rhs=_array(state['rhs'],(dimension,))
    prior=_array(state.get('prior_mean',np.zeros(dimension)),(dimension,))
    if not np.allclose(design,design.T) or not np.allclose(inverse,inverse.T):
        raise ValueError('Frozen design and inverse must be symmetric')
    try:
        np.linalg.cholesky(design);np.linalg.cholesky(inverse)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Frozen ridge precision must be positive definite') from exc
    if not np.allclose(design@inverse,np.eye(dimension),rtol=1e-6,atol=1e-6):
        raise ValueError('Frozen inverse is inconsistent with design')
    if not np.allclose(design@theta,rhs,rtol=1e-6,atol=1e-6):
        raise ValueError('Frozen theta is inconsistent with rhs')
    labels=state.get('arrived_original_labels')
    if not isinstance(labels,list):raise ValueError('Frozen original-label ledger is required')
    seen=set();contexts=set();x=[];y=[]
    for item in labels:
        if (item.get('kind')!='original' or not item.get('label_id')
                or item['label_id'] in seen or not item.get('question_id')):
            raise ValueError('Duplicate, pseudo or invalid original-label ledger')
        if _int(item['available_at'])>tick:raise ValueError('Unarrived feedback entered frozen parameters')
        value=float(item['value'])
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid observed label quality')
        x.append(_array(item['features'],(dimension,)));y.append(value)
        seen.add(item['label_id']);contexts.add(item['question_id'])
    expected_design=np.eye(dimension)*ridge;expected_rhs=ridge*prior
    if x:
        a=np.stack(x);b=np.asarray(y);expected_design+=a.T@a/noise;expected_rhs+=a.T@b/noise
    if (not np.allclose(design,expected_design,rtol=1e-6,atol=1e-6)
            or not np.allclose(rhs,expected_rhs,rtol=1e-6,atol=1e-6)):
        raise ValueError('Frozen parameters do not match paid original labels and explicit prior')
    theta.flags.writeable=False;inverse.flags.writeable=False
    return theta,inverse,contexts


class FrozenPolicy:
    """No learning interface; independent stateless Thompson RNG per question.

    Evaluation SHA streams do not consume training RNG or queued paid labels.
    The policy parameters remain frozen even when evaluating different orders.
    """
    def __init__(self,checkpoint,*,_production=False,evaluation_seed=20261031):
        self._checkpoint=deepcopy(checkpoint);cp=self._checkpoint;checkpoint_digest=canonical_digest(cp)
        self.software_test_only=not _production
        if cp.get('schema')!='restricted_feedback_policy_checkpoint_v1':
            raise ValueError('Unsupported frozen policy checkpoint')
        if cp.get('software_test_only') is not (not _production):
            raise ValueError('Software-only checkpoints cannot become production policies')
        if cp.get('pending_feedback_flushed') is not False:
            raise ValueError('Final evaluation must not flush pending feedback')
        self.seed=_int(cp['seed']);self.evaluation_seed=_int(evaluation_seed)
        self.strategy=str(cp['strategy']);self.serving_policy=cp['serving_policy']
        self.next_tick=_int(cp['next_tick'],1);tick=_int(cp['frozen_as_of_tick'])
        if tick!=self.next_tick-1:raise ValueError('Checkpoint must freeze at the last completed decision')
        fm=cp.get('feature_map',{});self.mapper=SubsetFeatureMap(6,3)
        self.phi_dimension=_int(fm.get('phi_dimension'),1)
        if (fm.get('n_models')!=6 or fm.get('k')!=3 or fm.get('basis_dim')!=22
                or fm.get('combinations')!=[list(v) for v in self.mapper.subsets]):
            raise ValueError('Frozen six-model K3 feature map changed')
        if _production and (self.phi_dimension!=17 or self.next_tick!=600 or self.seed not in SEEDS):
            raise ValueError('Production final policies require original 600/17-dimensional state')
        for name in ('policy','warmup','bootstrap'):
            state=cp.get('rng_states',{}).get(name)
            if not isinstance(state,dict) or state.get('bit_generator')!='PCG64':
                raise ValueError('Missing or changed frozen training RNG identity')
            try:np.random.PCG64().state=state
            except (ValueError,TypeError,KeyError) as exc:raise ValueError('Invalid frozen RNG state') from exc
        self.training_question_ids=set(cp.get('stream_question_order',[]))
        if len(self.training_question_ids)!=self.next_tick:
            raise ValueError('Checkpoint stream order must be complete and unique')
        self._theta,self._inverse,arrived=_ridge_state(cp['shared_ridge'],self.phi_dimension*22,tick)
        self.training_question_ids.update(arrived)
        for name in ('feedback','warmup_feedback'):
            broker=cp.get(name,{})
            if broker.get('pending_values_serialized') is not False:raise ValueError('Pending hidden feedback values forbidden')
            pending=broker.get('pending_paid_requests_without_values')
            if not isinstance(pending,list):raise ValueError('Missing pending feedback metadata boundary')
            count=0
            for request in pending:
                if set(request)!=PENDING_KEYS or _int(request['available_at'])<self.next_tick:
                    raise ValueError('Pending feedback contains values or an already arrived request')
                indices=request['combination_indices']
                if (not isinstance(indices,list) or not indices or len(set(indices))!=len(indices)
                        or any(_int(i)>=20 for i in indices) or len(request['label_ids'])!=len(indices)):
                    raise ValueError('Invalid pending combination request metadata')
                self.training_question_ids.add(request['question_id']);count+=len(indices)
            if broker.get('pending_paid_labels')!=count:raise ValueError('Pending paid-label count mismatch')
        self._member_thetas=None;self._fixed_index=None
        if self.serving_policy=='independent_topk':
            member=cp.get('member_predictor',{})
            if len(member.get('models',[]))!=6:raise ValueError('Six frozen member regressors required')
            states=[_ridge_state(v,self.phi_dimension,tick) for v in member['models']]
            self._member_thetas=np.stack([v[0] for v in states])
            for state in states:self.training_question_ids.update(state[2])
        elif self.serving_policy=='fixed':
            self._fixed_index=_int(cp.get('fixed_selection',{}).get('selected_index'))
            if self._fixed_index>=20:raise ValueError('Invalid frozen fixed combination')
        elif self.serving_policy not in ('greedy','linucb','thompson'):
            raise ValueError('Unsupported frozen service policy')
        self._alpha=float(cp.get('linucb_alpha',1.))
        if not np.isfinite(self._alpha) or self._alpha<0:raise ValueError('Invalid frozen LinUCB alpha')
        self._thompson_factor=None
        if self.serving_policy=='thompson':
            eigval,eigvec=np.linalg.eigh((self._inverse+self._inverse.T)/2)
            self._thompson_factor=eigvec*np.sqrt(np.maximum(0.,eigval))[None,:]
        self.checkpoint_state_sha256=checkpoint_digest
        self.arrived_original_label_count=len(cp['shared_ridge']['arrived_original_labels'])
        self._metadata=_compact_checkpoint_metadata(cp)
        if self.serving_policy!='linucb':self._inverse=None
        del self._checkpoint

    @property
    def checkpoint_metadata(self):return deepcopy(self._metadata)

    @classmethod
    def for_software_test(cls,checkpoint,*,evaluation_seed=20261031):
        return cls(checkpoint,evaluation_seed=evaluation_seed)

    @classmethod
    def from_bound_checkpoint(cls,path,*,expected_sha256,expected_provenance,
            expected_seed,expected_strategy,algorithm_sources,evaluation_seed=20261031):
        path=Path(path)
        if not _digest(expected_sha256) or sha_file(path)!=expected_sha256:raise ValueError('Frozen checkpoint SHA mismatch')
        if not set(SOURCE_FILES).issubset(algorithm_sources):raise ValueError('Complete bound training source identities required')
        for name,ref in algorithm_sources.items():
            if verified_reference(ref)!=(PROJECT/name).resolve():raise ValueError('Training source path differs from current project')
        cp=read_json(path)
        if cp.get('seed')!=expected_seed or cp.get('strategy')!=expected_strategy:
            raise ValueError('Frozen checkpoint method/seed mismatch')
        for split in ('online','calibration'):
            expected=expected_provenance[split];observed=cp.get(split+'_provenance',{})
            if not all(k in expected for k in PROVENANCE_KEYS):raise ValueError('Full frozen training provenance required')
            if any(observed.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen training provenance mismatch')
        return cls(cp,_production=True,evaluation_seed=evaluation_seed)

    def select(self,phi,*,question_id):
        if not isinstance(question_id,str) or not question_id:raise ValueError('A unique final question ID required')
        if question_id in self.training_question_ids or anonymous_question_id(question_id) in self.training_question_ids:
            raise ValueError('Final question leaks from paid training history')
        phi=_array(phi,(self.phi_dimension,))
        if phi[0]!=1.:raise ValueError('Frozen phi must retain leading constant')
        if self._fixed_index is not None:
            main=self._fixed_index;means=None;values=None
        elif self._member_thetas is not None:
            member_scores=self._member_thetas@phi
            selected=tuple(sorted(np.argsort(-member_scores,kind='stable')[:3].tolist()))
            main=self.mapper.subsets.index(selected)
            means=np.asarray([np.mean(member_scores[list(v)]) for v in self.mapper.subsets]);values=means
        else:
            features=self.mapper.transform(phi);means=features@self._theta;values=means
            if self.serving_policy=='linucb':
                variance=np.maximum(0.,np.einsum('...i,ij,...j->...',features,self._inverse,features))
                values=means+self._alpha*np.sqrt(variance)
            elif self.serving_policy=='thompson':
                entropy=hashlib.sha256((str(self.evaluation_seed)+':'+str(self.seed)+':'+question_id).encode()).digest()
                rng=np.random.default_rng(np.frombuffer(entropy,dtype='<u4').tolist())
                theta=self._theta+self._thompson_factor@rng.normal(size=len(self._theta))
                values=features@theta
            main=int(np.argmax(values))
        return {'main_index':int(main),'main_members':list(self.mapper.subsets[main]),
            'predicted_scores':None if means is None else means.tolist(),
            'selection_values':None if values is None else values.tolist(),
            'prediction_semantics':'member_mean_ranking_proxy' if self._member_thetas is not None else 'fixed_paid_warmup_selection' if self._fixed_index is not None else 'frozen_aggregate_quality_ridge',
            'parameters_updated':False,'pending_feedback_consumed':False,
            'evaluation_randomness':'stateless_sha256_question_seed_independent_of_training_rng'}


def _resolve_references(release):
    required={'config','split_manifest','pca_artifact','pool_config','features','feature_manifest'}
    if set(release.get('artifacts',{}))!=required:raise ValueError('FINAL_RELEASE exact artifact references required')
    return {name:verified_reference(record) for name,record in release['artifacts'].items()}


def validate_encoder_feature_sources(fm,feature_spec):
    encoder=read_json(verified_reference(fm.get('encoder_source_receipt')))
    if (encoder.get('role')!='encoder' or encoder.get('repo_id')!=feature_spec.get('encoder_id')
            or encoder.get('revision')!=feature_spec.get('encoder_revision') or not encoder.get('files')):
        raise ValueError('Final encoder receipt does not bind the actual frozen encoder')
    model_root=Path(encoder.get('path',''))
    if not model_root.is_absolute():model_root=PROJECT/model_root
    model_root=model_root.resolve()
    if not encoder.get('path') or not model_root.is_dir():raise ValueError('Frozen encoder model directory missing')
    encoder_paths=set();has_weights=False
    for item in encoder['files']:
        record=dict(item);source=Path(record['path'])
        if not source.is_absolute():source=PROJECT/source
        record['path']=str(source.resolve());source=verified_reference(record)
        if not source.is_relative_to(model_root):raise ValueError('Encoder load directory differs from hashed source files')
        if source in encoder_paths:raise ValueError('Duplicate encoder source receipt file')
        encoder_paths.add(source);has_weights=has_weights or source.name in ('model.safetensors','pytorch_model.bin')
        if item.get('lfs_sha256') is not None and item['lfs_sha256']!=record['sha256']:
            raise ValueError('Encoder source SHA differs from its pinned LFS identity')
    if not has_weights or not {model_root/name for name in ('model.safetensors','config.json','tokenizer.json')}.issubset(encoder_paths):
        raise ValueError('Frozen encoder load-root weights/config/tokenizer provenance missing')
    generation_source=verified_reference(fm.get('feature_generation_source'))
    if generation_source.suffix!='.py':raise ValueError('Final feature generation code identity missing')
    return encoder


def _validate_policy_groups(release,*,final_ids,sources,final_sources,retain_policies=True,authorization_context=None,batch_authorizations=None):
    policies=[];validated_count=0;groups=release.get('policy_groups',[])
    context=authorization_context or release
    batches={} if batch_authorizations is None else batch_authorizations
    if not groups or len({g.get('method') for g in groups})!=len(groups):
        raise ValueError('Uniquely named predeclared final method groups required')
    for group in groups:
        group_strategy=group.get('strategy',group['method'])
        group_rate=_int(group.get('label_rate'),1);group_delay=_int(group.get('feedback_delay'))
        if group_rate not in (10,25,50,100) or group_delay not in (0,5,20):
            raise ValueError('Predeclared final method group budget/delay differs from protocol')
        entries=group.get('checkpoints',[])
        if sorted(e.get('seed',-1) for e in entries)!=list(SEEDS):
            raise ValueError('Each final method requires exactly five frozen seeds')
        for entry in entries:
            cp_path=verified_reference(entry['checkpoint']);receipt_path=verified_reference(entry['episode_receipt'])
            grid_path=verified_reference(entry['grid_plan']);identity=read_json(grid_path)
            neighbor=identity.get('schema')=='neighbor_grid_identity_v1'
            e5=bool(identity.get('e5_condition'));flex=identity.get('schema')=='flex_grid_identity_v1'
            if flex:
                required_flex={'src/budget_router/flex_training.py','src/budget_router/flex_simulation.py','src/budget_router/flex_validation.py'}
                if not required_flex.issubset(final_sources):raise ValueError('Flex final restoration source closure missing')
                from .flex_validation import validate_flex_episode,authorize_flex_final_batch
                if 'flex' not in batches:
                    batches['flex']=authorize_flex_final_batch(verified_reference(context['e4_full_gate']),context['inputs'])
                result=validate_flex_episode(entry['checkpoint'],entry['episode_receipt'],entry['grid_plan'],batch_authorization=batches['flex'])
            elif e5:
                if 'src/budget_router/e5_validation.py' not in final_sources:
                    raise ValueError('E5 final checkpoint validator source identity missing')
                from .e5_validation import validate_e5_episode,authorize_e5_final_batch
                if 'e5' not in batches:
                    batches['e5']=authorize_e5_final_batch(verified_reference(context['e5_pre_final_gate']),context['inputs'])
                result=validate_e5_episode(entry['checkpoint'],entry['episode_receipt'],entry['grid_plan'],batch_authorization=batches['e5'])
            elif neighbor:
                if not set(NEIGHBOR_FINAL_SOURCES).issubset(final_sources):
                    raise ValueError('Neighbor final restoration/validator source closure missing')
                from .neighbor_validation import validate_neighbor_episode,authorize_neighbor_final_batch
                cache=batches.setdefault('neighbor',{});key=entry['grid_plan']['sha256']
                if key not in cache:cache[key]=(authorize_neighbor_final_batch(entry['grid_plan']),entry['grid_plan'])
                result=validate_neighbor_episode(entry['checkpoint'],entry['episode_receipt'],entry['grid_plan'],batch_authorization=cache[key][0])
            else:
                validate_run_identity(identity)
                identity_sources=identity['inputs']['algorithm_sources']
                if {k:v['sha256'] for k,v in identity_sources.items()}!={k:v['sha256'] for k,v in sources.items()}:
                    raise ValueError('Checkpoint engine differs from frozen source identity')
                result=None
            receipt=read_json(receipt_path);spec=receipt.get('spec',{})
            if (spec.get('seed')!=entry['seed'] or spec.get('strategy')!=group_strategy
                    or spec.get('label_rate')!=group_rate or spec.get('feedback_delay')!=group_delay):
                raise ValueError('Final group/episode method/budget/delay identity differs')
            if receipt_path.parent/'policy_checkpoint.json'!=cp_path or receipt_path.parent.parent/'grid_plan.json'!=grid_path:
                raise ValueError('Checkpoint and training receipts must bind the same episode')
            if not neighbor and not e5 and not flex:result=EpisodeStore(grid_path.parent,identity,resume=True).load(spec)
            if result is None:raise ValueError('Frozen policy training episode incomplete')
            cp=result['policy_checkpoint']
            for key in ('warmup_mode','stream_order'):
                if key in group and cp.get('settings',{}).get(key)!=group[key]:
                    raise ValueError('Frozen policy settings differ from declared group: '+key)
            for name in ('config','split_manifest','pca_artifact','pool_config'):
                if neighbor and result.get('final_protocol_bindings'):
                    actual=result['final_protocol_bindings'].get(name+'_sha256')
                else:
                    ref=identity['inputs']['assets']['config'] if name=='config' else identity['inputs']['locked_provenance'][name]
                    actual=ref['sha256']
                if actual!=release['artifacts'][name]['sha256']:
                    raise ValueError('Policy trained under different final protocol: '+name)
            expected_provenance={}
            for split_name in ('online','calibration'):
                inputs=identity['inputs'];locked=inputs['locked_provenance'];assets=inputs['assets']
                original_manifest=read_json(verified_reference(assets[split_name+'_manifest']))
                expected_provenance[split_name]={'matrix_sha256':assets[split_name+'_matrix']['sha256'],
                    'e0_lock_sha256':locked['e0_lock']['sha256'],'e1_gate_sha256':locked['e1_gate']['sha256'],
                    'pca_sha256':locked['pca_artifact']['sha256'],
                    **{key:original_manifest[key] for key in ('model_pool','cache_revision','aggregator','scorer')}}
            policy_class=FrozenFlexPolicy if flex else FrozenNeighborPolicy if neighbor else FrozenPolicy
            policy=policy_class.from_bound_checkpoint(cp_path,expected_sha256=entry['checkpoint']['sha256'],
                expected_provenance=expected_provenance,expected_seed=entry['seed'],expected_strategy=group_strategy,
                algorithm_sources=sources,evaluation_seed=_int(release.get('evaluation_seed')))
            if {anonymous_question_id(q) for q in final_ids}&policy.training_question_ids:
                raise ValueError('Sealed final question entered frozen training history')
            validated_count+=1
            if retain_policies:policies.append((group['method'],entry,policy))
    _finish_final_batches(batches)
    return policies if retain_policies else validated_count


def _finish_final_batches(batches):
    for name in ('e5','flex'):
        if name in batches:batches[name].verify_unchanged()
    if batches.get('neighbor'):
        from .neighbor_validation import verify_identity
        for _,ref in batches['neighbor'].values():verify_identity(read_json(verified_reference(ref)),parent=False)


def validate_final_release(path):
    """Validate immutable full-scale release without opening final score data.

    E5_PRE_FINAL covers pre-holdout ablations. Final independent-cache receipts
    follow this release, avoiding a circular requirement for seen test labels.
    """
    path=Path(path).resolve();release=read_json(path)
    core={k:v for k,v in release.items() if k!='release_sha256'}
    if (release.get('schema')!='frozen_final_release_v1' or release.get('status')!='FINAL_RELEASED'
            or release.get('software_test_only') is not False
            or release.get('release_sha256')!=canonical_digest(core)):
        raise ValueError('Invalid immutable FINAL_RELEASE identity')
    if (release.get('pending_feedback_policy')!='never_flush_or_update'
            or release.get('final_label_use')!='evaluation_only_after_decisions_frozen'):
        raise ValueError('FINAL_RELEASE feedback boundary missing')
    paths=_resolve_references(release);config=read_json(paths['config']);split=read_json(paths['split_manifest'])
    if (config.get('n_models')!=6 or config.get('k')!=3
            or config.get('split',{}).get('final_holdout_total')!=200
            or config.get('split',{}).get('online_total')!=600
            or config.get('statistics',{}).get('algorithm_stream_seeds')!=list(SEEDS)):
        raise ValueError('FINAL_RELEASE original scale and all five stream seeds required')
    records=split.get('records',[]);canonical=split.get('canonical_question_fingerprints',{})
    all_ids=[r['question_id'] for r in records]
    if (len(set(all_ids))!=len(all_ids) or set(all_ids)!=set(canonical)
            or not all(_digest(v) for v in canonical.values()) or len(set(canonical.values()))!=len(all_ids)):
        raise ValueError('Global final/train question isolation failed')
    final=[r for r in records if r['split']=='sealed_test']
    if len(final)!=200 or Counter(r['task'] for r in final)!={'gsm8k':100,'bbh':100}:
        raise ValueError('Exact full-scale balanced final 200 membership required')
    expected=[{'question_id':r['question_id'],'task':r['task'],'canonical_fingerprint':canonical[r['question_id']]} for r in final]
    if release.get('final_questions')!=expected:raise ValueError('Released final order/membership differs from locked split')
    sources=release.get('algorithm_sources',{})
    if not set(SOURCE_FILES).issubset(sources):raise ValueError('Incomplete frozen training source inventory')
    for name,ref in sources.items():
        if verified_reference(ref)!=(PROJECT/name).resolve():raise ValueError('Frozen algorithm source path differs from current project')
    final_sources=release.get('final_sources',{})
    allowed_final_sources=set(FINAL_SOURCES)|set(NEIGHBOR_FINAL_SOURCES)|{'src/budget_router/e5_validation.py',
        'scripts/prepare_final_features.py','src/budget_router/encoding.py',
        'src/budget_router/flex_validation.py','src/budget_router/flex_training.py','src/budget_router/flex_simulation.py',
        'scripts/run_flex_reference.py','scripts/build_final_release.py','scripts/prepare_second_aggregator.py','scripts/download_second_aggregator.py'}|set(FINAL_GENERATION_SOURCES)
    if not set(FINAL_SOURCES).issubset(final_sources) or set(final_sources)-allowed_final_sources:
        raise ValueError('Final evaluator source identity missing or unknown')
    for name,ref in final_sources.items():
        p=verified_reference(ref)
        if p!=(PROJECT/name).resolve():raise ValueError('Final source path differs from current project')
        if name==FINAL_SOURCES[0] and p.resolve()!=Path(__file__).resolve():
            raise ValueError('Release points to a different final evaluator implementation')
    bindings={name+'_sha256':release['artifacts'][name]['sha256'] for name in ('config','split_manifest','pca_artifact','pool_config')}
    bindings['algorithm_sources_sha256']=canonical_digest({k:v['sha256'] for k,v in sources.items()})
    stages=release.get('stage_evidence',{})
    if set(stages)!={'E4','E5_PRE_FINAL'}:raise ValueError('E4 and pre-final E5 technical evidence required')
    for stage,ref in stages.items():
        evidence=read_json(verified_reference(ref))
        actual_schema='e4_full_technical_gate_v1' if stage=='E4' else 'e5_ablations_pre_final_gate_v1'
        if evidence.get('schema')!=actual_schema and (evidence.get('stage')!=stage or evidence.get('technical_complete') is not True
                or evidence.get('bindings')!=bindings or evidence.get('software_test_only') is not False):
            raise ValueError('Incomplete or differently bound pre-final evidence: '+stage)
    fm=read_json(paths['feature_manifest']);feature_spec=config.get('features',{})
    if (fm.get('schema')!='sealed_features_only_v1' or fm.get('split')!='sealed_test' or fm.get('no_labels') is not True
            or fm.get('features_sha256')!=release['artifacts']['features']['sha256']
            or fm.get('pca_sha256')!=release['artifacts']['pca_artifact']['sha256']
            or any(fm.get(k)!=feature_spec.get(k) for k in ('encoder_id','encoder_revision','pooling'))):
        raise ValueError('Final feature/PCA/encoder binding mismatch')
    validate_encoder_feature_sources(fm,feature_spec)
    with np.load(paths['features'],allow_pickle=False) as data:
        if set(data.files)!={'question_ids','task_labels','phi','embeddings'}:
            raise ValueError('Final features may not contain labels/answers')
        features={k:data[k] for k in data.files}
    ids=features['question_ids'];tasks=features['task_labels']
    if (ids.dtype.kind not in 'US' or tasks.dtype.kind not in 'US'
            or ids.tolist()!=[r['question_id'] for r in final] or tasks.tolist()!=[r['task'] for r in final]):
        raise ValueError('Final feature order differs from released membership')
    phi=_array(features['phi'],(200,17));embeddings=np.asarray(features['embeddings'],dtype=float)
    if (embeddings.ndim!=2 or embeddings.shape[0]!=200 or not np.isfinite(embeddings).all()
            or not np.allclose(np.linalg.norm(embeddings,axis=1),1.,rtol=1e-5,atol=1e-5)):
        raise ValueError('Final frozen encoder embeddings invalid')
    with np.load(paths['pca_artifact'],allow_pickle=False) as data:
        components=_array(data['components'],(16,embeddings.shape[1]));mean=_array(data['mean'],(embeddings.shape[1],))
        fit=data['fit_question_ids'].tolist()
    cal_ids={r['question_id'] for r in records if r['split']=='online_calibration'}
    if len(fit)!=200 or len(set(fit))!=200 or set(fit)!=cal_ids:
        raise ValueError('Final projection must reuse exact original calibration PCA')
    if not np.allclose(phi,np.column_stack((np.ones(200),(embeddings-mean)@components.T)),rtol=1e-6,atol=1e-6):
        raise ValueError('Final phi differs from unchanged train-only PCA')
    if not release.get('policy_groups'):raise ValueError('Uniquely named predeclared final method groups required')
    preparation_ref=fm.get('preparation_release');preparation_path=verified_reference(preparation_ref)
    prepared=validate_preparation_release(preparation_path,_validate_policies=False)
    pre=prepared['release']
    if fm.get('preparation_release_sha256')!=pre['preparation_release_sha256']:
        raise ValueError('Final features bind a different pre-final preparation release')
    for name in ('config','split_manifest','pca_artifact','pool_config'):
        if release['artifacts'][name]['sha256']!=pre['artifacts'][name]['sha256']:
            raise ValueError('Final release changed the pre-encoding protocol: '+name)
    if (release.get('evaluation_conditions')!=pre.get('evaluation_conditions')
            or release.get('robustness_plan')!=pre.get('robustness_plan')
            or release.get('aggregator_configs')!=pre.get('aggregator_configs')):
        raise ValueError('Final generation/aggregator conditions changed after feature preparation')
    if release.get('calibration_encoder_replay_sources')!=pre.get('calibration_encoder_replay_sources'):
        raise ValueError('Original calibration replay sources changed after feature preparation')
    validate_calibration_replay_record(fm.get('calibration_encoder_replay'),pre,prepared['calibration_question_ids'])
    if canonical_digest(release['policy_groups'])!=canonical_digest(pre['policy_groups']):
        raise ValueError('Final policy checkpoints changed after feature preparation was authorized')
    _validate_declared_group_plan(release,read_json(prepared['paths']['final_policy_plan']))
    for stage,key in (('E4','e4_full_gate'),('E5_PRE_FINAL','e5_pre_final_gate')):
        if release['stage_evidence'][stage]!=pre[key]:raise ValueError('Final release stage gates differ from feature preparation')
    for name,ref in pre['final_sources'].items():
        if final_sources.get(name)!=ref:raise ValueError('Final source changed after question-only preparation')
    dependencies=fm.get('feature_dependency_sources',{})
    if set(dependencies)!={'src/budget_router/encoding.py'}:
        raise ValueError('Final encoder dependency source identity missing')
    if (dependencies['src/budget_router/encoding.py']!=pre['final_sources']['src/budget_router/encoding.py']
            or fm['feature_generation_source']!=pre['final_sources']['scripts/prepare_final_features.py']):
        raise ValueError('Final feature code differs from authorized preparation implementation')
    policies=_validate_policy_groups(release,final_ids=ids.tolist(),sources=sources,final_sources=final_sources,
        authorization_context=pre,batch_authorizations=prepared['batch_authorizations'])
    conditions=release.get('evaluation_conditions')
    if (not isinstance(conditions,list) or not conditions
            or len({c.get('condition_id') for c in conditions})!=len(conditions)):
        raise ValueError('Uniquely named predeclared final generation/aggregator conditions required')
    base=policies[0][2]._metadata['online_provenance']['aggregator'];has_primary=False
    for condition in conditions:
        agg=condition.get('aggregator',{})
        if (not isinstance(condition.get('condition_id'),str) or not condition['condition_id']
                or condition.get('frozen_policies_only') is not True or condition.get('retuning_allowed') is not False
                or not agg.get('model_id') or not agg.get('revision')
                or not _digest(agg.get('prompt_sha256')) or not _digest(agg.get('fingerprint'))):
            raise ValueError('Invalid or retunable frozen final condition')
        generation_seed=_int(condition.get('generation_seed'))
        has_primary=has_primary or (agg==base and generation_seed==42)
    if not has_primary:raise ValueError('Original aggregator/cache42 primary final condition required')
    return {'release':release,'path':path,'paths':paths,'features':features,'policies':policies,'bindings':bindings}


def _decision_record(method,entry,policy,features,index,qid):
    return {'method':method,'seed':entry['seed'],'question_id':qid,'task':str(features['task_labels'][index]),
        'checkpoint_sha256':entry['checkpoint']['sha256'],'phi_sha256':canonical_digest(features['phi'][index].tolist()),
        **policy.select(features['phi'][index],question_id=qid)}


def freeze_final_decisions(release_path,output_directory):
    """Create-only decision bundle; this function has no matrix/score argument."""
    validated=validate_final_release(release_path);release=validated['release'];features=validated['features']
    output=Path(output_directory).resolve()
    if output.exists():raise FileExistsError('Frozen final decisions cannot be overwritten')
    temporary=output.with_name('.'+output.name+'.'+uuid.uuid4().hex+'.pending');temporary.mkdir(parents=True)
    count=0;decisions_path=temporary/'decisions.jsonl'
    with decisions_path.open('x',encoding='utf-8',newline='\n') as stream:
        for method,entry,policy in validated['policies']:
            for i,qid in enumerate(features['question_ids'].tolist()):
                record=_decision_record(method,entry,policy,features,i,qid)
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');count+=1
            if hasattr(policy,'assert_unchanged'):policy.assert_unchanged()
        stream.flush();os.fsync(stream.fileno())
    receipt={'schema':'frozen_final_decisions_v1','status':'DECISIONS_FROZEN_NO_FINAL_LABELS_READ',
        'release':artifact_reference(release_path),'release_sha256':release['release_sha256'],
        'decisions_sha256':sha_file(decisions_path),'records':count,'pending_feedback_consumed':False,
        'parameter_updates':0,'evaluation_seed':release['evaluation_seed'],'software_test_only':False}
    with (temporary/'decision_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.rename(temporary,output)
    return receipt


def _read_frozen_decisions(directory,validated):
    directory=Path(directory);receipt=read_json(directory/'decision_receipt.json')
    if (receipt.get('schema')!='frozen_final_decisions_v1'
            or receipt.get('status')!='DECISIONS_FROZEN_NO_FINAL_LABELS_READ'
            or receipt.get('release_sha256')!=validated['release']['release_sha256']
            or receipt.get('software_test_only') is not False
            or receipt.get('pending_feedback_consumed') is not False or receipt.get('parameter_updates')!=0
            or sha_file(directory/'decisions.jsonl')!=receipt.get('decisions_sha256')):
        raise ValueError('Missing or changed immutable final decisions')
    if verified_reference(receipt['release'])!=validated['path']:
        raise ValueError('Final decisions bind a different release path')
    lines=(directory/'decisions.jsonl').read_text(encoding='utf-8').split('\n')
    def bad_constant(value):raise ValueError('Nonfinite decision JSON')
    records=[json.loads(line,parse_constant=bad_constant) for line in lines if line]
    expected=((method,entry,policy,i,qid) for method,entry,policy in validated['policies']
        for i,qid in enumerate(validated['features']['question_ids'].tolist()))
    expected_count=len(validated['policies'])*len(validated['features']['question_ids'])
    if len(records)!=expected_count or receipt.get('records')!=expected_count:
        raise ValueError('Final decision bundle incomplete')
    for record,(method,entry,policy,i,qid) in zip(records,expected):
        if record!=_decision_record(method,entry,policy,validated['features'],i,qid):
            raise ValueError('Frozen decisions differ from released no-update policy')
    for _,_,policy in validated['policies']:
        if hasattr(policy,'assert_unchanged'):policy.assert_unchanged()
    return records,receipt


def evaluate_final_matrix(release_path,decisions_directory,matrix_path,manifest_path):
    """Read real scores only after immutable decisions verify; never update policies."""
    validated=validate_final_release(release_path)
    decisions,decision_receipt=_read_frozen_decisions(decisions_directory,validated)
    manifest=read_json(manifest_path);release=validated['release']
    if (manifest.get('schema')!='sealed_real_matrix_v1' or manifest.get('split')!='sealed_test'
            or manifest.get('software_test_only') is not False or manifest.get('matrix_sha256')!=sha_file(matrix_path)
            or manifest.get('release_sha256')!=release['release_sha256']
            or manifest.get('frozen_decisions_sha256')!=decision_receipt['decisions_sha256']):
        raise ValueError('Final matrix not bound to released frozen decisions')
    for flag in ('all_combinations_executed','real_frozen_llm_aggregation','aggregate_scores_from_executed_aggregator',
            'failure_costs_included','no_unknown_costs','gold_excluded_from_model_inputs'):
        if manifest.get(flag) is not True:raise ValueError('Missing real execution/cost guarantee: '+flag)
    if manifest.get('combinations')!=[list(v) for v in SubsetFeatureMap(6,3).subsets]:
        raise ValueError('Final aggregation column order differs')
    for name in ('split_manifest','pca_artifact','pool_config'):
        record=manifest.get(name)
        if not isinstance(record,dict) or record.get('sha256')!=release['artifacts'][name]['sha256']:
            raise ValueError('Final matrix protocol provenance mismatch: '+name)
        verified_reference(record)
    sources=manifest.get('sources',[])
    if not {'member_outputs','aggregate_outputs','scorer_source'}.issubset({r.get('kind') for r in sources}):
        raise ValueError('Final real execution/scoring source evidence incomplete')
    for ref in sources:verified_reference(ref)
    reference=validated['policies'][0][2]._metadata['online_provenance']
    pool=read_json(validated['paths']['pool_config']).get('models',[])
    if (manifest.get('model_pool')!=reference['model_pool'] or len(pool)!=6
            or [(v.get('model_id'),v.get('revision')) for v in pool]
            !=[(v.get('model_id'),v.get('revision')) for v in manifest['model_pool']]):
        raise ValueError('Final matrix model identity/order differs from frozen pool')
    generation_seed=_int(manifest.get('generation_seed'));condition_id=manifest.get('condition_id')
    condition=next((c for c in release['evaluation_conditions'] if c['condition_id']==condition_id),None)
    if (condition is None or condition['generation_seed']!=generation_seed
            or condition['aggregator']!=manifest.get('aggregator') or manifest.get('scorer')!=reference['scorer']):
        raise ValueError('Final matrix differs from predeclared generation/aggregator condition')
    with np.load(matrix_path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    ids=arrays.get('question_ids',np.array([]));tasks=arrays.get('task_labels',np.array([]))
    if ids.tolist()!=validated['features']['question_ids'].tolist() or tasks.tolist()!=validated['features']['task_labels'].tolist():
        raise ValueError('Final matrix membership/order differs from sealed features')
    if manifest.get('question_fingerprints')!=[q['canonical_fingerprint'] for q in release['final_questions']]:
        raise ValueError('Final canonical question provenance differs')
    scores=_array(arrays['aggregate_scores'],(200,20));members=_array(arrays['member_scores'],(200,6))
    if np.any((scores<0)|(scores>1)) or np.any((members<0)|(members>1)):
        raise ValueError('Invalid final verified score range')
    caps=read_json(validated['paths']['config']).get('budget',{})
    for stem,width in (('member',6),('aggregate',20)):
        for field in ('prompt_tokens','completion_tokens','execution_counts'):
            value=arrays.get(stem+'_'+field)
            if value is None or value.shape!=(200,width) or value.dtype.kind not in 'iu' or (value<(1 if field=='execution_counts' else 0)).any():
                raise ValueError('Missing actual final billed executions/tokens')
        counts=arrays[stem+'_execution_counts']
        call_cap=caps.get('member_call_cap' if stem=='member' else 'aggregator_call_cap',1)
        prompt_cap=caps.get('input_cap_tokens',4096) if stem=='member' else caps.get('aggregator_input_cap_tokens',caps.get('input_cap_tokens',4096))
        completion_cap=caps.get('member_max_output_tokens',256) if stem=='member' else caps.get('aggregator_max_output_tokens',256)
        if (np.any(counts>call_cap) or np.any(arrays[stem+'_prompt_tokens']>counts*prompt_cap)
                or np.any(arrays[stem+'_completion_tokens']>counts*completion_cap)):
            raise ValueError('Final real execution exceeds locked call/token caps')
        wall=arrays.get(stem+'_wall_time_s')
        if wall is not None:_array(wall,(200,width))
        if wall is not None and (wall<0).any():raise ValueError('Negative final execution wall time')
    def cost(stem,row,index):
        return {'executions':int(arrays[stem+'_execution_counts'][row,index]),'labels':0,
            'tokens':int(arrays[stem+'_prompt_tokens'][row,index]+arrays[stem+'_completion_tokens'][row,index])}
    def add(total,other):
        for key in total:total[key]+=other[key]
    physical={'executions':0,'labels':0,'tokens':0}
    for stem,width in (('member',6),('aggregate',20)):
        for row in range(200):
            for index in range(width):add(physical,cost(stem,row,index))
    lookup={qid:i for i,qid in enumerate(ids.tolist())};groups={}
    trained_costs={(method,entry['seed']):policy._metadata['budget']['spent']
        for method,entry,policy in validated['policies']}
    for record in decisions:
        key=(record['method'],record['seed']);row=lookup[record['question_id']];index=record['main_index']
        item=groups.setdefault(key,{'rows':[],'logical_service_cost':{'executions':0,'labels':0,'tokens':0},'source_execution_wall_time_s':0.})
        selected=record['main_members'];coverage=float(np.max(members[row,selected]));quality=float(scores[row,index])
        item['rows'].append({'question_id':anonymous_question_id(record['question_id']),'task':record['task'],
            'quality':quality,'coverage':coverage,'oracle_regret':float(np.max(scores[row])-quality),
            'aggregation_miss_given_coverage':float(coverage>0 and quality==0)})
        for j in selected:add(item['logical_service_cost'],cost('member',row,j))
        add(item['logical_service_cost'],cost('aggregate',row,index))
        if 'member_wall_time_s' in arrays and 'aggregate_wall_time_s' in arrays:
            item['source_execution_wall_time_s']+=float(np.sum(arrays['member_wall_time_s'][row,selected])+arrays['aggregate_wall_time_s'][row,index])
        else:item['source_execution_wall_time_s']=None
    summaries=[]
    for (method,seed),item in groups.items():
        per_task={task:float(np.mean([r['quality'] for r in item['rows'] if r['task']==task])) for task in ('gsm8k','bbh')}
        summaries.append({'method':method,'seed':seed,'n_questions':200,
            'quality_micro':float(np.mean([r['quality'] for r in item['rows']])),
            'quality_task_macro':float(np.mean(list(per_task.values()))),'per_task_quality':per_task,
            'logical_service_cost':item['logical_service_cost'],'training_and_warmup_logical_cost':trained_costs[(method,seed)],
            'end_to_end_logical_cost':{key:trained_costs[(method,seed)][key]+item['logical_service_cost'][key] for key in ('executions','labels','tokens')},
            'source_execution_wall_time_s':item['source_execution_wall_time_s'],
            'candidate_coverage':float(np.mean([r['coverage'] for r in item['rows']])),
            'mean_oracle_regret':float(np.mean([r['oracle_regret'] for r in item['rows']])),
            'aggregation_miss_given_coverage_count':sum(int(r['aggregation_miss_given_coverage']) for r in item['rows']),
            'evaluation_question_rows':item['rows']})
    return {'schema':'frozen_final_matrix_evaluation_v1','status':'EVALUATED_FROZEN_FINAL_POLICIES',
        'research_complete':False,'release':artifact_reference(release_path),
        'decisions':artifact_reference(Path(decisions_directory)/'decision_receipt.json'),
        'matrix':artifact_reference(matrix_path),'matrix_manifest':artifact_reference(manifest_path),
        'condition_id':condition_id,'generation_seed':generation_seed,'generation_replicates':1,'stream_seeds_are_not_new_generations':True,
        'physical_full_cache_cost':physical,'physical_cost_includes_all_20_aggregations':True,
        'physical_cost_scope':'One condition only; shared member caches across conditions must be deduplicated by actual job_id/record SHA before global totals.',
        'member_output_source_sha256':[ref['sha256'] for ref in sources if ref['kind']=='member_outputs'],
        'physical_offline_grading_work':validate_physical_grading_work(manifest,arrays),
        'evaluation_feedback_cost':{'policy_feedback_grants':0,
            'scope':'Private verifier work is reported separately from LM executions/tokens and does not grant free online labels.'},
        'pending_feedback_consumed':False,'parameter_updates':0,'summaries':summaries,
        'limitations':['Conditional on one actual generation cache; independent generation replication is separate.',
            'Source wall time sums actual attempts, not GPU kernel seconds or full-service latency.',
            'End-to-end logical cost adds the bound training/warmup ledger to frozen final service; physical shared caches need separate global deduplication.']}


def paired_final_comparisons(evaluation,*,reference='none',replicates=2000,confidence=.95,seed=20261031):
    """Task-stratified paired question bootstrap, after averaging algorithm seeds.

    All seed policies see the same generated answers, so algorithm seeds are
    not treated as independent questions or independent generation replicates.
    No winner is selected here; every available method-reference pair is kept.
    """
    _int(replicates,2);_int(seed)
    if not np.isfinite(confidence) or not 0<confidence<1:raise ValueError('Invalid confidence level')
    if evaluation.get('schema')!='frozen_final_matrix_evaluation_v1':raise ValueError('Wrong final evaluation schema')
    methods={};tasks={}
    for summary in evaluation.get('summaries',[]):
        method=summary['method'];stream_seed=summary['seed']
        if stream_seed not in SEEDS or stream_seed in methods.setdefault(method,{}):
            raise ValueError('Duplicate or unknown final algorithm seed')
        rows=summary['evaluation_question_rows'];by_question={}
        for row in rows:
            qid=row['question_id'];task=row['task'];quality=float(row['quality'])
            if qid in by_question or task not in ('gsm8k','bbh') or not np.isfinite(quality) or not 0<=quality<=1:
                raise ValueError('Invalid paired final question row')
            if qid in tasks and tasks[qid]!=task:raise ValueError('Question task changed across policies')
            tasks[qid]=task;by_question[qid]=quality
        methods[method][stream_seed]=by_question
    if reference not in methods:raise ValueError('Predeclared final reference is absent')
    expected_ids=set(next(iter(methods[reference].values())))
    if len(expected_ids)!=200 or Counter(tasks[q] for q in expected_ids)!={'gsm8k':100,'bbh':100}:
        raise ValueError('Paired inference requires exact balanced final 200 support')
    averaged={}
    for method,runs in methods.items():
        if set(runs)!=set(SEEDS) or any(set(rows)!=expected_ids for rows in runs.values()):
            raise ValueError('Paired methods require the same 200 questions and all five seeds')
        averaged[method]={qid:float(np.mean([runs[s][qid] for s in SEEDS])) for qid in expected_ids}
    ordered={task:sorted(q for q in expected_ids if tasks[q]==task) for task in ('gsm8k','bbh')}
    rng=np.random.default_rng(seed);indices={task:rng.integers(0,len(ids),size=(replicates,len(ids))) for task,ids in ordered.items()}
    alpha=(1-confidence)/2;results=[]
    for method in sorted(set(methods)-{reference}):
        differences={task:np.array([averaged[method][q]-averaged[reference][q] for q in ids]) for task,ids in ordered.items()}
        distributions={task:values[indices[task]].mean(axis=1) for task,values in differences.items()}
        distribution=(distributions['gsm8k']+distributions['bbh'])/2
        delta=float(np.mean([v.mean() for v in differences.values()]))
        results.append({'method':method,'reference':reference,'n_questions':200,'algorithm_seeds':list(SEEDS),
            'generation_replicates':1,'quality_task_macro_difference':delta,
            'paired_question_ci':np.quantile(distribution,[alpha,1-alpha]).tolist(),
            'per_task':{task:{'n_questions':len(ordered[task]),'difference':float(differences[task].mean()),
                'paired_question_ci':np.quantile(distributions[task],[alpha,1-alpha]).tolist()} for task in ordered},
            'confidence_level':confidence,'bootstrap_replicates':replicates,
            'scope':'Descriptive paired evidence conditional on one pinned generation cache; no multiple-comparison adjustment.'})
    return {'schema':'paired_frozen_final_comparisons_v1','reference':reference,'comparisons':results,
        'statistical_unit':'question after averaging five algorithm-seed policies; stratified by task',
        'condition_id':evaluation.get('condition_id'),'generation_seed':evaluation.get('generation_seed'),'no_winner_selected':True}

class FrozenNeighborPolicy:
    """Frozen CES/C2 adapters; independent final RNG and a fixed training clock.

    This class restores policy state only. Production release integration also
    requires the separate neighbor episode validator, not an invented E2 receipt.
    """
    def __init__(self,checkpoint,*,_production=False,evaluation_seed=20261031):
        from .neighbor_policies import CESPaidWarmupPolicy,C2MABVPolicy
        cp=deepcopy(checkpoint);checkpoint_digest=canonical_digest(cp);self._checkpoint=cp
        if cp.get('schema')!='neighbor_restricted_feedback_policy_checkpoint_v1':
            raise ValueError('Unsupported frozen neighbor checkpoint')
        if cp.get('software_test_only') is not (not _production):
            raise ValueError('Neighbor software checkpoints cannot become production')
        if cp.get('pending_feedback_flushed') is not False:raise ValueError('Cannot flush neighbor pending feedback')
        self.software_test_only=not _production;self.seed=_int(cp['seed']);self.evaluation_seed=_int(evaluation_seed)
        self.next_tick=_int(cp['next_tick'],1);self.frozen_as_of_tick=_int(cp['frozen_as_of_tick'])
        if self.frozen_as_of_tick!=self.next_tick-1:raise ValueError('Neighbor freeze clock mismatch')
        self.strategy=cp['strategy'];self.mapper=SubsetFeatureMap(6,3)
        fm=cp.get('feature_map',{});self.phi_dimension=_int(fm.get('phi_dimension'),1)
        if fm.get('n_models')!=6 or fm.get('k')!=3:raise ValueError('Frozen neighbor requires N6/K3')
        if _production and (self.phi_dimension!=17 or self.next_tick!=600 or self.seed not in SEEDS):
            raise ValueError('Production neighbor final evaluation requires full original scale')
        for name,rng_state in cp.get('rng_states',{}).items():
            if not isinstance(rng_state,dict) or rng_state.get('bit_generator')!='PCG64':raise ValueError('Invalid neighbor RNG identity')
            try:np.random.PCG64().state=rng_state
            except (ValueError,KeyError,TypeError) as exc:raise ValueError('Invalid neighbor RNG state') from exc
        if 'policy' not in cp.get('rng_states',{}):raise ValueError('Frozen neighbor policy RNG missing')
        feedback=cp.get('feedback',{})
        if feedback.get('pending_values_serialized') is not False:raise ValueError('Pending neighbor values forbidden')
        for request in feedback.get('pending_paid_requests_without_values',[]):
            if set(request)!=PENDING_KEYS or _int(request['available_at'])<self.next_tick:
                raise ValueError('Pending neighbor feedback contains hidden or arrived values')
        state=cp['policy_state'];self.training_question_ids=set(cp.get('stream_question_order',[]))
        if len(self.training_question_ids)!=self.next_tick:raise ValueError('Neighbor trained question ledger incomplete')
        if self.strategy=='ces':
            if state.get('lam_cost')!=0.:raise ValueError('Frozen shared CES has predeclared lam_cost=0')
            self._policy=CESPaidWarmupPolicy(self.phi_dimension,alpha=state['alpha'],ridge=state['ridge'],
                projection_seed=state['projection_seed'],lam_cost=state['lam_cost'],cost_upper_bound=state['cost_upper_bound'])
            self._policy.load_state_dict(state)
            if self._policy.profile is None:raise ValueError('CES requires already frozen paid warmup profile')
            for row in state.get('arrived_member_keys',[]):self.training_question_ids.add(row[0])
            for row in state.get('arrived_combo_keys',[]):self.training_question_ids.add(row[0])
            for row in state.get('executed_bindings',[]):self.training_question_ids.add(row[0])
        elif self.strategy in ('c2mabv_awc','c2mabv_suc','c2mabv_aic'):
            if state.get('reward_model')!=self.strategy.rsplit('_',1)[1].upper():raise ValueError('C2 objective/checkpoint name mismatch')
            if _int(state.get('latest_available_at'))>self.frozen_as_of_tick:raise ValueError('C2 consumed unarrived future observation')
            self._policy=C2MABVPolicy(reward_model=state['reward_model'],budget_threshold=state['budget_threshold'],
                delta=state['delta'],alpha_mu=state['alpha_mu'],alpha_cost=state['alpha_cost'],cost_upper_bound=state['cost_upper_bound'])
            self._policy.load_state_dict(state)
            for row in state.get('arrived_member_keys',[]):self.training_question_ids.add(row[0])
        else:raise ValueError('Unsupported frozen neighbor strategy')
        self.checkpoint_state_sha256=checkpoint_digest
        self._policy_state_sha256=canonical_digest(self._policy.state_dict())
        self._metadata=_compact_checkpoint_metadata(cp);del self._checkpoint

    @property
    def checkpoint_metadata(self):return deepcopy(self._metadata)

    def assert_unchanged(self):
        if canonical_digest(self._policy.state_dict())!=self._policy_state_sha256:
            raise RuntimeError('Frozen neighbor unexpectedly changed learned policy state')

    @classmethod
    def for_software_test(cls,checkpoint,*,evaluation_seed=20261031):
        return cls(checkpoint,evaluation_seed=evaluation_seed)

    @classmethod
    def from_bound_checkpoint(cls,path,*,expected_sha256,expected_provenance,expected_seed,
            expected_strategy,algorithm_sources,evaluation_seed=20261031):
        path=Path(path)
        if not _digest(expected_sha256) or sha_file(path)!=expected_sha256:raise ValueError('Neighbor checkpoint SHA mismatch')
        for name,ref in algorithm_sources.items():
            if verified_reference(ref)!=(PROJECT/name).resolve():raise ValueError('Neighbor bound source path differs')
        cp=read_json(path)
        if cp.get('seed')!=expected_seed or cp.get('strategy')!=expected_strategy:raise ValueError('Neighbor method/seed mismatch')
        for split in ('online','calibration'):
            expected=expected_provenance[split]
            if not all(k in expected for k in PROVENANCE_KEYS):raise ValueError('Full neighbor training provenance missing')
            if any(cp.get(split+'_provenance',{}).get(k)!=v for k,v in expected.items()):
                raise ValueError('Neighbor training provenance mismatch')
        return cls(cp,_production=True,evaluation_seed=evaluation_seed)

    def select(self,phi,*,question_id):
        phi=_array(phi,(self.phi_dimension,))
        if not isinstance(question_id,str) or not question_id or phi[0]!=1.:raise ValueError('Invalid frozen neighbor question feature')
        if question_id in self.training_question_ids or anonymous_question_id(question_id) in self.training_question_ids:
            raise ValueError('Final neighbor question leaks from training history')
        if self.strategy=='ces':members=self._policy.select(phi)
        else:
            entropy=hashlib.sha256((str(self.evaluation_seed)+':'+str(self.seed)+':'+question_id).encode()).digest()
            rng=np.random.default_rng(np.frombuffer(entropy,dtype='<u4').tolist())
            members=self._policy.select(now=self.frozen_as_of_tick,rng=rng)['members']
        members=tuple(sorted(int(i) for i in members));index=self.mapper.subsets.index(members)
        return {'main_index':index,'main_members':list(members),'predicted_scores':None,'selection_values':None,
            'prediction_semantics':'native_set_ucb_or_structural_selection_proxy_not_aggregate_prediction',
            'parameters_updated':False,'pending_feedback_consumed':False,
            'evaluation_randomness':'stateless_sha256_question_seed_independent_of_training_rng',
            'fixed_training_clock':self.frozen_as_of_tick,'native_system':'NOT_RUN'}


def project_sealed_embeddings(embeddings,pca_path,*,calibration_question_ids):
    """Apply existing train-only PCA; never fit or mutate any projection state."""
    embeddings=np.asarray(embeddings,dtype=float)
    if embeddings.ndim!=2 or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings,axis=1),1.,rtol=1e-5,atol=1e-5):
        raise ValueError('Final embeddings must be finite frozen L2 vectors')
    with np.load(pca_path,allow_pickle=False) as data:
        if set(data.files)!={'components','mean','fit_question_ids','explained_variance'}:
            raise ValueError('Saved calibration PCA schema changed')
        components=_array(data['components'],(16,embeddings.shape[1]));mean=_array(data['mean'],(embeddings.shape[1],))
        fit=data['fit_question_ids'].tolist();variance=_array(data['explained_variance'],(16,))
    if (len(fit)!=200 or len(set(fit))!=200 or set(fit)!=set(calibration_question_ids)
            or len(set(calibration_question_ids))!=200 or np.any(variance<0)
            or not np.allclose(components@components.T,np.eye(16),rtol=1e-6,atol=1e-6)):
        raise ValueError('PCA must retain original exact 200-calibration fit and orthonormal components')
    return np.column_stack((np.ones(len(embeddings)),(embeddings-mean)@components.T))


def _validate_declared_group_plan(release,plan):
    if (plan.get('schema')!='frozen_final_policy_groups_plan_v1' or plan.get('seeds')!=list(SEEDS)
            or plan.get('final_questions')!=200):
        raise ValueError('Original predeclared final method/seed plan missing')
    expected=plan.get('expected_groups',[]);groups=release.get('policy_groups',[])
    if not expected or len({g.get('method') for g in expected})!=len(expected):
        raise ValueError('Predeclared final groups are empty or duplicated')
    actual={g.get('method'):g for g in groups}
    if len(actual)!=len(groups) or set(actual)!={g['method'] for g in expected}:
        raise ValueError('Released groups must match every predeclared final group exactly')
    for group in expected:
        if any(actual[group['method']].get(key)!=value for key,value in group.items()):
            raise ValueError('Released final group differs from predeclared method/budget/settings')


def validate_preparation_release(path,*,_validate_policies=True,_batch_authorizations=None):
    """Authorize question-only CPU final encoding after complete pre-final gates.

    No feature file or final score matrix is a dependency, so this gate cannot
    authorize itself via already observed test features or labels. Checkpoints
    are verified one at a time without retaining their full Python ledgers.
    """
    from .stage_validation import verify_input_identity
    path=Path(path).resolve();release=read_json(path)
    core={k:v for k,v in release.items() if k!='preparation_release_sha256'}
    if (release.get('schema')!='pre_final_feature_release_v1' or release.get('status')!='PRE_FINAL_FEATURES_RELEASED'
            or release.get('software_test_only') is not False
            or release.get('preparation_release_sha256')!=canonical_digest(core)):
        raise ValueError('Invalid immutable pre-final feature release')
    if release.get('final_label_use')!='evaluation_only_after_decisions_frozen':
        raise ValueError('Pre-final feature release cannot authorize label use')
    required={'config','split_manifest','pca_artifact','pool_config','encoder_source_receipt','final_policy_plan'}
    if set(release.get('artifacts',{}))!=required:raise ValueError('Pre-final feature artifacts are incomplete')
    paths={name:verified_reference(ref) for name,ref in release['artifacts'].items()}
    inputs=release.get('inputs',{});verify_input_identity(inputs)
    for name in ('config','split_manifest','pca_artifact','pool_config'):
        ref=inputs['assets']['config'] if name=='config' else inputs['locked_provenance'][name]
        if ref['sha256']!=release['artifacts'][name]['sha256']:
            raise ValueError('Feature release differs from already gated protocol: '+name)
    sources=release.get('algorithm_sources',{})
    if {k:v['sha256'] for k,v in sources.items()}!={k:v['sha256'] for k,v in inputs['algorithm_sources'].items()}:
        raise ValueError('Pre-final feature release changes frozen training source identity')
    for name,ref in sources.items():
        if verified_reference(ref)!=(PROJECT/name).resolve():raise ValueError('Feature release source path changed')
    final_sources=release.get('final_sources',{})
    required_sources=set(FINAL_SOURCES)|{'scripts/prepare_final_features.py','src/budget_router/encoding.py',
        'src/budget_router/e5_validation.py','src/budget_router/flex_validation.py','scripts/build_final_release.py'}
    if not required_sources.issubset(final_sources):raise ValueError('Pre-final feature source closure incomplete')
    for name,ref in final_sources.items():
        if verified_reference(ref)!=(PROJECT/name).resolve():raise ValueError('Pre-final feature implementation source path changed')
    e4_path=verified_reference(release.get('e4_full_gate'));e5_path=verified_reference(release.get('e5_pre_final_gate'))
    from .e5_validation import authorize_e5_final_batch,_E5FinalBatchAuthorization
    batches={} if _batch_authorizations is None else _batch_authorizations
    if 'e5' not in batches:batches['e5']=authorize_e5_final_batch(e5_path,inputs)
    if not isinstance(batches['e5'],_E5FinalBatchAuthorization):raise ValueError('An actual opaque E5 batch authorization is required')
    batches['e5'].verify_unchanged(full=False);e5_gate=batches['e5'].gate
    if e5_gate!=read_json(e5_path) or e5_gate.get('input_identity_sha256')!=canonical_digest(inputs):
        raise ValueError('E5 cached batch ticket belongs to a different feature release')
    if (verified_reference(e5_gate.get('e4_full_gate'))!=e4_path
            or e5_gate['e4_full_gate']['sha256']!=release['e4_full_gate']['sha256']):
        raise ValueError('Feature preparation must use the same complete E4 gate as E5')
    config=read_json(paths['config']);split=read_json(paths['split_manifest']);plan=read_json(paths['final_policy_plan'])
    if (config.get('n_models')!=6 or config.get('k')!=3 or config.get('split',{}).get('final_holdout_total')!=200
            or config.get('features',{}).get('pca_dimensions')!=16 or config.get('features',{}).get('frozen') is not True):
        raise ValueError('Final feature preparation must preserve frozen N6/K3 and PCA16')
    records=split.get('records',[]);canonical=split.get('canonical_question_fingerprints',{})
    ids=[r['question_id'] for r in records]
    if (len(set(ids))!=len(ids) or set(ids)!=set(canonical) or not all(_digest(v) for v in canonical.values())
            or len(set(canonical.values()))!=len(ids)):
        raise ValueError('Final feature release has global canonical question leakage')
    final=[r for r in records if r['split']=='sealed_test'];cal=[r['question_id'] for r in records if r['split']=='online_calibration']
    if len(final)!=200 or Counter(r['task'] for r in final)!={'gsm8k':100,'bbh':100} or len(cal)!=200:
        raise ValueError('Final feature release must contain exact original 200/200 balanced split')
    if sha_file(paths['final_policy_plan'])!=FINAL_GROUP_PLAN_SHA256 or len(plan.get('expected_groups',[]))!=216:
        raise ValueError('Final method plan differs from the original locked 216 groups')
    _validate_declared_group_plan(release,plan)
    robust_path=verified_reference(release.get('robustness_plan'))
    if release['robustness_plan']!=e5_gate['e5_plan']:
        raise ValueError('Final conditions differ from the actual E5 preregistration')
    conditions=release_conditions_from_plan(read_json(robust_path),release.get('aggregator_configs',{}))
    if release.get('evaluation_conditions')!=conditions:
        raise ValueError('Final conditions differ from the four predeclared generation/aggregator settings')
    original_online_manifest=read_json(verified_reference(inputs['assets']['online_manifest']))
    if conditions[0]['aggregator']!=original_online_manifest.get('aggregator'):
        raise ValueError('Primary final aggregator differs from the original learned online environment')
    replay_sources=release.get('calibration_encoder_replay_sources',{})
    if set(replay_sources)!={'frozen_features','feature_audit'}:raise ValueError('Original calibration replay source closure missing')
    for kind,ref in replay_sources.items():
        expected_refs=[r for r in original_online_manifest['sources'] if r.get('kind')==kind]
        if len(expected_refs)!=1 or ref['sha256']!=expected_refs[0]['sha256']:
            raise ValueError('Calibration replay source differs from original gated feature provenance')
        original=dict(expected_refs[0]);source=Path(original['path'])
        if not source.is_absolute():source=Path(inputs['assets']['online_manifest']['path']).parent/source
        original['path']=str(source.resolve())
        if verified_reference(ref)!=verified_reference(original):raise ValueError('Calibration replay source path differs')
    if _validate_policies:
        count=_validate_policy_groups(release,final_ids=[r['question_id'] for r in final],sources=sources,
            final_sources=final_sources,retain_policies=False,authorization_context=release,batch_authorizations=batches)
        if count!=len(plan['expected_groups'])*len(SEEDS):raise ValueError('Frozen final checkpoint inventory incomplete')
    question_refs=release.get('question_sources',[])
    if len(question_refs)!=2:raise ValueError('Both original frozen question sources are required')
    expected_sources={r['path']:r for r in split.get('source_files',[])};question_paths=[]
    for ref in question_refs:
        source=verified_reference(ref)
        if not source.is_relative_to(PROJECT):raise ValueError('Question source escaped experiment root')
        relative=source.relative_to(PROJECT).as_posix()
        if relative not in expected_sources or ref['sha256']!=expected_sources[relative]['sha256']:
            raise ValueError('Final question source differs from original frozen split')
        question_paths.append(source)
    if {p.name for p in question_paths}!={'available_records.jsonl','gsm8k_questions.jsonl'}:
        raise ValueError('Unexpected final public question source')
    encoder=validate_encoder_feature_sources({'encoder_source_receipt':release['artifacts']['encoder_source_receipt'],
        'feature_generation_source':final_sources['scripts/prepare_final_features.py']},config['features'])
    return {'release':release,'path':path,'paths':paths,'config':config,'split':split,'final_records':final,
        'calibration_question_ids':cal,'question_paths':question_paths,'encoder':encoder,'batch_authorizations':batches}


class FrozenFlexPolicy(FrozenNeighborPolicy):
    """Restore the named smoothed/PCA/fixed-K reference, with no final fitting."""
    def __init__(self,checkpoint,*,_production=False,evaluation_seed=20261031):
        from .flex_training import FrozenPCASmoothedFlexPolicy,FLEX_VARIANT
        cp=deepcopy(checkpoint);digest=canonical_digest(cp)
        if (cp.get('schema')!='flex_smoothed_frozen_policy_checkpoint_v1'
                or cp.get('strategy')!='flex_smoothed_adapted' or cp.get('method_variant')!=FLEX_VARIANT
                or cp.get('software_test_only') is not (not _production)
                or cp.get('online_updates') is not False or cp.get('native_system')!='NOT_RUN'
                or cp.get('pending_feedback_flushed') is not False):
            raise ValueError('Invalid or non-frozen explicit Flex reference checkpoint')
        self.software_test_only=not _production;self.strategy=cp['strategy'];self.seed=_int(cp['seed'])
        self.evaluation_seed=_int(evaluation_seed);self.next_tick=_int(cp['next_tick'],1)
        self.frozen_as_of_tick=_int(cp['frozen_as_of_tick']);fm=cp.get('feature_map',{})
        self.phi_dimension=_int(fm.get('phi_dimension'),3);self.mapper=SubsetFeatureMap(6,3)
        if (self.frozen_as_of_tick!=self.next_tick-1 or fm.get('n_models')!=6 or fm.get('k')!=3
                or fm.get('combinations')!=[list(v) for v in self.mapper.subsets] or fm.get('basis_dim')!=22):
            raise ValueError('Flex frozen scale, clock or feature map changed')
        if _production and (self.phi_dimension!=17 or self.next_tick!=600 or self.seed not in SEEDS):
            raise ValueError('Production Flex holdout requires exact original scale')
        for name in ('policy','warmup'):
            state=cp.get('rng_states',{}).get(name)
            if not isinstance(state,dict) or state.get('bit_generator')!='PCG64':raise ValueError('Missing Flex training RNG identity')
            try:np.random.PCG64().state=state
            except (ValueError,TypeError,KeyError) as exc:raise ValueError('Invalid Flex training RNG state') from exc
        for name in ('feedback','warmup_feedback'):
            broker=cp.get(name,{})
            if (broker.get('pending_values_serialized') is not False
                    or broker.get('pending_paid_labels')!=0 or broker.get('pending_paid_requests_without_values')!=[]):
                raise ValueError('Static Flex may not consume pending final/online feedback')
        self._policy=FrozenPCASmoothedFlexPolicy.from_state_dict(cp['policy_state'])
        if self._policy.feature_dim!=self.phi_dimension or self._policy.training_report.get('seed')!=self.seed:
            raise ValueError('Flex trained feature/seed identity mismatch')
        self.training_question_ids=set(cp.get('stream_question_order',[]))
        if len(self.training_question_ids)!=self.next_tick:raise ValueError('Flex complete training stream identity required')
        self.training_question_ids.update(self._policy.training_report.get('training_question_ids',[]))
        for item in cp.get('paid_unused_combo_labels',[]):
            if isinstance(item,dict):self.training_question_ids.add(item['question_id'])
        self.checkpoint_state_sha256=digest;self._policy_state_sha256=canonical_digest(self._policy.state_dict())
        self._metadata=_compact_checkpoint_metadata(cp)

    def select(self,phi,*,question_id):
        phi=_array(phi,(self.phi_dimension,))
        if not isinstance(question_id,str) or not question_id or phi[0]!=1.:raise ValueError('Invalid frozen Flex question feature')
        if question_id in self.training_question_ids or anonymous_question_id(question_id) in self.training_question_ids:
            raise ValueError('Final Flex question leaks from training history')
        members=tuple(sorted(int(i) for i in self._policy.select(phi)));index=self.mapper.subsets.index(members)
        return {'main_index':index,'main_members':list(members),'predicted_scores':None,'selection_values':None,
            'prediction_semantics':'smoothed_coverage_dpp_selection_not_final_aggregate_prediction',
            'parameters_updated':False,'pending_feedback_consumed':False,'fixed_training_clock':self.frozen_as_of_tick,
            'evaluation_randomness':'deterministic_frozen_policy_no_final_rng_consumption',
            'method_variant':self._policy.name,'native_system':'NOT_RUN'}


def release_conditions_from_plan(plan,aggregator_configs,*,root=PROJECT):
    """Resolve the four predeclared conditions using the actual pinned configs.

    Only configuration metadata is read; this never loads weights, question
    text, outputs, reference answers or any final score arrays.
    """
    from .inference import ModelSpec,configuration_fingerprint
    from .execution import SYSTEM_PROMPT,AGGREGATION_PROMPT_VERSION,TOKEN_POLICY_VERSION
    root=Path(root).resolve();requested=plan.get('final_frozen_policy_conditions',[])
    if (plan.get('schema')!='e5_robustness_preregistration_v1' or len(requested)!=4
            or len({c.get('condition_id') for c in requested})!=4
            or [c.get('generation_seed') for c in requested]!=[42,137,2024,42]
            or plan.get('final_condition_rules',{}).get('retuning_allowed') is not False
            or plan.get('final_condition_rules',{}).get('policy_update_from_final_labels_allowed') is not False):
        raise ValueError('The four original fixed-policy robustness conditions are required')
    paths={c.get('aggregator_config') for c in requested}
    if paths!={'configs/aggregator_v1.json','configs/aggregator_mistral_v03.json'} or set(aggregator_configs)!=paths:
        raise ValueError('Both exact frozen aggregator configurations are required')
    descriptors={};specs={}
    for relative,ref in aggregator_configs.items():
        path=verified_reference(ref)
        if path!=(root/relative).resolve():raise ValueError('Aggregator config path differs from preregistration')
        spec=ModelSpec(**read_json(path));spec.validate()
        if (spec.backend!='transformers' or spec.quantization!='nf4' or spec.dtype!='float16'
                or spec.max_input_tokens!=6144 or spec.max_new_tokens!=256 or spec.temperature!=0 or spec.top_p!=1.):
            raise ValueError('Frozen aggregator generation/quantization protocol changed')
        descriptors[relative]={'model_id':spec.model_id,'revision':spec.revision,'fingerprint':configuration_fingerprint(spec),
            'prompt_sha256':canonical_digest({'system':SYSTEM_PROMPT,'prompt_version':AGGREGATION_PROMPT_VERSION,'token_policy':TOKEN_POLICY_VERSION})}
        specs[relative]=spec
    if [c['aggregator_config'] for c in requested]!=['configs/aggregator_v1.json']*3+['configs/aggregator_mistral_v03.json']:
        raise ValueError('Generation replication must keep the original aggregator; second aggregator reuses base42')
    second=plan.get('second_aggregator',{});spec=specs['configs/aggregator_mistral_v03.json']
    if (spec.model_id,spec.revision)!=(second.get('model_id'),second.get('revision')):
        raise ValueError('Second aggregator identity differs from the independent locked choice')
    metadata=root/second.get('official_metadata','')
    if not metadata.is_file() or sha_file(metadata)!=second.get('official_metadata_sha256'):
        raise ValueError('Second aggregator official source identity evidence changed')
    return [{'condition_id':c['condition_id'],'generation_seed':c['generation_seed'],
        'aggregator':descriptors[c['aggregator_config']],'aggregator_config':dict(aggregator_configs[c['aggregator_config']]),
        'frozen_policies_only':True,'retuning_allowed':False} for c in requested]


def validate_physical_grading_work(manifest,arrays):
    """Validate recorded CPU grading attempts; missing work remains unknown.

    There is no conversion to model tokens or free policy feedback. Shared
    member generations and repeated CPU score attempts have distinct ledgers.
    """
    ref=manifest.get('actual_grading_work')
    if ref is None:
        return {'status':'MISSING_ACTUAL_GRADING_MEASUREMENT','local_verifier_calls':None,
            'process_time_s':None,'wall_time_s':None,'policy_feedback_grants':0,
            'scope':'Missing measurement is not zero; final research acceptance must require actual grading evidence.'}
    path=verified_reference(ref);work=read_json(path)
    if work.get('schema')!='actual_final_grading_work_v2' or work.get('scorer_version')!=manifest['scorer']['version']:
        raise ValueError('Actual final CPU grading identity/schema mismatch')
    selected=verified_reference(work['selected_complete_attempt'])
    if selected.parents[2].name!='.grading_attempts' or selected.parents[1].name!=manifest['condition_id']:
        raise ValueError('Actual grading attempt escaped its recorded condition')
    directory=selected.parents[3]
    if not directory.is_relative_to((PROJECT/'results/raw/final').resolve()):
        raise ValueError('Actual grading work escaped the final output subtree')
    from .final_generation import _grading_inventory,_score_digest
    expected=_grading_inventory(directory,manifest['condition_id'],manifest['grading_input_identity_sha256'],
        selected.parent,_score_digest(arrays))
    if work!=expected:raise ValueError('Actual grading CPU/call inventory differs from its immutable attempt records')
    return {'status':'RECORDED_WITH_UNOBSERVED_INTERRUPTED_WORK' if work['unfully_observed_attempts'] else 'ACTUAL_GRADING_WORK_VERIFIED',
        'artifact':ref,'completed_matrix_member_labels':1200,'completed_matrix_aggregate_labels':4000,
        'unique_reference_questions':200,'physical_member_label_evaluations':work['physical_member_label_evaluations'],
        'physical_aggregate_label_evaluations':work['physical_aggregate_label_evaluations'],
        'physical_member_label_evaluations_bounds':work['physical_member_label_evaluations_bounds'],
        'physical_aggregate_label_evaluations_bounds':work['physical_aggregate_label_evaluations_bounds'],
        'process_time_s':work['process_time_s'],'wall_time_s':work['wall_time_s'],
        'known_process_time_lower_bound_s':work['known_process_time_lower_bound_s'],
        'known_wall_time_lower_bound_s':work['known_wall_time_lower_bound_s'],
        'unfully_observed_attempts':work['unfully_observed_attempts'],'attempts':work['attempts'],
        'policy_feedback_grants':0,'model_judge_calls':0,
        'scope':'Actual private evaluator CPU work, including retries. Do not add to LM tokens or count as online label grants.'}


def validate_calibration_replay_record(reference,preparation,calibration_ids):
    report=read_json(verified_reference(reference))
    if (report.get('schema')!='calibration_encoder_replay_v1' or report.get('status')!='CALIBRATION_ENCODER_REPLAY_PASS'
            or report.get('questions')!=200 or report.get('feature_dimensions')!=17 or report.get('embedding_dimensions')!=384
            or report.get('max_abs_tolerance')!=5e-6 or report.get('fit_or_refit_performed') is not False
            or report.get('labels_used') is not False or report.get('sealed_final_encoded_during_replay') is not False
            or report.get('only_original_calibration_question_text_encoded') is not True
            or report.get('device')!='cpu' or report.get('encoder_max_tokens')!=256 or report.get('batch_size')!=8
            or report.get('calibration_question_ids')!=calibration_ids):
        raise ValueError('Mandatory original-calibration encoder replay was not accepted under the fixed protocol')
    for key in ('max_abs_embedding_difference','max_abs_phi_difference','process_time_s','wall_time_s'):
        value=report.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
            raise ValueError('Missing actual calibration replay difference/CPU clock')
    if max(report['max_abs_embedding_difference'],report['max_abs_phi_difference'])>5e-6:
        raise ValueError('Calibration replay feature drift exceeds the independently fixed tolerance')
    refs={'source_features':preparation['calibration_encoder_replay_sources']['frozen_features'],
        'source_feature_audit':preparation['calibration_encoder_replay_sources']['feature_audit'],
        'pca_artifact':preparation['artifacts']['pca_artifact'],
        'encoder_source_receipt':preparation['artifacts']['encoder_source_receipt']}
    for key,expected in refs.items():
        if report.get(key)!=expected:raise ValueError('Calibration replay source identity changed')
        verified_reference(expected)
    expected_code={name:preparation['final_sources'][name] for name in ('scripts/prepare_final_features.py','src/budget_router/encoding.py')}
    if report.get('source_code')!=expected_code:raise ValueError('Calibration replay encoder implementation changed')
    for ref in expected_code.values():verified_reference(ref)
    if not isinstance(report.get('runtime'),dict) or not all(report['runtime'].get(k) for k in ('python','torch','transformers','numpy')):
        raise ValueError('Calibration replay runtime provenance missing')
    return report
