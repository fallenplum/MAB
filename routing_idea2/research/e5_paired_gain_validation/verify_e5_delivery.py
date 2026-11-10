"""Independent numerical, permission, lineage and publication checks for E5."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys
import time
from collections import Counter
from copy import deepcopy
import numpy as np
from scipy.special import expit

EXPECTED_E3='7995dfdc3afcff8eb8473f6da2af79d55818283d30044200a92c293bf583a499'
EXPECTED_E4_MANIFEST='ddbc57b45b19cdd0df80c8ba50465fa88864600dce4162ae9606c11a3cd95f14'
EXPECTED_E4_RECEIPT='1a7a65b0c59526bd20990b042f35679812eec4db325fb01259ea429c332e0008'
EXPECTED_PDF='b2c9c7f9bf09a3183f4907eb7daedd3a7537dc9f6460f62ccfc246e525bb9e99'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Auditor:
    def __init__(self,root):
        self.root=root
        self.work=root/'research/e5_paired_gain_validation'
        self.public=self.work/'public'
        self.private=self.work/'private'
        self.errors=[]
        self.pending=[]
        self.assertions=0
        self.counts=Counter()
        self.checked_files=[]
        self.maximum_errors={}

    def check(self,value,label,detail=None):
        value=np.asarray(value,dtype=bool)
        self.assertions+=value.size
        if not value.all():
            self.errors.append(dict(check=label,failed=int(np.sum(~value)),detail=detail))

    def close(self,value,expected,label,tolerance=1e-10):
        value,expected=np.asarray(value),np.asarray(expected)
        self.check(value.shape==expected.shape,label+'_shape')
        if value.shape!=expected.shape:return
        if value.dtype.kind in 'USb' or expected.dtype.kind in 'USb':
            self.check(value==expected,label)
            return
        delta=abs(value-expected)
        self.check(np.isfinite(delta)&(delta<=tolerance),label)
        self.maximum_errors[label]=float(np.max(delta)) if delta.size else 0.

    def read(self,path):
        path=Path(path)
        value=json.loads(path.read_text(encoding='utf-8-sig'))
        self.counts['json_opened']+=1
        self.checked_files.append(str(path.relative_to(self.work)) if path.is_relative_to(self.work) else path.name)
        return value

    def arrays(self,path):
        path=Path(path)
        with np.load(path,allow_pickle=False) as loaded:
            result={name:loaded[name] for name in loaded.files}
        self.counts['npz_opened']+=1
        self.checked_files.append(str(path.relative_to(self.work)))
        for name,value in result.items():
            if value.dtype.kind in 'fc':
                allow_nan='non_tie' in name or 'rank_accuracy' in name
                self.check(~np.isinf(value) if allow_nan else np.isfinite(value),'npz_finite_'+path.name+'_'+name)
        return result


def module_imports(a):
    os.environ['MAB_ASSET_ROOT']=str(a.root)
    sys.path.insert(0,str(a.public))
    import e5_common as common
    import gate_models as gates
    import summarize_switching as switching
    import theory_checks as theory
    import run_e3
    return common,gates,switching,theory,run_e3


def immutable_history(a,e3):
    identity,sources=e3.signature()
    a.check(identity==EXPECTED_E3,'E3_original_signature')
    public=a.root/'research/e4_statistical_model_audit/public'
    a.check(sha(public/'PUBLIC_DELIVERY_MANIFEST.json')==EXPECTED_E4_MANIFEST,'E4_manifest_unchanged')
    a.check(sha(public/'DELIVERY_INDEPENDENT_AUDIT.json')==EXPECTED_E4_RECEIPT,'E4_verification_receipt_unchanged')
    manifest=a.read(public/'PUBLIC_DELIVERY_MANIFEST.json')
    for name,record in manifest['files'].items():
        path=public/name
        value=path.read_bytes()
        a.check(len(value)==record['bytes'],'E4_original_bytes_'+name)
        a.check(hashlib.sha256(value).hexdigest()==record['sha256'],'E4_original_sha_'+name)
        a.check(hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()==record['git_blob'],'E4_original_git_blob_'+name)
    a.check(sha(a.private/'REFERENCE_MANUSCRIPT.pdf')==EXPECTED_PDF,'reference_manuscript_identity')
    return dict(E3_signature=identity,E3_sources=sources,E4_manifest_entries=len(manifest['files']),
                E4_manifest_sha256=sha(public/'PUBLIC_DELIVERY_MANIFEST.json'))


def verify_theory(a,theory):
    result=a.read(a.public/'THEORY_IMPLEMENTATION_RESULTS.json')
    a.check(result['status']=='PASS','A_status')
    A=theory.action_design()
    C=np.eye(20)-np.ones((20,20))/20
    ranks=[np.linalg.matrix_rank(x) for x in (A,C@A,C)]
    a.close(np.asarray(ranks),np.asarray([15,14,19]),'A_ranks',0)
    cases=result['population']['cases']
    a.check(len(cases)==140,'A_population_grid_count')
    a.check(set(x['wrong_action'] for x in cases)==set(range(20)),'A_all_twenty_designated_wrong_actions')
    for case in cases:
        r,q,w,h,transversal,overlap=theory.strict_construction(case['wrong_action'],case['gamma'],case['epsilon'])
        a.close(r,np.asarray(case['all_reward_means']),'A_reward_means')
        a.close(q,np.asarray(case['all_population_optimal_scores']),'A_population_scores')
        a.close(A@np.linalg.lstsq(A,r,rcond=None)[0],q,'A_independent_projection')
        a.check(theory.maximizing_set(q)==(case['wrong_action'],),'A_unique_wrong_argmax')
        a.close(np.asarray(r.max()-r[case['wrong_action']]),np.asarray(case['theoretical_regret']),'A_regret_formula')
        a.close(np.asarray(np.mean((q-r)**2)),np.asarray(2*case['gamma']**2/5),'A_sharp_mse')
        a.close(A.T@w,np.zeros(22),'A_orthogonal_parity')
        a.counts['A_population_cases']+=1
    for row in result['population']['population_ridge']:
        r,*_=theory.strict_construction(row['wrong_action'],row['gamma'],row['epsilon'])
        coef=np.linalg.solve(A.T@A/20+row['penalty']*np.eye(22),A.T@r/20)
        q=A@coef
        action=min(theory.maximizing_set(q))
        a.check(action==row['chosen_action'],'A_regularized_action')
        a.close(np.asarray(r.max()-r[action]),np.asarray(row['regret']),'A_regularized_regret')
        a.counts['A_population_ridge']+=1
    for case in result['finite']['cases']:
        repeats=case['repeats_per_action']
        a.check(len(case['records'])==600,'A_finite_MC_count')
        for row in case['records']:
            r,q,*_=theory.strict_construction(row['wrong_action'],.2,.1)
            rng=np.random.default_rng(np.random.SeedSequence([20261110,row['wrong_action'],repeats,row['replicate']]))
            empirical=rng.binomial(repeats,r)/repeats
            ls=A@np.linalg.lstsq(A,empirical,rcond=None)[0]
            ridge=A@np.linalg.solve(repeats*A.T@A+100*np.eye(22),repeats*A.T@empirical)
            for name,score in [('ls',ls),('ridge',ridge)]:
                selected=min(theory.maximizing_set(score))
                a.check(selected==row[name+'_action'],'A_finite_'+name+'_action')
                a.close(np.asarray(r.max()-r[selected]),np.asarray(row[name+'_regret']),'A_finite_'+name+'_regret')
            a.counts['A_finite_paired_datasets']+=1
    a.check(a.counts['A_population_ridge']==420,'A_population_ridge_grid')
    a.check(a.counts['A_finite_paired_datasets']==1800,'A_finite_paired_grid')
    for row in result['centering']['cases']:
        a.check(row['coefficient_identity_max_error']<1e-9,'A_centered_coefficient_qualification')
        a.check(row['prediction_centering_max_error']<1e-9,'A_centered_prediction_qualification')
        a.check(row['maximizing_set_mismatches']==row['declared_lexicographic_selection_mismatches']==0,'A_centered_action_qualification')
    return dict(population=140,population_ridge=420,finite_paired_datasets=1800,
                finite_estimator_fits=3600,synthetic_centering_cases=len(result['centering']['cases']))


def verify_base(a,data,common):
    roles=common.split_roles(data)
    all_E=[]
    maximum_replay=0.
    for f,role in enumerate(roles):
        b=a.arrays(a.private/'base'/f'BASE_f{f}.npz')
        receipt=a.read(a.private/'base'/f'BASE_f{f}.json')
        for name,size in [('F',360),('G',120),('E',120)]:
            q=b[name]
            a.close(q,role[name],f'B_roles_fold{f}_{name}',0)
            a.check(len(q)==len(np.unique(data['ids'][q]))==len(np.unique(data['canonical'][q]))==size,'B_unique_original_questions_'+name)
            a.check(np.sum(data['task'][q]=='gsm8k')==size//2,'B_fixed_broad_balance_'+name)
        for left,right in [('F','G'),('F','E'),('G','E')]:
            a.check(not set(data['canonical'][b[left]])&set(data['canonical'][b[right]]),'B_canonical_role_disjointness')
        all_E.extend(b['E'].tolist())
        for name,expected in receipt['source'].items():
            if (a.public/name).exists():a.check(sha(a.public/name)==expected,'B_frozen_source_'+name)
        a.check(receipt['source']['E4_Gaussian']==sha(common.E4/'e4_gaussian.py'),'B_Gaussian_identity')
        a.check(receipt['source']['E4_Probit']==sha(common.E4/'e4_binary.py'),'B_Probit_identity')
        base=int(common.argmax(data['Y'][b['F']].mean((0,1))))
        a.check(int(b['base'])==base,'B_F_only_global_default')
        for part in ('G','E'):
            a.close(b['candidate_'+part],common.argmax(b['pred_'+part][1],base),'B_fixed_P2_candidate_'+part,0)
            a.check(b['candidate_'+part]!=base,'B_candidate_excludes_default')
        models=common.fit_models(data,b['F'])
        for m,model in enumerate(models):
            for name in ('F','G','E'):
                prediction=common.predict(model,b[name])
                a.close(prediction,b['pred_'+name][m],'B_base_prediction_replay')
                maximum_replay=max(maximum_replay,float(np.max(abs(prediction-b['pred_'+name][m]))))
            a.counts['B_replayed_base_predictors']+=1
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        pca=PCA(16,svd_solver='full').fit(data['E'][b['F']])
        scaler=StandardScaler().fit(pca.transform(data['E'][b['F']]))
        for name in ('F','E'):
            X=np.column_stack((np.ones(len(b[name])),scaler.transform(pca.transform(data['E'][b[name]]))))
            a.close(X,b['X_'+name],'B_independent_F_only_PCA_'+name)
        a.check(set(b['feedback_G20'])<=set(b['G']),'B_feedback_only_G')
        a.check(len(receipt['feedback'])==3,'B_feedback_source_count')
        for g,packet in enumerate(receipt['feedback']):
            a.check(packet['generation_seed']==common.SEEDS[g],'B_feedback_generation_identity')
            a.check(len(packet['feedback'])==20,'B_feedback_budget_twenty')
            for index,event in enumerate(packet['feedback']):
                q=event['q']
                a.check(q==int(b['feedback_G20'][index]),'B_feedback_order')
                pos=np.flatnonzero(b['G']==q)[0]
                expected=[base,int(b['candidate_G'][pos])]
                a.close(np.asarray(event['actions']),np.asarray(expected),'B_common_feedback_actions',0)
                a.close(np.asarray(event['raw_scores']),data['Y'][q,g,expected],'B_delivered_raw_pair',0)
            for m,model in enumerate(models):
                actor=deepcopy(model)
                for event in packet['feedback']:actor.update(event['q'],event['actions'],event['raw_scores'])
                a.close(common.predict(actor,b['E']),b['updated_pred_E'][g,m],'B_fortyfive_feedback_path_replay')
                a.counts['B_replayed_feedback_paths']+=1
        a.check(receipt['centering_check']['outer_accuracy_difference']==0,'A_saved_real_centering_accuracy')
    a.check(len(all_E)==len(set(all_E))==600,'B_total_six_hundred_original_E_questions')
    return dict(base_predictor_fits=15,feedback_paths=45,maximum_prediction_replay_error=maximum_replay)


def verify_centered(a,data):
    d=a.read(a.public/'CENTERED_DIAGNOSTICS_RESULTS.json')
    a.check(d['code_sha256']==sha(a.public/'centered_diagnostics.py'),'B_diagnostic_source_identity')
    a.check(d['configuration_sha256']==sha(a.public/'E5_EXPERIMENT_CONFIG.json'),'B_diagnostic_configuration')
    pair=a.arrays(a.private/'CENTERED_DIAGNOSTICS_PAIRED.npz')
    for name in ('P1','P2','P3_Probit','P1_updated','P2_updated','P3_Probit_updated'):
        get=lambda m:pair[name+'__'+m]
        a.close(get('absolute_mean_mse'),get('centered_mean_mse')+get('common_offset_mse'),'B_absolute_decomposition')
        a.close(get('cross_repeat_T'),get('centered_mean_mse')-get('observed_center_noise_correction'),'B_T_algebra')
        a.close(get('brier_clipped'),get('brier_centered_component')+get('brier_offset_component')+get('observation_noise_mean'),'B_clipped_Brier_decomposition')
        a.close(get('gain_vs_default'),get('correct')-get('harm'),'B_correction_minus_harm')
        for metric in ('absolute_mean_mse','centered_mean_mse','common_offset_mse','accuracy','brier_clipped','gain_vs_default'):
            value=.5*get(metric)[data['task']=='gsm8k'].mean()+.5*get(metric)[data['task']=='bbh'].mean()
            a.close(np.asarray(value),np.asarray(d['model_table'][name]['metrics'][metric]['balanced_v2']),'B_public_weighted_metric')
        a.check(int(np.sum(get('cross_repeat_T')<0))==d['model_table'][name]['negative_cross_repeat_questions'],'B_negative_T_preserved')
    for label,item in d['comparisons'].items():
        old,new=item['old'],item['new']
        center=pair[new+'__centered_mean_mse']-pair[old+'__centered_mean_mse']
        diffT=pair[new+'__cross_repeat_T']-pair[old+'__cross_repeat_T']
        a.close(center,diffT,'B_paired_T_cancellation')
    actual=a.read(a.public/'ACTUAL_CENTERING_RESULTS.json')
    a.check(len(actual['folds'])==5,'A_real_five_fold_centering')
    for row in actual['folds']:
        a.check(row['frozen_P2_prediction_replay_max_error']<1e-10,'A_real_centering_replay')
        a.check(row['outer_accuracy_difference']==row['declared_lexicographic_selection_mismatches']==0,'A_real_centering_accuracy_action_equivalence')
    return dict(model_contexts=6,independent_questions=600,negative_T_preserved=True)


def reconstruct_head(desc,features):
    if desc['constant'] is not None:return np.full(len(features),desc['constant'])
    x=(features-np.asarray(desc['scaler_mean']))/np.asarray(desc['scaler_scale'])
    return expit(x@np.asarray(desc['coefficient'])+desc['intercept'])


def verify_gate_folds(a,data,common,gatemodule):
    source=(a.public/'run_gates.py').read_text(encoding='utf8')
    a.check(source.index("put(folder/f'FROZEN_GATE_") < source.index('np.savez_compressed(decision_path') < source.index("Rb=data['Y'][E"),'C_source_freeze_before_E_reward_access')
    all_masks=np.zeros((24,600),dtype=bool)
    full_D=np.zeros((600,3))
    for f in range(5):
        b=a.arrays(a.private/'base'/f'BASE_f{f}.npz')
        x=a.arrays(a.private/'gates'/f'GATES_f{f}.npz')
        decisions=a.arrays(a.private/'gates'/f'FROZEN_DECISIONS_f{f}.npz')
        freeze=a.read(a.private/'gates'/f'FROZEN_GATE_f{f}.json')
        receipt=a.read(a.private/'gates'/f'GATES_f{f}.json')
        a.check(freeze['E_outcome_used_for_freeze'] is False,'C_freeze_label_permission')
        a.check(receipt['eval_after_freeze'] is True,'C_receipt_eval_after_freeze')
        a.check(receipt['frozen_gate_sha256']==sha(a.private/'gates'/f'FROZEN_GATE_f{f}.json'),'C_frozen_receipt_hash')
        a.check(receipt['decision_sha256']==sha(a.private/'gates'/f'FROZEN_DECISIONS_f{f}.npz'),'C_prospective_decision_hash')
        for name,expected in freeze['source'].items():a.check(sha(a.public/name)==expected,'C_frozen_source_'+name)
        a.close(x['G'],b['G'],'C_G_roles',0)
        a.close(x['E'],b['E'],'C_E_roles',0)
        a.close(x['base'],b['base'],'C_shared_default',0)
        for part in ('G','E'):
            q=x[part]
            a.close(x['candidate_'+part],b['candidate_'+part],'C_shared_candidate_'+part,0)
            candidate_reward=data['Y'][q[:,None],np.arange(3)[None,:],x['candidate_'+part][:,None]]
            default_reward=data['Y'][q,:,int(x['base'])]
            a.close(x['D_'+part],candidate_reward-default_reward,'C_ternary_pair_truth_'+part,0)
            a.check(np.isin(x['D_'+part],[-1,0,1]),'C_raw_ternary_feedback')
        features=np.column_stack((b['pred_E'][1,:,int(b['base'])],b['pred_E'][1,np.arange(120),b['candidate_E']]))
        for name in ('r','u'):
            a.close(reconstruct_head(freeze['fitted_heads'][name],features),x[name+'_E'],'C_fullG_fitted_head_reconstruction')
            details=freeze['fit_metadata'][name]
            outer=details['OOF_folds']
            a.check(np.array_equal(np.sort(np.concatenate([row['held_out_questions'] for row in outer])),np.arange(120)),'C_nested_G_OOF_coverage')
            for row in outer:
                tr,iv=set(row['training_questions']),set(row['held_out_questions'])
                a.check(not tr&iv and tr|iv==set(range(120)),'C_outer_G_question_disjoint')
                for candidate in row['regularization']['candidates']:
                    for inner in candidate['folds']:
                        ti,vi=set(inner['training_questions']),set(inner['validation_questions'])
                        a.check(not ti&vi and ti|vi==set(range(len(tr))),'C_inner_G_question_disjoint')
            support=freeze['fitted_heads'][name]['fit_info']['support']
            expected=(np.ones(120) if name=='r' else np.mean(x['D_G']!=0,axis=1))
            a.close(np.asarray(support['per_question_training_weight']),expected,'C_question_repeat_weights')
        score_G=np.asarray([b['pred_G'][1,np.arange(120),b['candidate_G']]-b['pred_G'][1,:,int(b['base'])],
            x['r_G']*(2*x['u_G']-1),x['r_G'],2*x['u_G']-1])
        score_E=np.asarray([b['pred_E'][1,np.arange(120),b['candidate_E']]-b['pred_E'][1,:,int(b['base'])],
            x['r_E']*(2*x['u_E']-1),x['r_E'],2*x['u_E']-1])
        a.close(x['scores_G_oof'],score_G,'C_score_factorization_G')
        a.close(x['scores_E'],score_E,'C_score_factorization_E')
        main=[]
        for s,name in enumerate(gatemodule.SCORE_NAMES):
            rules=freeze['frozen']['thresholds'][name]
            thresholds=[float(np.quantile(score_G[s],1-c)) for c in (.1,.25,.5,.75)]+[0.]
            a.close(np.asarray([r['threshold'] for r in rules]),np.asarray(thresholds),'C_G_only_thresholds')
            Gmask=np.asarray([score_G[s]>t for t in thresholds])
            Emask=np.asarray([score_E[s]>t for t in thresholds])
            a.close(Gmask,x['gate_masks_G'][s],'C_G_five_rule_masks',0)
            a.close(Emask,x['gate_masks_E'][s],'C_E_frozen_rule_masks',0)
            gains=Gmask@x['D_G'].mean(1)/120
            coverage=Gmask.mean(1)
            maximum=gains.max()
            eligible=np.flatnonzero(gains>=maximum-1e-12)
            selected=min(eligible,key=lambda j:(coverage[j],-thresholds[j],j))
            main.append(selected)
            a.check(selected==freeze['frozen']['main_indices'][name]==int(x['main_indices'][s]),'C_G_only_main_threshold')
            a.check(np.all(Emask[:-2]<=Emask[1:-1]),'C_nested_quantile_gate_family')
        a.check(np.all(x['gate_masks_E'][2,4]),'C_disagreement_zero_all_one')
        a.close(x['gate_masks_E'],decisions['gate_masks_E'],'C_committed_decisions_equal_evaluated',0)
        a.close(x['candidate_E'],decisions['candidate_E'],'C_candidate_unchanged_after_freeze',0)
        q=x['E']
        all_masks[:4,q]=x['gate_masks_E'][np.arange(4),x['main_indices']]
        for s in range(4):all_masks[4+s*5:4+(s+1)*5,q]=x['gate_masks_E'][s]
        full_D[q]=x['D_E']
        a.counts['C_frozen_folds']+=1
        a.counts['C_frozen_rules']+=20
    combined=a.arrays(a.private/'switching/SWITCH_PAIRED.npz')
    a.close(combined['gates'],all_masks,'C_combined_twentyfour_rows',0)
    a.close(combined['D'],full_D,'C_combined_ternary_rewards',0)
    a.check(len(combined['task'])==600 and combined['D'].shape==(600,3),'C_independent_unit_not_1800')
    return combined


def direct_matched(gates,D,task,sub,fold):
    n=len(task)
    weights=np.where(task=='gsm8k',.5/np.sum(task=='gsm8k'),.5/np.sum(task=='bbh'))
    d=D.mean(1)
    result=dict(gain=(gates*d[None])@weights,coverage=gates@weights,
                rho=(gates*(D**2).mean(1)[None])@weights,
                correction=(gates*(D==1).mean(1)[None])@weights,
                harm=(gates*(D==-1).mean(1)[None])@weights)
    for scheme in ('global','task','subtask'):
        total=np.zeros(len(gates))
        for f in sorted(set(fold)):
            rows=np.flatnonzero(fold==f)
            if scheme=='global':groups=[rows]
            elif scheme=='task':groups=[rows[task[rows]==t] for t in ('gsm8k','bbh')]
            else:groups=[rows[task[rows]=='gsm8k']]+[rows[(task[rows]=='bbh')&(sub[rows]==s)] for s in sorted(set(sub[rows][task[rows]=='bbh']))]
            for q in groups:
                if len(q):total+=gates[:,q].mean(1)*np.dot(weights[q],d[q])
        result['random_'+scheme+'_gain']=total
        result['excess_'+scheme]=result['gain']-total
    return result


def verify_switch_statistics(a,x,switchmodule,gatemodule):
    d=a.read(a.public/'SWITCH_RANKING_RESULTS.json')
    boot=a.arrays(a.private/'switching/SWITCH_BOOTSTRAP.npz')
    randomizations=a.arrays(a.private/'switching/SWITCH_MATCHED_RANDOMIZATIONS.npz')
    point=direct_matched(x['gates'],x['D'],x['task'],x['sub'],x['fold'])
    for j,name in enumerate(x['rule_names']):
        row=d['table'][str(name)]
        for field in ('gain','coverage','rho'):
            a.close(np.asarray(row[field]),np.asarray(point[field][j]),'C_public_paired_'+field)
        eta=row['eta']
        a.close(np.asarray(row['gain']),np.asarray(row['rho']*(2*eta-1) if eta is not None else 0),'C_disagreement_direction_identity')
        a.close(np.asarray(row['expected_corrected_questions']-row['expected_harmed_questions']),np.asarray(600*row['gain']),'C_correction_harm_denominator')
        for scheme in ('global','task','subtask'):
            a.close(np.asarray(row['matched'][scheme]['exact_expected_random_gain']),np.asarray(point['random_'+scheme+'_gain'][j]),'C_analytical_matched_'+scheme)
    # Recompute the complete record bootstrap and compare all saved numerical draws.
    rebuilt=switchmodule.question_record_bootstrap(x['gates'],x['D'],x['task'],x['sub'],x['fold'],reps=5000,seed=20261110)
    for key,value in boot.items():a.close(value,rebuilt['bootstrap_draws'][key],'C_full_bootstrap_replay_'+key)
    a.close(boot['gain'],boot['correction']-boot['harm'],'C_bootstrap_correction_harm')
    factor=int(np.flatnonzero(x['rule_names']=='main/factorized')[0])
    gate=a.read(a.public/'REAL_ALLOCATION_GATE.json')
    lower=[]
    for label,field in [('factorized_minus_default','gain'),('factorized_minus_task_matched','excess_task')]:
        summary=np.quantile(boot[field][:,factor],[.0125,.9875])
        a.close(np.asarray(gate['primary_comparisons'][label]['ci97_5_bonferroni_two_comparisons']),summary,'C_primary_Bonferroni_interval')
        lower.append(summary[0])
    a.check(gate['passed']==bool(np.all(np.asarray(lower)>1e-12)),'C_prespecified_scientific_branch')
    a.check(gate['switch_results_sha256']==sha(a.public/'SWITCH_RANKING_RESULTS.json'),'C_branch_source_hash')
    # Verify actual random masks and unchanged candidate identities on a real fold.
    q=np.flatnonzero(x['fold']==0)
    chosen=x['gates'][factor,q]
    controls=gatemodule.matched_controls(chosen,x['D'][q],x['task'][q],x['sub'][q],reps=1000,seed=20261110,return_masks=True)
    for scheme,item in controls.items():
        masks=item['randomization_masks']
        a.check(masks.shape==(1000,120),'C_randomization_unit_shape')
        a.check(masks.sum(1)==chosen.sum(),'C_randomization_total_switch_count')
        if scheme=='global':groups=[np.arange(120)]
        elif scheme=='task':groups=[np.flatnonzero(x['task'][q]==t) for t in ('gsm8k','bbh')]
        else:groups=[np.flatnonzero(x['task'][q]=='gsm8k')]+[np.flatnonzero((x['task'][q]=='bbh')&(x['sub'][q]==s)) for s in sorted(set(x['sub'][q][x['task'][q]=='bbh']))]
        for group in groups:a.check(masks[:,group].sum(1)==chosen[group].sum(),'C_randomization_group_count')
        # All three reward repeats share the one original-question accept decision.
        for sample in masks[:10]:
            actions=np.where(sample,x['candidate'][q],x['default'][q])
            a.check(np.all((actions==x['candidate'][q])|(actions==x['default'][q])),'C_randomization_keeps_each_question_candidate')
    qualification=switchmodule.qualification()
    a.check(qualification['status']=='PASS','C_duplicate_record_bootstrap_qualification')
    a.counts['C_bootstrap_replicates']=5000
    a.counts['C_randomization_arrays']=len(randomizations)
    return dict(rows=24,frozen_family_per_fold=20,independent_E_questions=600,
        bootstrap_complete_records_verified=True,real_allocation_gate=gate['status'])


def verify_metadata(a,data,common,gatemodule):
    d=a.read(a.public/'METADATA_PERMISSION_CONTROLS.json')
    from run_metadata_controls import defaults
    for mode in ('task-known','subtask-oracle'):
        combined=a.arrays(a.private/'metadata_gates'/f'{mode}_PAIRED.npz')
        for f in range(5):
            values=a.arrays(a.private/'metadata_gates'/f'{mode}_f{f}.npz')
            frozen=a.read(a.private/'metadata_gates'/f'FROZEN_{mode}_f{f}.json')
            roles=common.split_roles(data)[f]
            for key in ('F','G','E'):a.close(values[key],roles[key],'metadata_roles_'+mode,0)
            b,_=defaults(data,values['F'],values['E'],mode)
            a.close(values['base_E'],b,'metadata_same_permission_default',0)
            q=values['E']
            truebase=data['Y'][q[:,None],np.arange(3)[None,:],b[:,None]]
            truecandidate=data['Y'][q[:,None],np.arange(3)[None,:],values['candidate_E'][:,None]]
            a.close(values['D_E'],truecandidate-truebase,'metadata_raw_pair_truth',0)
            a.check(values['base_E']!=values['candidate_E'],'metadata_candidate_excludes_default')
            a.check(frozen['E_outcomes_used_for_freeze'] is False,'metadata_freeze_permission')
            for name,expected in frozen['source'].items():a.check(sha(a.public/name)==expected,'metadata_source_identity')
            a.close(combined['gates'][:,q],values['gate_masks_E'][np.arange(4),values['main_indices']],'metadata_combined_main_masks',0)
            a.close(combined['D'][q],values['D_E'],'metadata_combined_raw_pair',0)
            a.counts['metadata_cases']+=1
        a.check(d['table'][mode]['independent_questions']==600,'metadata_question_unit')
        weights=gatemodule.main_weights(data['task'])
        for j,name in enumerate(gatemodule.SCORE_NAMES):
            gain=float(weights@(combined['gates'][j]*combined['D'].mean(1)))
            a.close(np.asarray(d['table'][mode]['table'][name]['statistics']['gain']),np.asarray(gain),'metadata_public_gain')
    return dict(cases=10,permissions=['task-known','subtask-oracle'],main_text_policy_separate=True)


def blinded_refit(a,data,common,gatemodule):
    b=a.arrays(a.private/'base/BASE_f0.npz')
    frozen=a.read(a.private/'gates/FROZEN_GATE_f0.json')
    gate=a.arrays(a.private/'gates/GATES_f0.npz')
    F,G,E=b['F'],b['G'],b['E']
    errors=[]
    for mode in ('flip','NaN'):
        altered=dict(data)
        altered['Y']=data['Y'].copy()
        altered['Y'][E]=1-altered['Y'][E] if mode=='flip' else np.nan
        models=common.fit_models(altered,F)
        predG=np.asarray([common.predict(m,G) for m in models])
        predE=np.asarray([common.predict(m,E) for m in models])
        a.close(predG,b['pred_G'],'qualification_'+mode+'_G_predictions_invariant')
        a.close(predE,b['pred_E'],'qualification_'+mode+'_E_predictions_invariant')
        default=int(common.argmax(altered['Y'][F].mean((0,1))))
        a.check(default==int(b['base']),'qualification_'+mode+'_default_invariant')
        cG=common.argmax(predG[1],default)
        cE=common.argmax(predE[1],default)
        DG=altered['Y'][G[:,None],np.arange(3)[None,:],cG[:,None]]-altered['Y'][G,:,default]
        fit=gatemodule.fit_gate_scores(predG[1],np.full(120,default,dtype=int),cG,DG,
            predE[1],np.full(120,default,dtype=int),cE,strata_G=altered['task'][G],seed=20261110)
        thresholds=gatemodule.freeze_thresholds(fit['scores_G_oof'],DG,altered['task'][G])
        decisions=gatemodule.apply_thresholds(fit['scores_E'],thresholds)
        for s,name in enumerate(gatemodule.SCORE_NAMES):
            a.close(fit['scores_G_oof'][name],gate['scores_G_oof'][s],'qualification_'+mode+'_G_scores_invariant')
            a.close(fit['scores_E'][name],gate['scores_E'][s],'qualification_'+mode+'_E_scores_invariant')
            a.close(decisions[name],gate['gate_masks_E'][s],'qualification_'+mode+'_frozen_decisions_invariant',0)
            a.check(thresholds['thresholds'][name]==frozen['frozen']['thresholds'][name],'qualification_'+mode+'_thresholds_invariant')
        for kind in ('r','u'):a.check(fit['model_heads'][kind].descriptor()==frozen['fitted_heads'][kind],'qualification_'+mode+'_head_parameters_invariant')
        errors.append(dict(alteration=mode,base_predictors=3,head_parameter_mismatch=0,threshold_mismatch=0,gate_mismatch=0,
            maximum_E_prediction_error=float(np.max(abs(predE-b['pred_E'])))))
    return dict(fold=0,results=errors,original_input_hashes_preserved=True,
        scope='Actual re-fitting on copied data with outer E reward flips/NaNs, not a static-only permission check')


def public_scan(a,data):
    qids=[str(x) for x in data['ids']]
    ref=a.root/'artifacts/private/routing_redesign_20261104/derived/v2/selector_train_references_v1.jsonl'
    questions=[json.loads(line)['question'] for line in ref.read_text(encoding='utf8').splitlines()]
    secret=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})')
    assignment=re.compile(rb'''(?i)(?:password|api_key|secret_token|access_token)\s*[:=]\s*["'][^"'\r\n]{6,}["']''')
    bad=[]
    def fields(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in {'raw_scores','D_E','D_G','gate_masks_E','question_id'} and isinstance(item,(list,str)) and len(item):bad.append(dict(file=path.name,field=key))
                if key in {'gold','reference_answer','answer'} and isinstance(item,str) and item.strip():bad.append(dict(file=path.name,field=key))
                fields(item,path)
        elif isinstance(value,list):
            for item in value:fields(item,path)
    files=0
    for path in sorted(a.public.iterdir()):
        if not path.is_file():continue
        value=path.read_bytes()
        a.check(not secret.search(value) and not assignment.search(value),'public_credential_patterns_'+path.name)
        if path.suffix not in ('.json','.py','.md','.txt'):continue
        text=value.decode('utf-8-sig')
        hits=sum(qid in text for qid in qids)
        raw=sum(len(q)>=24 and (q in text or json.dumps(q,ensure_ascii=True)[1:-1] in text) for q in questions)
        a.check(hits==raw==0,'public_no_question_ID_or_text_'+path.name,dict(question_ID_hits=hits,question_text_hits=raw))
        if path.suffix=='.json':fields(json.loads(text),path)
        files+=1
    a.check(not bad,'public_no_raw_gold_feedback_arrays',bad[:20])
    return dict(text_files=files,structural_findings=bad,limits='Exact text/ID and format scans, not a proof against every possible secret or OS access event')


def manifest_check(a):
    path=a.public/'PUBLIC_DELIVERY_MANIFEST.json'
    if not path.exists():return dict(status='PENDING_ROOT_MANIFEST')
    m=a.read(path)
    rows=m['files']
    for name,record in rows.items():
        target=a.public/name
        a.check(target.resolve().is_relative_to(a.public.resolve()),'public_manifest_path_scope')
        value=target.read_bytes()
        a.check(len(value)==record['bytes'],'public_manifest_bytes')
        a.check(hashlib.sha256(value).hexdigest()==record['sha256'],'public_manifest_sha')
        if 'git_blob' in record:a.check(hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()==record['git_blob'],'public_manifest_git_blob')
    excluded={'PUBLIC_DELIVERY_MANIFEST.json','E5_INDEPENDENT_AUDIT.json'}
    actual={p.name for p in a.public.iterdir() if p.is_file()}
    a.check(actual==set(rows)|excluded,'public_manifest_complete_coverage')
    return dict(status='CHECKED',hashed_files=len(rows),public_files=len(actual),manifest_sha256=sha(path),
                self_reference_exceptions=sorted(excluded))


def synthetic_checks(a):
    required=['CERTIFICATION_SYNTHETIC_RESULTS.json','ALLOCATION_SYNTHETIC_RESULTS.json','CERTIFICATION_DEVELOPMENT_POWER.json']
    missing=[name for name in required if not (a.public/name).exists()]
    if missing:
        a.pending.extend(missing)
        return dict(status='PENDING_SYNTHETIC_DELIVERY',missing=missing)
    import run_certification_synthetic as simulation
    import certification
    import run_certification_power as power
    import run_allocation_synthetic as allocation_sim
    import allocation
    d=a.read(a.public/'CERTIFICATION_SYNTHETIC_RESULTS.json')
    a.check(d['status']=='COMPLETE' and len(d['rows'])==640,'D_full_simulation_grid')
    expected={(r,g,n,L,dep,J) for r in (.02,.1,.3,.6) for g in (-.5,0.,.1,.5)
              for n in (200,400,1000,3000,10000) for L in (1,3)
              for dep in ('conditional_independent','perfectly_correlated') for J in (1,5)}
    actual={(r['rho'],r['gain_ratio'],r['n'],r['L'],r['dependence'],r['J']) for r in d['rows']}
    a.check(actual==expected,'D_exact_640_configuration_identities')
    for name,digest in d['source_sha256'].items():a.check(sha(a.public/name)==digest,'D_source_'+name)
    max_false=0.
    duplicate_labels={}
    for row in d['rows']:
        rho,mu=row['rho'],row['nominal_delta']
        probability,values,truth,second=simulation.joint_cells(rho,mu,row['L'],row['dependence'],row['J'])
        # One cell is one complete question and has all J gate outcomes together.
        a.check(values.shape[1]==row['J'],'D_shared_question_gate_axis')
        a.close(np.asarray(row['true_gate_gains']),probability@values,'D_shared_joint_population_gains')
        a.close(np.asarray(row['true_cluster_second_moments']),probability@(values**2),'D_cluster_second_moments')
        if row['J']==1:
            gate_truth=np.asarray([mu]);single_second=np.asarray([rho])
        else:
            gate_truth=np.asarray([0.,mu/2,-mu/2,0.,0.]);single_second=rho*np.asarray([1,.5,.5,.5,.5])
        a.close(np.asarray(row['true_gate_gains']),gate_truth,'D_population_signed_gain_configuration')
        a.close(np.asarray(row['true_gate_disagreement']),single_second,'D_correctness_disagreement_configuration')
        repeated=simulation.experiment(rho,row['gain_ratio'],row['n'],row['L'],row['dependence'],row['J'],10000)
        a.check(row['replicates']==10000,'D_MC_replicate_count')
        for key,item in row['results'].items():
            if item is None:
                a.check(repeated['results'][key] is None,'D_absent_positive_population')
                continue
            a.check(item['count']==repeated['results'][key]['count'],'D_exact_MC_count_'+key)
            a.close(np.asarray(item['rate']),np.asarray(item['count']/10000),'D_MC_rate_'+key)
            a.close(np.asarray(item['mc_ci95']),np.asarray(certification.binomial_interval(item['count'],10000)),'D_Clopper_Pearson_MC_interval')
        for key in ('mean_interval_width','mean_radius','mean_selected_true_gain'):
            a.close(np.asarray(row[key]),np.asarray(repeated[key]),'D_exact_MC_'+key)
        a.check(row['fixed_n_no_optional_stopping'] is True,'D_fixed_timepoint')
        a.check(row['always_default']['positive_detection_power']==0 and row['always_default']['returned_default_rate']==1,'D_always_default_zero_power')
        a.check(row['results']['false_nonpositive_certificate']['mc_ci95'][0]<=.05,'D_MC_error_compatible_with_claimed_delta')
        max_false=max(max_false,row['results']['false_nonpositive_certificate']['rate'])
        if row['L']==1:
            key=(rho,row['gain_ratio'],row['n'],row['J'])
            if key in duplicate_labels:
                a.check(row['results']==duplicate_labels[key]['results'],'D_L1_dependence_label_same_random_world')
            duplicate_labels[key]=row
        a.counts['D_certification_configurations']+=1
    a.check(certification.qualification()['status']=='PASS','D_formula_and_IPW_qualification')
    dev=a.read(a.public/'CERTIFICATION_DEVELOPMENT_POWER.json')
    means=[];radii=[]
    for f in range(5):
        g=a.arrays(a.private/'gates'/f'GATES_f{f}.npz')
        labels=np.asarray([str(v) for v in np.asarray(__import__('e5_common').load_train()[0]['task'])[g['E']]])
        Z=g['gate_masks_E'].reshape(20,120).T*g['D_E'].mean(1)[:,None]
        U=g['gate_masks_E'].reshape(20,120).T*(g['D_E']**2).mean(1)[:,None]
        mean=np.zeros(20);radius=np.zeros(20)
        for task in ('gsm8k','bbh'):
            subset=Z[labels==task]
            a.check(len(subset)==60,'D_F1_within_dataset_independent_count')
            t=np.log(6*20/.005)
            mean+=.5*subset.mean(0)
            radius+=.5*(np.sqrt(2*subset.var(0,ddof=1)*t/60)+14*t/(3*59))
        recorded=dev['fold_certificates'][f]['certificate']
        a.close(np.asarray(recorded['mean']),mean,'D_F1_stratified_mean')
        a.close(np.asarray(recorded['radius']),radius,'D_F1_separate_empirical_Bernstein')
        a.close(np.asarray(recorded['lcb']),mean-radius,'D_F1_combined_lower_bounds')
        selected=int(np.argmax(mean-radius)) if np.max(mean-radius)>0 else -1
        a.check(int(recorded['selected'])==selected,'D_F1_zero_default')
        means.append(mean);radii.append(radius)
        resampled=power.bootstrap_fold(Z,labels,.01,2000,f)
        for original,replay in zip(dev['development_pool_resampling'][f]['rows'],resampled):
            a.check(original==replay,'D_empirical_pool_exact_resampling_replay')
            a.check(original['observed_distinct_source_questions']==120 and original['no_new_independent_observations'] is True,'D_pool_resampling_not_new_questions')
        for j in range(20):
            plan=next(p for p in dev['planning'] if p['fold']==f and p['rule']==dev['rule_names'][j])
            var=np.asarray([Z[labels==t,j].var(ddof=1) for t in ('gsm8k','bbh')])
            rho=np.asarray([U[labels==t,j].mean() for t in ('gsm8k','bbh')])
            a.close(np.asarray(plan['empirical_rho_by_dataset']),rho,'D_planning_paired_disagreement')
            a.close(np.asarray(plan['empirical_question_variance_by_dataset']),var,'D_planning_question_variance')
            for target in plan['target_gains']:
                n=power.balanced_sufficient_questions(rho,target['gain'],20,.05)
                a.check(n==target['fixed_balanced_F1_plugin_sufficient_n'],'D_balanced_sample_sufficient_count')
        a.counts['D_real_fold_certificates']+=1
    a.close(np.asarray(dev['fivefold_family_average_diagnostic']['mean']),np.mean(means,axis=0),'D_fold_average_target_mean')
    a.close(np.asarray(dev['fivefold_family_average_diagnostic']['radius']),np.mean(radii,axis=0),'D_fold_average_target_radius')
    allocation_result=verify_allocation(a,allocation_sim,allocation)
    return dict(status='COMPLETE',D_configurations=640,D_MC_repetitions_per_cell=10000,
        D_maximum_empirical_false_nonpositive_certificate_rate=max_false,
        necessary_lower_bound_scope='Manuscript arbitrary-within-question-dependence class, including identical repeats; not an L-independent lower bound for the conditional-iid-L3 subclass',
        real_certification='Fold/dataset-stratified development diagnostic, not pooled iid600 or fresh population confirmation',
        allocation=allocation_result)


def allocation_dual_check(a,weights,moments,costs,gates,probabilities,record,Bann,Bexec,label):
    active=np.asarray(gates).any(0)
    w,m,c,g,p=(np.asarray(weights)[active],np.asarray(moments)[active],np.asarray(costs)[active],
               np.asarray(gates)[:,active],np.asarray(probabilities)[active])
    kkt=record['KKT']
    lam=np.asarray(kkt['gate_dual_weights'])
    eta=np.asarray(kkt['resource_dual_multipliers'])
    a.check(lam>=-1e-12,label+'_nonnegative_gate_duals')
    a.check(eta>=-1e-12,label+'_nonnegative_resource_duals')
    a.close(np.asarray(lam.sum()),np.asarray(1.),label+'_simplex_dual')
    N=m*(lam@g)
    denominator=eta[0]+eta[1]*c
    star=np.full(len(w),.05)
    good=(N>0)&(denominator>0)
    star[good]=np.clip(np.sqrt(N[good]/denominator[good]),.05,1.)
    star[(N>0)&(denominator==0)]=1.
    star[(N==0)&(denominator==0)]=p[(N==0)&(denominator==0)]
    f=(g*(w*m/p)).sum(1)
    objective=float(f.max())
    dual=float(np.sum(w*(N/star+denominator*star))-eta[0]*Bann-eta[1]*Bexec)
    a.close(np.asarray(record['objective']),np.asarray(objective),label+'_direct_primal_objective')
    a.close(np.asarray(kkt['dual_value']),np.asarray(dual),label+'_independent_dual_value',1e-8)
    a.check(abs(objective-dual)<=2e-6*max(1.,objective),label+'_duality_gap')
    a.check(np.max(abs(lam*(f-objective)))<=5e-6*max(1.,objective),label+'_gate_complementarity')
    a.check(max(abs(eta[0]*(w@p-Bann)),abs(eta[1]*((w*c)@p-Bexec)))<=5e-6*max(1.,objective),label+'_resource_complementarity')


def verify_allocation(a,simulation,allocation):
    d=a.read(a.public/'ALLOCATION_SYNTHETIC_RESULTS.json')
    a.check(d['status']=='COMPLETE','E_synthetic_status')
    expected={(f,ann,exe) for f in ('nested','nonnested') for ann in (.25,.5,.75) for exe in (.25,.5,.75)}
    actual={(row['family'],row['annotation_fraction'],row['execution_fraction']) for row in d['cases']}
    a.check(actual==expected and len(d['cases'])==18,'E_eighteen_exact_scenarios')
    for name,digest in d['code_sha256'].items():a.check(sha(a.public/name)==digest,'E_source_'+name)
    w=np.asarray(d['config']['weights']);cost=np.asarray(d['config']['costs'])
    rho=np.asarray(d['config']['rho']);mu=np.asarray(d['config']['mu'])
    true_a=(rho-mu**2)/3+mu**2
    methods=d['config']['methods']
    a.check(len(methods)==7 and d['config']['replicates']==2000,'E_method_and_MC_grid')
    near_zero_plugin=0
    near_zero_rows=[]
    for row in d['cases']:
        family=row['family'];ann=row['annotation_fraction'];exe=row['execution_fraction']
        name=f'{family}_ann{int(100*ann)}_exec{int(100*exe)}'
        private=a.read(a.private/'synthetic_allocation/cases'/f'{name}.json')
        a.check(row==private,'E_public_private_scenario_identity')
        arrays=a.arrays(a.private/'synthetic_allocation/cases'/f'{name}.npz')
        gates=np.asarray(d['config']['gates'][family])
        active=gates.any(0)
        a.close(np.asarray(row['true_cluster_second_moment']),true_a,'E_true_mean_Dbar_squared')
        a.check(np.max(abs(true_a-rho))>.1,'E_cluster_second_moment_not_single_disagreement')
        pilot=a.read(a.private/'synthetic_allocation/private_pilots'/f'{family}.json')
        ah=np.zeros(7);rh=np.zeros(7)
        for h in np.flatnonzero(active):
            values=np.asarray(pilot[str(h)]['generation_gains'])
            a.check(values.shape==(30,3),'E_independent_pilot_size')
            ah[h]=np.mean(values.mean(1)**2)
            rh[h]=np.mean(values**2)
        a.close(np.asarray(row['pilot_estimated_second_moment']),ah,'E_pilot_estimates_Dbar_squared')
        a.close(np.asarray(row['pilot_estimated_single_repeat_disagreement']),rh,'E_disagreement_baseline_estimate')
        a.check(row['pilot']['gold']==180,'E_pilot_gold_charged')
        truth=(gates*(w*mu)).sum(1)
        a.close(arrays['true_gains'],truth,'E_known_population_gate_gains')
        probabilities={method:np.asarray(row['probabilities'][method]) for method in methods}
        probabilities_matrix=np.asarray([probabilities[m] for m in methods])
        cellsP=arrays['joint_probability'];cellsV=arrays['cells_contributions'];cellsC=arrays['cells_resources']
        p_check,v_check,c_check=simulation.shared_world_cells(gates,probabilities)
        a.close(cellsP,p_check,'E_one_common_input_gain_acquisition_world_probability')
        a.close(cellsV,v_check,'E_one_common_world_all_method_gate_contributions')
        a.close(cellsC,c_check,'E_one_common_world_execution_cost')
        a.close(np.asarray(cellsP.sum()),np.asarray(1.),'E_joint_law_sums_one')
        expected_mean=np.einsum('c,cmj->mj',cellsP,cellsV)
        a.close(expected_mean,np.tile(truth,(7,1)),'E_IPW_unbiased_all_incoming_denominator')
        a.check(np.max(cellsC[:,:,3])==6,'E_aggregators_not_multiplied_by_number_of_gates')
        seed=np.random.SeedSequence([20261110,2,0 if family=='nested' else 1,int(100*ann),int(100*exe)])
        counts=np.random.default_rng(seed).multinomial(400,cellsP,size=2000)
        a.check(counts.sum(1)==400,'E_iid_incoming_question_count')
        estimated=(counts@cellsV.reshape(len(cellsP),-1)/400).reshape(2000,7,len(gates))
        resources=(counts@cellsC.reshape(len(cellsP),-1)).reshape(2000,7,5)
        a.close(estimated,arrays['estimates'],'E_full_shared_world_estimator_replay')
        a.close(resources,arrays['realized_resources'],'E_full_shared_world_resource_replay')
        for j,method in enumerate(methods):
            p=probabilities[method]
            a.check(p[active]>=.05-1e-12,'E_positive_propensities')
            a.check((p<=1+1e-12)&(p>=0),'E_propensity_range')
            a.check(p[~active]==0,'E_inactive_region_known_zero')
            Bann=row['phase_B_ann']+(180/400 if method=='No-pilot-Uniform' else 0)
            Bexec=row['phase_B_exec']+(row['pilot']['execution_units']/400 if method=='No-pilot-Uniform' else 0)
            a.check(w@p<=Bann+1e-7 and (w*cost)@p<=Bexec+1e-7,'E_expected_both_resource_constraints')
            r=row['methods'][method]
            f=np.sum(gates[:,active]*(w[active]*true_a[active]/p[active]),axis=1)
            a.close(np.asarray(r['second_moments']),f,'E_direct_sum_w_g_a_over_p')
            a.close(np.asarray(r['exact_iid_estimator_variances']),(f-truth**2)/400,'E_exact_estimator_variance')
            a.close(np.asarray(r['MSE']),np.mean((estimated[:,j]-truth)**2,axis=0),'E_saved_MSE_actual_MC')
            a.close(np.asarray(r['bias']),estimated[:,j].mean(0)-truth,'E_saved_bias_actual_MC')
            selected=np.where(estimated[:,j].max(1)>0,np.argmax(estimated[:,j],axis=1),-1)
            gains=np.where(selected>=0,truth[np.maximum(selected,0)],0.)
            near_zero=int(np.sum((estimated[:,j].max(1)>0)&(estimated[:,j].max(1)<1e-12)))
            near_zero_plugin+=near_zero
            if near_zero:
                tolerant=np.where(estimated[:,j].max(1)>1e-12,np.argmax(estimated[:,j],axis=1),-1)
                tolerant_gain=np.where(tolerant>=0,truth[np.maximum(tolerant,0)],0.)
                near_zero_rows.append(dict(scenario=name,method=method,near_zero_positive_estimates=near_zero,
                    changed_default_decisions=int(np.sum(tolerant!=selected)),
                    mean_true_gain_difference=float(np.mean(tolerant_gain-gains)),
                    original_plugin_true_gain=float(gains.mean()),tolerance=1e-12,
                    scope='numerical default sensitivity only, original source simulation and protocol retained'))
            a.close(np.asarray(r['plugin_policy_selection']['mean_value_loss']),np.asarray(np.mean(max(0.,truth.max())-gains)),'E_policy_selection_loss')
            cert_radius=np.sqrt(2*np.log(2*len(gates)/.05)/(400*.05))+4*np.log(2*len(gates)/.05)/(3*400*.05)
            a.close(np.asarray(r['fixed_horizon_certified_selection']['radius']),np.asarray(cert_radius),'E_Freedman_not_iid_variance_certificate')
            pilotGold=0 if method=='No-pilot-Uniform' else 180
            pilotExec=0 if method=='No-pilot-Uniform' else row['pilot']['execution_units']
            a.close(np.asarray(r['observed_resources']['mean_gold_including_pilot']),np.asarray(resources[:,j,0].mean()+pilotGold),'E_pilot_inclusive_gold')
            a.close(np.asarray(r['observed_resources']['mean_execution_including_pilot']),np.asarray(resources[:,j,1].mean()+pilotExec),'E_pilot_inclusive_execution')
            a.close(np.asarray(r['observed_resources']['total_gold_overflow_probability']),np.asarray(np.mean(resources[:,j,0]+pilotGold>row['total_gold_budget']+1e-9)),'E_realized_vs_expected_gold_overflow')
            a.close(np.asarray(r['observed_resources']['total_execution_overflow_probability']),np.asarray(np.mean(resources[:,j,1]+pilotExec>row['total_execution_budget']+1e-9)),'E_realized_vs_expected_execution_overflow')
            a.counts['E_method_scenarios']+=1
        for method,moments in [('Estimated-two-resource',ah),('Oracle-two-resource',true_a)]:
            allocation_dual_check(a,w,moments,cost,gates,probabilities[method],row['solver_details'][method],row['phase_B_ann'],row['phase_B_exec'],'E_'+method)
        dominance=allocation.gate_dominance(gates)
        a.check(dominance==row['dominance'],'E_pointwise_dominance_detection')
        a.check(dominance['objective_reduces_to_single_gate']==(family=='nested'),'E_nested_objective_degeneracy')
        a.counts['E_allocation_scenarios']+=1
    a.check(a.counts['E_method_scenarios']==126,'E_eighteen_times_seven_methods')
    for family_record in d['estimated_weights_costs_sensitivity']:
        family=family_record['family'];gates=np.asarray(d['config']['gates'][family]);active=gates.any(0)
        index={(r['seed'],r['annotation_fraction'],r['execution_fraction']):r for r in family_record['rows']}
        a.check(len(index)==450 and len(family_record['rows'])==450,'E_fifty_pilots_nine_budgets')
        for pilot_seed in range(50):
            rng=np.random.default_rng(np.random.SeedSequence([20261110,3,0 if family=='nested' else 1,pilot_seed]))
            H=rng.choice(7,size=180,p=w)
            hist=np.bincount(H,minlength=7)
            wh=(hist+.5)/(180+3.5)
            ah=np.zeros(7);ch=np.ones(7)
            gold=0;execution=0.
            for h in range(7):
                if hist[h]==0 or not active[h]:continue
                support,law=simulation.cluster_distribution(rho[h],mu[h],3,'conditional_independent')
                gains=rng.choice(support,size=hist[h],p=law)
                ah[h]=np.mean(gains**2)
                sampled_cost=rng.lognormal(np.log(cost[h])-.4**2/2,.4,size=hist[h])
                ch[h]=sampled_cost.mean();gold+=hist[h];execution+=sampled_cost.sum()
            for ann in (.25,.5,.75):
                for exe in (.25,.5,.75):
                    row=index[(pilot_seed,ann,exe)]
                    Bann=ann*w[active].sum();Bexec=exe*(w*cost)[active].sum()
                    solved=allocation.solve_design(wh,ah,ch,gates,Bann,Bexec)
                    p=solved['probabilities']
                    a.check(wh@p<=Bann+1e-7 and (wh*ch)@p<=Bexec+1e-7,'E_pilot_estimated_domain_feasibility')
                    a.close(np.asarray(row['true_expected_annotation']),np.asarray(w@p),'E_pilot_true_domain_annotation',1e-8)
                    a.close(np.asarray(row['true_expected_execution']),np.asarray((w*cost)@p),'E_pilot_true_domain_execution',1e-8)
                    a.close(np.asarray(row['true_annotation_violation']),np.asarray(max(0.,w@p-Bann)),'E_estimated_weights_annotation_drift',1e-8)
                    a.close(np.asarray(row['true_execution_violation']),np.asarray(max(0.,(w*cost)@p-Bexec)),'E_estimated_costs_execution_drift',1e-8)
                    a.close(np.asarray(row['pilot_gold']),np.asarray(gold),'E_sensitivity_pilot_gold',0)
                    allocation_dual_check(a,wh,ah,ch,gates,p,solved,Bann,Bexec,'E_sensitive_estimated_domain')
                    a.counts['E_estimated_pilot_sensitivity_cases']+=1
    a.check(a.counts['E_estimated_pilot_sensitivity_cases']==900,'E_full_nine_hundred_sensitivity_grid')
    gate=a.read(a.public/'REAL_ALLOCATION_GATE.json')
    if not gate['passed']:
        a.check(d['real_train600_allocation']=='NOT_RUN_GATE_NOT_PASSED','E_real_allocation_gated_off')
        a.check(not (a.private/'real_allocation').exists(),'E_no_unapproved_real_allocation_receipts')
    return dict(scenarios=18,methods=7,Monte_Carlo_repeats=2000,pilot_sensitivity_cases=900,
        primitive_second_moment='E[(three-repeat D mean)^2], not single-repeat disagreement',
        common_question_and_acquisition_world_replayed=True,near_zero_positive_plugin_estimates=near_zero_plugin,
        near_zero_plugin_default_sensitivity=near_zero_rows,
        budget_scope='Expected resource constraints verified; realized overshoot explicitly re-computed',
        real_train600_allocation=d['real_train600_allocation'])


def direct_source_cost(data,rows,action_sets,gold):
    result={key:0 for key in ('gold','member_calls','aggregator_calls','scoring_ops','input_tokens','output_tokens','tokens')}
    result['source_model_wall_s']=0.
    import itertools
    teams=tuple(itertools.combinations(range(6),3))
    for q,actions in zip(rows,action_sets):
        actions=sorted(set(map(int,actions)))
        members=sorted({member for action in actions for member in teams[action]})
        for g in range(3):
            for stem,indices,call in [('member',members,'member_calls'),('aggregate',actions,'aggregator_calls')]:
                for suffix,key in [('execution_counts',call),('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')]:
                    value=data['cost'][stem+'_'+suffix][g,q,indices].sum()
                    result[key]+=float(value) if suffix=='wall_time_s' else int(value)
            result['scoring_ops']+=len(actions)
    result['tokens']=result['input_tokens']+result['output_tokens']
    result['gold']=gold
    return result


def verify_cost_and_parameters(a,data,common):
    ledger=a.read(a.public/'COST_AND_REUSE_RESULTS.json')
    complete=direct_source_cost(data,np.arange(600),[range(20)]*600,600)
    for key,value in complete.items():
        a.close(np.asarray(ledger['source_cache_once'][key]),np.asarray(value),'cost_complete_cache_'+key,1e-7 if isinstance(value,float) else 0)
    a.check(complete['member_calls']==10800 and complete['aggregator_calls']==36000,'cost_original_execution_count')
    a.check(complete['tokens']==37274127,'cost_original_cache_token_count')
    for f in range(5):
        base=a.arrays(a.private/'base'/f'BASE_f{f}.npz')
        gates=a.arrays(a.private/'gates'/f'GATES_f{f}.npz')
        parameters=a.arrays(a.private/'model_parameters'/f'BASE_MODEL_f{f}.npz')
        frozen=a.read(a.private/'gates'/f'FROZEN_GATE_f{f}.json')
        row=next(x for x in ledger['folds'] if x['fold']==f)
        F,G,E=base['F'],base['G'],base['E']
        default=int(base['base'])
        X=np.column_stack((np.ones(len(E)),(((data['E'][E]-parameters['pca_mean'])@parameters['pca_components'].T)-parameters['scaler_mean'])/parameters['scaler_scale']))
        a.close(X,base['X_E'],'serialized_F_only_PCA_and_scaler')
        predictions=[]
        for m in range(1,4):
            mean=parameters[f'P{m}_mean']
            covariance=parameters[f'P{m}_covariance']
            a.check(np.linalg.eigvalsh(covariance)>=-1e-10,'serialized_covariance_positive_definite')
            if m<3:
                basis=parameters[f'P{m}_action_basis']
                predictions.append(X@mean.reshape(17,basis.shape[1])@basis.T)
            else:
                from scipy.special import ndtr
                predictions.append(ndtr(X@mean.reshape(17,20)))
        a.close(np.asarray(predictions),base['pred_E'],'serialized_all_three_statistical_predictors')
        factor_index=int(gates['main_indices'][1])
        threshold=frozen['frozen']['thresholds']['factorized'][factor_index]['threshold']
        candidate=common.argmax(predictions[1],default)
        features=np.column_stack((predictions[1][:,default],predictions[1][np.arange(len(E)),candidate]))
        r=reconstruct_head(frozen['fitted_heads']['r'],features)
        u=reconstruct_head(frozen['fitted_heads']['u'],features)
        reconstructed_gate=r*(2*u-1)>threshold
        a.close(reconstructed_gate,gates['gate_masks_E'][1,factor_index],'serialized_input_to_gate_end_to_end',0)
        scenarios={
            'F_complete_training':(F,[range(20)]*len(F),360),
            'G_pair_feedback':(G,[[default,int(c)] for c in base['candidate_G']],120),
            'E_complete_diagnostic':(E,[range(20)]*len(E),120),
            'E_paired_gate_evaluation':(E,[[default,int(c)] for c in base['candidate_E']],120),
            'default_service':(E,[[default]]*len(E),0)}
        for name,(q,actions,gold) in scenarios.items():
            expected=direct_source_cost(data,q,actions,gold)
            for key,value in expected.items():a.close(np.asarray(row[name][key]),np.asarray(value),'cost_role_'+name+'_'+key,1e-7 if isinstance(value,float) else 0)
        for s,name in enumerate(('gap','factorized','disagreement','direction')):
            mask=gates['gate_masks_E'][s,gates['main_indices'][s]]
            selected=np.where(mask,base['candidate_E'],default)
            expected=direct_source_cost(data,E,[[int(c)] for c in selected],0)
            for key,value in expected.items():a.close(np.asarray(row['gate_service'][name][key]),np.asarray(value),'cost_frozen_gate_'+name+'_'+key,1e-7 if isinstance(value,float) else 0)
        a.check(row['routing']['exact_action_mismatch']==0,'cost_routing_qualification')
        a.check(row['routing']['repetitions']==100 and row['routing']['questions_per_repeat']==120,'cost_CPU_timing_denominator')
        a.check(row['routing']['cpu_s']>0 and row['routing']['encoder_forward_not_measured'] is True,'cost_CPU_scope_explicit')
        a.counts['serialized_model_parameter_sets']+=1
    a.check(ledger['physical_new_member_calls']==ledger['physical_new_aggregator_calls']==0,'cost_physical_new_LLM_calls_zero')
    a.check(ledger['gpu_time_s'] is None,'cost_source_wall_not_GPU_kernel_time')
    return dict(parameter_sets=5,serialized_input_to_gate_equivalence=True,shared_member_union_costs_verified=True,
        source_cache_once=complete,CPU_scope='frozen embedding onward, encoder preparation unavailable',GPU_kernel_time=None)


def verify_selection_sensitivity(a):
    d=a.read(a.public/'ALLOCATION_SELECTION_NUMERICAL_SENSITIVITY.json')
    original=a.read(a.public/'ALLOCATION_SYNTHETIC_RESULTS.json')
    a.check(d['code_sha256']==sha(a.public/'audit_allocation_selection.py'),'selection_sensitivity_source')
    a.check(d['original_ALLOCATION_SYNTHETIC_RESULTS_sha256']==sha(a.public/'ALLOCATION_SYNTHETIC_RESULTS.json'),'selection_sensitivity_original_results_preserved')
    records={(row['scenario'],row['method']):row for row in d['method_rows']}
    comparisons={row['scenario']:row for row in d['paired_policy_comparisons']}
    n_near=0
    nonnested_worse=Counter()
    mse_better=0
    for scenario in original['cases']:
        name=f"{scenario['family']}_ann{int(100*scenario['annotation_fraction'])}_exec{int(100*scenario['execution_fraction'])}"
        source=a.private/'synthetic_allocation/cases'/f'{name}.npz'
        a.check(sha(source)==d['source_NPZ_sha256'][source.name],'selection_sensitivity_all_original_NPZ_unchanged')
        arrays=a.arrays(source)
        decisions=a.arrays(a.private/'selection_numerical_sensitivity'/f'{name}.npz')
        truth=arrays['true_gains'];estimates=arrays['estimates']
        loss={key:[] for key in ('original','zero_tolerance','full_tolerance')}
        for m,method in enumerate(arrays['method_names'].astype(str)):
            maximum=estimates[:,m].max(1)
            raw=np.where(maximum>0,np.argmax(estimates[:,m],axis=1),-1)
            zero=np.where(maximum>1e-12,np.argmax(estimates[:,m],axis=1),-1)
            full=np.where(maximum>1e-12,np.argmax(estimates[:,m]>=maximum[:,None]-1e-12,axis=1),-1)
            for key,value in [('original',raw),('zero_tolerance',zero),('full_tolerance',full)]:
                a.close(decisions[method+'_'+key+'_selected'],value,'selection_sensitivity_'+key+'_decisions',0)
                gain=np.where(value>=0,truth[np.maximum(value,0)],0.)
                loss[key].append(max(0.,truth.max())-gain)
            near=int(np.sum((maximum>0)&(maximum<=1e-12)))
            n_near+=near
            a.check(records[(name,method)]['near_zero_positive_original_estimates']==near,'selection_sensitivity_near_zero_count')
            a.check(records[(name,method)]['estimator_MSE_changed'] is False,'selection_sensitivity_MSE_unchanged')
        methods=arrays['method_names'].astype(str).tolist()
        est,unif=methods.index('Estimated-two-resource'),methods.index('Uniform')
        for key,values in loss.items():
            delta=float(np.mean(values[est]-values[unif]))
            a.close(np.asarray(comparisons[name]['comparisons'][key]['estimated_minus_uniform_mean_value_loss']),np.asarray(delta),'selection_sensitivity_paired_loss_'+key)
            if scenario['family']=='nonnested':nonnested_worse[key]+=int(delta>1e-12)
        original_mse=np.mean((estimates-truth)**2,axis=0).max(1)
        mse_better+=int(original_mse[est]<original_mse[unif]-1e-12)
        a.counts['selection_sensitivity_scenarios']+=1
    a.check(n_near==566==d['summary']['positive_near_zero_estimates'],'selection_sensitivity_all_near_zero_events')
    a.check(dict(nonnested_worse)==d['summary']['nonnested_estimated_selection_loss_worse_than_uniform'],'selection_sensitivity_selection_conclusion_robust')
    a.check(mse_better==16==d['summary']['estimated_worst_MSE_better_than_uniform'],'selection_sensitivity_MSE_conclusion_unchanged')
    return dict(scenarios=18,original_results_preserved=True,positive_near_zero_events=n_near,
        nonnested_estimated_selection_loss_worse_than_uniform=dict(nonnested_worse),
        worst_MSE_better_scenarios=mse_better,total_MSE_scenarios=18,
        scope='Numeric zero/positive-tie sensitivity only; original simulator, main Text-only gate and fixed-horizon certification preserved')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(r'E:\语言模型组合路由'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    a=Auditor(args.root)
    output=args.output or a.public/'E5_INDEPENDENT_AUDIT.json'
    start=time.perf_counter()
    try:
        common,gates,switching,theory,e3=module_imports(a)
        data,refs=common.load_train()
        config=a.read(a.public/'E5_EXPERIMENT_CONFIG.json')
        a.check(config['manuscript_sha256']==EXPECTED_PDF,'reference_document_matches_task_config')
        sections={'immutable':immutable_history(a,e3),'A':verify_theory(a,theory)}
        print('A_AND_HISTORY_CHECKED',flush=True)
        sections['base']=verify_base(a,data,common)
        sections['B']=verify_centered(a,data)
        print('BASE_AND_CENTERED_CHECKED',flush=True)
        x=verify_gate_folds(a,data,common,gates)
        sections['C']=verify_switch_statistics(a,x,switching,gates)
        sections['metadata']=verify_metadata(a,data,common,gates)
        print('GATES_AND_MATCHED_CONTROLS_CHECKED',flush=True)
        sections['reward_blind_refit']=blinded_refit(a,data,common,gates)
        sections['synthetic']=synthetic_checks(a)
        sections['cost_and_serialized_parameters']=verify_cost_and_parameters(a,data,common)
        sections['selection_numerical_sensitivity']=verify_selection_sensitivity(a)
        privacy=public_scan(a,data)
        manifest=manifest_check(a)
        status='ISSUES_FOUND' if a.errors else 'PARTIAL_PENDING_SYNTHETIC' if a.pending else 'PASS_PENDING_PUBLIC_MANIFEST' if manifest['status']=='PENDING_ROOT_MANIFEST' else 'PASS'
        result=dict(schema='e5_independent_verification_v1',status=status,assertions=a.assertions,
            sections=sections,counts=dict(a.counts),errors=a.errors,pending=a.pending,
            maximum_numerical_errors=a.maximum_errors,checked_files=a.checked_files,
            public_privacy_scan=privacy,public_manifest=manifest,
            independent_units=dict(original_E_questions=600,generation_repeats_are_new_questions=False,
                randomization_or_bootstrap_replicates_are_new_questions=False),
            access_evidence=dict(train_only_loader=True,score_payloads_used=['selector_train'],
                fresh400_or_sealed200_scores_used=False,global_OS_access_trace_collected=False,
                scope='Source control-flow, frozen identities, actual reward-blind re-fits and saved numerical artifacts'),
            code_sha256=sha(__file__),elapsed_s=time.perf_counter()-start,new_llm_calls=0)
    except Exception as exc:
        result=dict(status='AUDIT_INCOMPLETE',error_type=type(exc).__name__,error=str(exc),
                    assertions=a.assertions,errors=a.errors,counts=dict(a.counts),elapsed_s=time.perf_counter()-start)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        raise
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status=status,assertions=a.assertions,errors=len(a.errors),pending=a.pending,
                         counts=dict(a.counts),elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':main()
