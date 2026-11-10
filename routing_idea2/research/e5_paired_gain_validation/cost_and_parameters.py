"""Separate original cache, fitting, gate feedback, evaluation and deployment costs."""
from e5_common import *
from gate_models import SCORE_NAMES
from scipy.special import expit
from time import perf_counter,process_time


def resources(data,rows,action_sets,*,gold=0):
    out={k:0 for k in ('gold','member_calls','aggregator_calls','scoring_ops','input_tokens','output_tokens','tokens')}
    out['source_model_wall_s']=0.
    for q,actions in zip(rows,action_sets):
        actions=list(sorted(set(map(int,actions))))
        members=list(sorted(set().union(*(set(TEAMS[a]) for a in actions))))
        for g in range(3):
            for stem,selected,call in [('member',members,'member_calls'),('aggregate',actions,'aggregator_calls')]:
                for suffix,key in [('execution_counts',call),('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')]:
                    value=data['cost'][stem+'_'+suffix][g,q,selected].sum()
                    out[key]+=float(value) if suffix=='wall_time_s' else int(value)
            out['scoring_ops']+=len(actions)
    out['tokens']=out['input_tokens']+out['output_tokens']
    out['gold']=int(gold)
    return out


def descriptor_probability(x,head):
    if head['constant'] is not None:return float(head['constant'])
    z=(x-np.asarray(head['scaler_mean']))/np.asarray(head['scaler_scale'])
    return float(expit(z@np.asarray(head['coefficient'])+head['intercept']))


def main():
    data,_=load_train()
    rows=[]
    parameter_folder=PRIVATE/'model_parameters'
    parameter_folder.mkdir(parents=True,exist_ok=True)
    for f in range(5):
        with np.load(PRIVATE/'base'/f'BASE_f{f}.npz') as a:
            F,G,E=a['F'],a['G'],a['E']
            base=int(a['base'])
            cG,cE=a['candidate_G'],a['candidate_E']
            saved_pred=a['pred_E']
        with np.load(PRIVATE/'gates'/f'GATES_f{f}.npz') as a:
            masks=a['gate_masks_E'][np.arange(4),a['main_indices']]
        frozen=json.loads((PRIVATE/'gates'/f'FROZEN_GATE_f{f}.json').read_text())
        # Reconstruct and serialize small statistical parameters without touching old actors.
        fitting_start=process_time()
        models=fit_models(data,F)
        reconstruction_cpu=process_time()-fitting_start
        error=max(float(np.max(abs(predict(model,E)-saved_pred[j]))) for j,model in enumerate(models))
        assert error<1e-10
        t=models[1].template
        arrays=dict(pca_components=t.pca.components_,pca_mean=t.pca.mean_,
                    scaler_mean=t.scaler.mean_,scaler_scale=t.scaler.scale_)
        for j,m in enumerate(models):
            arrays[f'P{j+1}_mean']=m.mean
            arrays[f'P{j+1}_covariance']=m.cov
            if j<2:arrays[f'P{j+1}_action_basis']=m.template.B
        np.savez_compressed(parameter_folder/f'BASE_MODEL_f{f}.npz',**arrays)
        W=models[1].mean.reshape(t.question_dimension,t.action_dimension)
        head_r,head_u=frozen['fitted_heads']['r'],frozen['fitted_heads']['u']
        chosen_threshold=frozen['frozen']['thresholds']['factorized'][frozen['frozen']['main_indices']['factorized']]['threshold']
        def route(embedding):
            x=np.r_[1.,t.scaler.transform(t.pca.transform(embedding[None]))[0]]
            q=x@W
            candidate=argmax(q,base)
            features=np.asarray([q[base],q[candidate]])
            score=descriptor_probability(features,head_r)*(2*descriptor_probability(features,head_u)-1)
            return candidate if score>chosen_threshold else base,score
        checked=[route(data['E'][q]) for q in E]
        expected=np.where(masks[1],cE,base)
        mismatch=int(np.sum(np.asarray([a for a,s in checked])!=expected))
        assert mismatch==0
        wall_start,cpu_start=perf_counter(),process_time()
        for _ in range(100):
            for q in E:route(data['E'][q])
        cpu,wall=process_time()-cpu_start,perf_counter()-wall_start
        row=dict(fold=f,F_complete_training=resources(data,F,[range(20)]*len(F),gold=len(F)),
            G_pair_feedback=resources(data,G,[[base,int(c)] for c in cG],gold=len(G)),
            E_complete_diagnostic=resources(data,E,[range(20)]*len(E),gold=len(E)),
            E_paired_gate_evaluation=resources(data,E,[[base,int(c)] for c in cE],gold=len(E)),
            default_service=resources(data,E,[[base]]*len(E)),
            gate_service={name:resources(data,E,[[int(cE[i]) if masks[j,i] else base] for i in range(len(E))]) for j,name in enumerate(SCORE_NAMES)},
            routing=dict(repetitions=100,questions_per_repeat=len(E),cpu_s=cpu,wall_s=wall,cpu_s_per_question=cpu/(100*len(E)),
                includes=['PCA/standardize from frozen384 embedding','P2 twenty-head prediction','proposal selection','two gate heads','threshold decision'],
                encoder_forward_not_measured=True,new_embedding_model_calls=0,exact_action_mismatch=mismatch),
            parameter_reconstruction=dict(prediction_max_error=error,cpu_s=reconstruction_cpu,
                sha256=sha(parameter_folder/f'BASE_MODEL_f{f}.npz'),role='independent serialization/audit, not a new policy'),
            new_llm_calls=0)
        rows.append(row)
    # The full source cache is paid once across all overlapping-fold diagnoses.
    cache=resources(data,np.arange(600),[range(20)]*600,gold=600)
    put(PUBLIC/'COST_AND_REUSE_RESULTS.json',dict(status='COMPLETE',source_cache_once=cache,folds=rows,
        physical_new_member_calls=0,physical_new_aggregator_calls=0,
        resource_roles='Counterfactual source inference costs are separate from new physical calls. Folds overlap and their fitting accounts are not added as actual repeated cache preparation.',
        shared_feedback='Each G pair supplies every gate and both heads, with one gold and unique member executions per question/seed. No repeated gold fee per rule.',
        evaluation_cost='Whole-matrix diagnostic, pairwise gate evaluation and one-combination deployment are separate accounts.',
        cpu_scope='Routing is measured from existing embeddings with process CPU time. Original encoder preparation cost is unavailable.',
        gpu_time_s=None,gpu_note='Source model wall sums are proxies, not measured GPU kernel time.'))
    print(json.dumps(dict(status='COMPLETE',source_cache_tokens=cache['tokens'],folds=5)))


if __name__=='__main__':main()
