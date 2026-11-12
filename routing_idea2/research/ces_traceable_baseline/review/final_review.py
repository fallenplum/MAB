"""Independent migration statistics, permission/cost and frozen-E7 review."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/ces_traceable_baseline'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
E7=ROOT/'research/e7_opportunity_and_feedback_learning'
sys.path.insert(0,str(PUBLIC))
from ces_common import load_train,resources,splits,TEAMS,sha

TRACKS=['ProfileOnly-SingleCombo','HistoryFit-SingleCombo','HistoryFit-Full20']
SEEDS=[17,29,43]
BUDGETS=[0,20,40,60]


def native_budget_review():
    asset=json.loads((PUBLIC/'CES_PUBLIC_ASSET_AUDIT.json').read_text())
    native=json.loads((PUBLIC/'NATIVE_AUDIT.json').read_text())
    required=[(200,2,600),(805,2,4025),(80,2,720),(89,2,445)]
    rows=asset['nominal_successful_call_budgets']
    for expected,row in zip(required,rows):
        assert (row['questions'],row['k'],row['nominal_online_calls'])==expected
        assert sum(row[key] for key in ['selected_member_calls','aggregator_calls','reference_generation_calls','online_judge_calls'])==row['nominal_online_calls']
    assert [row['logical_calls_before_external_evaluation'] for row in native['native_call_budget']]==[600,4025,720,445,800,4830,880,534]
    assert 'profile' in native['native_call_budget'][3]['caveat']
    assert asset['native_main_score_reproduction']['status']=='NOT_REPRODUCED_MISSING_PER_QUESTION_ARTIFACTS'
    assert asset['profile_reaggregation_summary']['arithmetically_consistent_models']==14
    return dict(status='PASS',README_k2_nominal_calls=[600,4025,720,445],
        native_paper_k3_nominal_calls=[800,4830,880,534],
        FLASK_successful_run_precondition='Missing published profiles must first be resolved; unmodified current runner raises KeyError before producing scores',
        native_main_performance='NOT_REPRODUCED',
        source_audit_completion='COMPLETE',upstream_profile_consistency='14 consistent,1 invalid noninteger correct-count model')


def independent_bootstrap(values,task,fold,reps=5000):
    generator=np.random.default_rng(20261112)
    cells=[np.flatnonzero((fold==f)&(task==t)) for f in range(5) for t in ['gsm8k','bbh']]
    out=[]
    for _ in range(reps):
        indices=np.concatenate([generator.choice(group,len(group),replace=True) for group in cells])
        weights=np.where(task[indices]=='gsm8k',.5/np.sum(task[indices]=='gsm8k'),.5/np.sum(task[indices]=='bbh'))
        out.append(values[...,indices]@weights)
    return np.asarray(out)


def main():
    checks=0
    maximum=0.
    def equal(left,right,tol=1e-11):
        nonlocal checks,maximum
        a,b=np.asarray(left),np.asarray(right)
        assert a.shape==b.shape,(a.shape,b.shape)
        if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
            error=float(np.max(abs(a-b))) if a.size else 0.
            maximum=max(maximum,error)
            assert error<tol,error
        else:
            assert np.array_equal(a,b)
        checks+=max(1,a.size)
    data,_=load_train()
    score=np.zeros((3,3,2,4,600,3))
    action=np.zeros((3,3,2,4,600),int)
    cost=np.zeros((5,3,3,4,8))
    query_cost=np.zeros((5,3,3,2,4,8))
    initial_cost=np.zeros((5,3,3,8))
    predicted=np.zeros((3,3,4,600,20))
    allQ=[]
    alltask=[]
    allfold=[]
    allsub=[]
    source_cost_keys=None
    for f in range(5):
        roles=splits(f)
        E,I,U=(roles[k] for k in ['E','I','U'])
        pos=slice(f*120,(f+1)*120)
        allQ.extend(E.tolist())
        alltask.extend(data['task'][E].tolist())
        allfold.extend([f]*120)
        allsub.extend(data['sub'][E].tolist())
        for ti,track in enumerate(TRACKS):
            for si,seed in enumerate(SEEDS):
                folder=PRIVATE/f'replay/f{f}_{track}_s{seed}'
                receipt=json.loads((folder/'FROZEN_PATH.json').read_text())
                equal(receipt['protocol_sha256'],sha(PUBLIC/'CES_PROTOCOL.json'))
                equal(receipt['record_sha256'],sha(folder/'FROZEN_PATH.npz'))
                assert receipt['new_LLM_calls']==0
                with np.load(folder/'FROZEN_PATH.npz') as a:
                    equal(a['I'],I)
                    equal(a['E'],E)
                    equal(a['U'],U)
                    purchased=a['purchased'].copy()
                    equal(a['checkpoints'],BUDGETS)
                    assert len(set(purchased))==60 and set(purchased)<=set(U)
                    with np.load(E7/f'private/c/f{f}_P1_Random_s{seed}/PATH_RECORDS.npz') as old:
                        equal(purchased,old['purchased'])
                    for mode,key in enumerate(['E_actions','E_mean_actions']):
                        chosen=a[key]
                        assert chosen.shape==(4,120) and (chosen>=0).all() and (chosen<20).all()
                        action[ti,si,mode,:,pos]=chosen
                        score[ti,si,mode,:,pos]=data['Y'][E[None,:,None],np.arange(3)[None,None,:],chosen[:,:,None]]
                    service=a['U_selected'].copy()
                    predicted[ti,si,:,pos]=a['E_predicted_means']
                logs=json.loads((folder/'FEEDBACK_ACCESS_AND_COST.json').read_text())
                assert len(logs['labels'])==len(logs['log'])==len(logs['cost'])==60
                assert logs['query_Y_reads']==0
                source_cost_keys=list(logs['cost'][0])
                assert len(source_cost_keys)==8,source_cost_keys
                rows=[]
                for step,(entry,observed,expectedcost) in enumerate(zip(logs['log'],logs['labels'],logs['cost'])):
                    q=int(purchased[step])
                    equal(entry['question'],q)
                    legal=tuple(range(20)) if track=='HistoryFit-Full20' else (int(service[step]),)
                    equal(entry['actions'],legal)
                    equal(observed['actions'],legal)
                    equal(observed['raw'],data['Y'][q][:,legal])
                    actual=resources(data,[q],[legal],gold=1)
                    equal([expectedcost[k] for k in source_cost_keys],[actual[k] for k in source_cost_keys])
                    expected_calls=(18,60) if track=='HistoryFit-Full20' else (9,3)
                    equal([actual['member_calls'],actual['aggregator_calls']],expected_calls)
                    rows.append([actual[k] for k in source_cost_keys])
                rows=np.asarray(rows)
                for bi,b in enumerate(BUDGETS):
                    cost[f,ti,si,bi]=rows[:b].sum(axis=0)
                    for mode in range(2):
                        qcost=resources(data,E,[[int(q)] for q in action[ti,si,mode,bi,pos]],gold=0)
                        query_cost[f,ti,si,mode,bi]=[qcost[k] for k in source_cost_keys]
                initial=logs['initial_cost']
                initial_cost[f,ti,si]=[initial[k] for k in source_cost_keys]
                if track=='ProfileOnly-SingleCombo':
                    equal([initial[k] for k in ['gold','member_calls','aggregator_calls','scoring_ops']],[240,4320,0,4320])
                    for suffix,target in [('prompt_tokens','input_tokens'),('completion_tokens','output_tokens'),('wall_time_s','source_model_wall_s')]:
                        equal(initial[target],data['cost']['member_'+suffix][:,I].sum())
                else:
                    expected=resources(data,I,[range(20)]*240,gold=240)
                    expected['scoring_ops']+=4320
                    equal([initial[k] for k in source_cost_keys],[expected[k] for k in source_cost_keys])
                for snap,expectedsha in receipt['snapshot_sha256'].items():
                    equal(sha(folder/snap),expectedsha)
    task=np.asarray(alltask)
    fold=np.asarray(allfold)
    sub=np.asarray(allsub)
    perq=score.mean(axis=(1,5))
    with np.load(E7/'private/c/E7_C_PAIRED.npz') as old:
        equal(old['Q'],np.asarray(allQ))
        frozenE7=old['score']
    contrasts=np.stack([perq[2,0,3]-frozenE7[k,0,:,3].mean(axis=(0,2)) for k in range(2)])
    draws=independent_bootstrap(contrasts,task,fold)
    weights=np.where(task=='gsm8k',.5/np.sum(task=='gsm8k'),.5/np.sum(task=='bbh'))
    point=contrasts@weights
    p=(1+np.sum(abs(draws-point)>=abs(point)-1e-15,axis=0))/(len(draws)+1)
    order=np.argsort(p)
    corrected=np.zeros(2)
    last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,(2-rank)*p[index]))
        corrected[index]=last
    review_private=PRIVATE/'final_review'
    review_private.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(review_private/'INDEPENDENT_PAIRED_RECONSTRUCTION.npz',score=score,action=action,perq=perq,
        cost=cost,query_cost=query_cost,Q=allQ,task=task,fold=fold,sub=sub,
        primary_contributions=contrasts,primary_draws=draws)
    primary=[dict(point=float(point[i]),ci95=np.quantile(draws[:,i],[.025,.975]).tolist(),
        ci97_5=np.quantile(draws[:,i],[.0125,.9875]).tolist(),centered_p=float(p[i]),Holm_p=float(corrected[i])) for i in range(2)]
    migration_path=PUBLIC/'MIGRATION_RESULTS.json'
    summary_checks=0
    if migration_path.exists():
        migration=json.loads(migration_path.read_text())
        with np.load(PRIVATE/'MIGRATION_PAIRED.npz') as a:
            equal(a['score'],score.transpose(0,2,1,3,4,5))
            equal(a['actions'],action.transpose(0,2,1,3,4))
            equal(a['perq'],perq)
            equal(a['predicted_means'],predicted)
            truth=data['Y'][np.asarray(allQ)].mean(axis=1)
            residual=predicted-truth[None,None,None]
            common=residual.mean(axis=-1)
            centered=((residual-common[...,None])**2).mean(axis=-1)
            equal(a['centered_error'],centered)
            equal(a['common_offset_error'],common**2)
            equal(a['Q'],np.asarray(allQ))
            equal(a['primary_boot'],draws.T)
            equal(a['feedback_cost'],cost.mean(axis=(0,2)))
            equal(a['initial_cost'],initial_cost.mean(axis=(0,2)))
            equal(a['E_service_cost'],query_cost.mean(axis=(0,2)))
            equal(a['costnames'],source_cost_keys)
        for i,row in enumerate(migration['primary_comparisons']):
            equal(row['point'],primary[i]['point'])
            equal(row['ci95'],primary[i]['ci95'])
            equal(row['ci_family_bonferroni'],primary[i]['ci97_5'])
            equal(row['centered_bootstrap_p'],primary[i]['centered_p'])
            equal(row['Holm_p'],primary[i]['Holm_p'])
        joint=independent_bootstrap(perq,task,fold)
        initial_boot=independent_bootstrap(perq-perq[:,:,:1],task,fold)
        for ti,track in enumerate(TRACKS):
            for mode,name in enumerate(['UCB','MeanOnlyDiagnostic']):
                for bi,b in enumerate(BUDGETS):
                    row=migration['table'][track][name][str(b)]
                    qscore=perq[ti,mode,bi]
                    equal(row['accuracy'],qscore@weights)
                    equal(row['ci95'],np.quantile(joint[:,ti,mode,bi],[.025,.975]))
                    gain=qscore-perq[ti,mode,0]
                    equal(row['gain_own_initial']['point'],gain@weights)
                    equal(row['gain_own_initial']['ci95'],np.quantile(initial_boot[:,ti,mode,bi],[.025,.975]))
                    eqseed=score[ti,:,mode,bi]
                    equal(row['seed_accuracy'],eqseed.mean(axis=(0,1)))
                    equal(row['algorithm_seed_accuracy'],eqseed.mean(axis=(1,2)))
                    equal(row['corrected_question_equivalents'],float((eqseed>score[ti,:,mode,0]).mean(axis=(0,2)).sum()))
                    equal(row['harmed_question_equivalents'],float((eqseed<score[ti,:,mode,0]).mean(axis=(0,2)).sum()))
                    equal(row['changed_from_initial'],np.sum(action[ti,:,mode,bi]!=action[ti,:,mode,0])/3)
                    equal(row['centered_MSE'],centered[ti,:,bi].mean())
                    equal(row['common_offset_MSE'],(common[ti,:,bi]**2).mean())
                    for t in ['gsm8k','bbh']:
                        equal(row['GSM_BBH'][t],qscore[task==t].mean())
                    subvalues=[]
                    for s in sorted(set(sub[task=='bbh'])):
                        actual=qscore[(task=='bbh')&(sub==s)].mean()
                        equal(row['BBH_subtasks'][s],actual)
                        subvalues.append(actual)
                    equal(row['GSM_BBH27_macro'],.5*qscore[task=='gsm8k'].mean()+.5*np.mean(subvalues))
                    feedback=cost[:,ti,:,bi].mean(axis=(0,1))
                    initial=initial_cost[:,ti].mean(axis=(0,1))
                    service=query_cost[:,ti,:,mode,bi].mean(axis=(0,1))
                    equal([row['feedback_cost'][k] for k in source_cost_keys],feedback)
                    equal([row['initial_cost'][k] for k in source_cost_keys],initial)
                    equal([row['E_service_cost'][k] for k in source_cost_keys],service)
                    equal([row['complete_system_cost'][k] for k in source_cost_keys],initial+feedback+service)
                    summary_checks+=1
    result=dict(status='PASS',scope='Independent source/feedback/cost, original-question statistics, frozenE7 and public privacy review',
        scalar_checks=checks,maximum_reconstruction_error=maximum,formal_paths=45,snapshots=180,
        reconstructed_service_reward_values=int(score.size),original_E_questions=600,
        primary_independent=primary,source_cost_keys=source_cost_keys,full_summary_rows_reconstructed=summary_checks,
        mean_perfold_feedback_cost=cost.mean(axis=(0,2)).tolist(),mean_perfold_query_cost=query_cost.mean(axis=(0,2)).tolist(),
        native_budget_review=native_budget_review(),actual_model_API_calls=0)
    # Parent output schema is reviewed once its summary has been written.
    if migration_path.exists():
        result['migration_result_sha256']=sha(migration_path)
        result['migration_summary_available']=True
    else:
        result['migration_summary_available']=False
    required=['01_NATIVE_REPRODUCIBILITY_AUDIT.md','02_DATA_FEEDBACK_AND_COST_PERMISSIONS.md',
        '03_ADAPTATION_CONTRACT_AND_OPERATOR_AUDIT.md','04_MIGRATION_RESULTS_AND_COST.md',
        '05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md','REPRODUCE.md','CES_COMPLETE_RESULTS.json','verify_delivery.py']
    missing=[name for name in required if not (PUBLIC/name).is_file()]
    if not missing:
        complete=json.loads((PUBLIC/'CES_COMPLETE_RESULTS.json').read_text())
        assert complete['native_benchmark']=='NOT_REPRODUCED_WITH_RELEASED_ARTIFACTS'
        assert complete['extra_CES_I_side_information'] is True
        assert complete['formal_paths']==45 and complete['model_snapshots']==180
        assert complete['protocol_sha256']==sha(PUBLIC/'CES_PROTOCOL.json')
        text=(PUBLIC/'05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md').read_text()
        assert '尚未独立复现' in text and '额外' in text and '先验' in text
        assert '9' in (PUBLIC/'01_NATIVE_REPRODUCIBILITY_AUDIT.md').read_text()
    frozen=json.loads((PUBLIC/'E7_FREEZE_RECEIPT.json').read_text())
    for name,digest in frozen['files'].items():
        equal(sha(E7/name),digest)
    e7manifest=json.loads((E7/'public/PUBLIC_DELIVERY_MANIFEST.json').read_text())
    for name,record in e7manifest['files'].items():
        equal(sha(E7/'public'/name),record['sha256'])
    ids=list(map(str,data['ids']))+list(map(str,data['canonical']))
    private_text=[]
    qidpattern=re.compile(r'(?<!\\)"question_id"\s*:\s*("(?:[^"\\]|\\.)*")')
    qpattern=re.compile(r'(?<!\\)"question"\s*:\s*("(?:[^"\\]|\\.)*")')
    targets=set(map(str,data['ids']))
    for rel in ['data/derived/available_records.jsonl','data/derived/gsm8k_questions.jsonl']:
        with (ROOT/rel).open(encoding='utf-8') as stream:
            for line in stream:
                match=qidpattern.search(line)
                if match and json.loads(match.group(1)) in targets:
                    value=qpattern.search(line)
                    if value:
                        private_text.append(json.loads(value.group(1)))
    leaks=[]
    count=0
    compiled=0
    for path in PUBLIC.rglob('*'):
        if not path.is_file() or path.suffix not in ['.py','.md','.json']:
            continue
        relative=path.relative_to(PUBLIC).as_posix()
        content=path.read_text('utf-8-sig')
        if any(s in content for s in ids if len(s)>8):
            leaks.append(relative+' actual identifier')
        if any(s in content for s in private_text if len(s)>40):
            leaks.append(relative+' actual question')
        if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',content):
            leaks.append(relative+' credential pattern')
        if path.suffix=='.py':
            compile(content,str(path),'exec')
            compiled+=1
        if path.suffix=='.json' and not relative.startswith('native/'):
            if re.search(r'"(?:w_hat|V_inv|W_x|W_m|E_predictions|E_predicted_means)"\s*:\s*\[',content):
                fixture=json.loads(content)
                synthetic=fixture.get('set_UCB_counterexample',{})
                allowed=(relative=='adapter/CES_ADAPTER_OPERATOR_CHECKS.json'
                         and synthetic.get('source')=='synthetic valid build_phi and consistent offline state'
                         and np.asarray(synthetic.get('w_hat')).shape==(6,)
                         and np.asarray(synthetic.get('V_inv')).shape==(6,6))
                if not allowed:
                    leaks.append(relative+' private fitted matrix')
        count+=1
    assert not leaks,leaks
    result.update(status='PASS',scalar_checks=checks,maximum_reconstruction_error=maximum,
        E7_frozen_receipt_records=len(frozen['files']),E7_public_manifest_records_unchanged=len(e7manifest['files']),
        privacy=dict(public_files=count,actual_identifiers_scanned=len(ids),question_strings_scanned=len(set(private_text)),
            matches=leaks,model_parameters_public=False),python_sources_compiled=compiled,
        reviewed_file_hash_exclusions=['CES_FINAL_REVIEW.json','PUBLIC_DELIVERY_MANIFEST.json','CES_DELIVERY_AUDIT.json'],
        reviewed_files={str(path.relative_to(PUBLIC)).replace('\\','/'):sha(path) for path in PUBLIC.rglob('*') if path.is_file()
                       and path.name not in ['CES_FINAL_REVIEW.json','PUBLIC_DELIVERY_MANIFEST.json','CES_DELIVERY_AUDIT.json']},pending=[])
    if not migration_path.exists() or missing:
        result['status']='PARTIAL_SUMMARY_PENDING'
        result['pending']=missing+([] if migration_path.exists() else ['MIGRATION_RESULTS.json'])
    (PUBLIC/'CES_FINAL_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['reviewed_files','mean_perfold_feedback_cost','mean_perfold_query_cost']},ensure_ascii=False))


if __name__=='__main__':
    main()
