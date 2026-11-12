"""Independent final E7 statistics, costs, source-preservation and privacy review."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys
import hashlib
import json
import re
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
sys.path.insert(0,str(PUBLIC))
from e7_common import load_train,splits,put


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks=0
    max_error=0.
    def close(a,b,label,tolerance=1e-10):
        nonlocal checks,max_error
        a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape,(label,a.shape,b.shape)
        error=float(np.max(abs(a-b))) if a.size else 0.
        assert error<tolerance or error==0,(label,error)
        checks+=a.size
        max_error=max(max_error,error)
    required=['01_CANDIDATE_OPPORTUNITY.md','02_INPUT_SIGNAL_AND_LEARNING_CURVES.md',
        '03_ACTIVE_FEEDBACK_LEARNING.md','04_COST_AND_STATISTICS.md','05_E7_RESEARCH_DECISION.md',
        'RELATED_WORK_MAPPING.md','E7_PROTOCOL.json','E7_COMPLETE_RESULTS.json','REPRODUCE.md',
        'prepare_e7.py','verify_e7_delivery.py']
    assert all((PUBLIC/name).is_file() for name in required)
    data,_=load_train()
    protocol=json.loads((PUBLIC/'E7_PROTOCOL.json').read_text())
    guard=json.loads((PRIVATE/'PROTOCOL_FROZEN.json').read_text())
    assert guard['protocol_sha256']==sha(PUBLIC/'E7_PROTOCOL.json')
    assert guard['split_sha256']==sha(PRIVATE/'E7_SPLITS_FROZEN.npz')
    with np.load(PRIVATE/'b/PRIMARY_B_COMPARISONS.npz') as a:
        original=a['question_indices'];task=a['task'];fold=a['fold']
        b_contribution=[a['learning_contribution'],a['content_contribution']]
    with np.load(PRIVATE/'c/E7_C_PAIRED.npz') as a:
        Q=a['Q'];score=a['score'];perq=a['perq'];chosen=a['actions']
        equal_query=np.array_equal(Q,original)
        close(score.mean(axis=(2,5)),perq,'C algorithm/generation axes')
        assert score.shape==(2,3,3,4,600,3) and chosen.shape==(2,3,3,4,600)
    assert equal_query and len(set(Q))==600
    actual=data['Y'][Q]
    rebuilt=actual[np.arange(600)[None,None,None,None,:,None],
                   np.arange(3)[None,None,None,None,None,:],chosen[...,None]]
    close(rebuilt,score,'all final route scores')
    contributions=np.asarray(b_contribution+[perq[1,2,3]-perq[1,0,3],perq[1,2,3]-perq[1,1,3]])
    with np.load(PRIVATE/'E7_PRIMARY_PAIRED.npz') as a:
        close(contributions,a['contributions'],'primary contributions')
        saved_boot=a['bootstrap']
    rng=np.random.default_rng(20261112)
    bootstrap=[]
    for _ in range(5000):
        ix=np.concatenate([rng.choice(ids,len(ids),replace=True) for f in range(5)
            for name in ('gsm8k','bbh') for ids in [np.flatnonzero((fold==f)&(task==name))]])
        bootstrap.append(contributions[:,ix].mean(axis=1))
    bootstrap=np.asarray(bootstrap).T
    close(bootstrap,saved_boot,'all four bootstrap arrays')
    stats=json.loads((PUBLIC/'E7_PRIMARY_STATISTICS.json').read_text())
    pvalues=[]
    for j,row in enumerate(stats['main_comparisons']):
        point=float(contributions[j].mean())
        close(point,row['point'],'primary point')
        close(np.quantile(bootstrap[j],[.025,.975]),row['ci95'],'primary95')
        close(np.quantile(bootstrap[j],[.05/8,1-.05/8]),row['ci_family_bonferroni'],'family98.75')
        p=float((1+np.sum(abs(bootstrap[j]-point)>=abs(point)-1e-15))/5001)
        pvalues.append(p)
        close(p,row['centered_bootstrap_p'],'primary p')
    previous=0.
    for rank,index in enumerate(np.argsort(pvalues)):
        previous=max(previous,min(1.,(4-rank)*pvalues[index]))
        close(previous,stats['main_comparisons'][index]['Holm_p'],'Holm4')
    assert all(not row['positive_evidence_after_family_control'] for row in stats['main_comparisons'])
    C=json.loads((PUBLIC/'c/E7_C_RESULTS.json').read_text())
    for ki,kind in enumerate(['P1','P3']):
        for mi,method in enumerate(['Random','ParameterIG','DecisionAware']):
            close(perq[ki,mi,3].mean(),C['table'][kind][method]['60']['accuracy'],'C60 accuracy')
    costs=json.loads((PUBLIC/'E7_COST_SUMMARY.json').read_text())
    token_per_package=18*(4096+2048)+60*(6144+2048)
    assert token_per_package==602112
    assert costs['packet']['Token_upper_bound']==token_per_package
    assert costs['Gold60']['Token_upper_bound']==token_per_package*60
    assert costs['physical_new_LLM_calls']==0
    source=data['cost']
    package_checks=0
    snapshot_checks=0
    all_paid=[]
    for f in range(5):
        role=splits(f)
        for kind in ['P1','P3']:
            for method in ['Random','ParameterIG','DecisionAware']:
                for seed in [17,29,43]:
                    name=f'f{f}_{kind}_{method}_s{seed}'
                    directory=PRIVATE/'c'/name
                    receipt=json.loads((directory/'FROZEN_PATH.json').read_text())
                    assert receipt['record_sha256']==sha(directory/'PATH_RECORDS.npz')
                    assert receipt['steps']==60 and receipt['checkpoints']==[0,20,40,60]
                    assert receipt['E_rewards_used_for_any_purchase'] is False
                    for filename,digest in receipt['snapshots'].items():
                        assert sha(directory/filename)==digest
                        snapshot_checks+=1
                    logs=json.loads((directory/'ACCESS_AND_COST_LOG.json').read_text())
                    with np.load(directory/'PATH_RECORDS.npz') as a:
                        paid=a['purchased']
                    assert len(set(paid))==60 and set(paid)<=set(role['U'])
                    assert not set(paid)&set(role['E'])
                    access=logs['access_log']
                    assert access[0]['role']=='historical_initialization'
                    assert access[0]['questions']==role['I'].tolist()
                    legal=[row for row in access if row['role']=='paid_feedback']
                    assert [row['question'] for row in legal]==paid.tolist()
                    assert len(access)==61
                    for q,row in zip(paid,logs['ledger']):
                        expected=dict(gold=1,member_calls=18,aggregator_calls=60,scoring_ops=60,
                            input_tokens=float(source['member_prompt_tokens'][:,q].sum()+source['aggregate_prompt_tokens'][:,q].sum()),
                            output_tokens=float(source['member_completion_tokens'][:,q].sum()+source['aggregate_completion_tokens'][:,q].sum()),
                            source_model_wall_s=float(source['member_wall_time_s'][:,q].sum()+source['aggregate_wall_time_s'][:,q].sum()))
                        expected['tokens']=expected['input_tokens']+expected['output_tokens']
                        for key,value in expected.items():
                            close(value,row[key],'actual paid packet '+key,tolerance=1e-8)
                        assert row['tokens']<=token_per_package
                        package_checks+=1
                    all_paid.extend(paid.tolist())
    assert package_checks==5400 and snapshot_checks==360
    decision=(PUBLIC/'05_E7_RESEARCH_DECISION.md').read_text('utf-8')
    for sentence in ['好组合有没有进入可选择范围？','只看题目，能不能学到哪些组合更好？','在同一个学习器下，我们的采集是否比随机反馈更有效？']:
        assert sentence in decision
    for phrase in ['45.6853','1.6111','0.7333','85.4259','85.3889','86.0000']:
        assert phrase in decision
    assert '独立' in decision and '开发' in decision
    legacy=[]
    e1=ROOT/'release/e1_20261103/github_payload'
    e1manifest=json.loads((e1/'release_manifest.json').read_text())
    for item in e1manifest['files']:
        assert sha(e1/item['path'])==item['sha256'],('E1',item['path'])
    legacy.append(dict(stage='E1',verified_records=len(e1manifest['files']),scope='E1 immutable delivery snapshot'))
    e2=ROOT/'release/e2_20261104'
    e2manifest=json.loads((e2/'DELIVERY_MANIFEST.json').read_text())
    e2count=0
    for item in e2manifest['files']:
        prefix='routing_idea2/release/e2_20261104/'
        if item['path'].startswith(prefix):
            assert sha(e2/item['path'][len(prefix):])==item['sha256']
            e2count+=1
    legacy.append(dict(stage='E2',verified_records=e2count,scope='Three locally available published result/index records',
        excluded_manifest_paths=['README.md','routing_idea2/README.md'],
        reason='These two historical Git-only README versions are not present in this local release snapshot; base tree inheritance is checked by publisher'))
    for name,subfolder,manifest_name,key in [
        ('E3','decision_acquisition_e3','PUBLICATION_MANIFEST.json','publication_files'),
        ('E4','e4_statistical_model_audit','PUBLIC_DELIVERY_MANIFEST.json','files'),
        ('E5','e5_paired_gain_validation','PUBLIC_DELIVERY_MANIFEST.json','files'),
        ('E6','e6_contrast_directed_feedback','PUBLIC_DELIVERY_MANIFEST.json','files')]:
        folder=ROOT/'research'/subfolder/'public'
        manifest=json.loads((folder/manifest_name).read_text())
        for filename,item in manifest[key].items():
            assert sha(folder/filename)==item['sha256'],(name,filename)
        legacy.append(dict(stage=name,verified_records=len(manifest[key]),scope='Prior public manifest records'))
    excluded={'E7_FINAL_REVIEW.json','PUBLIC_DELIVERY_MANIFEST.json','E7_DELIVERY_AUDIT.json'}
    files=[path for path in PUBLIC.rglob('*') if path.is_file()]
    compile_count=0
    private_ids=list(map(str,data['ids']))+list(map(str,data['canonical']))
    leaks=[]
    fitted=[]
    secrets=[r'\bgh[pousr]_[0-9A-Za-z]{20,}\b',r'github_pat_[0-9A-Za-z_]{20,}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    def fitted_parameters(value,trail=''):
        if isinstance(value,dict):
            for key,item in value.items():
                if key.lower() in ['beta','coefficient','coefficients','pca_components','pca_mean','scaler_mean','scaler_scale','covariance','offset'] and isinstance(item,list):
                    fitted.append(trail+'/'+key)
                fitted_parameters(item,trail+'/'+key)
        elif isinstance(value,list):
            for item in value:
                fitted_parameters(item,trail+'[]')
    for path in files:
        relative=path.relative_to(PUBLIC).as_posix()
        assert path.suffix in ['.py','.md','.json'],relative
        text=path.read_text('utf-8-sig')
        if path.suffix=='.py':
            compile(text,str(path),'exec')
            compile_count+=1
        if any(identifier in text for identifier in private_ids if len(identifier)>=8):
            leaks.append(relative+' private-question-ID')
        if any(re.search(pattern,text) for pattern in secrets):
            leaks.append(relative+' credential-pattern')
        if path.suffix=='.json':
            fitted_parameters(json.loads(text),relative)
    assert not leaks,leaks
    assert not fitted,fitted
    stage_audits=['a/E7_A_INDEPENDENT_AUDIT.json','b/E7_B_INDEPENDENT_AUDIT.json',
        'c/E7_C_REPORT_AUDIT.json','theory/ACTIVE_FEEDBACK_OPERATOR_ACCESS_AUDIT.json','theory/ACTIVE_FEEDBACK_STATISTICS_AUDIT.json']
    assert all(json.loads((PUBLIC/name).read_text())['status']=='PASS' for name in stage_audits)
    result=dict(status='PASS',scalar_comparisons=int(checks),maximum_absolute_error=max_error,
        original_evaluation_questions=600,independent_primary_comparisons=4,bootstrap_repetitions=5000,
        all_generation_and_algorithm_repeats_kept_inside_original_question=True,
        formal_feedback_paths=90,paid_packet_ledgers_rebuilt=package_checks,model_snapshot_hashes_verified=snapshot_checks,
        packet_gold=1,packet_member_calls=18,packet_aggregator_calls=60,packet_Token_hard_upper_bound=602112,
        unique_paid_pool_questions_across_counterfactual_paths=len(set(all_paid)),
        physical_new_LLM_calls=0,legacy_manifest_coverage=legacy,
        private_identifiers_checked=len(private_ids),privacy_matches=leaks,real_model_parameters_public=fitted,
        python_sources_compiled=compile_count,public_files_at_review=len(files),
        reviewed_file_sha256={path.relative_to(PUBLIC).as_posix():sha(path) for path in files if path.name not in excluded},
        hash_exclusions=sorted(excluded),pending=[],
        multiple_testing='Unified Holm4 and separate Bonferroni98.75 intervals rebuilt from complete original-question records',
        conclusions='Opportunity loss both before candidate gate and within it; exploratory independent-supervision growth, no family-supported within-task content gain or acquisition benefit',
        boundary='Existing train600 development, overlapping training histories, no independent confirmation. Legacy checks cover available prior manifest records, not full OS write forensics.')
    put(PUBLIC/'E7_FINAL_REVIEW.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='reviewed_file_sha256'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
