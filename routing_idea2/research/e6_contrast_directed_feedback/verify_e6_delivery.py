"""Verify the complete public delivery and its private frozen artifacts."""
import os
from pathlib import Path
import hashlib
import json
import sys
import re
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e6_contrast_directed_feedback'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
EXCLUDED={'PUBLIC_DELIVERY_MANIFEST.json','E6_DELIVERY_AUDIT.json'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record(p):
    raw=p.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                git_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def main():
    failures=[]
    checks=0
    def require(condition,message):
        nonlocal checks
        checks+=1
        if not condition:
            failures.append(message)
    required=['EXPERIMENT_SUMMARY.md','RELATED_WORK_MAPPING.md','THEORY_AND_NUMERICAL_CHECKS.md',
        'E6_RUN_CONFIG.json','E6_COMPLETE_RESULTS.json','E6_PRIMARY_STATISTICS.json','04_PRIMARY_STATISTICAL_ANALYSIS.md',
        'REPRODUCE.md','E6_IMPLEMENTATION_DECISIONS.json','E6_PROVENANCE_AND_COST.json','E6_FINAL_REVIEW.json',
        'a/A_RESULTS.json','b/E6_B_RESULTS.json','c/E6_C_RESULTS.json',
        'a/01_SWITCH_INFORMATION_AND_SUPERVISION.md','b/E6_B_SYNTHETIC_ALLOCATION.md',
        'c/03_REAL_CACHE_POLICY_IDENTIFICATION.md']
    for name in required:
        require((PUBLIC/name).is_file(),'missing '+name)
    manifest=json.loads((PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json').read_text())
    actual={p.relative_to(PUBLIC).as_posix():p for p in PUBLIC.rglob('*') if p.is_file() and p.name not in EXCLUDED}
    require(set(actual)==set(manifest['files']),'manifest exact file set')
    compile_count=0
    for name,path in actual.items():
        require(record(path)==manifest['files'].get(name), 'hash '+name)
        if path.suffix=='.py':
            try:
                compile(path.read_text('utf-8-sig'),str(path),'exec')
                compile_count+=1
            except Exception as exc:
                failures.append('compile '+name+' '+str(exc))
    config=json.loads((PUBLIC/'E6_RUN_CONFIG.json').read_text())
    frozen=json.loads((PRIVATE/'PROTOCOL_FROZEN.json').read_text())
    require(frozen['config_sha256']==sha(PUBLIC/'E6_RUN_CONFIG.json'),'config freeze hash')
    require(frozen['split_sha256']==sha(PRIVATE/'E6_FGE_PQ_FROZEN.npz'),'split freeze hash')
    require(config['statistics']['primary_comparisons']==['A3 minus task-matched random','C Contrast minus Marginal at gold36','C Contrast minus Uniform at gold36'],'primary comparisons frozen')
    for name in ['a/A_RESULTS.json','b/E6_B_RESULTS.json','c/E6_C_RESULTS.json','E6_COMPLETE_RESULTS.json']:
        require(json.loads((PUBLIC/name).read_text())['status']=='COMPLETE','stage status '+name)
    audits=['b/E6_B_AUDIT.json','theory/INDEPENDENT_OPERATOR_AUDIT.json','theory/INDEPENDENT_A_STATISTICS_AUDIT.json',
            'theory/INDEPENDENT_REAL_POOL_AUDIT.json','theory/INDEPENDENT_SOLVER_AUDIT.json','E6_FINAL_REVIEW.json']
    for name in audits:
        require(json.loads((PUBLIC/name).read_text())['status']=='PASS','independent audit '+name)
    inherited=ROOT/'research/e5_paired_gain_validation/public'
    old_manifest=json.loads((inherited/'PUBLIC_DELIVERY_MANIFEST.json').read_text())
    for name,ref in old_manifest['files'].items():
        require(sha(inherited/name)==ref['sha256'],'E5 preserved '+name)
    old_count=len(old_manifest['files'])
    sys.path.insert(0,str(inherited))
    from e5_common import load_train
    data,_=load_train()
    # Match actual private identifiers. Public SHA provenance alone does not
    # reveal question identities and is not a generic long-hex false positive.
    identifiers=list(map(str,data['ids']))+list(map(str,data['canonical']))
    leaks=[]
    secret_patterns=[r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',
                     r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----']
    for name,path in actual.items():
        text=path.read_text('utf-8-sig')
        if any(identifier in text for identifier in identifiers if len(identifier)>=8):
            leaks.append(name+' question identifier')
        if any(re.search(pattern,text) for pattern in secret_patterns):
            leaks.append(name+' secret pattern')
    require(not leaks,'privacy scan '+str(leaks))
    allP=[]
    allQ=[]
    decisions=0
    maxHT=0.
    maxcalls=0.
    for f in range(5):
        raw_path=PRIVATE/f'c/C_f{f}.npz'
        receipt=json.loads((PRIVATE/f'c/DECISIONS_C_f{f}.json').read_text())
        require(sha(raw_path)==receipt['decision_sha256'],'C frozen receipt '+str(f))
        with np.load(raw_path) as a:
            masks=a['masksP']
            purchase=a['purchases']
            pi=a['inclusion']
            estimates=np.einsum('bmri,ji,i->bmrj',purchase/pi,masks,a['DP'].mean(axis=1))/60
            error=float(np.max(abs(estimates-a['estimates'])))
            maxHT=max(maxHT,error)
            require(error<1e-12,'HT '+str(f))
            selected=np.argmax(estimates>=estimates.max(axis=-1,keepdims=True)-1e-12,axis=-1)
            require(np.array_equal(selected,a['selected']),'chosen '+str(f))
            require(np.array_equal(selected[:,1],selected[:,2]),'objective identity '+str(f))
            decisions+=selected.size
            for bi,budget in enumerate(a['budgets']):
                require(np.all(purchase[bi].sum(axis=-1)==budget),'gold count '+str((f,budget)))
            allP.extend(a['P'].tolist())
            allQ.extend(a['Q'].tolist())
    require(len(set(allP))==300 and len(set(allQ))==300 and not set(allP)&set(allQ),'distinct P/Q')
    stats=json.loads((PUBLIC/'E6_PRIMARY_STATISTICS.json').read_text())
    p=np.array([row['centered_bootstrap_p'] for row in stats['main_comparisons']])
    order=np.argsort(p)
    previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(p)-rank)*p[index]))
        require(abs(previous-stats['main_comparisons'][index]['Holm_p'])<1e-15,'Holm '+str(index))
    require(all(not r['positive_evidence_after_family_control'] for r in stats['main_comparisons']),'scientific outcome')
    payload=dict(status='PASS' if not failures else 'FAIL',scalar_or_group_assertions=checks,
        python_sources_compiled=compile_count,manifest_files_verified=len(actual),
        E5_manifest_records_unchanged=old_count,independently_rebuilt_C_choices=decisions,
        maximum_HT_reconstruction_error=maxHT,private_question_identifiers_scanned=len(identifiers),
        privacy_failures=leaks,failures=failures,
        independent_audit_sha256={name:sha(PUBLIC/name) for name in audits},
        manifest_sha256=sha(PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json'),
        scientific_gate='NO_POSITIVE_PRIMARY_FAMILY_EVIDENCE',work_status='COMPLETE',
        separation='A passed numerical audit does not establish a scientific benefit',
        outstanding=[])
    (PUBLIC/'E6_DELIVERY_AUDIT.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()
