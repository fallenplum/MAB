"""Final E7 file, provenance, decision and private-identifier verification."""
import os
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np

ROOT=Path(os.environ.get('MAB_ASSET_ROOT',r'E:\语言模型组合路由'))
WORK=ROOT/'research/e7_opportunity_and_feedback_learning'
PUBLIC=WORK/'public'
PRIVATE=WORK/'private'
EXCLUDE={'PUBLIC_DELIVERY_MANIFEST.json','E7_DELIVERY_AUDIT.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    raw=path.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        git_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def main():
    checks=0
    errors=[]
    def require(condition,label):
        nonlocal checks
        checks+=1
        if not condition:
            errors.append(label)
    required=['01_CANDIDATE_OPPORTUNITY.md','02_INPUT_SIGNAL_AND_LEARNING_CURVES.md','03_ACTIVE_FEEDBACK_LEARNING.md',
        '04_COST_AND_STATISTICS.md','05_E7_RESEARCH_DECISION.md','RELATED_WORK_MAPPING.md','E7_PROTOCOL.json',
        'E7_COMPLETE_RESULTS.json','REPRODUCE.md','E7_FINAL_REVIEW.json']
    for name in required:
        require((PUBLIC/name).is_file(),'required '+name)
    manifest=json.loads((PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json').read_text())
    current={p.relative_to(PUBLIC).as_posix():p for p in PUBLIC.rglob('*') if p.is_file() and p.name not in EXCLUDE}
    require(set(current)==set(manifest['files']),'exact manifest set')
    compiled=0
    for name,path in current.items():
        require(file_record(path)==manifest['files'].get(name),'hash '+name)
        if path.suffix=='.py':
            try:
                compile(path.read_text('utf-8-sig'),str(path),'exec')
                compiled+=1
            except Exception as exc:
                errors.append('compile '+name+' '+str(exc))
    config=json.loads((PUBLIC/'E7_PROTOCOL.json').read_text())
    frozen=json.loads((PRIVATE/'PROTOCOL_FROZEN.json').read_text())
    require(sha(PUBLIC/'E7_PROTOCOL.json')==frozen['protocol_sha256'],'protocol freeze')
    require(sha(PRIVATE/'E7_SPLITS_FROZEN.npz')==frozen['split_sha256'],'role freeze')
    for p in ['a/E7_A_RESULTS.json','b/E7_B_RESULTS.json','c/E7_C_RESULTS.json','E7_COMPLETE_RESULTS.json']:
        require(json.loads((PUBLIC/p).read_text())['status']=='COMPLETE','stage '+p)
    audits=['a/E7_A_INDEPENDENT_AUDIT.json','b/E7_B_INDEPENDENT_AUDIT.json','c/E7_C_REPORT_AUDIT.json',
        'theory/WORKING_POSTERIOR_OPERATOR_CHECKS.json','theory/ACTUAL_MODEL_OPERATOR_AUDIT.json',
        'theory/CANDIDATE_OPPORTUNITY_AUDIT.json','theory/ACTIVE_FEEDBACK_OPERATOR_ACCESS_AUDIT.json',
        'theory/ACTIVE_FEEDBACK_STATISTICS_AUDIT.json','E7_FINAL_REVIEW.json']
    for p in audits:
        require(json.loads((PUBLIC/p).read_text())['status']=='PASS','audit '+p)
    old_records=0
    for branch in ['e5_paired_gain_validation','e6_contrast_directed_feedback']:
        directory=ROOT/'research'/branch/'public'
        old=json.loads((directory/'PUBLIC_DELIVERY_MANIFEST.json').read_text())
        for p,record in old['files'].items():
            require(sha(directory/p)==record['sha256'],'old retained '+branch+'/'+p)
            old_records+=1
    sys.path.insert(0,str(PUBLIC))
    sys.path.insert(0,str(PUBLIC/'b'))
    from e7_common import load_train,splits,holm
    from e7_models import argmax
    data,_=load_train()
    identifiers=list(map(str,data['ids']))+list(map(str,data['canonical']))
    matches=[]
    for name,path in current.items():
        value=path.read_text('utf-8-sig')
        if any(identifier in value for identifier in identifiers if len(identifier)>=8):
            matches.append(name+' actual private identifier')
        if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',value):
            matches.append(name+' secret pattern')
    require(not matches,'privacy')
    paths=list((PRIVATE/'c').glob('f*_*/FROZEN_PATH.json'))
    require(len(paths)==90,'formal path count')
    snapshots=0
    predictions=0
    purchases=0
    for path in paths:
        receipt=json.loads(path.read_text())
        folder=path.parent
        require(receipt['status']=='FROZEN_BEFORE_E_REWARD_EVALUATION','path freeze '+folder.name)
        require(receipt['protocol_sha256']==sha(PUBLIC/'E7_PROTOCOL.json'),'path protocol '+folder.name)
        require(receipt['record_sha256']==sha(folder/'PATH_RECORDS.npz'),'path record '+folder.name)
        for name,expected in receipt['snapshots'].items():
            require(sha(folder/name)==expected,'snapshot '+folder.name+'/'+name)
            snapshots+=1
        with np.load(folder/'PATH_RECORDS.npz') as a:
            bought=a['purchased']
            require(len(bought)==len(set(bought))==60 and set(bought)<=set(a['U']),'purchase permissions '+folder.name)
            require(not set(bought)&set(a['E']),'E not purchased '+folder.name)
            require(a['E_predictions'].shape==(4,120,20),'prediction shape '+folder.name)
            require(argmax(a['E_predictions']).shape==(4,120),'argmax shape '+folder.name)
            predictions+=4*120
            purchases+=len(bought)
    stats=json.loads((PUBLIC/'E7_PRIMARY_STATISTICS.json').read_text())
    require(len(stats['main_comparisons'])==4,'primary family')
    actual=holm([r['centered_bootstrap_p'] for r in stats['main_comparisons']])
    for r,p in zip(stats['main_comparisons'],actual):
        require(abs(r['Holm_p']-p)<1e-15,'Holm '+r['comparison'])
    require(not any(r['positive_evidence_after_family_control'] for r in stats['main_comparisons']),'scientific outcome retained')
    split_ids=[]
    for f in range(5):
        roles=splits(f)
        require(len(roles['I'])==len(roles['U'])==240 and len(roles['E'])==120,'role sizes '+str(f))
        require(len(set(roles['I'])|set(roles['U'])|set(roles['E']))==600,'disjoint roles '+str(f))
        split_ids.extend(roles['E'].tolist())
    require(len(set(split_ids))==600,'E original questions')
    result=dict(status='PASS' if not errors else 'FAIL',checks=checks,python_files_compiled=compiled,
        public_manifest_records=len(current),old_E5_E6_hash_records_preserved=old_records,
        formal_paths=90,snapshots=snapshots,evaluation_decisions=predictions,
        feedback_purchase_events=purchases,distinct_original_questions=600,
        new_physical_LLM_calls=0,privacy_matches=matches,errors=errors,pending=[],
        manifest_sha256=sha(PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json'),
        audit_sha256={p:sha(PUBLIC/p) for p in audits},
        work_status='COMPLETE',science_status='NO_FAMILY_SUPPORTED_POSITIVE_PRIMARY_CONTRAST',
        scope='Artifact integrity and numerical audits are separate from scientific benefit')
    (PUBLIC/'E7_DELIVERY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
