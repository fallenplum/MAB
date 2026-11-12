"""Verify CES traceability delivery and the frozen E7 baseline."""
import os
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
EXCLUDE={'PUBLIC_DELIVERY_MANIFEST.json','CES_DELIVERY_AUDIT.json'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record(p):
    raw=p.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        git_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())


def main():
    checks=0
    errors=[]
    def require(value,label):
        nonlocal checks
        checks+=1
        if not value:
            errors.append(label)
    expected=['01_NATIVE_REPRODUCIBILITY_AUDIT.md','02_DATA_FEEDBACK_AND_COST_PERMISSIONS.md',
        '03_ADAPTATION_CONTRACT_AND_OPERATOR_AUDIT.md','04_MIGRATION_RESULTS_AND_COST.md',
        '05_EVIDENCE_CHAIN_AND_NEXT_STEPS.md','CES_PROTOCOL.json','CES_COMPLETE_RESULTS.json',
        'REPRODUCE.md','CES_FINAL_REVIEW.json']
    for name in expected:
        require((PUBLIC/name).is_file(),'required '+name)
    manifest=json.loads((PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json').read_text())
    current={p.relative_to(PUBLIC).as_posix():p for p in PUBLIC.rglob('*') if p.is_file() and p.name not in EXCLUDE}
    require(set(current)==set(manifest['files']),'exact manifest coverage')
    python_count=0
    for name,path in current.items():
        require(record(path)==manifest['files'].get(name),'public hash '+name)
        if path.suffix=='.py':
            try:
                compile(path.read_text('utf-8-sig'),str(path),'exec')
                python_count+=1
            except Exception as exc:
                errors.append('compile '+name+' '+str(exc))
    config=json.loads((PUBLIC/'CES_PROTOCOL.json').read_text())
    frozen=json.loads((PRIVATE/'PROTOCOL_FROZEN.json').read_text())
    require(sha(PUBLIC/'CES_PROTOCOL.json')==frozen['protocol_sha256'],'protocol frozen')
    e7=ROOT/'research/e7_opportunity_and_feedback_learning'
    protected=json.loads((PUBLIC/'E7_FREEZE_RECEIPT.json').read_text())
    for name,digest in protected['files'].items():
        require(sha(e7/name)==digest,'E7 freeze '+name)
    prior=json.loads((e7/'public/PUBLIC_DELIVERY_MANIFEST.json').read_text())
    for name,entry in prior['files'].items():
        require(sha(e7/'public'/name)==entry['sha256'],'E7 old manifest '+name)
    audits=['NATIVE_OPERATOR_QUALIFICATION.json','CES_DATA_AUDIT_VERIFICATION.json',
        'adapter/CES_ADAPTER_OPERATOR_CHECKS.json','adapter/CES_PROFILE_SOURCE_AUDIT.json',
        'adapter/CES_MIGRATION_OPERATOR_ACCESS_AUDIT.json','CES_FINAL_REVIEW.json']
    for name in audits:
        require(json.loads((PUBLIC/name).read_text())['status']=='PASS','audit '+name)
    complete=json.loads((PUBLIC/'CES_COMPLETE_RESULTS.json').read_text())
    require(complete['status']=='COMPLETE_FOR_AUDIT_AND_CACHE_TRANSFER','bounded completion status')
    require(complete['native_benchmark']=='NOT_REPRODUCED_WITH_RELEASED_ARTIFACTS','native status retained')
    require(complete['extra_CES_I_side_information'] is True,'extra member profile disclosure')
    require(complete['physical_new_model_API_calls']==0,'new API count')
    paths=list((PRIVATE/'replay').glob('f*_*/FROZEN_PATH.json'))
    require(len(paths)==45,'path count')
    snapshots=0
    for path in paths:
        receipt=json.loads(path.read_text())
        folder=path.parent
        require(receipt['status']=='FROZEN_BEFORE_EVALUATION','path state '+folder.name)
        require(receipt['record_sha256']==sha(folder/'FROZEN_PATH.npz'),'path record '+folder.name)
        for name,digest in receipt['snapshot_sha256'].items():
            require(sha(folder/name)==digest,'snapshot '+folder.name+'/'+name)
            snapshots+=1
        with np.load(folder/'FROZEN_PATH.npz') as a:
            require(a['E_actions'].shape==a['E_mean_actions'].shape==(4,120),'action shape '+folder.name)
            require(len(set(a['purchased']))==60 and set(a['purchased'])<=set(a['U']),'legal purchases '+folder.name)
            require(not set(a['purchased'])&set(a['E']),'E disjoint '+folder.name)
    sys.path.insert(0,str(PUBLIC))
    from ces_common import load_train
    data,_=load_train()
    ids=list(map(str,data['ids']))+list(map(str,data['canonical']))
    leaks=[]
    for name,path in current.items():
        text=path.read_text('utf-8-sig')
        if any(identity in text for identity in ids if len(identity)>=8):
            leaks.append(name+' private question identifier')
        if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',text):
            leaks.append(name+' secret pattern')
    require(not leaks,'public privacy')
    result=dict(status='PASS' if not errors else 'FAIL',checks=checks,manifest_records=len(current),
        python_sources_compiled=python_count,frozen_E7_entries=len(protected['files']),
        E7_manifest_records_unchanged=len(prior['files']),formal_paths=len(paths),model_snapshots=snapshots,
        audits={name:sha(PUBLIC/name) for name in audits},privacy_leaks=leaks,errors=errors,pending=[],
        source_scope='Original algorithm operators verified, original paper main performance remains unreplicated',
        native_paper_status=complete['native_benchmark'],physical_new_model_API_calls=0,
        complete_scope=complete['status'],manifest_sha256=sha(PUBLIC/'PUBLIC_DELIVERY_MANIFEST.json'))
    (PUBLIC/'CES_DELIVERY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
