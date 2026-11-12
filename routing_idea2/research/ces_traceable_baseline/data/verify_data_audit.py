"""Verify CES data-audit facts without model calls or new downloads."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

WORK=Path(r'E:\语言模型组合路由\research/ces_traceable_baseline')
PUBLIC=WORK/'public'
PRIVATE=WORK/'private/public_assets'


def main():
    archive=PRIVATE/'CES-v1.0.0-kdd2026.zip'
    raw=archive.read_bytes()
    assert len(raw)==39524383
    assert hashlib.md5(raw).hexdigest()=='ba087d464f99dbefcebeb618b6b0b413'
    tag=json.loads((PRIVATE/'TAG_GIT_TREE_RAW.json').read_text())
    official={r['path']:r['sha'] for r in tag['tree'] if r['type']=='blob'}
    checks=0
    with zipfile.ZipFile(archive) as a:
        for info in a.infolist():
            if info.is_dir():
                continue
            p='/'.join(info.filename.split('/')[1:])
            b=a.read(info)
            assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==official[p]
            checks+=1
    assert checks==925
    profile=json.loads((PRIVATE/'archive_core/artifacts/mmlu_meta_10d.json').read_text())
    recomputed=json.loads((PUBLIC/'CES_CAPABILITY_AGGREGATE_RECHECK.json').read_text())
    for row in recomputed['rows']:
        p=profile['models'][row['name']]
        correct=np.array(p['cap_correct'],float)
        total=np.array(p['cap_total'],float)
        assert np.allclose(correct/total,row['recomputed_cap_accuracy'],atol=1e-15,rtol=0)
        assert abs(correct.sum()/total.sum()-row['recomputed_overall'])<1e-15
    assert recomputed['consistent_models']==14
    invalid=[r for r in recomputed['rows'] if not r['correct_fields_are_integer_counts']]
    assert len(invalid)==1 and invalid[0]['name']=='qwen-2.5-32b-instruct'
    assert recomputed['status']=='REAGGREGATED_WITH_DISCREPANCIES'
    result=dict(status='PASS',archive_git_blob_matches=checks,profile_models_recomputed=15,
        arithmetically_consistent_profiles=14,known_invalid_count_profile_preserved=True,
        public_asset_audit_sha256=hashlib.sha256((PUBLIC/'CES_PUBLIC_ASSET_AUDIT.json').read_bytes()).hexdigest(),
        source_profile_sha256=hashlib.sha256((PRIVATE/'archive_core/artifacts/mmlu_meta_10d.json').read_bytes()).hexdigest(),
        audit_note='PASS verifies the audit faithfully reports the original discrepancy; it does not certify the upstream profile or original CES main metrics.',
        model_API_calls=0)
    (PUBLIC/'CES_DATA_AUDIT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
