"""Fetch official Git metadata and cross-check the complete Zenodo tree."""
from pathlib import Path
import json
import hashlib
import requests
import zipfile

WORK=Path(r'E:\语言模型组合路由\research/ces_traceable_baseline')
PRIVATE=WORK/'private/public_assets'


def get(url,name):
    response=requests.get(url,timeout=60,headers={'User-Agent':'MAB-CES-reproducibility-audit/1.0'})
    response.raise_for_status()
    raw=response.content
    (PRIVATE/name).write_bytes(raw)
    return response.json(),dict(url=url,path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    repo,repo_ref=get('https://api.github.com/repos/JinkunXu/CES','REPO_METADATA_RAW.json')
    current,current_ref=get('https://api.github.com/repos/JinkunXu/CES/branches/master','MASTER_REF_RAW.json')
    release,release_ref=get('https://api.github.com/repos/JinkunXu/CES/git/ref/tags/v1.0.0-kdd2026','TAG_REF_RAW.json')
    assert current['commit']['sha']=='9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7'
    assert release['object']['sha']=='778ffd9e24ac7e2ee0a3790bc2a52711296dd59f'
    master,master_ref=get('https://api.github.com/repos/JinkunXu/CES/git/trees/1451ea798bd66b75b9b451ab56a322a5576a58e4?recursive=1','MASTER_GIT_TREE_RAW.json')
    tag,tag_ref=get('https://api.github.com/repos/JinkunXu/CES/git/trees/778ffd9e24ac7e2ee0a3790bc2a52711296dd59f?recursive=1','TAG_GIT_TREE_RAW.json')
    assert not master['truncated'] and not tag['truncated']
    tm={row['path']:row for row in tag['tree'] if row['type']=='blob'}
    mm={row['path']:row for row in master['tree'] if row['type']=='blob'}
    matches=[]
    with zipfile.ZipFile(PRIVATE/'CES-v1.0.0-kdd2026.zip') as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative='/'.join(member.filename.split('/')[1:])
            raw=archive.read(member)
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            assert relative in tm and tm[relative]['sha']==blob
            matches.append(relative)
    assert len(matches)==len(tm)==925
    diff=[dict(path=p,tag_blob=tm[p]['sha'],master_blob=mm[p]['sha']) for p in tm if p in mm and tm[p]['sha']!=mm[p]['sha']]
    result=dict(status='PASS',master_commit='9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7',
        master_tree='1451ea798bd66b75b9b451ab56a322a5576a58e4',tag_commit='778ffd9e24ac7e2ee0a3790bc2a52711296dd59f',
        master_blobs=len(mm),tag_blobs=len(tm),archive_blob_matches=len(matches),
        modified_tag_files=diff,master_only_files=sorted(set(mm)-set(tm)),tag_only_files=sorted(set(tm)-set(mm)),
        fetched_metadata=[repo_ref,current_ref,release_ref,master_ref,tag_ref],no_truncated_tree=True,physical_model_API_calls=0)
    (PRIVATE/'SOURCE_TREE_ARCHIVE_CROSSCHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
