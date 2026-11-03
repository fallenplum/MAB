"""Download immutable public six-model pool into the authorized experiment root."""
from pathlib import Path
import argparse, hashlib, json, time, sys
from huggingface_hub import HfApi, snapshot_download
from download_inference_models import PATTERNS
ROOT=Path(__file__).resolve().parents[1]

def download(entry):
    alias,repo,revision=entry['alias'],entry['repo_id'],entry['revision']
    api=HfApi(); info=api.model_info(repo,revision=revision,files_metadata=True)
    if info.sha!=revision: raise ValueError('Pinned revision mismatch')
    dest=ROOT/'models'/repo.replace('/','--')
    print(json.dumps({'event':'download_start','alias':alias,'repo':repo,'revision':revision}),flush=True)
    snapshot_download(repo_id=repo,revision=revision,allow_patterns=PATTERNS,
                      local_dir=dest,cache_dir=ROOT/'.cache/huggingface',max_workers=2)
    files=[]
    for path in sorted(dest.rglob('*')):
        if not path.is_file() or '.cache' in path.parts: continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''): h.update(block)
        source=next((x for x in info.siblings if x.rfilename==path.relative_to(dest).as_posix()),None)
        lfs=getattr(source,'lfs',None) if source else None
        expected=getattr(lfs,'sha256',None) if lfs else None
        if expected and h.hexdigest()!=expected:raise ValueError('Pinned LFS file hash mismatch: '+path.name)
        files.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':h.hexdigest(),'lfs_sha256':expected})
    receipt={'role':'member','model_alias':alias,'repo_id':repo,'revision':revision,'path':str(dest),'files':files,'completed_at_unix':time.time()}
    target=ROOT/'artifacts/audit'/f'member_{alias.lower()}_model_download.json'
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'event':'download_complete','alias':alias,'files':len(files),'bytes':sum(x['bytes'] for x in files)}),flush=True)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--alias',default='all');a=p.parse_args()
    pool=json.loads((ROOT/'artifacts/audit/candidate_model_identities.json').read_text('utf-8-sig'))
    selected=[x for x in pool if a.alias in ('all',x['alias'])]
    if not selected:raise ValueError('Alias not in preregistered candidate pool')
    for entry in selected:download(entry)
if __name__=='__main__':main()
