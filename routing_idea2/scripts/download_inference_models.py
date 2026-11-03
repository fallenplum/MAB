"""Download only public pinned inference assets; all cache and receipts stay in project root."""
from pathlib import Path
import argparse,hashlib,json,time
from huggingface_hub import HfApi,snapshot_download
ROOT=Path(__file__).resolve().parents[1]
SPECS=[
 ('aggregator','Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28'),
 ('encoder','sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41')]
PATTERNS=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
          'vocab.txt','vocab.json','merges.txt','model*.safetensors','model.safetensors.index.json','README.md','modules.json','1_Pooling/config.json']

def main():
 p=argparse.ArgumentParser();p.add_argument('--role',choices=['aggregator','encoder','all'],default='all');args=p.parse_args()
 api=HfApi()
 for role,repo,revision in SPECS:
  if args.role not in {'all',role}:continue
  info=api.model_info(repo,revision=revision,files_metadata=True)
  if info.sha!=revision:raise ValueError('Revision moved unexpectedly')
  dest=ROOT/'models'/repo.replace('/','--')
  print(json.dumps({'event':'download_start','role':role,'model':repo,'revision':revision,'destination':str(dest)}),flush=True)
  snapshot_download(repo_id=repo,revision=revision,allow_patterns=PATTERNS,
                    local_dir=dest,cache_dir=ROOT/'.cache/huggingface',max_workers=2)
  files=[]
  for f in sorted(dest.rglob('*')):
   if not f.is_file() or '.cache' in f.parts:continue
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
   metadata=next((s for s in info.siblings if s.rfilename==f.relative_to(dest).as_posix()),None)
   lfs=getattr(metadata,'lfs',None) if metadata else None
   expected=getattr(lfs,'sha256',None) if lfs else None
   if expected and h.hexdigest()!=expected:raise ValueError('Downloaded LFS hash mismatch: '+f.name)
   files.append({'path':f.relative_to(ROOT).as_posix(),'bytes':f.stat().st_size,'sha256':h.hexdigest(),'lfs_sha256':expected})
  receipt={'role':role,'repo_id':repo,'revision':revision,'path':str(dest),'files':files,'completed_at_unix':time.time()}
  target=ROOT/'artifacts/audit'/f'{role}_model_download.json';target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps({'event':'download_complete','role':role,'files':len(files),'bytes':sum(f['bytes'] for f in files)}),flush=True)
if __name__=='__main__':main()
