"""Bounded retry for public immutable asset downloads, never model inference."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,uuid
ROOT=Path(__file__).resolve().parents[1]


def _state(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with temporary.open('w',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['aggregator','members'],required=True)
    p.add_argument('--attempts',type=int,default=10);a=p.parse_args(argv)
    if a.attempts<=0:raise ValueError('Positive bounded download retry count required')
    target=ROOT/'artifacts/audit'/f'download_{a.role}_status.json'
    command=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts/download_models_resumable.py'),'--role',a.role]
    env=os.environ.copy();env.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_DISABLE_XET='1',
        HF_HUB_DOWNLOAD_TIMEOUT='120',HF_HUB_ETAG_TIMEOUT='30')
    for attempt in range(1,a.attempts+1):
        state={'role':a.role,'status':'RUNNING','attempt':attempt,'wrapper_pid':os.getpid(),
               'updated_at_unix':time.time(),'immutable_revision_and_sha_verification':True,
               'resume_policy':'rehash completed receipts locally; reuse digest-anchored pinned metadata; persistent Range partials'}
        _state(target,state);print(json.dumps(state),flush=True)
        result=subprocess.run(command,cwd=ROOT,env=env)
        if result.returncode==0:
            state.update(status='COMPLETED',updated_at_unix=time.time());_state(target,state);return 0
        terminal=result.returncode in {3,4}
        state.update(status='FAILED' if terminal or attempt==a.attempts else 'RETRYING',
            last_child_exit=result.returncode,updated_at_unix=time.time(),
            failure_kind='asset_validation_or_lock' if terminal else 'download_or_network')
        _state(target,state)
        if terminal:return result.returncode
        if attempt<a.attempts:time.sleep(min(attempt*10,60))
    return 2

if __name__=='__main__':raise SystemExit(main())
