"""Finish all locked head/control grids with two CPU workers."""
from pathlib import Path
import concurrent.futures
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).parent/'code'))
from e9_common import ROOT, now, put, sha, verify_sources


def one(fold):
    folder=ROOT/'private/head_execution'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'fold{fold}.log'
    with path.open('x',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).parent/'code/e9_train.py'),
            '--fold',str(fold)],stdout=stream,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        raise RuntimeError('Head subprocess failed fold'+str(fold))
    return dict(fold=fold,log_sha256=sha(path))


def main():
    verify_sources('HEAD_SOURCE_LOCK.json')
    if not (ROOT/'private/models/fold0/FOLD_FROZEN_RECEIPT.json').exists():
        raise RuntimeError('Formal fold0 pilot must complete first')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(one,range(1,5)))
    put(ROOT/'private/head_execution/ALL_FOLDS_COMPLETE.json',dict(status='ALL_MODELS_AND_CONTROLS_FROZEN',
        completed_utc=now(),records=results,fold0_previously_complete=True,max_workers=2,
        source_sha256=sha(__file__)))
    print('ALL_HEADS_AND_CONTROLS_COMPLETE',flush=True)


if __name__=='__main__':main()
