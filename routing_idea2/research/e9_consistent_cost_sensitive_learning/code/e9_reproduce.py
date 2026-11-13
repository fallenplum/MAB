"""Verify immutable E9 assets and reproduce scoring or full legal refitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import shutil
import sys
import time
import e9_common as common
from e9_common import ROOT, ASSETS, archive, put, read, sha


def new_directory(label):
    path=ROOT/'private/reproductions'/(label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    path.mkdir(parents=True,exist_ok=False)
    return path


def verify():
    snapshot=read(ROOT/'private/PARENT_SNAPSHOTS.json')
    for name,entries in snapshot.items():
        folder=ASSETS/'research'/name
        files={p.relative_to(folder).as_posix():p for p in folder.rglob('*') if p.is_file()}
        if set(files)!=set(entries):raise AssertionError('Historical parent inventory changed '+name)
        for key,digest in entries.items():
            if sha(files[key])!=digest:raise AssertionError('Historical parent bytes changed '+key)
    common.verify_sources('TEACHER_SOURCE_LOCK.json')
    common.verify_sources('HEAD_SOURCE_LOCK.json')
    protocol=read(ROOT/'private/PROTOCOL_FREEZE_RECEIPT.json')
    if protocol['protocol_sha256']!=sha(ROOT/'public/E9_PROTOCOL.json'):
        raise AssertionError('Immutable E9 protocol changed')
    import e9_evaluate
    _,_,logs,restorations,_=e9_evaluate.collect()
    for folder in (ROOT/'private/teachers').glob('fold*'):
        for receipt_path in folder.glob('*/FROZEN_RECEIPT.json'):
            receipt=read(receipt_path)
            if sha(receipt_path.parent/'TEACHER_MEAN.npz')!=receipt['artifact_sha256']:
                raise AssertionError('Teacher mean artifact changed')
            if sha(receipt_path.parent/'PRIOR_AND_SPLITS.npz')!=receipt['prior_artifact_sha256']:
                raise AssertionError('Teacher prior artifact changed')
    result=dict(status='PASS',historical_parent_files_unchanged={k:len(v) for k,v in snapshot.items()},
        restored_head_slots=len(restorations),actual_head_fits=sum(v['actual_head_fit'] for v in logs),
        derived_heads=sum(not v['actual_head_fit'] for v in logs),
        maximum_restoration_error=max(v['max_score_error'] for v in restorations),
        physical_LLM_calls=0,retained_outcome_reads=0)
    put(ROOT/'public/E9_ARCHIVE_VERIFICATION.json',result)
    return e9_evaluate,result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['verify','scoring-replay','full-refit'])
    args=parser.parse_args();evaluator,result=verify()
    if args.mode=='verify':print(result);return
    destination=new_directory(args.mode)
    start_cpu,start_wall=time.process_time(),time.perf_counter()
    if args.mode=='full-refit':
        import e9_teachers,e9_train
        for name in ('PARENT_SNAPSHOTS.json','PROTOCOL_FREEZE_RECEIPT.json','HEAD_SOURCE_LOCK.json',
                     'TEACHER_SOURCE_LOCK.json','EVALUATION_SOURCE_LOCK.json'):
            (destination/'private').mkdir(exist_ok=True)
            shutil.copy2(ROOT/'private'/name,destination/'private'/name)
        (destination/'public').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'public/E9_PROTOCOL.json',destination/'public/E9_PROTOCOL.json')
        # Relocate output directories only. Frozen operator bytes and old inputs remain fixed.
        common.ROOT=destination;e9_teachers.ROOT=destination;e9_train.ROOT=destination;evaluator.ROOT=destination
        for fold in range(5):
            e9_teachers.prepare_fold(fold,destination)
            e9_train.run_fold(fold)
    evaluator.main(destination)
    files=['E9_MAIN_RESULTS.json','E9_PRIMARY_STATISTICS.json','E9_CONTENT_CONTROLS.json']
    for name in files:
        expected,actual=read(ROOT/'public'/name),read(destination/'public'/name)
        if name=='E9_MAIN_RESULTS.json':
            expected.pop('model_restorations');actual.pop('model_restorations')
            if args.mode=='full-refit':
                expected.pop('computational_cost');actual.pop('computational_cost')
        if expected!=actual:raise AssertionError('Reproduced science differs '+name)
    receipt=dict(status='PASS_REPRODUCED',mode=args.mode,directory=str(destination),
        scientific_metrics_identical=True,models_refitted=args.mode=='full-refit',
        comparison_excludes='floating restoration diagnostics; full-refit also excludes new execution timing',
        cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.perf_counter()-start_wall,
        physical_LLM_calls=0,retained_outcome_reads=0)
    put(destination/'REPRODUCTION_VERIFIED.json',receipt)
    print(receipt)


if __name__=='__main__':main()
