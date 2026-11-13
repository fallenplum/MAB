"""Verify or replay E8 evaluation into a new private output directory."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import run_e8_evaluation as evaluator

BASE=evaluator.BASE


def verify():
    locks=['E8_SOURCE_LOCK.json','E8_RECOVERY_SOURCE_LOCK.json','E8_EVALUATION_SOURCE_LOCK.json']
    files={}
    for filename in locks:
        value=evaluator.read(BASE/'private'/filename)
        for path,digest in value['files'].items():
            if evaluator.sha(path)!=digest:raise AssertionError('Frozen source mismatch '+path)
            files[path]=digest
    result_lock=BASE/'private/E8_RESULT_FREEZE.json'
    if result_lock.exists():
        for relative,digest in evaluator.read(result_lock)['files'].items():
            if evaluator.sha(BASE/'public'/relative)!=digest:raise AssertionError('Frozen result mismatch '+relative)
    index=evaluator.read(BASE/'private/training/E8_FROZEN_TRAINING_INDEX.json')
    count=0
    for fold in index['folds']:
        for name,receipt in fold['variants'].items():
            folder=BASE/'private/training'/('fold'+str(fold['fold']))/name
            for filename,digest in receipt['artifact_hashes'].items():
                if evaluator.sha(folder/filename)!=digest:raise AssertionError('Frozen fit artifact mismatch')
            count+=1
    if not index['complete'] or count!=70:raise AssertionError('Incomplete frozen training')
    return dict(status='PASS',frozen_models=count,unique_locked_source_files=len(files),
        protocol_sha256=evaluator.sha(BASE/'public/E8_PROTOCOL.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['verify','evaluate-replay'])
    args=parser.parse_args();audit=verify()
    if args.command=='verify':print(json.dumps(audit));return
    parent=BASE/'private/evaluation_reproductions'
    destination=parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if destination.exists() or not destination.resolve().is_relative_to(parent.resolve()):
        raise FileExistsError('Replay requires a new private reproduction directory')
    destination.mkdir(parents=True)
    shutil.copytree(BASE/'public',destination/'public')
    shutil.copytree(BASE/'private/training',destination/'private/training')
    shutil.copytree(BASE/'private/qualification',destination/'private/qualification')
    for filename in ['E8_SOURCE_LOCK.json','E8_RECOVERY_SOURCE_LOCK.json','E8_EVALUATION_SOURCE_LOCK.json']:
        shutil.copy2(BASE/'private'/filename,destination/'private'/filename)
    reference=destination/'private/ORIGINAL_COMPLETE_RESULTS.json'
    (destination/'public/E8_COMPLETE_RESULTS.json').replace(reference)
    evaluator.BASE=destination
    evaluator.main()
    expected=evaluator.read(reference);actual=evaluator.read(destination/'public/E8_COMPLETE_RESULTS.json')
    expected.pop('evaluation_wall_seconds');actual.pop('evaluation_wall_seconds')
    if expected!=actual:raise AssertionError('Replayed complete metrics differ')
    if evaluator.read(BASE/'public/E8_PRIMARY_STATISTICS.json')!=evaluator.read(destination/'public/E8_PRIMARY_STATISTICS.json'):
        raise AssertionError('Replayed primary statistics differ')
    evaluator.put(destination/'REPRODUCTION_AUDIT.json',dict(status='PASS',
        original_outputs_overwritten=False,all14_metrics_identical=True,
        all_primary_statistics_identical=True,models_refitted=False,
        original_source_and_prediction_hashes_verified=True))
    print(json.dumps(dict(status='PASS_E8_REPRODUCED',directory=str(destination))))


if __name__=='__main__':main()
