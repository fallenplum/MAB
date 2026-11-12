"""Nonoverwriting replay entry point for the already frozen B1 experiment."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone

import run_b1
from audit_and_report import verify_main


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['verify','replay'])
    args=parser.parse_args()
    audit=verify_main()
    if args.command=='verify':
        print(json.dumps(audit),flush=True)
        return
    original=run_b1.BASE
    destination=original/'private/reproductions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing replay')
    destination.mkdir(parents=True)
    shutil.copytree(original/'public',destination/'public')
    shutil.copytree(original/'private/assets',destination/'private/assets')
    shutil.copytree(original/'private/design',destination/'private/design')
    run_b1.BASE=destination
    run_b1.evaluate()
    original_result=json.loads((original/'public/REAL_FIXED_POOL_RESULTS.json').read_text(encoding='utf-8'))
    replay_result=json.loads((destination/'public/REAL_FIXED_POOL_RESULTS.json').read_text(encoding='utf-8'))
    original_result.pop('wall_seconds')
    replay_result.pop('wall_seconds')
    if original_result!=replay_result:
        raise AssertionError('Replay metrics differ from the frozen original run')
    run_b1.put(destination/'REPRODUCTION_AUDIT.json',dict(status='PASS',
        all_27_cells_identical=True,original_artifacts_overwritten=False,
        frozen_source_and_design_verified_before_replay=True,
        original_directory=str(original),reproduction_directory=str(destination)))
    print(json.dumps(dict(status='PASS_REPRODUCED',directory=str(destination))),flush=True)


if __name__=='__main__':main()
