"""Seal the complete prediction-head path before any real E9 head fit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'code'))
from e9_common import ROOT, now, put, read, sha, verify_sources


def main():
    verify_sources('TEACHER_SOURCE_LOCK.json')
    stage=Path(__file__).parent
    if read(stage/'audit/E9_ASSET_A_REUSE_QUALIFICATION.json')['status']!='PASS_REUSE_ELIGIBLE':
        raise AssertionError('Complete A reuse qualification required')
    code=stage/'code'
    files=['e9_common.py','e9_teachers.py','csce_operators.py','e9_heads.py','e9_train.py']
    target=ROOT/'private/HEAD_SOURCE_LOCK.json'
    if target.exists():raise FileExistsError('Head source lock already exists')
    put(target,dict(created_utc=now(),protocol_sha256=sha(ROOT/'public/E9_PROTOCOL.json'),
        sources={name:sha(code/name) for name in files},
        A_reuse_qualification_sha256=sha(stage/'audit/E9_ASSET_A_REUSE_QUALIFICATION.json'),
        mathematical_qualification_sha256=sha(stage/'math/public/CSCE_QUALIFICATION.json'),
        head_qualification_sha256=sha(stage/'public_qualification/E9_HEADS_QUALIFICATION.json')))
    for name in files:
        destination=ROOT/'public/code'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((code/name).read_bytes())
    print('HEAD_SOURCE_FROZEN',sha(target))


if __name__=='__main__':main()
