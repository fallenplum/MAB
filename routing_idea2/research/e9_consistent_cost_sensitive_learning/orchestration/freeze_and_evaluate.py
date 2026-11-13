"""Freeze evaluator, qualify the full sealed collection, then score once."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'code'))
from e9_common import ROOT, now, put, sha


def main():
    code=Path(__file__).parent/'code/e9_evaluate.py'
    lock=ROOT/'private/EVALUATION_SOURCE_LOCK.json'
    if lock.exists():raise FileExistsError('Evaluation source already frozen')
    put(lock,dict(created_utc=now(),source_sha256=sha(code),
        protocol_sha256=sha(ROOT/'public/E9_PROTOCOL.json'),
        head_lock_sha256=sha(ROOT/'private/HEAD_SOURCE_LOCK.json'),
        predeclared_primary=['C-A','B-A']))
    destination=ROOT/'public/code/e9_evaluate.py'
    destination.write_bytes(code.read_bytes())
    import e9_evaluate
    e9_evaluate.main(ROOT)


if __name__=='__main__':main()
