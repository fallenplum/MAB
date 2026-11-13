"""Execute independent legal teacher regions with bounded CPU concurrency."""
from pathlib import Path
import concurrent.futures
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).parent / 'code'))
from e9_common import ROOT, now, put, sha


def one(fold):
    folder = ROOT / 'private/teacher_execution'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'fold{fold}.log'
    with path.open('x', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8',
            str(Path(__file__).parent / 'code/e9_teachers.py'), '--fold', str(fold)],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError('Teacher subprocess failed fold' + str(fold))
    return dict(fold=fold, exit_code=0, log_sha256=sha(path))


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(one, range(5)))
    put(ROOT / 'private/teacher_execution/ALL_COMPLETE.json', dict(status='PASS',
        completed_utc=now(), folds=results, source_sha256=sha(__file__), max_workers=2))
    print('ALL_TEACHERS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
