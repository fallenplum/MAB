"""Reproduce or resume the complete frozen E3 development pipeline."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    directory = Path(__file__).resolve().parent
    asset_root = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
    recorded_python = asset_root / '.venv/Scripts/python.exe'
    python = str(recorded_python if recorded_python.exists() else Path(sys.executable))

    def run(script, *arguments):
        command = [python, '-B', '-X', 'utf8', str(directory / script), *map(str, arguments)]
        print('Running ' + script + ' ' + ' '.join(map(str, arguments)), flush=True)
        subprocess.run(command, check=True)

    run('run_e3.py', '--stage', 'precheck', '--workers', min(args.workers, 4))
    run('audit_e3.py')
    for stage in ('A', 'B1', 'B2', 'J', 'sensitivity', 'D'):
        workers = min(args.workers, 8) if stage == 'sensitivity' else args.workers
        run('run_e3.py', '--stage', stage, '--workers', workers)
        run('summarize_e3.py', '--stage', stage.upper() if stage == 'sensitivity' else stage)
    run('run_e3.py', '--stage', 'delete', '--workers', min(args.workers, 6))
    run('primary_statistics.py')
    run('replay_mechanism_audit.py', '--stage', 'all', '--workers', min(args.workers, 4))
    run('audit_kg_precision.py')
    run('build_e3_delivery.py')
    run('verify_e3_delivery.py')
    print('Complete E3 development pipeline finished', flush=True)


if __name__ == '__main__':
    main()
