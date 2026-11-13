"""Verify the original archive or replay scoring into a new private directory."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ASSETS = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
ROOT = ASSETS / 'research/e8_m4_mechanism_audit'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def execute(script, arguments):
    subprocess.run([sys.executable, '-B', '-X', 'utf8', str(script), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'analysis-replay'])
    args = parser.parse_args()
    execute(ROOT / 'public/orchestration/verify_m4_audit.py', [])
    if args.mode == 'verify':
        return
    destination = ROOT / 'private/reproductions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    code = ROOT / 'public/code'
    execute(code / 'run_stable_gap_audit.py', ['--root', ASSETS, '--out', destination / 'stable'])
    execute(code / 'run_content_and_power.py', ['--asset-root', ASSETS, '--output', destination / 'content_power'])
    shutil.copytree(ROOT / 'private/distribution', destination / 'distribution/private')
    spec = importlib.util.spec_from_file_location('distribution_replay', code / 'run_distribution_audit.py')
    distribution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(distribution)
    distribution.OUT = destination / 'distribution'
    distribution.E8 = ASSETS / 'research/e8_decision_focused_combination_learning'
    distribution.evaluate()
    keys = [
        ('distribution', 'M4_DISTRIBUTION_AND_TIE_RESULTS.json'),
        ('stable', 'M4_STABLE_GAP_RESULTS.json'),
        ('content_power', 'M4_CONTENT_AND_POWER_RESULTS.json'),
    ]
    for module, filename in keys:
        expected = read(ROOT / 'public' / filename)
        actual = read(destination / module / 'public' / filename)
        if module == 'stable':
            for key in ('protocol_sha256', 'wall_seconds'):
                expected.pop(key)
                actual.pop(key)
        if expected != actual:
            raise AssertionError('Scientific replay differs ' + module)
    receipt = {
        'status': 'PASS_ALL_THREE_MODULES_REPLAYED',
        'directory': str(destination),
        'all_scientific_results_identical': True,
        'models_refitted': False, 'new_physical_LLM_calls': 0,
        'excluded_comparison_fields': {'stable': ['protocol_sha256', 'wall_seconds']},
    }
    (destination / 'REPRODUCTION_VERIFIED.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
