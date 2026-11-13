"""Verify the archive or replay frozen scoring into a new private directory."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

ASSET = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
ROOT = ASSET / 'research/m4_noise_aware_set_learning'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify():
    snapshots = read(ROOT / 'private/PARENT_SNAPSHOTS.json')
    for name, snapshot in snapshots.items():
        parent = ASSET / 'research' / name
        current = {p.relative_to(parent).as_posix(): p for p in parent.rglob('*') if p.is_file()}
        if set(current) != set(snapshot):
            raise AssertionError('Parent inventory changed ' + name)
        for relative, digest in snapshot.items():
            if sha(current[relative]) != digest:
                raise AssertionError('Parent artifact changed ' + name + '/' + relative)
    lock = read(ROOT / 'private/CORE_SOURCE_LOCK.json')
    for item in lock['files']:
        if sha(ROOT / item['public_path']) != item['sha256']:
            raise AssertionError('Core public source changed')
        parts = Path(item['execution_source']).parts
        position = parts.index('.ns_spo_staging')
        runtime = ROOT / 'private/runtime' / Path(*parts[position + 1:])
        if sha(runtime) != item['sha256']:
            raise AssertionError('Runtime source differs from public source')
    training = ROOT / 'private/runtime/training'
    posterior = ROOT / 'private/runtime/noise/private'
    sys.path.insert(0, str(ROOT / 'public/code'))
    import evaluate_ns
    _, _, _, fits, restores = evaluate_ns.collect(training, posterior)
    for fold in range(5):
        receipt = read(posterior / f'fold{fold}/POSTERIOR_FREEZE_RECEIPT.json')
        if sha(posterior / f'fold{fold}/POSTERIOR_TRAINING.npz') != receipt['posterior_sha256']:
            raise AssertionError('Noise target changed')
        packs = ASSET / 'research/e8_decision_focused_combination_learning/private/qualification/fold_packs' / f'fold{fold}'
        history_hash = sha(packs / 'history480.npz')
        input_hash = sha(packs / 'evaluation_inputs_only120.npz')
        if receipt['input_history_sha256'] != history_hash:
            raise AssertionError('Noise target history hash differs')
        variants = ['NS_main', 'NS_delta0', 'Davidson_main'] + (['NS_main_mc_alt'] if fold == 0 else [])
        for variant in variants:
            model_receipt = read(training / f'private/fold{fold}/{variant}/FROZEN_RECEIPT.json')
            if model_receipt['history_sha256'] != history_hash or model_receipt['evaluation_inputs_sha256'] != input_hash:
                raise AssertionError('Model fitting/input scope differs')
    qualified = len(fits) == 16 and all(fit['status'] in ('Solved', 'CONVERGED') for fit in fits)
    if not qualified:
        raise AssertionError('Some model fits failed qualification')
    result = {'status': 'PASS', 'parent_files_byte_identical': {k: len(v) for k, v in snapshots.items()},
              'core_sources_verified': len(lock['files']), 'models_restored': len(restores),
              'maximum_model_score_error': max(row['score_error'] for row in restores),
              'all_fit_statuses_qualified': qualified,
              'new_physical_LLM_calls': 0, 'retained_reward_reads': 0}
    put(ROOT / 'public/ARCHIVE_VERIFICATION.json', result)
    return training, posterior, evaluate_ns, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'scoring-replay'])
    args = parser.parse_args()
    training, posterior, evaluator, result = verify()
    if args.mode == 'verify':
        print(json.dumps(result))
        return
    destination = ROOT / 'private/reproductions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination.mkdir(parents=True, exist_ok=False)
    evaluator.evaluate(training, posterior, destination)
    for name in ('NS_TRAINING_RESULTS.json', 'NS_PRIMARY_STATISTICS.json'):
        if sha(ROOT / 'public' / name) != sha(destination / 'public' / name):
            raise AssertionError('Scientific replay differs ' + name)
    receipt = {'status': 'PASS_EXACT_SCORING_REPLAY', 'destination': str(destination),
               'results_identical': True, 'models_refitted': False, 'new_physical_LLM_calls': 0}
    put(destination / 'REPRODUCTION_VERIFIED.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
