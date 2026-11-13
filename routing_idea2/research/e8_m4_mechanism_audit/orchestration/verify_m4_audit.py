"""Independently check frozen artifacts and the read-only E8 lineage."""
from pathlib import Path
import hashlib
import json
import os
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np

ASSET = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
PARENT = ASSET / 'research/e8_decision_focused_combination_learning'
ROOT = ASSET / 'research/e8_m4_mechanism_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main():
    original = read(ROOT / 'private/lineage/PARENT_E8_SNAPSHOT.json')
    current = {p.relative_to(PARENT).as_posix(): p for p in PARENT.rglob('*') if p.is_file()}
    if set(current) != set(original['files']):
        raise AssertionError('Original E8 file inventory changed')
    for relative, spec in original['files'].items():
        if sha(current[relative]) != spec['sha256']:
            raise AssertionError('Original E8 changed ' + relative)

    distribution = ROOT / 'private/distribution'
    receipt = read(distribution / 'INTERVENTIONS_FROZEN_RECEIPT.json')
    if sha(ROOT / 'public/code/run_distribution_audit.py') != receipt['source_sha256']:
        raise AssertionError('Distribution source differs from its pre-run lock')
    interventions = arrays(distribution / 'INTERVENTIONS_FROZEN.npz')
    if sha(distribution / 'INTERVENTIONS_FROZEN.npz') != receipt['prediction_sha256']:
        raise AssertionError('Frozen intervention scores changed')
    sys.path.insert(0, str(ROOT / 'public/code'))
    import run_distribution_audit as code
    restorations = []
    for fit in receipt['fits']:
        model = distribution / 'models' / Path(fit['model_path']).name
        if sha(model) != fit['model_sha256']:
            raise AssertionError('Intervention model hash changed')
        fold = fit['fold']
        inputs_path = PARENT / f'private/qualification/fold_packs/fold{fold}/evaluation_inputs_only120.npz'
        history_path = PARENT / f'private/qualification/fold_packs/fold{fold}/history480.npz'
        if sha(inputs_path) != fit['inputs_sha256'] or sha(history_path) != fit['history_sha256']:
            raise AssertionError('Fold pack hash differs from intervention fit')
        inputs = arrays(inputs_path)
        state = arrays(model)
        if 'Y' in inputs:
            raise AssertionError('Evaluation input pack exposes rewards')
        is_constant = fit['variant'].startswith('constant_')
        phi = np.ones((120, 1)) if is_constant else code.feature_map(state).phi(inputs['E'])
        expected_shape = (1, 20) if is_constant else (17, 20)
        if state['coefficient'].shape != expected_shape:
            raise AssertionError('Unexpected intervention model capacity')
        scores = phi @ state['coefficient']
        stored = interventions['raw__' + fit['variant']][inputs['global_indices']]
        error = float(np.max(np.abs(scores - stored)))
        if error > 1e-12 or not np.array_equal(scores.argmax(1), stored.argmax(1)):
            raise AssertionError('Frozen intervention cannot be restored')
        if fit['status'] != 'Solved' or fit['primal_dual_gap'] > 1e-7:
            raise AssertionError('Intervention solver did not qualify')
        restorations.append({'fold': fold, 'variant': fit['variant'], 'score_error': error})
    if len(restorations) != 25:
        raise AssertionError('All 25 predefined fits are required')

    stable = arrays(ROOT / 'private/stable/STABLE_GAP_QUESTION_RECORDS.npz')
    content = arrays(ROOT / 'private/content_power/CONTENT_AND_POWER_PAIRED_VALUES.npz')
    difference = content['M4'] - content['M0_original']
    if not np.array_equal(difference, stable['difference']):
        raise AssertionError('Two independent evaluators disagree on paired outcomes')
    if int((difference > 0).sum()) != 95 or int((difference < 0).sum()) != 83:
        raise AssertionError('E8 pairwise correction/destruction totals differ')
    pair_mask = stable['mask__two_seed_stable_selected_pair']
    general_mask = stable['mask__two_seed_stable_action_pair']
    if int(pair_mask.sum()) != 50 or int(general_mask.sum()) != 344:
        raise AssertionError('Independent support reconstruction differs')
    result = {
        'status': 'PASS', 'original_E8_files_byte_identical': len(current),
        'intervention_models_restored': len(restorations),
        'maximum_intervention_score_error': max(item['score_error'] for item in restorations),
        'restorations': restorations,
        'independent_evaluators_agree': True,
        'corrections': 95, 'destructions': 83, 'net_events': 12,
        'scientific_inference': 'exploratory reused-development mechanism study',
        'preserved_outcome_splits_accessed': [], 'new_physical_LLM_calls': 0,
    }
    destination = ROOT / 'public/M4_AUDIT_VERIFICATION.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'restorations'}))


if __name__ == '__main__':
    main()
