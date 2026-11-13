"""Qualify the variable-size teacher and freeze its source before real builds."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'code'))
from e9_common import ROOT, array_hash, now, put, sha
from e9_teachers import build_mean
import numpy as np


def main():
    lock_path = ROOT / 'private/TEACHER_SOURCE_LOCK.json'
    if lock_path.exists():
        raise FileExistsError('Teacher source lock already exists')
    code = Path(__file__).parent / 'code'
    files = ['e9_common.py', 'e9_teachers.py', 'csce_operators.py']
    put(lock_path, dict(created_utc=now(), protocol_sha256=sha(ROOT / 'public/E9_PROTOCOL.json'),
        sources={name: sha(code / name) for name in files}))
    rng = np.random.default_rng(202611140900)
    n = 61
    region = dict(ids=np.array([f'synthetic{i}' for i in range(n)]),
        global_indices=np.arange(1000,1000+n), task=np.array(['gsm8k']*31+['bbh']*30),
        sub=np.array(['synthetic']*n), Y=rng.binomial(1,.6,(n,3,20)).astype(float))
    dest = ROOT / 'private/qualification/variable_teacher61'
    means, receipt = build_mean(region, 0, dest, 'synthetic_variable61')
    blocked = False
    try:
        build_mean(region, 0, dest / 'should_not_exist', 'forbidden', [region['ids'][0]])
    except ValueError:
        blocked = True
    assert blocked and means.shape == (61,20) and np.isfinite(means).all()
    assert sorted(item['fitting_questions'] for item in receipt['inner_fits']) == [40,41,41]
    assert sorted(item['target_questions'] for item in receipt['inner_fits']) == [20,20,21]
    put(ROOT / 'public/TEACHER_SOFTWARE_QUALIFICATION.json', dict(status='PASS',
        toy_questions=61, fitting_sizes=[40,41,41], target_sizes=[20,20,21],
        all512_mean=True, overlapping_evaluation_ids_rejected=blocked,
        source_sha256=sha(code / 'e9_teachers.py'), qualification_cpu_seconds=receipt['cpu_seconds'],
        qualification_wall_seconds=receipt['wall_seconds'], fallback_questions=receipt['fallback_questions'],
        actual_synthetic_prior_fits=3, actual_synthetic_sampling_calls=3,
        scientific_scope='software qualification only'))
    print('TEACHER_SOFTWARE_PASS', receipt['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
