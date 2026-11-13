"""Freeze actual mechanism operators before the first real model fit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

STAGE = Path(__file__).resolve().parent
ROOT = Path(r'E:\语言模型组合路由\research\m4_noise_aware_set_learning')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    files = ['noise/joint_noise_model.py', 'noise/run_joint_noise.py',
             'training/ns_training.py', 'training/evaluate_ns.py',
             'math_literature/public/set_spo_operators.py',
             'math_literature/public/qualify_set_losses.py']
    entries = []
    for relative in files:
        source = STAGE / relative
        destination = ROOT / 'public/code' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError('Core source is already frozen')
        shutil.copy2(source, destination)
        entries.append({'public_path': destination.relative_to(ROOT).as_posix(),
                        'sha256': sha(source), 'execution_source': str(source)})
    clarification = {
        'status': 'FROZEN_BEFORE_FIRST_REAL_PREDICTOR_FIT',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'main_delta': .05,
        'paired_MC_precision': 'NS-SPO+ and Davidson main training both use the identical eight joint latent draws per question',
        'large_MC_role': '512 posterior draws are for sampler/predictive diagnostics, not a stronger Davidson main target',
        'fixed_nu': 'historical marginal tie probability matched at zero margin, not a jointly fitted author-native Davidson likelihood',
        'frozen_execution_scope': 'member-generation seeds42/137/2024, aggregator seed42 remains fixed',
        'posterior_scope': 'bounded empirical-Bayes Gaussian working posterior with partial predictive miscalibration in pilot; no calibrated latent-equivalence claim',
        'mathematical_scope': 'set losses convex and nonnegative; no universal Bayes decision consistency, including an explicit exact-posterior counterexample',
        'prototype_fits': 16,
    }
    put(ROOT / 'public/NS_SPO_SCOPE_CLARIFICATION_01.json', clarification)
    put(ROOT / 'private/CORE_SOURCE_LOCK.json', {
        'created_utc': clarification['frozen_utc'], 'files': entries,
        'protocol_sha256': sha(ROOT / 'public/NS_SPO_PROTOCOL.json'),
        'clarification_sha256': sha(ROOT / 'public/NS_SPO_SCOPE_CLARIFICATION_01.json')})
    print(json.dumps({'status': 'CORE_FROZEN', 'source_files': len(entries),
                      'training_sha256': sha(STAGE / 'training/ns_training.py')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
