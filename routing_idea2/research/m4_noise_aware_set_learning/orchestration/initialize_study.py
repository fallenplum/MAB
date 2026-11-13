"""Initialize an isolated, cached NS-SPO+ mechanism study."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

STAGE = Path(__file__).resolve().parent
ASSET = Path(r'E:\语言模型组合路由')
ROOT = ASSET / 'research/m4_noise_aware_set_learning'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if ROOT.exists():
        raise FileExistsError('New study directory already exists')
    protocol = {
        'study': 'Noise-aware set-valued SPO+ exploratory mechanism validation',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_branch': 'codex/e8-m4-mechanism-audit',
        'parent_commit': '0dece5c291c5ec644f690f892638f81543aed691',
        'branch': 'codex/m4-noise-aware-set-learning',
        'data': 'original train600 only, five frozen480/120 folds, three cached generation repeats',
        'main_delta': .05, 'main_delta_authorization': 'user explicitly selected five percentage points',
        'sensitivity_delta': 0.,
        'capacity': 'frozen384 MiniLM -> historical PCA16/scaler, penalized constant, linear17x20 head',
        'regularization': 'reuse each outer fold original M4 selected lambda without new search',
        'noise_hyperparameters': 'three-way cross-fitting inside each480 history region, other320 fit pooled within-question covariance and empirical-Bayes random-effects prior',
        'noise_shrinkage': .2,
        'posterior': 'bounded joint truncated Gaussian working posterior, no claim of exact Bernoulli posterior',
        'posterior_diagnostics_draws': 512, 'training_MC_draws': 8,
        'numerical_posterior_fallback': 'explicit empirical mean samples with uniform near-best weights on sampling failure or maxRhat>1.1 or minESS<32',
        'posterior_validation': 'leave-one-generation posterior predictive diagnostics, never latent-equivalence calibration claims from one held generation',
        'new_objectives': ['NS-SPO delta.05', 'NS-SPO delta0', 'Davidson same posterior three-state targets delta.05'],
        'new_fits': {'main_and_sensitivity': 15, 'fold0_disjoint_MC_qualification': 1},
        'reference_models': ['frozen original Ridge', 'frozen M4 lexicographic', 'previous same-lambda empirical uniform true-tie TSPO'],
        'Davidson_nu': 'one historical posterior tie mass determines nu=p0/(1-p0), 2nu parameterization, clipped1e-6 to1e6 for numeric safety',
        'inference': 'raw-score exact numpy argmax, lexicographic exact ties',
        'statistics': '10000 original-question paired stratified bootstrap, main NS.05 vs Ridge and frozen M4 with Holm2, all exploratory reused-development evidence',
        'selection': 'no outer-reward-driven model, delta, posterior or epoch selection',
        'preserved_outcome_accesses': [], 'physical_LLM_calls': 0,
        'theory': 'proofs and unit tests first; no inheritance of unrestricted SPO calibration; document explicit inconsistency counterexample',
        'scope': 'mathematical qualification, working-posterior diagnostics and small fixed-capacity CPU experiment',
        'next_phase': 'requires review of results; no automatic feedback or retained-set evaluation',
    }
    put(ROOT / 'public/NS_SPO_PROTOCOL.json', protocol)
    parents = {}
    for name in ('e8_decision_focused_combination_learning', 'e8_m4_mechanism_audit'):
        folder = ASSET / 'research' / name
        parents[name] = {p.relative_to(folder).as_posix(): sha(p)
                         for p in sorted(folder.rglob('*')) if p.is_file()}
    put(ROOT / 'private/PARENT_SNAPSHOTS.json', parents)
    print(json.dumps({'root': str(ROOT), 'parent_file_counts': {k: len(v) for k, v in parents.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
