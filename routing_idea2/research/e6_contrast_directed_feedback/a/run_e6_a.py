"""E6-A paired direction learning. Training and evaluation are distinct commands."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ.get('MAB_ASSET_ROOT', r'E:\语言模型组合路由'))
WORK = ROOT / 'research/e6_contrast_directed_feedback'
E5 = ROOT / 'research/e5_paired_gain_validation'
sys.path.insert(0, str(E5 / 'public'))
from e5_common import load_train, put, sha, TEAMS
from cost_and_parameters import resources

PUBLIC = WORK / 'public/a'
PRIVATE = WORK / 'private/a'
LAMBDAS = (.01, .1, 1.)
NAMES = ('A0', 'A1', 'A2', 'A3', 'A4', 'gap', 'default', 'all')
POLICY_NAMES = ('default', 'all', 'gap', 'A0', 'A2', 'A3')
TIE = 1e-12


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.writing.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def svd_embedding(rewards):
    centered = rewards.mean(axis=1)
    centered = centered - centered.mean(axis=1, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    u = vt[:2].T.copy()
    for column in range(2):
        pivot = np.argmax(np.abs(u[:, column]))
        if u[pivot, column] < 0:
            u[:, column] *= -1
    return u, singular


def features(pred, base, actions, phi, embedding, level):
    actions = np.asarray(actions, dtype=int)
    if actions.ndim == 1:
        actions = actions[:, None]
    n, width = actions.shape
    rows = np.broadcast_to(np.arange(n)[:, None], (n, width))
    values = np.stack((np.broadcast_to(pred[:, base, None], (n, width)), pred[rows, actions]), axis=2)
    if level == 'A1':
        return values
    identity = np.eye(20)[actions] - np.eye(20)[base]
    if level == 'A2':
        return np.concatenate((values, identity), axis=2)
    text = np.broadcast_to(phi[:, None, :], (n, width, 8))
    diff = embedding[actions] - embedding[base]
    interactions = (text[:, :, :, None] * diff[:, :, None, :]).reshape(n, width, 16)
    result = np.concatenate((values, identity, text, interactions), axis=2)
    assert result.shape[-1] == 46
    return result


def weighted_training(features_current, differences_current, features_aux=None, differences_aux=None):
    """Return rows and direction counts whose weights retain D=0 normalization."""
    n, repeats = differences_current.shape
    x = features_current[:, 0, :]
    positive = (differences_current > 0).sum(axis=1) / repeats
    negative = (differences_current < 0).sum(axis=1) / repeats
    if features_aux is None:
        return x, positive / n, negative / n
    x_aux = features_aux.reshape(-1, x.shape[1])
    positive_aux = (differences_aux > 0).sum(axis=2).reshape(-1) / repeats
    negative_aux = (differences_aux < 0).sum(axis=2).reshape(-1) / repeats
    return (np.concatenate((x, x_aux)),
            np.r_[.5 * positive / n, .5 * positive_aux / (19 * n)],
            np.r_[.5 * negative / n, .5 * negative_aux / (19 * n)])


def fit_head(x, positive, negative, lam, question_count):
    feature_weight = np.ones(len(x))
    if len(x) == question_count * 20:
        feature_weight[:question_count] = .5
        feature_weight[question_count:] = .5 / 19
    scaler = StandardScaler().fit(x, sample_weight=feature_weight)
    design = np.column_stack((np.ones(len(x)), scaler.transform(x)))
    mass = positive + negative
    support = dict(positive_mass=float(positive.sum() * question_count),
                   negative_mass=float(negative.sum() * question_count),
                   normalization_questions=int(question_count))
    if positive.sum() == 0 or negative.sum() == 0:
        probability = (positive.sum() * question_count + 1) / (mass.sum() * question_count + 2)
        beta = np.zeros(design.shape[1])
        beta[0] = logit(probability)
        return dict(beta=beta, mean=scaler.mean_, scale=scaler.scale_, fallback='Beta(1,1)',
                    probability=float(probability), lam=lam, iterations=0, gradient_inf=None,
                    objective=None, support=support)
    beta = np.zeros(design.shape[1])
    def objective(b):
        score = design @ b
        return float(positive @ np.logaddexp(0, -score) + negative @ np.logaddexp(0, score) + lam * (b @ b))
    for iteration in range(100):
        probability = expit(design @ beta)
        gradient = design.T @ (mass * probability - positive) + 2 * lam * beta
        if np.max(np.abs(gradient)) <= 1e-11:
            break
        hessian = (design.T * (mass * probability * (1 - probability))) @ design
        hessian += 2 * lam * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        old = objective(beta)
        step_size = 1.
        while objective(beta - step_size * step) > old - 1e-4 * step_size * (gradient @ step):
            step_size *= .5
            if step_size < 1e-12:
                break
        beta -= step_size * step
    probability = expit(design @ beta)
    gradient = design.T @ (mass * probability - positive) + 2 * lam * beta
    assert np.max(np.abs(gradient)) < 1e-8
    return dict(beta=beta, mean=scaler.mean_, scale=scaler.scale_, fallback=None,
                probability=None, lam=lam, iterations=iteration + 1,
                gradient_inf=float(np.max(np.abs(gradient))), objective=objective(beta), support=support)


def predict_head(head, x):
    return head['beta'][0] + ((x - head['mean']) / head['scale']) @ head['beta'][1:]


def support(d_current, d_aux=None):
    def count(d):
        axes = tuple(range(1, d.ndim))
        return dict(independent_questions=len(d),
                    effective_direction_questions=int(np.any(d != 0, axis=axes).sum()),
                    positive_direction_questions=int(np.any(d > 0, axis=axes).sum()),
                    negative_direction_questions=int(np.any(d < 0, axis=axes).sum()),
                    paired_score_records=int(d.size), nonzero_direction_records=int(np.count_nonzero(d)),
                    positive_records=int((d > 0).sum()), negative_records=int((d < 0).sum()))
    result = dict(current_candidate=count(d_current))
    if d_aux is not None:
        result['all_nineteen_nondefault'] = count(d_aux)
        result['current_duplicate_in_auxiliary'] = True
    return result


def train_fold(fold, data):
    start = perf_counter()
    with np.load(E5 / f'private/base/BASE_f{fold}.npz') as archive:
        F, G, E = (archive[key].copy() for key in ('F', 'G', 'E'))
        base = int(archive['base'])
        candidate_g, candidate_e = archive['candidate_G'], archive['candidate_E']
        pred_g, pred_e = archive['pred_G'][1], archive['pred_E'][1]
    assert not set(F) & set(G) and not set(F) & set(E) and not set(G) & set(E)
    # F-only fitted representations. E scores are never indexed by this function.
    pca = PCA(n_components=8, svd_solver='full').fit(data['E'][F])
    phi_g, phi_e = pca.transform(data['E'][G]), pca.transform(data['E'][E])
    embedding, singular = svd_embedding(data['Y'][F])
    d_g = data['Y'][G[:, None], np.arange(3)[None, :], candidate_g[:, None]] - data['Y'][G, :, base]
    actions = np.asarray([a for a in range(20) if a != base])
    all_g = np.broadcast_to(actions, (len(G), 19))
    d_aux = data['Y'][G][:, :, actions].transpose(0, 2, 1) - data['Y'][G, :, base][:, None, :]
    splitter = list(StratifiedKFold(3, shuffle=True, random_state=20261112 + fold).split(G, data['task'][G]))
    all_scores = {}
    heads = {}
    cv = {}
    for name in ('A1', 'A2', 'A3', 'A4'):
        fg = features(pred_g, base, candidate_g, phi_g, embedding, name)
        fe = features(pred_e, base, candidate_e, phi_e, embedding, name)[:, 0, :]
        auxiliary = features(pred_g, base, all_g, phi_g, embedding, name) if name == 'A4' else None
        candidates = []
        for lam in LAMBDAS:
            oof = np.zeros(len(G))
            fold_rows = []
            for tr, va in splitter:
                train_x, pos, neg = weighted_training(fg[tr], d_g[tr],
                    None if auxiliary is None else auxiliary[tr], None if auxiliary is None else d_aux[tr])
                head = fit_head(train_x, pos, neg, lam, len(tr))
                oof[va] = predict_head(head, fg[va, 0])
                fold_rows.append(dict(train=G[tr], validation=G[va], fitted=head))
            gain = float(np.mean((oof > 0) * d_g.mean(axis=1)))
            candidates.append(dict(lam=lam, gain=gain, oof=oof, folds=fold_rows))
        best_gain = max(row['gain'] for row in candidates)
        chosen = max((row for row in candidates if row['gain'] >= best_gain - TIE), key=lambda row: row['lam'])
        train_x, pos, neg = weighted_training(fg, d_g, auxiliary, d_aux if auxiliary is not None else None)
        head = fit_head(train_x, pos, neg, chosen['lam'], len(G))
        all_scores[name] = predict_head(head, fe)
        heads[name] = head
        cv[name] = dict(chosen_lambda=chosen['lam'], chosen_gain=chosen['gain'], candidates=candidates,
                        support=support(d_g, d_aux if name == 'A4' else None))
    with np.load(E5 / f'private/gates/FROZEN_DECISIONS_f{fold}.npz') as archive:
        assert np.array_equal(archive['E'], E) and int(archive['base']) == base
        assert np.array_equal(archive['candidate_E'], candidate_e)
        original = archive['gate_masks_E']
        selected = archive['main_indices']
        a0 = original[1, selected[1]]
        gap = original[0, selected[0]]
    masks = dict(A0=a0, gap=gap, default=np.zeros(len(E), dtype=bool), all=np.ones(len(E), dtype=bool))
    masks.update({name: value > 0 for name, value in all_scores.items()})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    policy_path = PRIVATE / f'fold{fold}_policies.npz'
    atomic_npz(policy_path, E=E, base=base, candidate=candidate_e, names=np.asarray(POLICY_NAMES),
               masks=np.asarray([masks[name] for name in POLICY_NAMES]))
    decision_path = PRIVATE / f'FROZEN_A_f{fold}.npz'
    atomic_npz(decision_path, F=F, G=G, E=E, base=base, candidate=candidate_e,
               names=np.asarray(NAMES), masks=np.asarray([masks[name] for name in NAMES]),
               scores=np.asarray([all_scores[name] for name in ('A1', 'A2', 'A3', 'A4')]),
               pca_components=pca.components_, pca_mean=pca.mean_, combo_embedding=embedding)
    put(PRIVATE / f'FROZEN_A_f{fold}.json', serial(dict(fold=fold, status='FROZEN_BEFORE_E_OUTCOMES',
        source_sha256=sha(Path(__file__)), base_receipt=sha(E5 / f'private/base/BASE_f{fold}.json'),
        decision_sha256=sha(decision_path), policy_sha256=sha(policy_path),
        dimension=dict(A1=2, A2=22, A3=46, A4=46), heads=heads, cv=cv,
        representation_singular_values=singular, candidate_direction_support=support(d_g, d_aux),
        G_pair_feedback_cost=resources(data, G, [[base, int(a)] for a in candidate_g], gold=len(G)),
        G_all_feedback_cost=resources(data, G, [range(20)] * len(G), gold=len(G)),
        training_wall_s=perf_counter() - start, new_llm_calls=0)))
    return dict(fold=fold, status='FROZEN', selected_lambda={name: heads[name]['lam'] for name in heads},
                policy_sha256=sha(policy_path), training_wall_s=perf_counter() - start)


def train():
    # Root must persist P/Q and global E6 config before any E6-A train command.
    ready = WORK / 'private/PROTOCOL_FROZEN.json'
    if not ready.exists():
        raise RuntimeError('Parent protocol freeze receipt is required before E6-A training')
    data, refs = load_train()
    for fold in range(5):
        print(json.dumps(train_fold(fold, data)), flush=True)


def interval(draws, point, confidence=.95):
    alpha = (1 - confidence) / 2
    return dict(point=float(point), confidence=confidence,
                interval=np.quantile(draws, [alpha, 1 - alpha]).tolist())


def matched_gain(mask, differences, folds, tasks, subtasks, mode):
    """Exact conditional expectation of random gates, preserving fold counts."""
    expected = np.zeros(len(mask))
    if mode == 'global':
        groups = np.asarray([str(f) for f in folds])
    elif mode == 'task':
        groups = np.asarray([str(f) + '/' + t for f, t in zip(folds, tasks)])
    elif mode == 'subtask':
        groups = np.asarray([str(f) + '/' + (s if t == 'bbh' else t)
                             for f, t, s in zip(folds, tasks, subtasks)])
    else:
        raise ValueError(mode)
    for group in np.unique(groups):
        selected = groups == group
        expected[selected] = np.mean(mask[selected])
    return float(np.mean(expected * differences))


def evaluate():
    """Only this separately invoked evaluator indexes E outcomes."""
    data, refs = load_train()
    all_rows = []
    all_masks = []
    all_rb = []
    all_d = []
    fold_labels = []
    fold_details = []
    for fold in range(5):
        frozen_path = PRIVATE / f'FROZEN_A_f{fold}.npz'
        receipt = json.loads((PRIVATE / f'FROZEN_A_f{fold}.json').read_text(encoding='utf-8'))
        assert sha(frozen_path) == receipt['decision_sha256']
        with np.load(frozen_path) as archive:
            E = archive['E']
            base = int(archive['base'])
            candidate = archive['candidate']
            masks = archive['masks']
            assert tuple(archive['names']) == NAMES
        rb = data['Y'][E, :, base]
        rc = data['Y'][E[:, None], np.arange(3)[None, :], candidate[:, None]]
        all_rows.extend(E.tolist())
        all_masks.append(masks)
        all_rb.append(rb)
        all_d.append(rc - rb)
        fold_labels.extend([fold] * len(E))
        fold_details.append(dict(fold=fold, training=receipt,
            E_pair_evaluation_cost=resources(data, E, [[base, int(c)] for c in candidate], gold=len(E)),
            service_cost={name: resources(data, E,
                [[int(candidate[i]) if masks[j, i] else base] for i in range(len(E))])
                for j, name in enumerate(NAMES)}))
    rows = np.asarray(all_rows)
    assert len(rows) == len(set(rows)) == 600
    masks = np.concatenate(all_masks, axis=1)
    rb = np.concatenate(all_rb)
    d = np.concatenate(all_d)
    folds = np.asarray(fold_labels)
    tasks, subtasks = data['task'][rows], data['sub'][rows]
    dbar = d.mean(axis=1)
    accuracy = (rb[None] + masks[:, :, None] * d[None]).mean(axis=2)
    point = accuracy.mean(axis=1)
    gains = (masks * dbar).mean(axis=1)
    matched = {mode: np.asarray([matched_gain(mask, dbar, folds, tasks, subtasks, mode) for mask in masks])
               for mode in ('global', 'task', 'subtask')}
    rng = np.random.default_rng(20261112)
    cells = [np.flatnonzero((folds == fold) & (tasks == task))
             for fold in range(5) for task in ('gsm8k', 'bbh')]
    bootstrap_count = 5000
    indices = np.empty((bootstrap_count, len(rows)), dtype=int)
    boot_accuracy = np.empty((bootstrap_count, len(NAMES)))
    boot_gain = np.empty_like(boot_accuracy)
    boot_matched = {mode: np.empty_like(boot_accuracy) for mode in matched}
    for b in range(bootstrap_count):
        selected = np.concatenate([rng.choice(cell, len(cell), replace=True) for cell in cells])
        indices[b] = selected
        boot_accuracy[b] = accuracy[:, selected].mean(axis=1)
        boot_gain[b] = (masks[:, selected] * dbar[selected]).mean(axis=1)
        for mode in matched:
            boot_matched[mode][b] = [matched_gain(mask[selected], dbar[selected], folds[selected],
                tasks[selected], subtasks[selected], mode) for mask in masks]
    results = []
    for j, name in enumerate(NAMES):
        accepted = masks[j]
        signed = accepted[:, None] * d
        disagreements = int(np.sum(accepted[:, None] & (d != 0)))
        per_seed_gain = signed.mean(axis=0)
        per_task = {}
        for task in ('gsm8k', 'bbh'):
            ix = tasks == task
            per_task[task] = dict(questions=int(ix.sum()), accuracy=float(accuracy[j, ix].mean()),
                gain=float(signed[ix].mean()), switch_rate=float(accepted[ix].mean()),
                corrected=float(np.sum(signed[ix] > 0) / 3), harmed=float(np.sum(signed[ix] < 0) / 3))
        per_subtask = {}
        for sub in sorted(set(subtasks[tasks == 'bbh'])):
            ix = (tasks == 'bbh') & (subtasks == sub)
            per_subtask[sub] = dict(questions=int(ix.sum()), accuracy=float(accuracy[j, ix].mean()),
                gain=float(signed[ix].mean()), switch_rate=float(accepted[ix].mean()),
                corrected=float(np.sum(signed[ix] > 0) / 3), harmed=float(np.sum(signed[ix] < 0) / 3))
        results.append(dict(configuration=name, independent_E_questions=600,
            direction_support_by_fold=[row['training']['cv'][name]['support'] if name in ('A1', 'A2', 'A3', 'A4')
                else row['training']['candidate_direction_support'] for row in fold_details],
            accuracy=interval(boot_accuracy[:, j], point[j]),
            gain=interval(boot_gain[:, j], gains[j]), switch_questions=int(accepted.sum()),
            switch_rate=float(accepted.mean()), corrected=float(np.sum(signed > 0) / 3),
            harmed=float(np.sum(signed < 0) / 3), net_corrected=float(np.sum(signed) / 3),
            rho=float(np.mean(accepted[:, None] * d ** 2)),
            eta=None if disagreements == 0 else float(np.sum(signed > 0) / disagreements),
            seed_gain=per_seed_gain.tolist(), seed_accuracy=(rb[None] + masks[:, :, None] * d[None])[j].mean(axis=0).tolist(),
            tasks=per_task, bbh_subtasks=per_subtask,
            task_macro_accuracy=float(.5 * (per_task['gsm8k']['accuracy'] + per_task['bbh']['accuracy'])),
            gsm_bbh27_macro_accuracy=float(.5 * (per_task['gsm8k']['accuracy'] + np.mean([v['accuracy'] for v in per_subtask.values()]))),
            matched_excess={mode: interval(boot_gain[:, j] - boot_matched[mode][:, j], gains[j] - matched[mode][j])
                            for mode in matched} ))
    contrasts = {}
    for left, right in (('A3', 'A1'), ('A3', 'A0'), ('A4', 'A3')):
        j, k = NAMES.index(left), NAMES.index(right)
        contrasts[left + '_minus_' + right] = interval(boot_accuracy[:, j] - boot_accuracy[:, k], point[j] - point[k])
    PRIMARY = NAMES.index('A3')
    atomic_npz(PRIVATE / 'PAIRED_E_RESULTS.npz', rows=rows, folds=folds, tasks=tasks,
               subtasks=subtasks, names=np.asarray(NAMES), masks=masks, Rb=rb, D=d,
               accuracy=accuracy, bootstrap_indices=indices, bootstrap_accuracy=boot_accuracy,
               bootstrap_gain=boot_gain, bootstrap_matched_task=boot_matched['task'],
               bootstrap_matched_global=boot_matched['global'], bootstrap_matched_subtask=boot_matched['subtask'])
    # Parent joins this raw contrast with two C contrasts for unified multiplicity handling.
    atomic_npz(PRIVATE / 'PRIMARY_A_CONTRAST.npz', rows=rows,
               question_contribution=masks[PRIMARY] * dbar,
               observed=gains[PRIMARY] - matched['task'][PRIMARY],
               bootstrap=boot_gain[:, PRIMARY] - boot_matched['task'][:, PRIMARY])
    public_folds = []
    for row in fold_details:
        train_receipt = row['training']
        public_training = dict(
            status=train_receipt['status'],
            decision_sha256=train_receipt['decision_sha256'],
            policy_sha256=train_receipt['policy_sha256'],
            dimension=train_receipt['dimension'],
            training_wall_s=train_receipt['training_wall_s'],
            candidate_direction_support=train_receipt['candidate_direction_support'],
            G_pair_feedback_cost=train_receipt['G_pair_feedback_cost'],
            G_all_feedback_cost=train_receipt['G_all_feedback_cost'],
            cv={name: dict(chosen_lambda=info['chosen_lambda'], chosen_gain=info['chosen_gain'],
                support=info['support'], candidates=[dict(lam=c['lam'], gain=c['gain']) for c in info['candidates']])
                for name, info in train_receipt['cv'].items()},
            maximum_solver_gradient_inf=max((head['gradient_inf'] or 0.) for head in train_receipt['heads'].values()))
        public_folds.append({**row, 'training': public_training})
    output = dict(status='COMPLETE', scientific_role='existing train600 cross-fitted development evaluation',
        independent_E_questions=600, repeats=3, names=list(NAMES), configurations=results, contrasts=contrasts,
        bootstrap=dict(repetitions=bootstrap_count, seed=20261112, unit='original question',
            strata='outer fold x broad task', generation_repeats='kept together',
            matched_random='exact conditional mean, acceptance counts recomputed on every bootstrap',
            limitation='overlapping F/G training histories and reused development questions not refitted in conditional bootstrap'),
        folds=public_folds, new_llm_calls=0,
        interpretation='fractional corrected/harmed counts average three seed outcomes; not extra independent questions')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    put(PUBLIC / 'A_RESULTS.json', serial(output))
    print(json.dumps(dict(status='COMPLETE', configurations=[dict(name=r['configuration'], accuracy=r['accuracy']['point'],
        gain=r['gain']['point'], task_excess=r['matched_excess']['task']['point']) for r in results])), flush=True)


def qualify():
    rng = np.random.default_rng(20261112)
    assertions = 0
    for _ in range(100):
        d = rng.choice([-1., 0., 1.], (12, 3))
        g = rng.integers(0, 2, len(d))
        left = np.mean(np.abs(d) * (g[:, None] != (d > 0)))
        right = np.mean(np.maximum(d, 0)) - np.mean(g[:, None] * d)
        assert abs(left - right) < 1e-14
        x = rng.normal(size=(12, 1, 5))
        xx, positive, negative = weighted_training(x, d)
        assert np.all(positive + negative <= 1 / 12 + 1e-15)
        assert abs(positive.sum() + negative.sum() - np.mean(d != 0)) < 1e-14
        for lam in LAMBDAS:
            head = fit_head(xx, positive, negative, lam, len(d))
            assert head['gradient_inf'] is None or head['gradient_inf'] < 1e-8
            assert np.isfinite(predict_head(head, xx)).all()
            assertions += 2
        assertions += 3
    x = rng.normal(size=(12, 1, 46))
    xa = rng.normal(size=(12, 19, 46))
    d = rng.choice([-1., 0., 1.], (12, 3))
    da = rng.choice([-1., 0., 1.], (12, 19, 3))
    _, p, n = weighted_training(x, d, xa, da)
    assert abs(p.sum() + n.sum() - (.5 * np.mean(d != 0) + .5 * np.mean(da != 0))) < 1e-14
    assertions += 1
    PRIVATE.mkdir(parents=True, exist_ok=True)
    put(PRIVATE / 'LOSS_NUMERICAL_QUALIFICATION.json', dict(status='PASS', assertions=assertions,
        tests=['weighted 0/1 paired-loss identity', 'normalization includes zero differences',
               'convex logistic gradient convergence', 'A4 precise half/half auxiliary normalization']))
    print(json.dumps(dict(status='PASS', assertions=assertions)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('train', 'evaluate', 'qualify'))
    args = parser.parse_args()
    {'train': train, 'evaluate': evaluate, 'qualify': qualify}[args.stage]()
