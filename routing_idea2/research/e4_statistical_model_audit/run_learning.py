"""E4-A nested, question-held-out Ridge curves. Scoring is auditor-only."""
from e4_common import *
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from time import perf_counter
import argparse


def geometry(data, history, predictor):
    pca = PCA(n_components=16, svd_solver='full').fit(data['E'][history])
    scaler = StandardScaler().fit(pca.transform(data['E'][history]))
    X = np.column_stack((np.ones(len(data['E'])), scaler.transform(pca.transform(data['E']))))
    if predictor == 'P1':
        z = np.asarray([[int(j in team) for j in range(6)] for team in TEAMS])
        original = np.column_stack((np.ones(20), z, *[z[:, i]*z[:, j] for i,j in combinations(range(6),2)]))
        _, s, vh = np.linalg.svd(original, full_matrices=False)
        B = original @ vh[:15].T
        assert np.sum(s > 1e-10) == 15
    else:
        B = np.eye(20)
    return X, B


def fit_coeff(Xh, B, rewards, alpha):
    L = rewards.shape[1]
    evx, U = np.linalg.eigh(Xh.T @ Xh)
    eva, V = np.linalg.eigh(B.T @ B)
    spectral = L*np.maximum(evx[:,None],0)*np.maximum(eva[None,:],0)
    inverse = 1/(alpha+spectral)
    rhs = L*Xh.T @ rewards.mean(1) @ B
    W = U @ ((U.T @ rhs @ V)*inverse) @ V.T
    return W, spectral, inverse


def worker(job):
    fold_id, size, predictor = job
    start = perf_counter()
    data, _ = load_train()
    fold = e4_splits(data)['folds'][fold_id]
    history = fold['nested'][size]
    query = fold['stream']
    X, B = geometry(data, history, predictor)
    rows, predictions, train_predictions = [], [], []
    for L in (1,2,3):
        for subset in combinations(range(3), L):
            # Fit only the permitted history/seed block. Other labels remain evaluator-only.
            allowed = data['Y'][np.ix_(history, subset, np.arange(20))]
            for regularization in ('fixed', 'matched') if L < 3 else ('fixed',):
                alpha = 100. if regularization == 'fixed' else 100.*L/3
                W, spectrum, inverse = fit_coeff(X[history], B, allowed, alpha)
                pred = X[query] @ W @ B.T
                pred_train = X[history] @ W @ B.T
                rows.append(dict(L=L, subset=list(subset), alpha=alpha,
                                 regularization=regularization,
                                 train_raw_mse=float(np.mean((allowed-pred_train[:,None])**2)),
                                 train_mean_mse=float(np.mean((allowed.mean(1)-pred_train)**2)),
                                 effective_df=float(np.sum(spectrum*inverse)),
                                 observable_parameter_dimension=int(spectrum.size),
                                 original_parameter_dimension=int(17*(22 if predictor=='P1' else 20)),
                                 posterior_trace_observable=float(inverse.sum()),
                                 prior_only_null_trace=float(17*7/alpha if predictor=='P1' else 0),
                                 posterior_max_eigenvalue_original=float(max(inverse.max(), 1/alpha if predictor=='P1' else 0))))
                predictions.append(pred)
                # Different history sizes have distinct supports. Keep scalar train losses, not padded arrays.
    Xs = np.linalg.svd(X[history], compute_uv=False)
    Bs = np.linalg.svd(B, compute_uv=False)
    spectrum = (Xs[:,None]*Bs[None,:]).ravel()
    total = spectrum**2
    distribution = total/total.sum()
    info = dict(fold=fold_id, n=size, predictor=predictor, cases=rows,
                x_singular_values=Xs.tolist(), b_singular_values=Bs.tolist(),
                design_rank=int(np.sum(spectrum>1e-10)),
                nonzero_design_condition=float(spectrum.max()/spectrum[spectrum>1e-10].min()),
                entropy_effective_rank=float(np.exp(-np.sum(distribution*np.log(distribution)))),
                wall_s=perf_counter()-start, no_new_llm_calls=True)
    path = PRIVATE/'learning'/f'f{fold_id}_n{size}_{predictor}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, predictions=np.asarray(predictions), query=query, history=history)
    put(path.with_suffix('.json'), info)
    return dict(fold=fold_id,n=size,predictor=predictor,cases=len(rows),wall_s=info['wall_s'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    jobs=[(f,n,p) for f in range(5) for n in (60,120,240,360,480) for p in ('P1','P2')]
    if args.smoke:
        jobs=jobs[:2]
    jobs=[j for j in jobs if not (PRIVATE/'learning'/f'f{j[0]}_n{j[1]}_{j[2]}.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker,jobs):
            print(json.dumps(result),flush=True)
    put(PRIVATE/'LEARNING_DONE.json',dict(expected_groups=2 if args.smoke else 50,
                                        smoke=args.smoke, new_llm_calls=0))


if __name__=='__main__':
    main()
