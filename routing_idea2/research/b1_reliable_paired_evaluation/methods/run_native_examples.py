"""Execute audited upstream functions on author and transferred reference data.

Third-party bodies are loaded without edits from private snapshots using an AST
allow-list, avoiding import-time optional GPU/dataset dependencies.
"""
import ast
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(os.environ.get("B1_METHODS_WORK", str(Path(__file__).resolve().parent)))
sys.path[:0] = [str(HERE / "dependencies"), str(HERE / "upstream/cps-sampling/src")]
import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm
from sklearn.model_selection import train_test_split as SKtrain_test_split
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=RuntimeWarning)
RESULTS = HERE / "native_results"
RESULTS.mkdir(exist_ok=True)


def upstream_nodes(path, names, namespace):
    raw = path.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    found = {node.name for node in nodes}
    if found != set(names):
        raise ValueError(f"Missing upstream nodes {set(names) - found}")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def asi_alphafold():
    started = time.perf_counter()
    path = HERE / "upstream/active-inference/alphafold-ptm.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    data = np.load(HERE / "reference_data/alphafold.npz", allow_pickle=False)
    env = {"np": np, "pd": pd, "bernoulli": bernoulli, "norm": norm,
           "Y_total": data["Y"], "Yhat_total": data["Yhat"], "Z": data["phosphorylated"].astype(bool)}
    np.random.seed(20261113)
    for index in [2, 3, 4, 5]:
        exec(compile(cells[index], f"{path.name}:codecell{index}", "exec"), env)
    df = env["df"]
    df.to_csv(RESULTS / "ASI_ALPHAFOLD_NATIVE_DRAWS.csv", index=False)
    # The 2024 notebook initializes all columns as float, then appends numpy
    # bools and strings. Current pandas keeps object-bool aggregation semantics.
    # Cast the saved indicators explicitly before summarizing their arithmetic mean.
    df["coverage"] = df["coverage"].astype(float)
    summary = df.groupby(["estimator", "$n_b$"]).agg(coverage=("coverage", "mean"), width=("interval width", "mean")).reset_index()
    summary.to_csv(RESULTS / "ASI_ALPHAFOLD_NATIVE_SUMMARY.csv", index=False)
    result = {"status": "AUTHOR_EXAMPLE_EXECUTED", "author_data": True, "N": int(len(data["Y"])),
              "budgets": env["budgets"].tolist(), "num_trials": int(env["num_trials"]),
              "alpha": env["alpha"], "truth_odds_ratio": float(env["odds_ratio"]),
              "original_cells_executed": [2, 3, 4, 5], "changes": ["global random seed fixed to 20261113", "data retrieved through author ppi_py dataset ID", "plotting skipped"],
              "inference_scope": "author native asymptotic population delta-method CI", "wall_time_s": time.perf_counter()-started,
              "main_budget_rows": summary[summary["$n_b$"] == summary["$n_b$"].max()].to_dict("records")}
    return result


def active_testing_quadratic(runs=100):
    started = time.perf_counter()
    env = {"np": np, "OmegaConf": OmegaConf, "SKtrain_test_split": SKtrain_test_split,
           "warnings": warnings, "DEBUG_WEIGHTS": False}
    # NumPy >=1.24 removed aliases used by the 2021 author dataset implementation.
    if not hasattr(np, "int"):
        np.int = int
    base = HERE / "upstream/active-testing/activetesting"
    upstream_nodes(base / "loss.py", ["SELoss", "MSELoss", "RMSELoss"], env)
    env["maps"] = SimpleNamespace(loss={"SELoss": env["SELoss"]})
    upstream_nodes(base / "datasets.py", ["_Dataset", "_ActiveTestingDataset", "QuadraticDatasetForLinReg"], env)
    upstream_nodes(base / "models/skmodels.py", ["BaseModel", "SKLearnModel", "LinearRegressionModel", "GaussianProcessRegressor"], env)
    upstream_nodes(base / "risk_estimators.py", ["RiskEstimator", "FancyUnbiasedRiskEstimator", "BiasedRiskEstimator"], env)
    acquisition_classes = ["AcquisitionFunction", "RandomAcquisition", "DistanceBasedAcquisition", "_LossAcquisitionBase", "_SurrogateAcquisitionBase", "_GPSurrogateAcquisitionBase", "GPSurrogateAcquisitionMSE"]
    upstream_nodes(base / "acquisition.py", acquisition_classes, env)
    ds_cfg = OmegaConf.create(dict(test_proportion=.9, n_points=50, standardize=False))
    model_cfg = OmegaConf.create(dict(name="LinearRegressionModel", task_type="regression"))
    gp_cfg = OmegaConf.create(dict(task_type="regression", efficient=False, length_scale=1.0))
    acq_cfg = OmegaConf.create(dict(sample=True, animate=False, animate_until=0, lazy_save=False, uniform_clip=False))
    output = []
    for run in range(runs):
        np.random.seed(20261113 + run)
        dataset = env["QuadraticDatasetForLinReg"](ds_cfg)
        model = env["LinearRegressionModel"](model_cfg)
        model.fit(*dataset.train_data)
        truth = np.mean((model.predict(dataset.x[dataset.test_idxs]) - dataset.y[dataset.test_idxs])**2)
        for strategy in ["RandomAcquisition", "DistanceBasedAcquisition", "GPSurrogateAcquisitionMSE"]:
            dataset.restart()
            if strategy == "GPSurrogateAcquisitionMSE":
                acquisition = env[strategy]([acq_cfg, run], dataset, model, gp_cfg)
            else:
                acquisition = env[strategy]([acq_cfg, run], dataset)
            estimator = env["FancyUnbiasedRiskEstimator"]("SELoss", dataset)
            biased = env["BiasedRiskEstimator"]("SELoss", dataset)
            for count in range(1, len(dataset.test_idxs)+1):
                idx, _ = acquisition.acquire()
                dataset.observe(idx)
                observed = dataset.test_observed
                prediction = model.predict(dataset.x[observed])
                estimate = estimator.estimate(prediction, dataset.y[observed], acquisition.weights)
                naive = biased.estimate(prediction, dataset.y[observed], acquisition.weights)
                output.append({"run": run, "strategy": strategy, "labels": count, "truth": truth,
                               "LURE": float(estimate), "biased": float(naive),
                               "squared_error": float((estimate-truth)**2), "biased_squared_error": float((naive-truth)**2)})
        if (run+1) % 20 == 0:
            print(f"ActiveTesting author reference {run+1}/{runs}", flush=True)
    df = pd.DataFrame(output)
    df.to_csv(RESULTS / "ACTIVE_TESTING_QUADRATIC_NATIVE_DRAWS.csv", index=False)
    summary = df.groupby(["strategy", "labels"])[["squared_error", "biased_squared_error"]].mean().reset_index()
    summary.to_csv(RESULTS / "ACTIVE_TESTING_QUADRATIC_NATIVE_SUMMARY.csv", index=False)
    return {"status": "AUTHOR_STANDARD_SYNTHETIC_EXAMPLE_EXECUTED_REDUCED_REPEATS", "author_data": True,
            "author_config": "conf/paper/SyntheticQuadraticLinear.yaml", "N": 50, "historical": 5, "pool": 45,
            "runs": runs, "author_runs": 5000, "acquisitions": ["RandomAcquisition", "DistanceBasedAcquisition", "GPSurrogateAcquisitionMSE"],
            "changes": ["100 repeated author examples instead of 5000", "pure AST-loaded unchanged author classes avoid optional image/torch imports", "NumPy np.int compatibility alias", "GPU efficient wrapper disabled for sklearn prediction equivalence"],
            "wall_time_s": time.perf_counter()-started,
            "labels10": summary[summary.labels == 10].to_dict("records")}


def robust_operator_control():
    started = time.perf_counter()
    namespace = {"np": np, "Tuple": __import__("typing").Tuple}
    upstream_nodes(HERE / "upstream/Robust-Active-Statistical-Inference/utils_optimization.py", ["MinMaxOptimizer_l2"], namespace)
    from public_methods import robust_geometric_path
    v = np.array([.005, .01, .04, .09, .15, .2, .3, .5])
    rate, c = .25, .04
    uncertainty = np.sqrt(v)
    pi = np.clip(rate * uncertainty / uncertainty.mean(), 0, 1)
    author = namespace["MinMaxOptimizer_l2"](uncertainty, pi, rate*len(v), len(v), c*np.sqrt(len(v)))
    value, rho = author.optimize()
    clean = robust_geometric_path(v, rate, c)
    err = abs(clean["objective"]-value)
    assert err < 1e-12 and abs(clean["rho"]-rho) < 1e-12
    return {"status": "NATIVE_OPERATOR_CONTROL_ONLY", "author_data": False,
            "author_class": "utils_optimization.py::MinMaxOptimizer_l2", "rho": float(rho), "objective": float(value),
            "cleanroom_max_absolute_error": err, "wall_time_s": time.perf_counter()-started,
            "original_benchmark_status": "Pew source .sav absent from author repo. Native full data benchmark remains unreplicated."}


def fixedsize_reference():
    from cps_sampling import ConditionalPoisson, inclusion_from_matrix
    from cps_sampling.variance import ht_variance
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    N, n = 200, 20
    uncertainty = rng.uniform(.05, 1., N)
    target_pi = n * uncertainty / uncertainty.sum()
    design = ConditionalPoisson(pi=target_pi, n=n)
    actual_pi = inclusion_from_matrix(design.q)
    Pi = design.joint_inclusion()
    y = rng.normal(size=N)
    estimates = []
    variances = []
    counts = np.zeros(N)
    for seed in range(2000):
        sample = design.sample(seed)
        assert len(sample) == n
        counts[sample] += 1
        estimates.append(np.sum(y[sample]/actual_pi[sample])/N)
        from cps_sampling.variance import syg_variance
        variances.append(syg_variance(y, actual_pi, Pi, sample, N))
    return {"status": "AUTHOR_CPS_PACKAGE_REFERENCE_EXECUTED", "N": N, "n": n, "draws": 2000,
            "target_to_actual_pi_max_error": float(abs(actual_pi-target_pi).max()),
            "actual_pi_to_piij_diagonal_max_error": float(abs(actual_pi-np.diag(Pi)).max()),
            "fixedsize_identity_error": float(abs(Pi.sum(1)-n*actual_pi).max()),
            "bias": float(np.mean(estimates)-np.mean(y)), "mse": float(np.mean((np.array(estimates)-np.mean(y))**2)),
            "exact_design_variance": float(ht_variance(y, actual_pi, Pi, N=N)),
            "mean_SYG_variance": float(np.mean(variances)), "wall_time_s": time.perf_counter()-started}


if __name__ == "__main__":
    results = {}
    for name, function in [("ASI_AlphaFold", asi_alphafold), ("RobustASI_operator", robust_operator_control),
                           ("FixedSize_author_package", fixedsize_reference), ("ActiveTesting_author_quadratic", active_testing_quadratic)]:
        try:
            results[name] = function()
        except Exception as exc:
            import traceback
            results[name] = {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()}
        (RESULTS / "NATIVE_REFERENCE_RESULTS.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(name, json.dumps(results[name]), flush=True)
