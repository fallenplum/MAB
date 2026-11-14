"""Question-scoped utilities for the independent E9 Bayes-gate experiment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import psutil

WORKSPACE = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("MAB_BAYES_RESULTS", str(WORKSPACE / "results/e9_bayes_gate")))
PRIVATE = RESULTS / "private"
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
E8 = ASSETS / "research/e8_decision_focused_combination_learning"
OUTCOMES = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
INPUTS = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
MULTIPLIERS = (0.1, 1.0, 10.0)
SHRINKAGES = (0.0, 1.0, 3.0, 9.0)
BASE_SEED = 20261113
PERMUTATION_SEEDS = tuple(202611150000 + i for i in range(100))

sys.path.insert(0, str(E8 / "public/qualification"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite {path}")
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def validate_scope(history, evaluation):
    allowed = {"E", "ids", "canonical", "global_indices"}
    if set(evaluation) != allowed:
        raise ValueError("Evaluation input packet must contain only E, ids, canonical, global_indices")
    if "Y" not in history or len(history["Y"]) != len(history["ids"]):
        raise ValueError("A legal historical reward packet is required")
    for key in ("ids", "canonical", "global_indices"):
        if set(history[key]) & set(evaluation[key]):
            raise ValueError(f"History/evaluation overlap in {key}")
    if history["Y"].shape != (480, 3, 20):
        raise ValueError("Unexpected original history shape")
    if not np.isin(history["Y"], (0, 1)).all():
        raise ValueError("Historical final rewards must be complete real binary scores")


def legal_fold(fold):
    root = E8 / f"private/qualification/fold_packs/fold{fold}"
    history_path = root / "history480.npz"
    evaluation_path = root / "evaluation_inputs_only120.npz"
    manifest = read_json(E8 / "private/qualification/FOLD_PACKS_MANIFEST.json")["folds"][fold]
    for path, key in ((history_path, "history"), (evaluation_path, "evaluation_inputs")):
        if sha256(path) != manifest[key]["sha256"]:
            raise ValueError("Original fold packet hash changed")
    history, evaluation = load_npz(history_path), load_npz(evaluation_path)
    validate_scope(history, evaluation)
    return history, evaluation


def legal_metadata(fold, expected_indices):
    """Metadata permission is separate from text-only model input permission."""
    path = E8 / f"private/qualification/fold_packs/fold{fold}/evaluation_metadata_only120.npz"
    packet = load_npz(path)
    if set(packet) != {"task", "sub", "global_indices"}:
        raise ValueError("Evaluation metadata must contain no rewards or responses")
    if not np.array_equal(packet["global_indices"], expected_indices):
        raise ValueError("Evaluation metadata identity mismatch")
    return packet


def choice(scores, action_order=None):
    """Original exact argmax in the frozen semantic combination column order."""
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite action scores")
    if action_order is None:
        return np.argmax(scores, axis=-1)
    order = np.argsort(np.asarray(action_order), kind="stable")
    return order[np.argmax(scores[..., order], axis=-1)]


def balanced_accuracy(rewards, actions, task):
    values = np.asarray(rewards)[np.arange(len(actions)), :, np.asarray(actions)].mean(axis=1)
    return float(np.mean([values[task == t].mean() for t in ("gsm8k", "bbh")]))


def prior_scores(rewards, task, sub, query_task, query_sub, kind="subtask", min_support=2):
    means = np.asarray(rewards).mean(axis=1)
    global_mean = means.mean(axis=0)
    result = np.repeat(global_mean[None], len(query_task), axis=0)
    if kind == "global":
        return result
    for t in sorted(set(query_task)):
        train_rows, query_rows = task == t, query_task == t
        if np.any(train_rows):
            result[query_rows] = means[train_rows].mean(axis=0)
    if kind == "task":
        return result
    for s in sorted(set(query_sub[query_task == "bbh"])):
        train_rows = (task == "bbh") & (sub == s)
        query_rows = (query_task == "bbh") & (query_sub == s)
        if int(train_rows.sum()) >= min_support:
            result[query_rows] = means[train_rows].mean(axis=0)
    return result


def permute_blocks(rewards, task, sub, seed, fold, region):
    """Permute whole 3x20 reward blocks within this fitting region only."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(fold), int(region)]))
    permutation = np.arange(len(rewards))
    group = np.array([str(t) + "/" + (str(s) if t == "bbh" else "") for t, s in zip(task, sub)])
    for key in sorted(set(group)):
        rows = np.flatnonzero(group == key)
        permutation[rows] = rng.permutation(rows)
    return np.asarray(rewards)[permutation], permutation


def process_state():
    info = psutil.Process().memory_info()
    return {"rss_bytes": int(info.rss), "peak_rss_bytes": int(getattr(info, "peak_wset", info.rss))}


class Timer:
    def __enter__(self):
        self.cpu, self.wall = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *_):
        self.record = {"cpu_s": time.process_time() - self.cpu, "wall_s": time.perf_counter() - self.wall,
                       **process_state()}


def selected_candidate(records, family):
    candidates = [row for row in records if row["family"] == family]
    maximum = max(row["inner_accuracy"] for row in candidates)
    tied = [row for row in candidates if row["inner_accuracy"] >= maximum - 1e-12]
    # All tie preferences are reward-blind and fixed before outer scoring.
    return sorted(tied, key=lambda r: (-r["multiplier"], -r.get("shrinkage", 0),
                                      0 if r.get("loss") == "bce" else 1))[0]
