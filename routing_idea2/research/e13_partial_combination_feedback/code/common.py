"""Frozen identities and input-only preparation for E13 partial feedback."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import psutil

CODE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("MAB_ASSET_ROOT", r"E:\语言模型组合路由"))
STUDY = Path(os.environ.get("MAB_E13_ROOT", str(ASSETS / "research/e13_partial_combination_feedback")))
PUBLIC = STUDY / "public"
PRIVATE = STUDY / "private"
BUDGETS = (60, 120, 240, 300, 360, 480)
LOGGER_SEEDS = (17, 29, 43)
GENERATION_SEEDS = (42, 137, 2024)
SCENARIOS = ("Main", "Pair")
METHODS = ("Raw", "Group", "GroupResidual", "RawCentered", "SaveRouterFixedMask",
           "GlobalFixed", "TaskFixed", "SubtaskFixed", "TaskFixedOracle", "SubtaskFixedOracle")
REFERENCE_VERSION = "V2_selector_train_frozen_bundle_15095d4cf8b4f2d564b62f814102a63fe05a9a9e5c9702e14c9b8232b80649db"
INPUTS = ASSETS / "artifacts/private/routing_redesign_20261104/full_feedback_inputs/v1/selector_train_inputs.npz"
OUTCOMES = ASSETS / "artifacts/private/routing_redesign_20261104/ff_v1/tr/selector_train_outcomes.npz"
SOURCE_HASHES = {
    "inputs": "57e39747ec2756bcd79ab00885b3c08ad4f91fdb3ec900cbed15146420cd17c3",
    "outcomes": "a01230f2bed3e7f9d05713bf9e3e4f1cf44987b41c460f91e1d83bc1c29f191e",
    "E7_splits": "48e631682448fca1cd6d53dd2c2c29dadda33cdbeb86d136ae3379a74a94f29a",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refuse to overwrite " + str(path))
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def choice(scores):
    values = np.asarray(scores)
    if not np.isfinite(values).all():
        raise ValueError("Action scores must be finite")
    return np.argmax(values, axis=-1)


def process_state():
    memory = psutil.Process().memory_info()
    return dict(rss_bytes=int(memory.rss), peak_rss_bytes=int(getattr(memory, "peak_wset", memory.rss)))


class Timer:
    def __enter__(self):
        self.cpu, self.wall = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *_):
        self.record = dict(cpu_s=time.process_time()-self.cpu, wall_s=time.perf_counter()-self.wall, **process_state())


def source_manifest():
    e12 = ASSETS / "research/e12_shared_feedback_learning"
    e12_protocol = read_json(e12 / "public/E12_PROTOCOL.json")
    records = []
    def add(path, expected=None, role="read-only original source"):
        path = Path(path)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("Original source hash changed: " + str(path))
        records.append(dict(path=str(path), sha256=digest, bytes=path.stat().st_size, role=role))
    add(INPUTS, SOURCE_HASHES["inputs"], "train600 input-only identities/features/metadata")
    add(OUTCOMES, SOURCE_HASHES["outcomes"], "cache manager only; true executed final rewards and execution costs")
    add(ASSETS / "research/e7_opportunity_and_feedback_learning/private/E7_SPLITS_FROZEN.npz", SOURCE_HASHES["E7_splits"])
    add(e12 / "public/E12_PROTOCOL.json")
    add(e12 / "public/code/e12_shared_feedback.py", e12_protocol["source_sha256"])
    for fold in range(5):
        add(ASSETS / f"research/e8_decision_focused_combination_learning/private/training/fold{fold}/M0_tuned/model_state.npz",
            e12_protocol["source_assets"][f"M_model{fold}"], "read only four preprocessing arrays; never reuse old reward-fitted coefficients")
        add(e12 / f"private/fold{fold}/GROUP_MAPPINGS.npz", role="H480 input/metadata-only T_M and group keys")
        add(e12 / f"private/fold{fold}/MASKS.npz", role="six original nested question masks")
    return records


def input_features(fold, inputs):
    e12 = ASSETS / "research/e12_shared_feedback_learning/private" / f"fold{fold}"
    groups = load_npz(e12 / "GROUP_MAPPINGS.npz")
    masks = load_npz(e12 / "MASKS.npz")
    history, evaluation = groups["H"].astype(int), groups["E"].astype(int)
    if len(history) != 480 or len(evaluation) != 120 or set(history) & set(evaluation):
        raise ValueError("Original fold scope changed")
    old_roles = load_npz(ASSETS / "research/e7_opportunity_and_feedback_learning/private/E7_SPLITS_FROZEN.npz")
    if not np.array_equal(history, old_roles[f"f{fold}_H"]) or not np.array_equal(evaluation, old_roles[f"f{fold}_E"]):
        raise ValueError("E12 input fold identity differs from hash-locked E7 roles")
    initial, update = old_roles[f"f{fold}_I"], old_roles[f"f{fold}_U"]
    for budget, original in ((240, initial), (300, np.r_[initial, update[:60]]),
                             (360, np.r_[initial, update[:120]]), (480, history)):
        if set(masks[f"B{budget}"].tolist()) != set(original.tolist()):
            raise ValueError("E12 feedback mask differs from its original E7 role")
    state_path = ASSETS / f"research/e8_decision_focused_combination_learning/private/training/fold{fold}/M0_tuned/model_state.npz"
    # NPZ member reads are explicit. The dense-label coefficient is never decoded.
    with np.load(state_path, allow_pickle=False) as saved:
        preprocessing = {name: saved[name] for name in ("pca_mean", "pca_components", "scaler_mean", "scaler_scale")}
    def phi(rows):
        projected = (inputs["embeddings"][rows] - preprocessing["pca_mean"]) @ preprocessing["pca_components"].T
        scaled = (projected-preprocessing["scaler_mean"])/preprocessing["scaler_scale"]
        return np.column_stack((np.ones(len(rows)), scaled))
    lookup = {tuple(row): index for index, row in enumerate(groups["group_keys"].tolist())}
    def true_groups(rows):
        return np.array([lookup[(str(inputs["task_labels"][q]), str(inputs["subtask_labels"][q]))] for q in rows])
    phi_h, phi_e = phi(history), phi(evaluation)
    group_h = true_groups(history)
    t = groups["T_M"]
    z = np.eye(28)[group_h]
    reconstructed = np.linalg.solve(phi_h.T @ phi_h+(1000/3)*np.eye(17), phi_h.T @ z)
    if np.max(np.abs(reconstructed-t)) > 1e-12:
        raise ValueError("Reward-free group mapping does not restore")
    previous = set()
    for budget in BUDGETS:
        rows = masks[f"B{budget}"]
        selected = set(map(int, rows))
        if len(rows) != budget or not previous <= selected <= set(history):
            raise ValueError("Original nested mask changed")
        if any(int(np.sum(inputs["task_labels"][rows] == task)) != budget//2 for task in ("gsm8k", "bbh")):
            raise ValueError("Original task-balanced mask changed")
        if len(set(true_groups(rows))) != 28:
            raise ValueError("All 28 training groups must have input coverage at every budget")
        previous = selected
    return dict(H=history, E=evaluation, phi_h=phi_h, phi_e=phi_e, group_h=group_h,
                predicted_group_e=choice(phi_e @ t), oracle_group_e=true_groups(evaluation),
                group_keys=groups["group_keys"], group_to_task=groups["group_keys"][:, 0],
                T=t, **masks, **preprocessing)


def protocol():
    return read_json(PUBLIC / "E13_PROTOCOL.json")


def verify_source_lock():
    locked = protocol()
    for relative, expected in locked["source_hashes"].items():
        if sha256(CODE / relative) != expected:
            raise ValueError("Frozen E13 training source changed: " + relative)
    return locked
