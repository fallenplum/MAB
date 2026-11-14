"""Fixed-mask adapter around the unchanged, pinned SaveRouter estimator.

Training accepts sparse observations from one generation trajectory. The
upstream SparseSupervision and FixedKSparseRouter implementations run as-is.
The caller supplies the common reward-free group mapping for evaluation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import types

import numpy as np


UPSTREAM_COMMIT = "250b7ef35a741c598aaf1bb264930d1e0bfd1a61"
UPSTREAM_HASHES = {
    "router.py": "d3f97af559021f1e8b991c4f4bfe879bcbed889f4cd10e9c2a8edda531ce92b2",
    "data.py": "5c55ace20cda0deb48155b00ddf4cc8d7032178113c18521573621c86789d70c",
    "grouping.py": "1af306fb87ad3ca6856dbc514dacc104a790e10c364d42b46c20a6900057ec69",
    "profiles.py": "319d97271cf05fa7853e42ef3591127b9b4566d441f84a85e396560923c8109f",
}
DEFAULT_SOURCE = Path(
    r"E:\语言模型组合路由\research\e12_shared_feedback_learning\private\runtime\literature\saverouter_source\saverouter"
)
PROFILE_NAME = "llmrouterbench"
N_ACTIONS = 20


def _source_root(source_dir=None):
    return Path(source_dir or os.environ.get("MAB_SAVEROUTER_SOURCE", str(DEFAULT_SOURCE))).resolve()


@lru_cache(maxsize=4)
def load_upstream(source_dir=None):
    root = _source_root(source_dir)
    for name, expected in UPSTREAM_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned SaveRouter source changed: " + name)
    namespace = "_mab_e13_saverouter_" + UPSTREAM_COMMIT[:12]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(root)]
        package.__package__ = namespace
        sys.modules[namespace] = package
        for name in ("data", "grouping", "profiles", "router"):
            full_name = namespace + "." + name
            spec = importlib.util.spec_from_file_location(full_name, root / (name + ".py"))
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load pinned SaveRouter module " + name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
    data = sys.modules[namespace + ".data"]
    router = sys.modules[namespace + ".router"]
    profiles = sys.modules[namespace + ".profiles"]
    profile = profiles.get_profile(PROFILE_NAME)
    expected_profile = {
        "prior_strength": 40.0,
        "prior_ridge_alpha": 10.0,
        "residual_ridge_alpha": 200.0,
        "residual_gamma": 2.0,
        "min_model_observations": 8,
    }
    if any(profile[key] != expected for key, expected in expected_profile.items()):
        raise ValueError("Pinned SaveRouter profile no longer matches the E13 lock")
    return data.SparseSupervision, router.FixedKSparseRouter, expected_profile


def _design(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all():
        raise ValueError("SaveRouter receives the shared finite PCA16 features without a constant column")
    return values


def _observations(query_indices, action_indices, rewards, costs, n_history):
    query = np.asarray(query_indices)
    action = np.asarray(action_indices)
    reward = np.asarray(rewards, dtype=np.float64)
    if any(values.ndim != 1 for values in (query, action, reward)):
        raise ValueError("Only one-dimensional sparse query/action/reward observations are accepted")
    if not len(query) or len(query) != len(action) or len(query) != len(reward):
        raise ValueError("Nonempty observation vectors must have equal length")
    if not np.issubdtype(query.dtype, np.integer) or not np.issubdtype(action.dtype, np.integer):
        raise ValueError("Query and action indices must be integer vectors")
    query, action = query.astype(np.int64), action.astype(np.int64)
    if np.any((query < 0) | (query >= n_history)) or np.any((action < 0) | (action >= N_ACTIONS)):
        raise ValueError("Observed query or action is outside its frozen domain")
    if not np.isin(reward, [0.0, 1.0]).all():
        raise ValueError("Each observation is one generation's true final binary aggregate reward")
    cost = None if costs is None else np.asarray(costs, dtype=np.float64)
    if cost is not None and (cost.shape != reward.shape or not np.isfinite(cost).all() or np.any(cost < 0)):
        raise ValueError("Observed standalone service costs must align with the sparse rewards")
    return query, action, reward, cost


class SaveRouterFixedMask:
    """Use the original estimator with a common fixed mask and common groups."""

    def __init__(self, source_dir=None):
        self.source_dir = source_dir

    def fit(self, features_h, groups_h, query_indices, action_indices, rewards, costs=None):
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        features = _design(features_h)
        groups = np.asarray(groups_h)
        if groups.shape != (len(features),) or not np.issubdtype(groups.dtype, np.integer) or np.any(groups < 0):
            raise ValueError("One legal historical integer group is required per input")
        query, action, reward, cost = _observations(query_indices, action_indices, rewards, costs, len(features))
        selected = np.unique(query)
        local_query = np.searchsorted(selected, query)
        counts = np.bincount(local_query, minlength=len(selected))
        if not np.all(counts == counts[0]):
            raise ValueError("Original SaveRouter requires a common number of distinct actions per revealed question")
        SparseSupervision, NativeRouter, parameters = load_upstream(self.source_dir)
        supervision = SparseSupervision(
            n_queries=len(selected), n_models=N_ACTIONS, query_indices=local_query,
            model_indices=action, rewards=reward, group_ids=groups[selected],
            k=int(counts[0]), costs=cost, model_ids=[str(a) for a in range(N_ACTIONS)],
            metadata={"collector": "E13_common_frozen_logger", "single_generation_trajectory": True,
                      "available_pairs": len(selected)*N_ACTIONS},
        )
        self.native_ = NativeRouter(**parameters).fit(features[selected], supervision)
        self.profile_ = parameters
        self.selected_history_indices_ = selected
        self.supervision_ = supervision
        self.fit_diagnostics_ = {
            "method": "SaveRouter-FixedMask",
            "upstream_commit": UPSTREAM_COMMIT,
            "source_hashes": dict(UPSTREAM_HASHES),
            "profile_name": PROFILE_NAME,
            "profile": dict(parameters),
            "native_training_summary": dict(self.native_.training_summary_),
            "history_inputs": len(features),
            "revealed_original_questions": len(selected),
            "observed_single_generation_tuples": len(reward),
            "fixed_feedback_actions_per_question": int(counts[0]),
            "groups_with_feedback": len(np.unique(groups[selected])),
            "native_group_axis_size": self.native_.n_groups_,
            "residual_action_models": sorted(self.native_.residual_models_),
            "cost_permission": "observed standalone combination service tokens" if cost is not None else "quality-only feedback",
            "decision_rule": "quality-only argmax with common external semantic tie rule",
            "group_mapping": "common external reward-free text mapping supplied by runner",
            "native_estimator_source_modified": False,
            "cpu_s": time.process_time()-cpu_start,
            "wall_s": time.perf_counter()-wall_start,
        }
        return self

    def predict_scores(self, features, groups):
        return self.native_.predict_scores(_design(features), group_ids=np.asarray(groups, dtype=np.int64))

    def predict_costs(self, features, groups):
        return self.native_.predict_costs(_design(features), group_ids=np.asarray(groups, dtype=np.int64))

    def state_dict(self):
        model = self.native_
        coefficient = np.zeros((16, N_ACTIONS))
        intercept = np.zeros(N_ACTIONS)
        fitted = np.zeros(N_ACTIONS, dtype=bool)
        for action, residual in model.residual_models_.items():
            coefficient[:, action] = residual.coef_
            intercept[action] = residual.intercept_
            fitted[action] = True
        state = {
            "upstream_commit": np.array(UPSTREAM_COMMIT),
            "profile_name": np.array(PROFILE_NAME),
            "group_scores": model.group_scores_.copy(),
            "additive_prior": model.additive_prior_.copy(),
            "group_sums": model.group_sums_.copy(),
            "group_counts": model.group_counts_.copy(),
            "residual_coefficient": coefficient,
            "residual_intercept": intercept,
            "residual_fitted": fitted,
            "residual_gamma": np.array(model.residual_gamma),
            "selected_history_indices": self.selected_history_indices_.copy(),
        }
        if model.group_costs_ is not None:
            state["group_costs"] = model.group_costs_.copy()
            state["global_costs"] = model.global_costs_.copy()
        return state


def predict_from_state(state, features, groups):
    features = _design(features)
    groups = np.asarray(groups, dtype=np.int64)
    if groups.shape != (len(features),) or np.any((groups < 0) | (groups >= len(state["group_scores"]))):
        raise ValueError("Evaluation groups must be in the saved native group domain")
    scores = state["group_scores"][groups].copy()
    for action in np.flatnonzero(state["residual_fitted"]):
        coefficient = state["residual_coefficient"][:, action].copy()
        residual = features @ coefficient + state["residual_intercept"][action]
        scores[:, action] += float(state["residual_gamma"]) * residual
    return scores


def fit_saverouter(*, features_h, groups_h, query_indices, action_indices, rewards,
                  query_features, query_groups, costs=None, source_dir=None):
    adapter = SaveRouterFixedMask(source_dir=source_dir).fit(
        features_h, groups_h, query_indices, action_indices, rewards, costs,
    )
    result = {
        "scores": adapter.predict_scores(query_features, query_groups),
        "state": adapter.state_dict(),
        "diagnostics": adapter.fit_diagnostics_,
    }
    if costs is not None:
        result["predicted_service_costs"] = adapter.predict_costs(query_features, query_groups)
    return result
