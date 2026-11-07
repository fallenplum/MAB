"""History-only metadata feature ablations, without reading reward arrays."""

from pathlib import Path
import hashlib
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


MODES = ("text-only", "task-known", "subtask-oracle", "subtask-predicted")


def _hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _probabilities(train_x, labels, query_x, vocabulary, seed):
    """Fit on raw 384-dimensional frozen vectors and align absent class columns."""
    names = np.unique(labels)
    result = np.zeros((len(query_x), len(vocabulary)))
    if len(names) == 1:
        result[:, vocabulary.index(str(names[0]))] = 1.
        return result, {"iterations": 0, "constant_class": str(names[0]), "classes_seen": names.tolist()}
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    classifier.fit(train_x, labels)
    if np.max(classifier.n_iter_) >= classifier.max_iter:
        raise RuntimeError("Historical metadata classifier did not converge")
    probability = classifier.predict_proba(query_x)
    for j, name in enumerate(classifier.classes_):
        result[:, vocabulary.index(str(name))] = probability[:, j]
    return result, {"iterations": int(np.max(classifier.n_iter_)), "constant_class": None,
                    "classes_seen": classifier.classes_.tolist()}


def make_features(data, history, mode, seed=20261107):
    """Return all-question features plus serializable provenance and diagnostics.

    Predicted metadata uses out-of-fold probabilities at historical rows and a
    full-history classifier at external rows. External metadata labels are never
    consumed in this mode. Oracle metadata explicitly grants that privilege.
    """
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    if mode not in MODES:
        raise ValueError("Unknown metadata feature mode")
    embeddings = np.asarray(data["E"], dtype=float)
    history = np.asarray(history)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Finite frozen question embeddings are required")
    if history.ndim != 1 or history.dtype.kind not in "iu" or len(history) < 16:
        raise ValueError("Historical integer indices must support PCA16")
    history = history.astype(int, copy=True)
    if len(np.unique(history)) != len(history) or np.any(history < 0) or np.any(history >= len(embeddings)):
        raise ValueError("Invalid or duplicate historical indices")
    pca = PCA(n_components=16, svd_solver="full")
    reduced = pca.fit_transform(embeddings[history])
    scaler = StandardScaler().fit(reduced)
    text = np.column_stack((np.ones(len(embeddings)), scaler.transform(pca.transform(embeddings))))
    info = {"mode": mode, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": int(seed), "history_questions": len(history), "history_indices_sha256": _hash(history),
            "history_embedding_sha256": _hash(embeddings[history]),
            "pca_components": 16, "pca_and_standardization_fit": "historical questions only",
            "outcome_arrays_read": False, "query_metadata_permission": "none",
            "feature_blocks": [{"name": "constant_and_history_PCA16_standardized", "columns": 17}]}
    features = text
    if mode != "text-only":
        tasks = np.asarray(data["task"])
        if tasks.shape != (len(embeddings),):
            raise ValueError("Task labels must align with inputs")
        if mode == "task-known":
            features = np.column_stack((text, tasks == "bbh"))
            info["query_metadata_permission"] = "true broad task is supplied"
            info["feature_blocks"].append({"name": "known_BBH_indicator", "columns": 1})
        else:
            subtasks = np.asarray(data["sub"])
            if subtasks.shape != tasks.shape:
                raise ValueError("Subtask labels must align with inputs")
            historical_tasks = tasks[history]
            historical_subtasks = subtasks[history]
            names = tuple(sorted(set(historical_subtasks[historical_tasks == "bbh"].tolist())))
            info["historical_BBH_subtask_vocabulary"] = list(names)
            if mode == "subtask-oracle":
                bbh = tasks == "bbh"
                onehot = np.column_stack([bbh & (subtasks == name) for name in names]) if names else np.empty((len(tasks), 0))
                features = np.column_stack((text, bbh, onehot))
                info["query_metadata_permission"] = "privileged true broad task and BBH subtask"
                info["feature_blocks"].extend(({"name": "known_BBH_indicator", "columns": 1},
                                              {"name": "oracle_historical_subtask_onehot", "columns": len(names)}))
                info["unknown_subtask_rule"] = "BBH indicator only; no vocabulary fitted on external labels"
            else:
                if embeddings.shape[1] != 384:
                    raise ValueError("Predicted metadata classifier requires frozen 384-dimensional MiniLM")
                labels = np.asarray(["gsm8k" if task == "gsm8k" else "bbh/" + str(sub)
                                     for task, sub in zip(historical_tasks, historical_subtasks)])
                vocabulary = tuple(sorted(set(labels.tolist())))
                if any(np.sum(historical_tasks == task) < 3 for task in set(historical_tasks)):
                    raise ValueError("Three-fold OOF requires at least three historical questions per broad task")
                probabilities = np.zeros((len(embeddings), len(vocabulary)))
                oof = np.zeros((len(history), len(vocabulary)))
                fold_info = []
                splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed))
                for fold, (fit, held_out) in enumerate(splitter.split(history, historical_tasks)):
                    if set(history[fit]) & set(history[held_out]):
                        raise AssertionError("Historical OOF question identities overlap")
                    predicted, fitted = _probabilities(embeddings[history[fit]], labels[fit],
                                                     embeddings[history[held_out]], vocabulary, int(seed) + fold)
                    oof[held_out] = predicted
                    fold_info.append({"fold": fold, "fit_questions": len(fit), "held_out_questions": len(held_out),
                                      "fit_indices_sha256": _hash(history[fit]), "held_out_indices_sha256": _hash(history[held_out]),
                                      "missing_historical_classes": sorted(set(vocabulary) - set(fitted["classes_seen"])),
                                      **fitted})
                external = np.flatnonzero(~np.isin(np.arange(len(embeddings)), history))
                # Fit even if no external rows exist, so the recorded model rule is
                # well-defined. A single dummy historical input is discarded below.
                query = embeddings[external] if len(external) else embeddings[history[:1]]
                external_probs, full_info = _probabilities(embeddings[history], labels, query, vocabulary, int(seed))
                probabilities[history] = oof
                probabilities[external] = external_probs[:len(external)]
                if not np.allclose(probabilities.sum(axis=1), 1.) or np.any(probabilities < 0):
                    raise AssertionError("Metadata probability columns are invalid")
                predicted_labels = np.asarray(vocabulary)[np.argmax(oof, axis=1)]
                features = np.column_stack((text, probabilities))
                info.update({"query_metadata_permission": "none; predictions use external embeddings only",
                             "classifier": {"input": "raw frozen MiniLM384", "C": 1., "solver": "lbfgs",
                                            "objective": "multinomial for three or more historical classes",
                                            "class_weight": None, "max_iter": 1000},
                             "semantic_class_vocabulary": list(vocabulary),
                             "historical_probability_rule": "three-fold OOF, grouped by original question, broad-task stratified",
                             "external_probability_rule": "full-history classifier, no external task/subtask labels",
                             "fine_class_stratification": "not claimed; small BBH classes can be absent from an OOF fit and receive probability zero",
                             "folds": fold_info, "full_history_classifier": full_info,
                             "oof_accuracy": float(np.mean(predicted_labels == labels)),
                             "oof_confusion_matrix": confusion_matrix(labels, predicted_labels, labels=list(vocabulary)).tolist(),
                             "oof_confusion_order": list(vocabulary),
                             "stacking_boundary": "OOF history features and full-fit external features intentionally differ"})
                info["feature_blocks"].append({"name": "predicted_semantic_class_probabilities", "columns": len(vocabulary)})
    info["feature_dimension"] = features.shape[1]
    info["cpu_s"] = time.process_time() - started_cpu
    info["wall_s"] = time.perf_counter() - started_wall
    return features, info


def qualification():
    from e4_common import LegacyTemplate
    rng = np.random.default_rng(20261107)
    n = 72
    data = {"E": rng.normal(size=(n, 384)),
            "task": np.array(["gsm8k" if i % 2 else "bbh" for i in range(n)]),
            "sub": np.array(["" if i % 2 else ("logic" if i % 4 else "arithmetic") for i in range(n)]),
            "ids": np.array([str(i) for i in range(n)])}
    data["E"] /= np.linalg.norm(data["E"], axis=1, keepdims=True)
    history = np.arange(54)
    text, _ = make_features(data, history, "text-only")
    legacy_data = {**data, "Y": np.zeros((n, 3, 20))}
    legacy = LegacyTemplate(legacy_data, history, "P2", history[:3], {})
    assert np.array_equal(text, legacy.X)
    results = {}
    for mode in MODES:
        x, info = make_features(data, history, mode)
        assert np.isfinite(x).all() and np.array_equal(x[:, :17], text)
        changed = {key: value.copy() for key, value in data.items()}
        changed["Y"] = np.full((n, 3, 20), np.nan)
        changed["task"][54:] = "invalid"
        changed["sub"][54:] = "unseen"
        alternative, _ = make_features(changed, history, mode)
        if mode in ("text-only", "subtask-predicted"):
            assert np.array_equal(x, alternative)
        results[mode] = {"dimension": x.shape[1], "external_metadata_labels_used": mode in ("task-known", "subtask-oracle")}
    return {"status": "PASS", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "results": results, "checks": ["text_exact_E3_P2", "history_only_PCA_and_scaler",
                                             "all_modes_preserve_text_features", "no_Y_access",
                                             "predicted_external_label_invariance", "OOF_fold_question_disjointness"],
            "new_llm_calls": 0}


if __name__ == "__main__":
    import json
    print(json.dumps(qualification(), ensure_ascii=False, indent=2))
