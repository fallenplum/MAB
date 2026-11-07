"""History-only initialisation and matched support-feedback update rules."""
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

METHODS = [
    "no_update", "fixed", "online_ridge", "online_ridge_shuffle",
    "hierarchical", "hierarchical_shuffle", "knn", "knn_shuffle",
    "adapter2", "adapter2_shuffle", "adapter4", "adapter4_shuffle",
]


@dataclass
class HistoryState:
    history_indices: np.ndarray
    pca: PCA
    scaler: StandardScaler
    train_features: np.ndarray
    coefficients: np.ndarray
    precision: np.ndarray
    precision_inverse: np.ndarray
    action_embedding: np.ndarray
    fixed_action: int
    query_features: np.ndarray
    query_prediction: np.ndarray
    query_covariance: np.ndarray
    historical_neighbor_residuals: np.ndarray
    historical_neighbor_distances: np.ndarray
    query_distances: np.ndarray
    donor_indices: np.ndarray


def semantic_labels(data):
    return np.where(data['task'] == 'gsm8k', 'gsm8k', data['sub'])


def history_indices(train, task_name, mode):
    labels = semantic_labels(train)
    if mode == 'unseen' and task_name != 'gsm8k':
        return np.flatnonzero(labels != task_name)
    return np.arange(len(labels))


def initialize(train, evaluation_inputs, task_name, mode, alpha=100.0):
    """This function accepts historical labels and outcome-free query inputs."""
    indices = history_indices(train, task_name, mode)
    embeddings = train['E'][indices]
    means = train['Y'][indices].mean(axis=1)
    pca = PCA(n_components=16, svd_solver='full').fit(embeddings)
    reduced = pca.transform(embeddings)
    scaler = StandardScaler().fit(reduced)
    features = np.column_stack((np.ones(len(indices)), scaler.transform(reduced)))
    precision = 3 * features.T @ features + alpha * np.eye(features.shape[1])
    coefficients = np.linalg.solve(precision, 3 * features.T @ means)
    precision_inverse = np.linalg.inv(precision)
    query_reduced = scaler.transform(pca.transform(evaluation_inputs['E']))
    query_features = np.column_stack((np.ones(len(query_reduced)), query_reduced))
    prediction = query_features @ coefficients
    labels = semantic_labels(train)[indices]
    same = labels == task_name
    broad = train['task'][indices] == ('gsm8k' if task_name == 'gsm8k' else 'bbh')
    fixed_mean = means[same].mean(0) if same.any() else means[broad].mean(0)
    fixed_action = int(np.argmax(fixed_mean))

    # Action directions are learned only from historical semantic-task residuals.
    residuals = means - features @ coefficients
    group_residuals = np.stack([residuals[labels == name].mean(0) for name in sorted(set(labels))])
    group_residuals -= group_residuals.mean(axis=1, keepdims=True)
    _, _, right = np.linalg.svd(group_residuals, full_matrices=False)
    action_embedding = right[:4].T * np.sqrt(20.0)

    # Cosine distance in frozen normalized MiniLM space. Five historical neighbors
    # remain available for every action even when new support feedback is restricted.
    distances = np.maximum(0.0, 1.0 - evaluation_inputs['E'] @ embeddings.T)
    order = np.argsort(distances, axis=1)[:, :5]
    neighbor_distances = np.take_along_axis(distances, order, axis=1)
    neighbor_residuals = residuals[order]
    query_distances = np.maximum(0.0, 1.0 - evaluation_inputs['E'] @ evaluation_inputs['E'].T)

    # Wrong-task feedback donors come from the frozen historical pool. No query
    # label enters the negative control, including at k=1.
    donor_pool = indices[labels != task_name]
    if task_name != 'gsm8k':
        donor_pool = donor_pool[train['task'][donor_pool] == 'bbh']
    assert len(donor_pool)
    donor_indices = []
    for question_id in evaluation_inputs['ids']:
        digest = hashlib.sha256((str(question_id) + '|20261107').encode()).hexdigest()
        donor_indices.append(donor_pool[int(digest[:16], 16) % len(donor_pool)])
    return HistoryState(
        indices, pca, scaler, features, coefficients, precision, precision_inverse,
        action_embedding, fixed_action, query_features, prediction,
        query_features @ precision_inverse @ query_features.T,
        neighbor_residuals, neighbor_distances, query_distances, np.asarray(donor_indices),
    )


def posterior_prediction(state, support, rewards, observed):
    """Exact full-parameter historical Ridge posterior using Woodbury."""
    result = state.query_prediction.copy()
    residual = rewards - state.query_prediction[support]
    patterns = {tuple(np.flatnonzero(observed[:, action])) for action in range(20)}
    for pattern in patterns:
        if not pattern:
            continue
        columns = [action for action in range(20) if tuple(np.flatnonzero(observed[:, action])) == pattern]
        rows = np.asarray(pattern)
        support_rows = support[rows]
        covariance = state.query_covariance[np.ix_(support_rows, support_rows)]
        correction = np.linalg.solve(np.eye(len(rows)) + covariance, residual[np.ix_(rows, columns)])
        result[:, columns] += state.query_covariance[:, support_rows] @ correction
    return result


def bias_prediction(state, support, rewards, observed):
    residual = np.where(observed, rewards - state.query_prediction[support], 0.0)
    bias = residual.sum(0) / (observed.sum(0) + 4.0)
    return state.query_prediction + bias


def adapter_prediction(state, support, rewards, observed, dimension):
    embedding = state.action_embedding[:, :dimension]
    rows, actions = np.nonzero(observed)
    design = embedding[actions]
    targets = rewards[rows, actions] - state.query_prediction[support[rows], actions]
    posterior = np.linalg.solve(design.T @ design + 10.0 * np.eye(dimension), design.T @ targets)
    return state.query_prediction + embedding @ posterior


def knn_prediction(state, support, rewards, observed):
    """Shared Ridge plus change in historical+support neighbor residual estimate."""
    old_weights = 1.0 / np.maximum(state.historical_neighbor_distances, 0.01)
    old_estimate = (state.historical_neighbor_residuals * old_weights[:, :, None]).sum(1) / old_weights.sum(1)[:, None]
    support_residual = rewards - state.query_prediction[support]
    distances = np.column_stack((state.historical_neighbor_distances, state.query_distances[:, support]))
    updated = old_estimate.copy()
    for action in np.flatnonzero(observed.any(0)):
        available = np.r_[np.ones(5, dtype=bool), observed[:, action]]
        distance = np.where(available[None, :], distances, np.inf)
        order = np.argsort(distance, axis=1)[:, :5]
        selected_distance = np.take_along_axis(distance, order, axis=1)
        residual = np.column_stack((state.historical_neighbor_residuals[:, :, action], np.broadcast_to(support_residual[:, action], (len(distances), len(support)))))
        selected_residual = np.take_along_axis(residual, order, axis=1)
        weights = 1.0 / np.maximum(selected_distance, 0.01)
        updated[:, action] = (weights * selected_residual).sum(1) / weights.sum(1)
    return state.query_prediction + updated - old_estimate


def adapt(state, support, rewards, observed, method):
    if method == 'fixed':
        result = np.zeros_like(state.query_prediction)
        result[:, state.fixed_action] = 1.0
        return result
    if len(support) == 0 or method == 'no_update':
        return state.query_prediction.copy()
    if method == 'online_ridge':
        return posterior_prediction(state, support, rewards, observed)
    if method == 'hierarchical':
        return bias_prediction(state, support, rewards, observed)
    if method == 'knn':
        return knn_prediction(state, support, rewards, observed)
    if method in ('adapter2', 'adapter4'):
        return adapter_prediction(state, support, rewards, observed, int(method[-1]))
    raise ValueError(method)
