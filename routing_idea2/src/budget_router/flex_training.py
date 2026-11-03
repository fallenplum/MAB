"""Paid-only CPU FlexRouter engineering variant; not native reproduction.

The existing exact DPP loss is unchanged. This separate variant smooths hit
probability, freezes normalized PCA queries and learns only six embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np

from .neighbor_policies import (LegalMemberFeedback, FeedbackPermissionError,
    complete_labels_from_broker, flexrouter_kernel, fixed_k_dpp_greedy)

FLEX_VARIANT = "FlexRouter-fixedK3-frozenPCA-smoothedCoverage-adapted"
FLEX_STRATEGY = "flex_smoothed_adapted"
FLEX_PARAMETERS = {"epsilon": 1e-6, "steps": 200, "learning_rate": .01,
                   "weight_decay": .01, "auxiliary_weight": 1.,
                   "optimizer": "torch.optim.Adam", "dtype": "float64", "device": "cpu"}


@dataclass(frozen=True)
class PaidFullWarmupRow:
    question_id: str
    phi: np.ndarray
    member_feedback: tuple[LegalMemberFeedback, ...]


def normalized_frozen_pca(phi, feature_dim=None):
    value = np.asarray(phi, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or not len(value):
        raise ValueError("Finite frozen PCA vector required")
    if feature_dim is not None and value.shape != (feature_dim,):
        raise ValueError("Frozen PCA dimension changed")
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Frozen PCA context cannot have zero norm")
    return value/norm


def _training_data(rows, *, now):
    rows = tuple(rows)
    if not rows: raise FeedbackPermissionError("No complete paid warmup rows; do not scan free calibration labels")
    if len({row.question_id for row in rows}) != len(rows): raise ValueError("Duplicate paid training question")
    rows = tuple(sorted(rows, key=lambda row: row.question_id))
    x, y, identities = [], [], []
    dimension = None
    for row in rows:
        if not isinstance(row, PaidFullWarmupRow) or not row.question_id:
            raise FeedbackPermissionError("Typed paid full-row receipt required")
        records = tuple(row.member_feedback)
        if len(records) != 6 or any(not isinstance(record, LegalMemberFeedback) for record in records):
            raise FeedbackPermissionError("Exactly six authorized original member receipts required")
        if any(record.warmup is not True or record.split != "online_calibration" for record in records):
            raise FeedbackPermissionError("Only preplanned paid all-six calibration warmup can train this variant")
        if len({record.authorization_id for record in records}) != 1 or len({record.task for record in records}) != 1:
            raise FeedbackPermissionError("Full row requires one shared paid reference event and task")
        if records[0].task not in {"gsm8k", "bbh"}:
            raise ValueError("Frozen tasks are GSM8K and BBH")
        labels = complete_labels_from_broker(records, question_id=row.question_id, now=now)
        vector = normalized_frozen_pca(row.phi, dimension)
        dimension = len(vector)
        if dimension < 3: raise ValueError("Fixed three-member DPP requires at least three query/model dimensions")
        x.append(vector); y.append(labels)
        identities.append({"question_id": row.question_id, "normalized_phi": vector.tolist(),
            "labels": labels.tolist(), "authorization_id": records[0].authorization_id})
    digest = hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return np.asarray(x), np.asarray(y), identities, digest


def torch_exact_hit_probabilities(embeddings, normalized_phi, complete_labels):
    """Exact Eq7 hit probability, CPU FP64, without modifying the original loss."""
    import torch
    if (embeddings.device.type != "cpu" or embeddings.dtype != torch.float64
            or normalized_phi.device.type != "cpu" or normalized_phi.dtype != torch.float64):
        raise ValueError("This adapter permits CPU FP64 tensors only")
    if embeddings.ndim != 2 or embeddings.shape[0] != 6 or normalized_phi.ndim != 2 or normalized_phi.shape[1] != embeddings.shape[1]:
        raise ValueError("Six embeddings aligned to frozen PCA rows required")
    if complete_labels.shape != (len(normalized_phi), 6) or complete_labels.device.type != "cpu":
        raise FeedbackPermissionError("A complete six-label vector is required for every row")
    if (not torch.isfinite(embeddings).all() or not torch.isfinite(normalized_phi).all()
            or not torch.isfinite(complete_labels).all()
            or not torch.all((complete_labels == 0) | (complete_labels == 1))):
        raise ValueError("Finite embeddings, queries and complete binary labels required")
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if torch.any(norms <= 0): raise ValueError("Zero model embedding has undefined cosine kernel")
    logits = normalized_phi@embeddings.T
    competence = torch.sigmoid(logits)
    directions = embeddings/norms[:, None]
    weighted = competence[:, :, None]*directions[None, :, :]
    kernel = weighted@weighted.transpose(1, 2)
    identity = torch.eye(6, dtype=torch.float64, device="cpu")[None, :, :]
    failure = 1-complete_labels.to(dtype=torch.float64)
    failed_kernel = kernel*failure[:, :, None]*failure[:, None, :]
    sign_total, log_total = torch.linalg.slogdet(identity+kernel)
    sign_failed, log_failed = torch.linalg.slogdet(identity+failed_kernel)
    if not torch.all(sign_total > 0) or not torch.all(sign_failed > 0):
        raise RuntimeError("Invalid positive determinant in DPP normalizer")
    delta = log_failed-log_total
    if torch.any(delta > 1e-10): raise RuntimeError("Failure probability exceeds one beyond numerical roundoff")
    # Mathematical delta<=0; this only projects positive floating roundoff to0.
    delta = torch.minimum(delta, torch.zeros_like(delta))
    hit = -torch.expm1(delta)
    all_wrong = torch.all(complete_labels == 0, dim=1)
    # For F=all, determinant ratio is identically1. Set its exact zero explicitly
    # so coverage contributes no spurious finite-precision gradient.
    hit = torch.where(all_wrong, torch.zeros_like(hit), hit)
    return {"hit_probability": hit, "logits": logits, "competence": competence,
            "kernel": kernel, "all_wrong": all_wrong}


def torch_smoothed_flex_objective(embeddings, normalized_phi, complete_labels):
    """Named engineering loss: -log(eps+(1-eps)*exact_hit)+sum member BCE."""
    import torch
    import torch.nn.functional as functional
    exact = torch_exact_hit_probabilities(embeddings, normalized_phi, complete_labels)
    epsilon = FLEX_PARAMETERS["epsilon"]
    hit_terms = -torch.log(epsilon+(1-epsilon)*exact["hit_probability"])
    bce_terms = functional.binary_cross_entropy_with_logits(exact["logits"],
        complete_labels.to(dtype=torch.float64), reduction="none").sum(dim=1)
    loss = (hit_terms+bce_terms).mean()
    return {**exact, "smoothed_hit_terms": hit_terms, "bce_terms": bce_terms,
            "smoothed_hit_loss": hit_terms.mean(), "bce_loss": bce_terms.mean(), "loss": loss}


def require_finite_exact_training_rows(embeddings, normalized_phi, complete_labels):
    """Reject original infinite-loss rows; never route this to the smoothed loss."""
    import torch
    exact = torch_exact_hit_probabilities(embeddings, normalized_phi, complete_labels)
    if torch.any(exact["hit_probability"] <= 0):
        raise ValueError("Original exact FlexRouter hit loss is infinite; explicit smoothed variant required")
    return -torch.log(exact["hit_probability"])


class FrozenPCASmoothedFlexPolicy:
    name = FLEX_VARIANT

    def __init__(self, embeddings, training_report):
        values = np.asarray(embeddings, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 6 or values.shape[1] < 3 or not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) <= 0):
            raise ValueError("Six finite nonzero embeddings are required")
        self._embeddings = values.copy(); self._embeddings.flags.writeable = False
        self.feature_dim = values.shape[1]
        self._training_report = json.loads(json.dumps(training_report, allow_nan=False))

    def kernel(self, phi):
        return flexrouter_kernel(normalized_frozen_pca(phi, self.feature_dim), self._embeddings)

    def select(self, phi):
        kernel, _ = self.kernel(phi)
        # Existing exact Schur routine raises on singular selected blocks. No
        # jitter, pseudo-inverse, resampling or selection informed by labels.
        return fixed_k_dpp_greedy(kernel)["members"]

    def state_dict(self):
        return {"schema": "flex_smoothed_frozen_pca_policy_v1", "name": self.name,
                "feature_dim": self.feature_dim, "embeddings": self._embeddings.tolist(),
                "parameters": dict(FLEX_PARAMETERS), "training_report": self.training_report,
                "native_system": "NOT_RUN", "online_updates": False}

    @property
    def training_report(self):
        return json.loads(json.dumps(self._training_report, allow_nan=False))

    @classmethod
    def from_state_dict(cls, state):
        if (state.get("schema") != "flex_smoothed_frozen_pca_policy_v1" or state.get("name") != FLEX_VARIANT
                or state.get("parameters") != FLEX_PARAMETERS or state.get("online_updates") is not False
                or state.get("native_system") != "NOT_RUN"):
            raise ValueError("Frozen Flex variant/config binding mismatch")
        result = cls(state["embeddings"], state["training_report"])
        if state.get("feature_dim") != result.feature_dim: raise ValueError("Flex feature dimension mismatch")
        return result


def train_flex_adapted(rows, *, seed, now=0):
    """Train only on typed complete paid warmup; no environment/gold argument."""
    x, labels, identities, digest = _training_data(rows, now=now)
    import torch
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))
        parameter = torch.nn.Parameter(torch.randn((6, x.shape[1]), dtype=torch.float64,
            device="cpu", generator=generator)/np.sqrt(x.shape[1]))
        queries = torch.tensor(x, dtype=torch.float64, device="cpu")
        targets = torch.tensor(labels, dtype=torch.float64, device="cpu")
        optimizer = torch.optim.Adam([parameter], lr=FLEX_PARAMETERS["learning_rate"],
            weight_decay=FLEX_PARAMETERS["weight_decay"])
        losses = []
        for _ in range(FLEX_PARAMETERS["steps"]):
            optimizer.zero_grad(set_to_none=True)
            terms = torch_smoothed_flex_objective(parameter, queries, targets)
            if not torch.isfinite(terms["loss"]): raise RuntimeError("Non-finite explicit smoothed Flex training loss")
            losses.append(float(terms["loss"].detach()))
            terms["loss"].backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all(): raise RuntimeError("Non-finite Flex gradient")
            optimizer.step()
        with torch.no_grad():
            terms = torch_smoothed_flex_objective(parameter, queries, targets)
            if not torch.isfinite(terms["loss"]): raise RuntimeError("Non-finite final Flex loss")
            learned = parameter.detach().numpy().copy()
            exact_infinite = int((terms["hit_probability"] <= 0).sum())
            final_loss = float(terms["loss"])
        report = {"variant": FLEX_VARIANT, "seed": int(seed), "training_rows": len(x),
            "training_member_labels": int(labels.size), "training_question_ids": [r["question_id"] for r in identities],
            "training_reference_authorization_ids": [r["authorization_id"] for r in identities],
            "training_input_sha256": digest, "original_exact_infinite_loss_rows": exact_infinite,
            "all_wrong_full_rows": int(np.all(labels == 0, axis=1).sum()),
            "coverage_gradient_on_all_wrong_rows": "exactly zero; only auxiliary BCE trains those rows",
            "initial_smoothed_loss": losses[0], "final_smoothed_loss": final_loss,
            "optimization_loss_trace": losses, "parameters": dict(FLEX_PARAMETERS),
            "query_encoder": "frozen existing PCA row L2; no query encoder training",
            "initialization": "seeded CPU generator, N(0,1)/sqrt(phi_dimension)",
            "learned_parameters": int(6*x.shape[1]), "early_stopping": False,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "native_system": "NOT_RUN", "original_all_wrong_filter_rule_found_in_primary": False,
            "primary_source": "https://arxiv.org/html/2609.38585",
            "risk": "small paid warmup, smoothed loss, fixed PCA and K3 differ from original RoBERTa/RouterEval/adaptive policy"}
        return FrozenPCASmoothedFlexPolicy(learned, report)
    finally:
        torch.set_num_threads(old_threads); torch.use_deterministic_algorithms(old_deterministic)
