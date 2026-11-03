"""CPU-only E4 neighbours with explicit shared-protocol adaptations.

This module neither opens experiment matrices nor obtains labels.  The trusted
execution/broker adapter must issue the typed receipts below after payment,
execution and authorized reference release.  A string receipt is not itself a
cryptographic proof of payment.  No original-system score or theorem is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit


CES_COMMIT = "9e071c51b405f4fbfc36b0453ec6a91ddb36d1b7"
_CES_CORE_SHA = "ac066dfe08517d98180b7cf8792358493b3bfaf95bd38a26a22dd334ec95bc8e"
_CES_PROJECTOR_SHA = "397b78f6544239b88baed2e528f599dfcf21b97482ce35e54871efe0f833392f"
_CES_LICENSE_SHA = "1f7167285b17af5f1f4dbc60d25a4c1a2603149f963bf7a530b169256385b321"
_ALLOWED_SPLITS = {"online_calibration", "online_stream"}
_TASKS = ("gsm8k", "bbh")
_TOL = 1e-8


class FeedbackPermissionError(ValueError):
    pass


class RelaxationInfeasible(ValueError):
    pass


def _vector(value, width, name, *, low=None, high=None):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {width}")
    if low is not None and np.any(result < low - _TOL):
        raise ValueError(f"{name} below its locked lower bound")
    if high is not None and np.any(result > high + _TOL):
        raise ValueError(f"{name} above its locked upper bound")
    return result.copy()


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer tick")
    return int(value)


def _members(value, n, k=None):
    raw = tuple(value)
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in raw):
        raise ValueError("Member indices must be integers")
    result = tuple(sorted(int(i) for i in raw))
    if len(set(result)) != len(result) or any(i < 0 or i >= n for i in result):
        raise ValueError("Illegal or repeated member index")
    if k is not None and len(result) != k:
        raise ValueError("This controlled policy requires fixed cardinality")
    return result


def _permission(record, now):
    _time(now, "now")
    _time(record.available_at, "available_at")
    if record.available_at > now:
        raise FeedbackPermissionError("Feedback has not arrived")
    if record.split not in _ALLOWED_SPLITS:
        raise FeedbackPermissionError("E1 and sealed-test feedback are forbidden")
    if not record.question_id or not record.authorization_id:
        raise FeedbackPermissionError("Broker question and authorization IDs required")
    if record.paid_reference is not True or record.generated is not True:
        raise FeedbackPermissionError("Only paid authorized feedback of generated answers is legal")
    if not np.isfinite(record.value) or not 0 <= record.value <= 1:
        raise ValueError("Correctness must be in [0,1]")


@dataclass(frozen=True)
class LegalMemberFeedback:
    question_id: str
    member_index: int
    value: float
    task: str
    available_at: int
    authorization_id: str
    split: str
    warmup: bool = False
    paid_reference: bool = True
    generated: bool = True


@dataclass(frozen=True)
class LegalComboFeedback:
    label_id: str
    question_id: str
    members: tuple[int, ...]
    value: float
    available_at: int
    authorization_id: str
    split: str
    paid_reference: bool = True
    generated: bool = True


@dataclass(frozen=True)
class PastMemberExecution:
    execution_id: str
    question_id: str
    member_index: int
    cost: float
    observed_at: int
    split: str
    warmup: bool = False
    paid_execution: bool = True


def _execution(record, now, n, cost_upper_bound):
    if not isinstance(record, PastMemberExecution):
        raise FeedbackPermissionError("Typed execution receipt required")
    _time(now, "now")
    _time(record.observed_at, "observed_at")
    _members((record.member_index,), n)
    if (record.observed_at > now or record.split not in _ALLOWED_SPLITS
            or record.paid_execution is not True or not record.execution_id or not record.question_id):
        raise FeedbackPermissionError("Only already observed paid execution cost is legal")
    if not np.isfinite(record.cost) or not 0 <= record.cost <= cost_upper_bound:
        raise ValueError("Cost exceeds predeclared normalization bound; do not clip it")


@lru_cache(maxsize=8)
def _import_file(path, sha256):
    name = "_audited_ces_" + sha256
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_audited_ces(audit_root=None):
    """Import only the two pinned, hash-checked MIT pure-numpy source files."""
    root = Path(audit_root) if audit_root is not None else Path(__file__).resolve().parents[2] / "third_party" / "ces_source_audit"
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf8"))
    if manifest.get("repository") != "JinkunXu/CES" or manifest.get("commit") != CES_COMMIT:
        raise ValueError("CES upstream identity mismatch")
    entries = {row["path"]: row for row in manifest["files"]}
    modules, bindings = [], {}
    for relative in ("src/bandits/linucb_combinatorial.py", "src/embeddings/projectors.py"):
        path = root / "upstream" / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entries[relative]["sha256"]:
            raise ValueError("CES source hash mismatch: " + relative)
        if relative.endswith("linucb_combinatorial.py") and digest != _CES_CORE_SHA:
            raise ValueError("CES core is not the audited snapshot")
        if relative.endswith("projectors.py") and digest != _CES_PROJECTOR_SHA:
            raise ValueError("CES feature builder is not the audited snapshot")
        modules.append(_import_file(str(path.resolve()), digest))
        bindings[relative] = digest
    license_path = root / "upstream" / "LICENSE"
    if hashlib.sha256(license_path.read_bytes()).hexdigest() != _CES_LICENSE_SHA or entries["LICENSE"]["sha256"] != _CES_LICENSE_SHA:
        raise ValueError("CES MIT license identity mismatch")
    return modules[0].LinUCBSharedCombinatorial, modules[1].build_phi, bindings


class CESPaidWarmupPolicy:
    """Original shared set-UCB core; paid two-task profile replaces MMLU-10d.

    alpha=3.5 and ridge=.1 follow the audited objective runner.  PCA input,
    two-task profile/masks and seeded random metadata projection are controlled
    engineering adaptations, not claims of original-system reproduction.
    """
    name = "CES-shared-paid-warmup-adapted"

    def __init__(self, feature_dim, *, n_members=6, k=3, alpha=3.5, ridge=.1,
                 projection_seed=42, lam_cost=0., cost_upper_bound=1., audit_root=None):
        if feature_dim < 1 or n_members != 6 or k != 3:
            raise ValueError("Frozen experiment requires N6/K3 and positive feature dimension")
        if not np.isfinite([alpha, ridge, lam_cost, cost_upper_bound]).all() or min(alpha, lam_cost) < 0 or min(ridge, cost_upper_bound) <= 0:
            raise ValueError("Illegal CES parameters")
        core, self._build_phi, self.source_hashes = load_audited_ces(audit_root)
        self.feature_dim, self.n_members, self.k = int(feature_dim), n_members, k
        self.alpha, self.ridge, self.lam_cost = float(alpha), float(ridge), float(lam_cost)
        self.cost_upper_bound, self.projection_seed = float(cost_upper_bound), int(projection_seed)
        self.core = core(p=3*self.feature_dim, alpha=self.alpha, lam=self.ridge)
        self._cap_sum = np.zeros((n_members, 2)); self._cap_count = np.zeros((n_members, 2), dtype=int)
        self._cost_sum = np.zeros(n_members); self._cost_count = np.zeros(n_members, dtype=int)
        self._seen_members, self._seen_costs, self._seen_combos = {}, {}, {}
        self._executed, self._profile, self._projector, self._v = {}, None, None, None

    def observe_paid_warmup(self, records, *, now):
        if self._profile is not None:
            raise ValueError("Paid capability profile is frozen")
        for record in records:
            if not isinstance(record, LegalMemberFeedback):
                raise FeedbackPermissionError("Typed broker member receipt required")
            _permission(record, now); _members((record.member_index,), self.n_members)
            if record.warmup is not True or record.split != "online_calibration" or record.task not in _TASKS:
                raise FeedbackPermissionError("Profile uses only paid arrived calibration warmup")
            key = (record.question_id, record.member_index)
            value = (record.task, float(record.value))
            if key in self._seen_members:
                if self._seen_members[key] != value:
                    raise ValueError("Conflicting member feedback")
                continue
            self._seen_members[key] = value
            j, task = record.member_index, _TASKS.index(record.task)
            self._cap_sum[j, task] += record.value; self._cap_count[j, task] += 1

    def observe_warmup_costs(self, records, *, now):
        if self._profile is not None:
            raise ValueError("Paid capability profile is frozen")
        for record in records:
            _execution(record, now, self.n_members, self.cost_upper_bound)
            if record.warmup is not True or record.split != "online_calibration":
                raise FeedbackPermissionError("Profile costs require calibration warmup receipts")
            old = self._seen_costs.get(record.execution_id)
            value = (record.member_index, float(record.cost))
            if old is not None:
                if old != value: raise ValueError("Conflicting execution receipt")
                continue
            self._seen_costs[record.execution_id] = value
            j = record.member_index
            self._cost_sum[j] += record.cost / self.cost_upper_bound; self._cost_count[j] += 1

    def freeze_profile(self):
        if self._profile is not None:
            return self.profile
        if not self._seen_members:
            raise FeedbackPermissionError("No arrived paid warmup member supervision")
        cap = np.divide(self._cap_sum, self._cap_count, out=np.full_like(self._cap_sum, .5), where=self._cap_count > 0)
        cost = np.divide(self._cost_sum, self._cost_count, out=np.ones(self.n_members), where=self._cost_count > 0)
        # Missing .5 accuracy and upper-bound cost are uninformative engineering
        # placeholders, not measurements; explicit mask columns preserve that fact.
        self._profile = np.column_stack((cap, cost, self._cap_count > 0, self._cost_count > 0)).astype(float)
        self._projector = np.random.default_rng(self.projection_seed).normal(size=(self.feature_dim, 6)) / np.sqrt(6.)
        self._v = self._profile @ self._projector.T
        return self.profile

    @property
    def profile(self):
        return None if self._profile is None else self._profile.copy()

    def member_features(self, phi):
        if self._profile is None: raise ValueError("Freeze paid profile before routing")
        x = _vector(phi, self.feature_dim, "frozen PCA context")
        return np.asarray([self._build_phi(x, v, beta=3., gamma=3.) for v in self._v])

    def select(self, phi, *, past_normalized_costs=None):
        phis = self.member_features(phi)
        costs = None if past_normalized_costs is None else _vector(past_normalized_costs, self.n_members, "past cost", low=0., high=1.)
        if self.lam_cost > 0 and costs is None:
            raise ValueError("Nonzero cost penalty requires past/locked costs")
        selected = self.core.select_topk(list(phis), self.k, costs=costs, lam_cost=self.lam_cost)
        return tuple(sorted(int(i) for i in selected))

    def register_execution(self, question_id, phi, members, execution_id, *, paid, split,
                           realized_total_cost=None):
        if paid is not True or split not in _ALLOWED_SPLITS or not question_id or not execution_id:
            raise FeedbackPermissionError("Bind only an actual paid allowed execution")
        members = _members(members, self.n_members, self.k)
        key = (question_id, members)
        z = self.member_features(phi)[list(members)].sum(axis=0)
        if realized_total_cost is not None and (not np.isfinite(realized_total_cost) or realized_total_cost < 0):
            raise ValueError("Realized total combo cost must be finite and nonnegative")
        if self.lam_cost > 0 and realized_total_cost is None:
            raise ValueError("Reward cost penalty requires the actual paid combo execution cost")
        bound = (z, None if realized_total_cost is None else float(realized_total_cost))
        if key in self._executed and (not np.array_equal(self._executed[key][0], z) or self._executed[key][1] != bound[1]):
            raise ValueError("An executed combination cannot change context")
        self._executed[key] = bound

    def observe_combo(self, record, *, now):
        if not isinstance(record, LegalComboFeedback):
            raise FeedbackPermissionError("Typed original-combination receipt required")
        _permission(record, now)
        if not record.label_id: raise ValueError("Original label identity required")
        key = (record.question_id, _members(record.members, self.n_members, self.k))
        if key not in self._executed: raise FeedbackPermissionError("Combination was not registered as executed")
        if key in self._seen_combos:
            if self._seen_combos[key][1] != float(record.value): raise ValueError("Conflicting original combination label")
            return False
        if any(v[0] == record.label_id for v in self._seen_combos.values()):
            raise ValueError("A raw label ID cannot identify multiple combinations")
        z, realized = self._executed[key]
        reward = float(record.value) - self.lam_cost*(0. if realized is None else realized)
        self.core.update(z, reward)
        self._seen_combos[key] = (record.label_id, float(record.value))
        return True

    def state_dict(self):
        return {"schema": "ces_shared_paid_warmup_state_v1", "name": self.name,
                "feature_dim": self.feature_dim, "alpha": self.alpha, "ridge": self.ridge,
                "lam_cost": self.lam_cost, "projection_seed": self.projection_seed,
                "cost_upper_bound": self.cost_upper_bound, "source_hashes": self.source_hashes,
                "profile": None if self._profile is None else self._profile.tolist(),
                "projector": None if self._projector is None else self._projector.tolist(),
                "capability_sums": self._cap_sum.tolist(), "cost_sums": self._cost_sum.tolist(),
                "capability_counts": self._cap_count.tolist(), "cost_counts": self._cost_count.tolist(),
                "V_inv": self.core.V_inv.tolist(), "b": self.core.b.tolist(), "w_hat": self.core.w_hat.tolist(),
                "arrived_member_keys": [[q, i, *value] for (q, i), value in self._seen_members.items()],
                "arrived_combo_keys": [[q, list(team), *value] for (q, team), value in self._seen_combos.items()],
                "observed_costs": [[key, *value] for key, value in self._seen_costs.items()],
                "executed_bindings": [[q, list(team), z.tolist(), cost] for (q, team), (z, cost) in self._executed.items()],
                "native_system": "NOT_RUN", "pending_feedback_values": "never accepted or exported"}

    def load_state_dict(self, state):
        """Restore only a source/config-bound snapshot; no pending target values."""
        expected = (self.feature_dim, self.alpha, self.ridge, self.lam_cost, self.projection_seed, self.cost_upper_bound)
        received = tuple(state.get(key) for key in ("feature_dim", "alpha", "ridge", "lam_cost", "projection_seed", "cost_upper_bound"))
        if state.get("schema") != "ces_shared_paid_warmup_state_v1" or received != expected or state.get("source_hashes") != self.source_hashes:
            raise ValueError("CES checkpoint source/config binding mismatch")
        inverse = np.asarray(state["V_inv"], dtype=float)
        if inverse.shape != (3*self.feature_dim, 3*self.feature_dim) or not np.isfinite(inverse).all() or not np.allclose(inverse, inverse.T) or np.linalg.eigvalsh(inverse).min() <= 0:
            raise ValueError("Invalid CES posterior inverse")
        rhs = _vector(state["b"], 3*self.feature_dim, "posterior rhs")
        weights = _vector(state["w_hat"], 3*self.feature_dim, "posterior weights")
        if not np.allclose(inverse@rhs, weights): raise ValueError("Inconsistent CES posterior")
        profile = None if state["profile"] is None else np.asarray(state["profile"], dtype=float)
        projector = None if state["projector"] is None else np.asarray(state["projector"], dtype=float)
        if profile is not None and (profile.shape != (6, 6) or not np.isfinite(profile).all() or projector is None or projector.shape != (self.feature_dim, 6) or not np.isfinite(projector).all()):
            raise ValueError("Invalid CES frozen profile/projector")
        cap_sum = np.asarray(state["capability_sums"], dtype=float)
        cap_count = np.asarray(state["capability_counts"], dtype=int)
        cost_sum = _vector(state["cost_sums"], 6, "warmup cost sums", low=0.)
        cost_count = np.asarray(state["cost_counts"], dtype=int)
        if cap_sum.shape != (6, 2) or cap_count.shape != (6, 2) or cost_count.shape != (6,) or not np.isfinite(cap_sum).all() or np.any(cap_count < 0) or np.any(cost_count < 0) or np.any(cap_sum < 0) or np.any(cap_sum > cap_count+_TOL) or np.any(cost_sum > cost_count+_TOL):
            raise ValueError("Invalid CES warmup summary")
        executed = {}
        for q, team, z, cost in state["executed_bindings"]:
            key = (q, _members(team, 6, 3))
            if cost is not None and (not np.isfinite(cost) or cost < 0): raise ValueError("Invalid paid execution cost")
            executed[key] = (_vector(z, 3*self.feature_dim, "executed CES feature"), cost)
        self.core.load_state(V_inv=inverse, b=rhs, w_hat=weights)
        self._profile, self._projector = profile, projector
        self._v = None if profile is None else profile@projector.T
        self._cap_sum, self._cap_count, self._cost_sum, self._cost_count = cap_sum, cap_count, cost_sum, cost_count
        self._seen_members = {(q, int(i)): (task, float(v)) for q, i, task, v in state["arrived_member_keys"]}
        self._seen_combos = {(q, _members(team, 6, 3)): (label, float(v)) for q, team, label, v in state["arrived_combo_keys"]}
        self._seen_costs = {key: (int(i), float(v)) for key, i, v in state["observed_costs"]}
        self._executed = executed


def confidence_radius(counts, *, t, n_members, delta):
    """Paper section 4.1; zero count explicitly extends radius to infinity."""
    counts = np.asarray(counts)
    if counts.shape != (n_members,) or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError("Observation counts must be nonnegative integers")
    if t < 1 or int(t) != t or not 0 < delta <= 1:
        raise ValueError("Paper confidence bound requires t>=1, delta in (0,1]")
    result = np.full(n_members, np.inf)
    positive = counts > 0
    result[positive] = np.sqrt(np.log(2*np.pi**2*n_members*t**3/(3*delta))/(2*counts[positive]))
    return result


def structural_reward(members, means, model):
    means = np.asarray(means, dtype=float)
    selected = means[list(_members(members, len(means)))]
    if model == "AWC": return float(1-np.prod(1-selected))
    if model == "SUC": return float(np.sum(selected))
    if model == "AIC": return float(np.prod(selected))
    raise ValueError("Reward model must be AWC, SUC or AIC")


def relaxed_reward(z, means, model):
    z = _vector(z, len(means), "fractional selection", low=0., high=1.)
    means = _vector(means, len(z), "member means", low=0., high=1.)
    if model == "AWC": return float(1-np.prod(1-means*z))
    if model == "SUC": return float(z @ means)
    if model == "AIC":
        used = z > 0
        if np.any(means[used] == 0): return 0.
        return float(np.exp(z[used] @ np.log(means[used])))
    raise ValueError("Reward model must be AWC, SUC or AIC")


def _fractional(z, k):
    z = _vector(z, len(z), "fractional selection", low=0., high=1.)
    z = np.clip(z, 0., 1.)
    if abs(float(z.sum())-k) > _TOL:
        raise ValueError("Fixed-K rounding requires sum(z)=K")
    return z


def dependent_round(z, *, k, rng):
    """C2MAB-V Appendix B Alg3; preserves marginals and cardinality, not cost."""
    z = _fractional(z, k)
    for _ in range(len(z)+1):
        fractional = np.flatnonzero((z > 1e-10) & (z < 1-1e-10))
        if len(fractional) == 0: break
        if len(fractional) == 1:
            raise ValueError("Numerically inconsistent single fractional coordinate")
        i, j = fractional[:2]
        p, q = min(1-z[i], z[j]), min(z[i], 1-z[j])
        if rng.random() < q/(p+q): z[i] += p; z[j] -= p
        else: z[i] -= q; z[j] += q
    result = tuple(int(i) for i in np.flatnonzero(z > .5))
    if len(result) != k: raise RuntimeError("Dependent rounding did not preserve fixed cardinality")
    return result


def base_convex_decomposition(z, *, k):
    """Uniform-matroid decomposition, NOT exact discrete reward optimization.

    All bases are included irrespective of single-round cost. Removing costly
    bases would replace the paper's matroid polytope by a different integer hull.
    """
    z = _fractional(z, k)
    bases = tuple(itertools.combinations(range(len(z)), k))
    incidence = np.zeros((len(z), len(bases)))
    for j, base in enumerate(bases): incidence[list(base), j] = 1.
    solved = linprog(np.zeros(len(bases)), A_eq=np.vstack((incidence, np.ones(len(bases)))),
                     b_eq=np.r_[z, 1.], bounds=(0., None), method="highs")
    if not solved.success: raise RelaxationInfeasible("Uniform-base convex decomposition failed")
    keep = np.flatnonzero(solved.x > 1e-10)
    weights = solved.x[keep]; weights /= weights.sum()
    chosen = tuple(bases[i] for i in keep)
    if not np.allclose(incidence[:, keep] @ weights, z, atol=1e-8, rtol=0):
        raise RuntimeError("Convex decomposition changed selection marginals")
    return chosen, weights


def swap_round(z, *, k, rng, decomposition=None):
    """Alg2 specialized to equal-size bases: its augmentation G is empty."""
    z = _fractional(z, k)
    bases, weights = base_convex_decomposition(z, k=k) if decomposition is None else decomposition
    weights = _vector(weights, len(bases), "decomposition weights", low=0.)
    if not np.isclose(weights.sum(), 1., atol=1e-8, rtol=0): raise ValueError("Weights must sum to one")
    reconstructed = np.zeros(len(z))
    for base, weight in zip(bases, weights): reconstructed[list(_members(base, len(z), k))] += weight
    if not np.allclose(reconstructed, z, atol=1e-8, rtol=0): raise ValueError("Decomposition is not bound to z")
    active = [(set(base), float(w)) for base, w in zip(bases, weights) if w > 0]
    first, mass = active[0]
    for other, weight in active[1:]:
        while first != other:
            i, j = min(first-other), min(other-first)
            if rng.random() < weight/(mass+weight): first.remove(i); first.add(j)
            else: other.remove(j); other.add(i)
        mass += weight
    return tuple(sorted(first))


class C2MABVPolicy:
    """Paper-based fixed-parallel-K3 adaptation; no native guarantees.

    SUC/AIC solve the paper's LP/log-LP. AWC keeps Eq3's multilinear
    objective but uses a documented local SLSQP solver, NOT a global/1-1/e
    approximation claim. All modes impose equality K3 in this experiment.
    """
    name = "C2MAB-V-paper-based-shared-protocol-adapted"

    def __init__(self, *, reward_model, budget_threshold, n_members=6, k=3,
                 delta=.05, alpha_mu=.3, alpha_cost=.05, cost_upper_bound=1.):
        if n_members != 6 or k != 3 or reward_model not in {"AWC", "SUC", "AIC"}:
            raise ValueError("Frozen N6/K3 and AWC/SUC/AIC required")
        if not np.isfinite([budget_threshold, delta, alpha_mu, alpha_cost, cost_upper_bound]).all() or budget_threshold < 0 or not 0 < delta <= 1 or min(alpha_mu, alpha_cost, cost_upper_bound) <= 0:
            raise ValueError("Illegal C2MAB-V parameters")
        self.reward_model, self.budget_threshold = reward_model, float(budget_threshold)
        self.n_members, self.k = n_members, k
        self.delta, self.alpha_mu, self.alpha_cost = float(delta), float(alpha_mu), float(alpha_cost)
        self.cost_upper_bound = float(cost_upper_bound)
        self.reward_counts = np.zeros(n_members, dtype=int); self.cost_counts = np.zeros(n_members, dtype=int)
        self.reward_sums = np.zeros(n_members); self.cost_sums = np.zeros(n_members)
        self._seen_members, self._seen_costs = {}, {}
        self._latest_available_at = 0

    def observe_members(self, records, *, now):
        for record in records:
            if not isinstance(record, LegalMemberFeedback): raise FeedbackPermissionError("Typed broker receipt required")
            _permission(record, now); _members((record.member_index,), self.n_members)
            key, value = (record.question_id, record.member_index), float(record.value)
            if key in self._seen_members:
                if self._seen_members[key] != value: raise ValueError("Conflicting member feedback")
                continue
            self._seen_members[key] = value
            self.reward_counts[record.member_index] += 1; self.reward_sums[record.member_index] += value
            self._latest_available_at = max(self._latest_available_at, int(record.available_at))

    def observe_costs(self, records, *, now):
        for record in records:
            _execution(record, now, self.n_members, self.cost_upper_bound)
            value = (record.member_index, float(record.cost))
            if record.execution_id in self._seen_costs:
                if self._seen_costs[record.execution_id] != value: raise ValueError("Conflicting execution cost")
                continue
            self._seen_costs[record.execution_id] = value
            self.cost_counts[record.member_index] += 1
            self.cost_sums[record.member_index] += record.cost/self.cost_upper_bound
            self._latest_available_at = max(self._latest_available_at, int(record.observed_at))

    def confidence_bounds(self, *, now):
        t = _time(now, "now")+1
        if now < self._latest_available_at:
            raise FeedbackPermissionError("Cannot move policy clock behind consumed observation time")
        means = np.divide(self.reward_sums, self.reward_counts, out=np.zeros(self.n_members), where=self.reward_counts > 0)
        costs = np.divide(self.cost_sums, self.cost_counts, out=np.zeros(self.n_members), where=self.cost_counts > 0)
        r_mu = confidence_radius(self.reward_counts, t=t, n_members=self.n_members, delta=self.delta)
        r_cost = confidence_radius(self.cost_counts, t=t, n_members=self.n_members, delta=self.delta)
        return {"reward_means": means, "cost_means": costs,
                "reward_ucb": np.minimum(means+self.alpha_mu*r_mu, 1.),
                "cost_lcb": np.maximum(costs-self.alpha_cost*r_cost, 0.),
                "reward_radius": r_mu, "cost_radius": r_cost}

    def solve_relaxation(self, *, now):
        bounds = self.confidence_bounds(now=now); means, costs = bounds["reward_ucb"], bounds["cost_lcb"]
        arguments = dict(A_ub=costs[None, :], b_ub=[self.budget_threshold],
                         A_eq=np.ones((1, self.n_members)), b_eq=[self.k], bounds=[(0., 1.)]*self.n_members, method="highs")
        if self.reward_model == "AIC" and np.any(means == 0):
            raise NotImplementedError("Zero reward-UCB log-LP degeneracy is not silently floored")
        objective = -means if self.reward_model != "AIC" else -np.log(means)
        solved = linprog(objective, **arguments)
        if not solved.success: raise RelaxationInfeasible("No fixed-K3 point satisfies the optimistic cost constraint")
        z, solver = solved.x, "scipy.linprog-highs"
        if self.reward_model == "AWC":
            # The publication does not specify executable continuous-greedy
            # details. This is an explicit local solver adaptation of its Eq3.
            constraints = ({"type": "eq", "fun": lambda x: x.sum()-self.k},
                           {"type": "ineq", "fun": lambda x: self.budget_threshold-costs@x})
            candidate = minimize(lambda x: -relaxed_reward(x, means, "AWC"), z,
                                 method="SLSQP", bounds=[(0., 1.)]*self.n_members,
                                 constraints=constraints, options={"maxiter": 300, "ftol": 1e-11})
            if not candidate.success: raise RuntimeError("AWC local relaxation failed: " + candidate.message)
            z, solver = candidate.x, "scipy.SLSQP-local-no-global-or-approximation-guarantee"
        z = _fractional(z, self.k)
        if costs@z > self.budget_threshold+1e-7: raise RuntimeError("Relaxation violates its own cost constraint")
        return {**bounds, "fractional_selection": z, "solver": solver,
                "relaxed_structural_reward": relaxed_reward(z, means, self.reward_model),
                "relaxed_cost_lcb": float(costs@z)}

    def select(self, *, now, rng):
        result = self.solve_relaxation(now=now)
        z = result["fractional_selection"]
        chosen = swap_round(z, k=self.k, rng=rng) if self.reward_model == "AWC" else dependent_round(z, k=self.k, rng=rng)
        return {**result, "members": chosen, "rounding": "Alg2-fixed-base-swap" if self.reward_model == "AWC" else "Alg3-dependent-pair",
                "selected_structural_reward_ucb": structural_reward(chosen, result["reward_ucb"], self.reward_model),
                "selected_cost_lcb": float(result["cost_lcb"][list(chosen)].sum()),
                "cost_constraint_semantics": "expected/soft; external service-reserved hard-cap admission still required",
                "actual_aggregate_reward": None, "native_system": "NOT_RUN"}

    def state_dict(self):
        return {"schema": "c2mabv_shared_protocol_state_v1", "name": self.name,
                "reward_model": self.reward_model, "budget_threshold": self.budget_threshold,
                "delta": self.delta, "alpha_mu": self.alpha_mu, "alpha_cost": self.alpha_cost,
                "cost_upper_bound": self.cost_upper_bound,
                "latest_available_at": self._latest_available_at,
                "reward_counts": self.reward_counts.tolist(), "cost_counts": self.cost_counts.tolist(),
                "reward_sums": self.reward_sums.tolist(), "cost_sums": self.cost_sums.tolist(),
                "arrived_member_keys": [[q, i, v] for (q, i), v in self._seen_members.items()],
                "executed_costs": [[key, *value] for key, value in self._seen_costs.items()],
                "native_system": "NOT_RUN", "pending_feedback_values": "never accepted or exported"}

    def load_state_dict(self, state):
        keys = ("reward_model", "budget_threshold", "delta", "alpha_mu", "alpha_cost", "cost_upper_bound")
        if state.get("schema") != "c2mabv_shared_protocol_state_v1" or any(state.get(key) != getattr(self, key) for key in keys):
            raise ValueError("C2MAB-V checkpoint config binding mismatch")
        reward_counts = _vector(state["reward_counts"], 6, "reward counts", low=0.)
        cost_counts = _vector(state["cost_counts"], 6, "cost counts", low=0.)
        reward_sums = _vector(state["reward_sums"], 6, "reward sums", low=0.)
        cost_sums = _vector(state["cost_sums"], 6, "cost sums", low=0.)
        if np.any(reward_counts != np.rint(reward_counts)) or np.any(cost_counts != np.rint(cost_counts)) or np.any(reward_sums > reward_counts+_TOL) or np.any(cost_sums > cost_counts+_TOL):
            raise ValueError("Invalid C2MAB-V observation summary")
        self.reward_counts, self.cost_counts = reward_counts.astype(int), cost_counts.astype(int)
        self.reward_sums, self.cost_sums = reward_sums, cost_sums
        self._seen_members = {(q, int(i)): float(v) for q, i, v in state["arrived_member_keys"]}
        self._seen_costs = {key: (int(i), float(v)) for key, i, v in state["executed_costs"]}
        self._latest_available_at = _time(state["latest_available_at"], "latest_available_at")


def flexrouter_kernel(query_embedding, model_embeddings):
    """Paper Eq4-6: sigmoid competence and cosine-Gram PSD kernel."""
    models = np.asarray(model_embeddings, dtype=float)
    if models.ndim != 2 or not np.isfinite(models).all() or len(models) != 6:
        raise ValueError("Six finite model embeddings required")
    query = _vector(query_embedding, models.shape[1], "query embedding")
    norms = np.linalg.norm(models, axis=1)
    if np.any(norms <= 0): raise ValueError("Cosine model embeddings must be nonzero")
    q = expit(models@query)
    normalized = models/norms[:, None]
    kernel = (q[:, None]*normalized) @ (q[:, None]*normalized).T
    return kernel, q


def _dpp_kernel(kernel):
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1] or not np.isfinite(kernel).all():
        raise ValueError("Finite square DPP kernel required")
    if not np.allclose(kernel, kernel.T, atol=1e-10, rtol=0) or np.linalg.eigvalsh(kernel).min() < -1e-9:
        raise ValueError("DPP kernel must be symmetric PSD")
    return kernel


def dpp_coverage_probability(kernel, complete_labels):
    """Exact original all-model-label objective; missing labels never become 0."""
    kernel = _dpp_kernel(kernel)
    labels = np.asarray(complete_labels)
    if labels.shape != (len(kernel),) or not np.isin(labels, [0, 1]).all():
        raise FeedbackPermissionError("Coverage loss requires complete binary labels for every member")
    failure = np.flatnonzero(labels == 0)
    _, total = np.linalg.slogdet(np.eye(len(kernel))+kernel)
    _, failed = np.linalg.slogdet(np.eye(len(failure))+kernel[np.ix_(failure, failure)])
    delta = min(float(failed-total), 0.)
    return float(-np.expm1(delta))


def flexrouter_loss(kernel, complete_labels, *, competence=None, auxiliary_weight=1.):
    """Original hit loss is +inf for an all-wrong sample; no hidden clipping."""
    probability = dpp_coverage_probability(kernel, complete_labels)
    hit = float(-np.log(probability)) if probability > 0 else float("inf")
    if not np.isfinite(auxiliary_weight) or auxiliary_weight < 0: raise ValueError("Illegal auxiliary weight")
    if auxiliary_weight > 0 and competence is None:
        raise ValueError("The original auxiliary BCE requires explicit model competence")
    bce = 0.
    if competence is not None:
        q = _vector(competence, len(kernel), "competence", low=0., high=1.)
        y = np.asarray(complete_labels, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(y == 1, -np.log(q), -np.log1p(-q))
        bce = float(terms.sum())
    total = hit if auxiliary_weight == 0 else hit+auxiliary_weight*bce
    return {"coverage_probability": probability, "hit_loss": hit, "bce_loss": bce,
            "loss": total, "original_loss_finite": bool(np.isfinite(total))}


def complete_labels_from_broker(records, *, question_id, now, n_members=6):
    """Construct a complete training row only from arrived authorized receipts."""
    labels = {}
    for record in records:
        if not isinstance(record, LegalMemberFeedback): raise FeedbackPermissionError("Typed broker receipt required")
        _permission(record, now); _members((record.member_index,), n_members)
        if record.question_id != question_id: raise ValueError("Mixed question IDs in a training row")
        if record.value not in (0, 1): raise ValueError("FlexRouter requires binary member labels")
        if record.member_index in labels and labels[record.member_index] != record.value:
            raise ValueError("Conflicting complete-label row")
        labels[record.member_index] = record.value
    if set(labels) != set(range(n_members)):
        raise FeedbackPermissionError("Incomplete N6 row: unobserved members must not be filled with zero")
    return np.asarray([labels[i] for i in range(n_members)], dtype=int)


def require_finite_flexrouter_training_loss(kernel, complete_labels, *, competence, auxiliary_weight=1.):
    result = flexrouter_loss(kernel, complete_labels, competence=competence, auxiliary_weight=auxiliary_weight)
    if not result["original_loss_finite"]:
        raise ValueError("Original FlexRouter loss is non-finite; an all-wrong row requires an explicitly named training adaptation")
    return result


def fixed_k_dpp_greedy(kernel, *, k=3):
    """Fixed-K3 Schur-gain adaptation; not the paper's adaptive stopping rule."""
    kernel = _dpp_kernel(kernel)
    if len(kernel) != 6 or k != 3: raise ValueError("Frozen experiment is N6/K3")
    chosen, gains = [], []
    for _ in range(k):
        best = None
        for i in range(len(kernel)):
            if i in chosen: continue
            gain = float(kernel[i, i])
            if chosen:
                block = kernel[np.ix_(chosen, chosen)]
                if np.linalg.matrix_rank(block, tol=1e-12) < len(chosen):
                    raise np.linalg.LinAlgError("Singular selected DPP block; no silent jitter/pseudoinverse")
                gain -= float(kernel[i, chosen] @ np.linalg.solve(block, kernel[chosen, i]))
            if gain < -1e-8: raise RuntimeError("Negative Schur gain for PSD kernel")
            gain = max(gain, 0.)
            if best is None or gain > best[0]: best = (gain, i)
        gains.append(best[0]); chosen.append(best[1])
    return {"members": tuple(sorted(chosen)), "greedy_order": tuple(chosen), "schur_gains": tuple(gains),
            "native_system": "NOT_RUN", "trained_policy": "NOT_IMPLEMENTED_BY_THIS_MATHEMATICAL_REFERENCE"}
