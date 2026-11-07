"""Projected implementation of the frozen V2 one-step Gaussian KG operator."""
import time
import numpy as np
from scipy.special import ndtr
from budget_router_v2.acquisition import antithetic_fantasies


def fantasy_table(seed, pairs=8):
    one = antithetic_fantasies(1, pairs=pairs, seed=seed)
    # Conditional sign symmetry makes finite-sample extra-comparison KG nonnegative.
    draws = antithetic_fantasies(2, pairs=max(1, pairs // 2), seed=seed+1)
    two = draws[:len(draws)//2]
    positive = np.concatenate((two, two * np.array([1., -1.])))
    return {1: one, 2: np.concatenate((positive, -positive))}


def projected_scores(posterior, q, main, neighbors, weights, fantasies, include_neighbors=True, need_kg=True):
    """Raw joint observations and the very same posterior/noise used by update."""
    started = time.process_time()
    actions = [main] + list(neighbors)
    h = posterior.observation(q, actions)
    cross = posterior.cross(q, actions)
    latent = h @ cross
    reference_cross = posterior.reference_cross(q, actions) if need_kg else None
    current = posterior.predict(q)
    ref = posterior.ref_predictions
    before = float(ref.max(axis=1) @ weights) if need_kg else 0.
    noise = posterior.noise[np.ix_(actions, actions)]
    gap = np.maximum(0.0, current[main] - current[actions])
    contrast = np.maximum(0.0, np.diag(latent) + latent[0, 0] - 2 * latent[0])
    boundary = float(np.max(ndtr(-gap[1:] / np.sqrt(np.maximum(contrast[1:], 1e-15))))) if len(actions) > 1 else 0.0
    competitor = int(np.argmax(ndtr(-gap[1:] / np.sqrt(np.maximum(contrast[1:], 1e-15))))) + 1

    def boundary_reduction(indices):
        direction_covariance = latent[0, indices] - latent[competitor, indices]
        predictive = latent[np.ix_(indices, indices)] + noise[np.ix_(indices, indices)]
        reduction = direction_covariance @ np.linalg.solve(predictive, direction_covariance)
        return float(boundary * reduction / max(contrast[competitor], 1e-15))

    def score(indices, samples):
        cov = latent[np.ix_(indices, indices)] + noise[np.ix_(indices, indices)]
        r = noise[np.ix_(indices, indices)]
        chol = np.linalg.cholesky(cov)
        kg = 0.
        if need_kg:
            update = np.linalg.solve(chol, reference_cross[:, :, indices].reshape(-1, len(indices)).T).T
            changes = (samples @ update.T).reshape(len(samples), *ref.shape)
            values = (ref[None] + changes).max(axis=2) @ weights
            kg = max(0., float(values.mean() - before))
        ig = float(np.log(np.diag(chol)).sum() - np.log(np.diag(np.linalg.cholesky(r))).sum())
        return kg, max(0., ig)

    main_kg, main_ig = score([0], fantasies[1])
    result = [{'actions': [main], 'kg': main_kg, 'ig': main_ig,
               'variance': float(latent[0, 0]), 'value_variance': float(latent[0, 0]),
               'pair_variance': 0., 'boundary': boundary, 'boundary_reduction': boundary_reduction([0]),
               'extra_kg': 0., 'extra_ig': 0.}]
    for j in range(1, len(actions)) if include_neighbors else ():
        kg, ig = score([0, j], fantasies[2])
        matched_main_kg, matched_main_ig = score([0], fantasies[2][:, :1])
        result.append({'actions': [main, actions[j]], 'kg': kg, 'ig': ig,
                       'variance': float(latent[0, 0] + latent[j, j]), 'value_variance': float(latent[j, j]),
                       'pair_variance': float(contrast[j]), 'boundary_reduction': boundary_reduction([0, j]),
                       'boundary': float(ndtr(-gap[j] / np.sqrt(max(contrast[j], 1e-15)))),
                       'extra_kg': max(0., kg - matched_main_kg), 'extra_ig': max(0., ig - matched_main_ig)})
    return result, time.process_time() - started


def choose_candidate(candidates, method, estimated_costs, rng, stage):
    """Only prospective statistics and history-estimated prices enter this rule."""
    if method.startswith('random'):
        index = int(rng.integers(len(candidates)))
        return index, float(rng.random())
    scores = []
    for candidate, cost in zip(candidates, estimated_costs):
        if method.startswith('max_uncertainty'):
            value = candidate['variance']
        elif method == 'value_uncertainty':
            value = candidate['value_variance']
        elif method == 'pair_uncertainty':
            value = candidate['pair_variance']
        elif method.startswith('boundary'):
            value = candidate['boundary_reduction'] if stage in ('J', 'D') else candidate['boundary']
        elif method.startswith('ig'):
            value = candidate['extra_ig'] if stage == 'B2' else candidate['ig']
        else:
            value = candidate['extra_kg'] if stage == 'B2' else candidate['kg']
        if method.endswith('_cost'):
            value /= max(float(cost), 1e-12)
        scores.append(float(value))
    index = int(np.argmax(scores))
    return index, scores[index]


class BudgetPacing:
    """Shared non-lookahead record rule within prepaid equal-width gold slots."""
    def __init__(self, length, budget, seed, warmup=2):
        self.length = length
        self.budget = min(budget, length)
        self.warmup = min(warmup, self.budget)
        self.edges = np.linspace(self.warmup, length, self.budget - self.warmup + 1, dtype=int)
        self.random_positions = {}
        rng = np.random.default_rng(seed)
        for start, end in zip(self.edges[:-1], self.edges[1:]):
            self.random_positions[int(start)] = int(rng.integers(start, end))
        self.purchased = set()
        self.records = {}

    def eligible(self, tick):
        if tick < self.warmup:
            return True
        if len(self.edges) <= 1:
            return False
        slot = min(len(self.edges) - 2, int(np.searchsorted(self.edges, tick, side='right') - 1))
        return slot not in self.purchased

    def decide(self, tick, score, random=False):
        if tick < self.warmup:
            return True
        if len(self.edges) <= 1:
            return False
        slot = min(len(self.edges) - 2, int(np.searchsorted(self.edges, tick, side='right') - 1))
        start, end = map(int, self.edges[slot:slot+2])
        if slot in self.purchased:
            return False
        previous = self.records.get(slot, -np.inf)
        self.records[slot] = max(previous, score)
        observation_window = max(1, (end - start) // 3)
        if random:
            buy = tick == self.random_positions[start]
        else:
            buy = tick == end - 1 or (tick >= start + observation_window and score > previous)
        if buy:
            self.purchased.add(slot)
        return bool(buy)
