"""Exact conditional-Bernoulli sampling with fixed count and integer cost cap.

The law of a subset is proportional to its product of odds, conditional on
its size and an input-known cost bound. Inclusion probabilities are computed
by reverse differentiation of the finite partition function.
"""
import itertools
import numpy as np


class CappedSubsetDesign:
    def __init__(self, odds, count, costs, cost_cap):
        raw = np.asarray(odds, dtype=float)
        self.n = len(raw)
        self.count = int(count)
        self.costs = np.asarray(costs, dtype=int)
        self.cap = int(cost_cap)
        if raw.ndim != 1 or self.costs.shape != raw.shape or np.any(raw <= 0):
            raise ValueError('Strictly positive odds and aligned integer costs required')
        if not np.isfinite(raw).all() or np.any(self.costs < 0) or not 0 <= self.count <= self.n:
            raise ValueError('Invalid finite-pool subset design')
        self.weights = raw / max(float(raw.max()), 1.)
        self.layers = [np.zeros((self.count + 1, self.cap + 1))]
        self.layers[0][0, 0] = 1.
        for weight, cost in zip(self.weights, self.costs):
            previous = self.layers[-1]
            current = previous.copy()
            if cost <= self.cap:
                current[1:, cost:] += weight * previous[:-1, :self.cap + 1 - cost]
            self.layers.append(current)
        self.partition = float(self.layers[-1][self.count].sum())
        if not self.partition > 0:
            raise ValueError('No feasible subset under the fixed input-known caps')
        adjoint = np.zeros_like(self.layers[-1])
        adjoint[self.count] = 1.
        derivative = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            cost = self.costs[i]
            previous = self.layers[i]
            if cost <= self.cap:
                derivative[i] = np.sum(adjoint[1:, cost:] * previous[:-1, :self.cap + 1 - cost])
                next_adjoint = adjoint.copy()
                next_adjoint[:-1, :self.cap + 1 - cost] += self.weights[i] * adjoint[1:, cost:]
                adjoint = next_adjoint
        self.pi = self.weights * derivative / self.partition
        if np.any(self.pi < -1e-10) or np.any(self.pi > 1 + 1e-10):
            raise ArithmeticError('Invalid first-order inclusion probabilities')
        self.pi = np.clip(self.pi, 0, 1)
        if abs(self.pi.sum() - self.count) > 1e-8:
            raise ArithmeticError('Inclusion sum does not match the exact sample count')

    def sample(self, uniforms):
        uniforms = np.asarray(uniforms)
        if uniforms.shape != (self.n + 1,):
            raise ValueError('Supply one terminal-cost and n backward-sampling uniforms')
        mass = self.layers[-1][self.count]
        cumulative = np.cumsum(mass) / self.partition
        remaining_cost = int(np.searchsorted(cumulative, uniforms[0], side='right'))
        remaining_cost = min(remaining_cost, self.cap)
        remaining_count = self.count
        selected = np.zeros(self.n, dtype=bool)
        for i in range(self.n - 1, -1, -1):
            if remaining_count == 0:
                break
            cost = int(self.costs[i])
            denominator = self.layers[i + 1][remaining_count, remaining_cost]
            numerator = 0.
            if remaining_cost >= cost:
                numerator = self.weights[i] * self.layers[i][remaining_count - 1, remaining_cost - cost]
            probability = float(numerator / denominator) if denominator > 0 else 0.
            if uniforms[i + 1] < min(1., probability):
                selected[i] = True
                remaining_count -= 1
                remaining_cost -= cost
        if remaining_count or selected.sum() != self.count or self.costs @ selected > self.cap:
            raise ArithmeticError('Subset sampler violated its frozen caps')
        return selected


def qualification():
    odds = np.array([.2, 1.7, .9, 2., .4, .7])
    costs = np.array([0, 1, 2, 0, 1, 2])
    design = CappedSubsetDesign(odds, 3, costs, 3)
    subsets = []
    masses = []
    for ids in itertools.combinations(range(6), 3):
        mask = np.zeros(6, dtype=bool)
        mask[list(ids)] = True
        if costs @ mask <= 3:
            subsets.append(mask)
            masses.append(float(np.prod(odds[mask])))
    masks = np.asarray(subsets)
    mass = np.asarray(masses) / np.sum(masses)
    exact_pi = mass @ masks
    assert np.max(abs(exact_pi - design.pi)) < 1e-12
    values = np.array([1., -1, .3, -.2, .7, -.8])
    estimates = (masks / design.pi) @ values / 6
    exact_mean = float(mass @ estimates)
    assert abs(exact_mean - values.mean()) < 1e-12
    second = (masks.T * mass) @ masks
    covariance = second - np.outer(design.pi, design.pi)
    variance = float((values / design.pi) @ covariance @ (values / design.pi) / 36)
    assert abs(variance - mass @ ((estimates - exact_mean) ** 2)) < 1e-12
    rng = np.random.default_rng(20261112)
    draws = np.asarray([design.sample(rng.random(7)) for _ in range(20000)])
    max_error = float(np.max(abs(draws.mean(0) - exact_pi)))
    assert max_error < .02
    return dict(status='PASS', exact_inclusion_max_error=float(np.max(abs(exact_pi - design.pi))),
                exact_HT_bias=exact_mean - float(values.mean()), covariance_variance_error=float(variance - mass @ ((estimates - exact_mean) ** 2)),
                MC_draws=20000, MC_inclusion_max_error=max_error,
                qualification='Finite subset enumeration, exact dependent-sample inclusion and HT variance, coupled sampler frequencies')


if __name__ == '__main__':
    import json
    print(json.dumps(qualification(), indent=2))
