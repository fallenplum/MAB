"""Software-only checks for the KG-informed Gaussian bundle operator."""
import numpy as np
import pytest

from budget_router_v2.acquisition import antithetic_fantasies, decision_value, decision_values


def inputs():
    return dict(mean=np.zeros(2), covariance=np.eye(2), observations=np.array([[0.0, 1.0]]),
                reference_features=np.array([[[1.0, 1.0], [1.0, -1.0]]]),
                noise_covariance=np.eye(1), fantasies=antithetic_fantasies(1, pairs=64, seed=17))


def test_antithetic_fantasies_are_seeded_centered_and_leave_global_rng_untouched():
    np.random.seed(198)
    before = np.random.get_state()
    first = antithetic_fantasies(3, pairs=8, seed=42)
    assert np.array_equal(first, antithetic_fantasies(3, pairs=8, seed=42))
    assert np.array_equal(first[:8], -first[8:])
    assert np.allclose(first.mean(axis=0), 0)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_parameter_information_decoy_has_zero_decision_value():
    params = inputs()
    boundary = decision_value(**params)
    params["observations"] = np.array([[10.0, 0.0]])
    decoy = decision_value(**params)
    assert decoy["ig_theta"] > boundary["ig_theta"]
    assert decoy["decision_kg"] == pytest.approx(0.0, abs=1e-12)
    assert decoy["weighted_contrast_variance_reduction"] == pytest.approx(0.0)
    assert boundary["decision_kg"] > 0
    assert boundary["weighted_contrast_variance_reduction"] > 0


def test_general_three_observation_bundle_uses_joint_predictive_covariance():
    params = inputs()
    params["observations"] = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    params["noise_covariance"] = np.array([[1.0, .2, .1], [.2, 2.0, .3], [.1, .3, 1.5]])
    params["fantasies"] = antithetic_fantasies(3, pairs=32, seed=3)
    result = decision_value(**params)
    h, r = params["observations"], params["noise_covariance"]
    expected = .5 * (np.linalg.slogdet(r + h @ h.T)[1] - np.linalg.slogdet(r)[1])
    assert result["ig_theta"] == pytest.approx(expected)
    assert result["observation_rows"] == 3
    assert result["decision_kg"] >= 0
    assert result["planner_cpu_s"] >= 0 and result["planner_wall_s"] >= 0


def test_reference_weights_integrate_future_decision_value_without_reference_labels():
    params = inputs()
    first = decision_value(**params)
    params["reference_features"] = 2 * params["reference_features"]
    second = decision_value(**params)
    params["reference_features"] = np.concatenate((inputs()["reference_features"], params["reference_features"]))
    result = decision_value(**params, reference_weights=np.array([1.0, 3.0]))
    assert result["decision_kg"] == pytest.approx(.25 * first["decision_kg"] + .75 * second["decision_kg"])
    assert result["reference_inputs"] == 2


def test_extreme_finite_reference_weights_normalize_without_overflow():
    params = inputs()
    baseline = decision_value(**params)
    params["reference_features"] = np.repeat(params["reference_features"], 2, axis=0)
    result = decision_value(**params, reference_weights=np.array([1e308, 1e308]))
    assert result["decision_kg"] == pytest.approx(baseline["decision_kg"])


def test_finite_inputs_that_overflow_predictive_covariance_are_rejected():
    params = inputs()
    params["observations"] = np.array([[1e308, 1e308]])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite numerical range"):
            decision_value(**params)


def test_zero_posterior_uncertainty_returns_zero_information_and_no_finite_margin():
    params = inputs()
    params["covariance"] = np.zeros((2, 2))
    result = decision_value(**params)
    assert result["decision_kg"] == result["ig_theta"] == 0
    assert result["weighted_min_standardized_margin"] is None
    assert result["weighted_contrast_variance_reduction"] == 0


def test_fixed_twenty_action_shape_and_input_arrays_are_preserved():
    params = inputs()
    params["reference_features"] = np.array([[[1.0, value] for value in np.linspace(-1, 1, 20)]])
    originals = {name: value.copy() for name, value in params.items()}
    result = decision_value(**params)
    for name, value in params.items():
        assert np.array_equal(value, originals[name])
    assert result["reference_actions"] == 20


@pytest.mark.parametrize("name,value", [
    ("mean", np.zeros((2, 1))),
    ("covariance", np.array([[1.0, .5], [0.0, 1.0]])),
    ("covariance", np.diag([1.0, -1.0])),
    ("noise_covariance", np.zeros((1, 1))),
    ("observations", np.zeros((0, 2))),
    ("reference_features", np.zeros((1, 1, 2))),
    ("fantasies", np.ones((4, 1))),
    ("fantasies", np.array([[np.nan], [np.nan]])),
    ("reference_weights", np.array([0.0])),
    ("reference_weights", np.array([-1.0])),
])
def test_invalid_inputs_fail_before_any_observation_is_produced(name, value):
    params = inputs()
    params[name] = value
    with pytest.raises(ValueError):
        decision_value(**params)


@pytest.mark.parametrize("kwargs", [{"dimension": 0}, {"dimension": True},
                                     {"dimension": 1, "pairs": 0}, {"dimension": 1, "seed": -1}])
def test_invalid_fantasy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        antithetic_fantasies(**kwargs)


def test_batch_matches_single_values_for_different_bundle_sizes():
    params = inputs()
    candidates = []
    for rows in (1, 2, 3):
        h = np.array([[1.0, float(i + 1)] for i in range(rows)])
        candidates.append({"observations": h, "noise_covariance": np.eye(rows),
                           "fantasies": antithetic_fantasies(rows, pairs=16, seed=12)})
    references = np.repeat(params["reference_features"], 2, axis=0)
    weights = np.array([1.0, 4.0])
    batch = decision_values(params["mean"], params["covariance"], candidates, references, weights)
    assert batch["candidate_count"] == 3
    assert batch["planner_cpu_s"] >= 0 and batch["planner_wall_s"] >= 0
    for candidate, score in zip(candidates, batch["candidates"]):
        single = decision_value(params["mean"], params["covariance"],
                                reference_features=references, reference_weights=weights, **candidate)
        for name, value in score.items():
            if isinstance(value, float):
                assert value == pytest.approx(single[name], rel=1e-12, abs=1e-12)
            else:
                assert value == single[name]


def test_batch_validates_shared_posterior_only_once_without_cached_state(monkeypatch):
    import budget_router_v2.acquisition as acquisition
    params = inputs()
    candidate = {name: params[name] for name in ("observations", "noise_covariance", "fantasies")}
    original = acquisition._prepare_shared
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(acquisition, "_prepare_shared", counted)
    first = acquisition.decision_values(params["mean"], params["covariance"], [candidate] * 4,
                                        params["reference_features"])
    assert len(calls) == 1
    changed = acquisition.decision_values(params["mean"], params["covariance"] * 2, [candidate],
                                          params["reference_features"])
    assert len(calls) == 2
    assert changed["candidates"][0]["ig_theta"] > first["candidates"][0]["ig_theta"]


@pytest.mark.parametrize("mutation", ["weights", "posterior_psd", "posterior_finite",
                                     "candidate_noise", "candidate_finite", "label_field", "empty"])
def test_batch_rejects_invalid_shared_and_per_candidate_inputs(mutation):
    params = inputs()
    candidate = {name: params[name] for name in ("observations", "noise_covariance", "fantasies")}
    candidates = [candidate]
    weights = None
    if mutation == "weights":
        weights = np.array([-1.0])
    elif mutation == "posterior_psd":
        params["covariance"] = np.diag([1.0, -1.0])
    elif mutation == "posterior_finite":
        params["mean"] = np.array([np.nan, 0.0])
    elif mutation == "candidate_noise":
        candidates.append({**candidate, "noise_covariance": np.zeros((1, 1))})
    elif mutation == "candidate_finite":
        candidates.append({**candidate, "observations": np.array([[np.inf, 0.0]])})
    elif mutation == "label_field":
        candidates.append({**candidate, "labels": np.array([1.0])})
    else:
        candidates = []
    with pytest.raises(ValueError):
        decision_values(params["mean"], params["covariance"], candidates,
                        params["reference_features"], weights)


def test_batch_is_order_stable_deterministic_and_preserves_inputs():
    params = inputs()
    first = {name: params[name] for name in ("observations", "noise_covariance", "fantasies")}
    second = {**first, "observations": np.array([[10.0, 0.0]])}
    original_arrays = {name: value.copy() for name, value in params.items()}
    forward = decision_values(params["mean"], params["covariance"], [first, second, first],
                              params["reference_features"])
    reverse = decision_values(params["mean"], params["covariance"], [second, first],
                              params["reference_features"])
    repeated = decision_values(params["mean"], params["covariance"], [first, second, first],
                               params["reference_features"])
    assert forward["candidates"] == repeated["candidates"]
    assert forward["candidates"][0] == forward["candidates"][2] == reverse["candidates"][1]
    assert forward["candidates"][1] == reverse["candidates"][0]
    for name, value in params.items():
        assert np.array_equal(value, original_arrays[name])


def test_fantasy_prediction_overflow_is_rejected_before_nonnegative_clamp():
    references = np.array([[[1e308], [1e308]], [[-1e308], [-1e308]]])
    candidate = {"observations": np.ones((1, 1)), "noise_covariance": np.ones((1, 1)),
                 "fantasies": np.array([[1.0], [-1.0]])}
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite numerical range"):
            decision_values(np.array([1.5]), np.ones((1, 1)), [candidate], references,
                            reference_weights=np.array([.5, .5]))
