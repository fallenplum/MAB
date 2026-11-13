"""Compare independently implemented training and mathematical operators."""
from pathlib import Path
import importlib.util
import json
import numpy as np

import ns_training as training

source = Path(__file__).resolve().parents[1] / 'math_literature/public/set_spo_operators.py'
spec = importlib.util.spec_from_file_location('independent_math', source)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)
rng = np.random.default_rng(202611140321)
phi = np.column_stack((np.ones(9), rng.normal(size=(9, 3))))
weights = rng.normal(size=(4, 20))
samples = rng.uniform(.01, .99, size=(9, 8, 20))
target = training.near_optimal_weights(samples, .05)
actual = training.ns_value_gradient(weights, phi, samples, .05, .002)
expected = math.ns_spo_value_subgradient(weights, phi, samples, target, .002)
ns_value_error = abs(actual[0] - expected[0])
ns_gradient_error = float(abs(actual[1] - expected[1]).max())
p = rng.dirichlet(np.ones(3), size=(9, 190))
actual_davidson = training.davidson_value_gradient(weights, phi, p, .7, .002)
expected_davidson = math.davidson_value_gradient(weights, phi, p, log_nu=np.log(.7), regularization=.002)
davidson_value_error = abs(actual_davidson[0] - expected_davidson[0])
davidson_gradient_error = float(abs(actual_davidson[1] - expected_davidson[1]).max())
if max(ns_value_error, ns_gradient_error, davidson_value_error, davidson_gradient_error) > 1e-10:
    raise AssertionError('Independent objective implementations disagree')
result = dict(status='PASS', training_source_sha256=training.sha(Path(training.__file__)),
    mathematical_source_sha256=training.sha(source),
    ns_value_error=ns_value_error, ns_gradient_error=ns_gradient_error,
    davidson_value_error=davidson_value_error, davidson_gradient_error=davidson_gradient_error)
training.put(Path(__file__).parent / 'public/INDEPENDENT_OPERATOR_AGREEMENT.json', result)
print(json.dumps(result))
