"""Independent positive correlation and relabeling checks for the frozen sampler."""
from pathlib import Path
import hashlib
import json
import numpy as np
import joint_noise_model as model

HERE = Path(__file__).resolve().parent
scale = .10
rho = .55
covariance = scale ** 2 * (rho * np.ones((20, 20)) + (1 - rho) * np.eye(20))
prior = dict(mean=np.full(20, .5), prior_covariance=covariance, noise=covariance)
observed = np.full((8, 2, 20), .5)
sampled = model.sample_bounded(prior, observed, 20261114092)
values = sampled['draws'].reshape(-1, 20)
expected_covariance = covariance / 3
sample_covariance = np.cov(values, rowvar=False, ddof=1)
sample_correlation = np.corrcoef(values, rowvar=False)
off_diagonal = np.triu_indices(20, 1)
covariance_error = float(np.max(np.abs(sample_covariance - expected_covariance)))
correlation_error = float(np.max(np.abs(sample_correlation[off_diagonal] - rho)))
assert covariance_error < .00065 and correlation_error < .12
assert abs(values.mean() - .5) < .01
# A deliberately non-exchangeable parameterization catches coordinate mistakes.
prior['mean'] = np.linspace(.3, .7, 20)
prior['prior_covariance'] = covariance + np.diag(np.linspace(.001, .02, 20))
observed = np.stack((np.linspace(.2, .8, 20), np.linspace(.4, .6, 20)))[None]
permutation = np.random.default_rng(20261114093).permutation(20)
permuted = dict(mean=prior['mean'][permutation],
    prior_covariance=prior['prior_covariance'][np.ix_(permutation, permutation)],
    noise=prior['noise'][np.ix_(permutation, permutation)])
mean, covariance, precision = model.conjugate_parameters(prior, observed)
other_mean, other_covariance, other_precision = model.conjugate_parameters(permuted, observed[:, :, permutation])
mean_error = float(np.max(np.abs(other_mean - mean[:, permutation])))
covariance_permutation_error = float(np.max(np.abs(other_covariance - covariance[np.ix_(permutation, permutation)])))
precision_error = float(np.max(np.abs(other_precision - precision[np.ix_(permutation, permutation)])))
assert max(mean_error, covariance_permutation_error, precision_error) < 1e-10
result = dict(status='PASS_INDEPENDENT_CORRELATED_SAMPLER_CHECK',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    frozen_sampler_sha256=hashlib.sha256((HERE / 'joint_noise_model.py').read_bytes()).hexdigest(),
    expected_correlation=rho, measured_mean_correlation=float(sample_correlation[off_diagonal].mean()),
    maximum_correlation_error=correlation_error, maximum_covariance_error=covariance_error,
    maximum_parameter_relabeling_error=max(mean_error, covariance_permutation_error, precision_error),
    clipping_used=False, raw_samples_bounded=bool(np.all((values >= 0) & (values <= 1))),
    scope='software positive correlation check with truncation boundaries over8 standard deviations away; not real latent calibration')
(HERE / 'public/CORRELATED_SAMPLER_CHECK.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result))
