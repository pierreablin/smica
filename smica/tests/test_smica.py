import pytest
import numpy as np

from numpy.testing import assert_allclose

from smica import SMICA, compute_covariances


@pytest.mark.parametrize('avg_noise', [False, True])
def test_covs(avg_noise):
    n_c, p = 4, 3
    q = 2
    covs = np.array([x.dot(x.T)
                     for x in (np.random.randn(p, p) for _ in range(n_c))])
    smica = SMICA(q, avg_noise=avg_noise).fit(covs)
    sigma_shape = {True: (p,), False: (n_c, p)}[avg_noise]
    assert smica.powers_.shape == (n_c, q)
    assert smica.sigmas_.shape == sigma_shape
    assert smica.A_.shape == (p, q)
    assert np.all(smica.powers_ >= 0)
    assert np.all(smica.sigmas_ >= 0)
    assert smica.true_loss() >= 0
    assert smica.compute_approx_covs().shape == covs.shape


@pytest.mark.parametrize('avg_noise', [True, False])
def test_loss(avg_noise):
    p, q, n_epochs = 3, 2, 4
    rng = np.random.RandomState(0)
    A = rng.randn(p, q)
    if avg_noise:
        sigmas = np.abs(rng.randn(p))
    else:
        sigmas = np.abs(rng.randn(n_epochs, p))
    powers = np.abs(rng.randn(n_epochs, q))
    covs = compute_covariances(A, powers, sigmas, avg_noise)
    smica = SMICA(q, avg_noise=avg_noise, rng=0).fit(covs, max_iter=3000)
    covs_est = compute_covariances(smica.A_, smica.powers_, smica.sigmas_,
                                   avg_noise)
    assert_allclose(covs_est, covs, atol=1e-2)
    assert_allclose(smica.compute_approx_covs(), covs, atol=1e-2)
