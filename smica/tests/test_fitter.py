import pytest
import numpy as np

from numpy.testing import assert_allclose

from smica import CovarianceFit, compute_covariances


@pytest.mark.parametrize('avg_noise', [False, True])
@pytest.mark.parametrize('corr', [False, True])
def test_covs(avg_noise, corr):
    n_c, p = 4, 3
    q = 2
    rng = np.random.RandomState(0)
    covs = np.array([x.dot(x.T)
                     for x in (rng.randn(p, p) for _ in range(n_c))])
    covfit = CovarianceFit(q, avg_noise=avg_noise,
                           corr=corr).fit(covs, tol=1e-3, verbose=1)
    sigma_shape = {True: (p,), False: (n_c, p)}[avg_noise]
    if corr:
        assert covfit.powers_.shape == (n_c, q, q)
    else:
        assert covfit.powers_.shape == (n_c, q)
    assert covfit.sigmas_.shape == sigma_shape
    assert covfit.A_.shape == (p, q)
    if corr:
        eigs = np.array([np.linalg.eigvalsh(c) for c in covfit.powers_])
        assert np.all(eigs >= 0)
    else:
        assert np.all(covfit.powers_ >= 0)
    assert np.all(covfit.sigmas_ >= 0)
    assert covfit.true_loss() >= 0
    assert covfit.compute_approx_covs().shape == covs.shape


@pytest.mark.parametrize('avg_noise', [True, False])
@pytest.mark.parametrize('corr', [False, True])
def test_loss(avg_noise, corr):
    p, q, n_epochs = 3, 2, 5
    rng = np.random.RandomState(0)
    A = rng.randn(p, q)
    if avg_noise:
        sigmas = .1 + rng.rand(p)
    else:
        sigmas = .1 + rng.rand(n_epochs, p)
    if corr:
        powers = np.zeros((n_epochs, q, q))
        for i in range(n_epochs):
            M = rng.randn(q, q)
            powers[i] = .1 + M.dot(M.T)
    else:
        powers = .1 + rng.rand(n_epochs, q)
    covs = compute_covariances(A, powers, sigmas, avg_noise, corr)
    covfit = CovarianceFit(q, avg_noise=avg_noise, corr=corr,
                           rng=0).fit(covs, verbose=1)
    covs_est = compute_covariances(covfit.A_, covfit.powers_, covfit.sigmas_,
                                   avg_noise, corr)
    assert_allclose(covs_est, covs, atol=1e-1)
    assert_allclose(covfit.compute_approx_covs(), covs_est, atol=1e-7)
