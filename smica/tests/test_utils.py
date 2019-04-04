import pytest
import numpy as np

from smica import fourier_sampling, loss


def test_fourier():
    n_bins = 10
    sfreq = 20
    p, n = 2, 30
    freqs = np.linspace(1, 10, n_bins+1)
    X = np.random.randn(p, n)
    C, ft, idx = fourier_sampling(X, sfreq, freqs)
    assert C.shape == (n_bins, p, p)
    assert ft.shape == (p, n)
    assert idx.shape == (n_bins + 1,)


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
    if avg_noise:
        covs = np.array([np.dot(A, power[:, None] * A.T) + np.diag(sigmas)
                         for power in powers])
    else:
        covs = np.array([np.dot(A, power[:, None] * A.T) + np.diag(sigma)
                         for power, sigma in zip(powers, sigmas)])
    assert loss(covs, A, sigmas, powers, avg_noise, normalize=True) == 0
