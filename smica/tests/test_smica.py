import pytest
import numpy as np

from numpy.testing import assert_allclose

from smica import SMICA


@pytest.mark.parametrize('avg_noise', [True, False])
def test_dims(avg_noise):
    rng = np.random.RandomState(0)
    p, n = 4, 100
    n_components = 2
    X = rng.randn(p, n)
    sfreq = 10
    freqs = np.linspace(.1, 5, 10)
    smica = SMICA(n_components, freqs, sfreq,
                  avg_noise=avg_noise, rng=rng)
    smica.fit(X, tol=1e-1)
    n_mat = len(freqs) - 1
    assert smica.C_.shape == (n_mat, p, p)
    assert smica.A_.shape == (p, n_components)
    assert smica.ft_.shape == X.shape
    assert len(smica.freq_idx_) == len(freqs)
    assert smica.powers_.shape == (n_mat, n_components)


@pytest.mark.parametrize('avg_noise', [True, False])
@pytest.mark.parametrize('new_X', [True, False])
def test_sources(avg_noise, new_X):
    rng = np.random.RandomState(0)
    p, n = 4, 100
    n_components = 2
    X = rng.randn(p, n)
    if new_X:
        Y = rng.randn(p, 2 * n)
    else:
        Y = None
    sfreq = 10
    freqs = np.linspace(.1, 5, 10)
    smica = SMICA(n_components, freqs, sfreq,
                  avg_noise=avg_noise, rng=rng)
    smica.fit(X, tol=1e-1)
    sources = smica.compute_sources(Y)
    if not new_X:
        assert sources.shape == (n_components, n)
    else:
        assert sources.shape == (n_components, 2 * n)


@pytest.mark.parametrize('avg_noise', [True, False])
def test_filter(avg_noise):
    rng = np.random.RandomState(0)
    p, n = 4, 100
    n_components = 3
    X = rng.randn(p, n)
    sfreq = 10
    freqs = np.linspace(.1, 5, 10)
    smica = SMICA(n_components, freqs, sfreq,
                  avg_noise=avg_noise, rng=rng)
    smica.fit(X, tol=1e-1)
    bad_sources = [1, 2]
    X_f = smica.filter(bad_sources)
    assert X_f.shape == X.shape
