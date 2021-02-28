import numpy as np
from smica import sobi, SOBI
import pytest


def test_solver():
    p, n, m = 3, 30, 4
    n_components = 2
    rng = np.random.RandomState(0)
    X = rng.randn(p, n)
    lags = np.linspace(0, n // 2, m, dtype=int)
    W = sobi(X, lags, n_components)
    assert W.shape == (n_components, p)


def test_dims():
    rng = np.random.RandomState(0)
    p, n = 4, 10
    m = 5
    n_components = 3
    X = rng.randn(p, n)
    sfreq = 10
    freqs = np.linspace(.1, 5, 3)
    sobi = SOBI(m, n_components, freqs, sfreq, rng=rng)
    sobi.fit(X, tol=1e-1)
    n_mat = len(freqs) - 1
    assert sobi.C_.shape == (n_mat, p, p)
    assert sobi.A_.shape == (p, n_components)
    assert sobi.ft_.shape == X.shape
    assert len(sobi.freq_idx_) == len(freqs)
    assert sobi.powers_.shape == (n_mat, n_components)


@pytest.mark.parametrize('new_X', [True, False])
def test_sources(new_X):
    rng = np.random.RandomState(0)
    p, n = 4, 20
    m = 5
    n_components = 3
    X = rng.randn(p, n)
    if new_X:
        Y = rng.randn(p, 2 * n)
    else:
        Y = None
    sfreq = 10
    freqs = np.linspace(1, 5, 3)
    sobi = SOBI(m, n_components, freqs, sfreq, rng=rng)
    sobi.fit(X, tol=1e-1)
    sources = sobi.compute_sources(Y, method='pinv')
    if not new_X:
        assert sources.shape == (n_components, n)
    else:
        assert sources.shape == (n_components, 2 * n)


def test_filter():
    rng = np.random.RandomState(0)
    p, n = 4, 20
    n_components = 3
    m = 5
    X = rng.randn(p, n)
    sfreq = 10
    freqs = np.linspace(1, 5, 3)
    sobi = SOBI(m, n_components, freqs, sfreq, rng=rng)
    sobi.fit(X, tol=1e-1)
    bad_sources = [1, 2]
    X_f = sobi.filter(bad_sources=bad_sources)
    assert X_f.shape == X.shape
