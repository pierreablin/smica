import numpy as np
from smica import JDIAG
import pytest


def test_dims():
    rng = np.random.RandomState(0)
    p, n = 4, 10
    n_components = 3
    X = rng.randn(p, n)
    sfreq = 10
    freqs = np.linspace(.1, 5, 3)
    jdiag = JDIAG(n_components, freqs, sfreq, rng=rng)
    jdiag.fit(X, tol=1e-1, max_iter=10)
    n_mat = len(freqs) - 1
    assert jdiag.C_.shape == (n_mat, p, p)
    assert jdiag.A_.shape == (p, n_components)
    assert jdiag.ft_.shape == X.shape
    assert len(jdiag.freq_idx_) == len(freqs)
    assert jdiag.powers_.shape == (n_mat, n_components)


@pytest.mark.parametrize('new_X', [True, False])
def test_sources(new_X):
    rng = np.random.RandomState(0)
    p, n = 4, 20
    n_components = 3
    X = rng.randn(p, n)
    if new_X:
        Y = rng.randn(p, 2 * n)
    else:
        Y = None
    sfreq = 10
    freqs = np.linspace(1, 5, 3)
    jdiag = JDIAG(n_components, freqs, sfreq, rng=rng)
    jdiag.fit(X, tol=1e-1, max_iter=10)
    sources = jdiag.compute_sources(Y, method='pinv')
    if not new_X:
        assert sources.shape == (n_components, n)
    else:
        assert sources.shape == (n_components, 2 * n)


def test_filter():
    rng = np.random.RandomState(0)
    p, n = 4, 20
    n_components = 3
    X = rng.randn(p, n)
    sfreq = 10
    freqs = np.linspace(1, 5, 3)
    jdiag = JDIAG(n_components, freqs, sfreq,  rng=rng)
    jdiag.fit(X, tol=1e-1, max_iter=10)
    bad_sources = [1, 2]
    X_f = jdiag.filter(bad_sources)
    assert X_f.shape == X.shape
