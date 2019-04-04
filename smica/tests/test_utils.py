import pytest
import numpy as np

from smica import fourier_sampling


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
