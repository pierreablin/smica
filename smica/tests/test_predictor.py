import pytest
import numpy as np

from numpy.testing import assert_allclose

from smica import SMICAPredictor


@pytest.mark.parametrize('avg_noise', [True, False])
def test_transform(avg_noise):
    rng = np.random.RandomState(0)
    n, p, T = 3, 2, 10
    n_components = 2
    X = rng.randn(n, p, T)
    sfreq = 10
    freqs = np.linspace(.1, 5, 3)
    smica = SMICAPredictor(n_components, freqs, sfreq, avg_noise=avg_noise,
                           rng=rng)
    y = rng.randint(2, size=n)
    smica.fit(X, y, tol=1e-3)
    smica.predict(X)
