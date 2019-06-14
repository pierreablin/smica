import pytest
import numpy as np

from numpy.testing import assert_allclose

from smica import SMICATransformer


@pytest.mark.parametrize('avg_noise', [True, False])
@pytest.mark.parametrize('transformer', ['source_powers', 'likelihood'])
def test_transform(avg_noise, transformer):
    rng = np.random.RandomState(0)
    n, p, T = 3, 4, 10
    n_components = 2
    X = rng.randn(n, p, T)
    sfreq = 10
    freqs = np.linspace(1, 5, 3)
    smica = SMICATransformer(n_components, freqs, sfreq, avg_noise=avg_noise,
                             transformer=transformer, rng=rng)
    smica.fit(X, tol=1e-3)
    features = smica.transform(X)
    if transformer == 'source_powers':
        assert features.shape == (n, n_components)
    else:
        assert features.shape == (n, )
