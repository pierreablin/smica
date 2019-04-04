import pytest
import numpy as np

from smica import SMICA


@pytest.mark.parametrize('avg_noise', [False, True])
def test_covs(avg_noise):
    n_c, p = 4, 3
    q = 2
    covs = np.array([x.dot(x.T)
                     for x in (np.random.randn(p, p) for _ in range(n_c))])
    smica = SMICA(covs, q, avg_noise=avg_noise)
    smica.fit()
    sigma_shape = {True: (p,), False: (n_c, p)}[avg_noise]
    assert smica.powers.shape == (n_c, q)
    assert smica.sigmas.shape == sigma_shape
    assert smica.A.shape == (p, q)
    assert np.all(smica.powers >= 0)
    assert np.all(smica.sigmas >= 0)
    assert smica.true_loss() >= 0
    assert smica.compute_approx_covs().shape == covs.shape
