import numpy as np

from smica import CovarianceFit
import matplotlib.pyplot as plt


rng = np.random.RandomState(0)

n_bins = 10
n_sources = 3
n_channels = 5

source_powers = 1 + 2 * rng.rand(n_bins, n_sources)
noise_powers = 0.03 * (1 + rng.rand(n_bins, n_channels))
mixing = rng.randn(n_channels, n_sources)

covariances = [np.dot(mixing, p[:, None] * mixing.T) + np.diag(n)
               for p, n in zip(source_powers, noise_powers)]

covariances = np.array(covariances)

covfit = CovarianceFit(n_sources, rng=rng).fit(covariances)

estimated_powers = covfit.powers_
estimated_mixing = covfit.A_
estimated_noise = covfit.sigmas_

plt.matshow(np.abs(np.linalg.pinv(mixing).dot(estimated_mixing)))
plt.show()
