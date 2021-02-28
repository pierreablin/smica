"""
Simple SMICA Example
====================

"""
import numpy as np

from smica import SMICA


rng = np.random.RandomState(0)


n_samples = 1000
n_channels = 5

X = rng.randn(n_channels, n_samples)
sfreq = 1

freqs = np.array([0.1, 0.2, 0.3])


smica = SMICA(n_components=3, freqs=freqs, sfreq=sfreq).fit(X)


# So far SMICA saves the input X, might be better to avoid that?
estimated_sources = smica.compute_sources()
