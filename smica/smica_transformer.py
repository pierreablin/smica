import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from .core_smica import SMICA
from .utils import fourier_sampling


class SMICATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_sources, freqs, sfreq, avg_noise=False, rng=None):
        self.n_sources = n_sources
        self.freqs = freqs
        self.sfreq = sfreq
        self.rng = check_random_state(rng)
        self.avg_noise = avg_noise

    def fit(X, y, **kwargs):
        n_epochs, n_channels, n_times = X.shape
        C, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        smica = SMICA(self.n_sources, avg_noise=self.avg_noise, rng=self.rng)
        smica.fit(C, **kwargs)
        self.smica_ = smica
        return self

    def transform(X):
        return np.array([[smica.true_loss(x) for smica in self.smicas]
                        for x in X])
