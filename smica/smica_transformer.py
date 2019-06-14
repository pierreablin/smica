import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from .core_smica import SMICA


class SMICATransformer(BaseEstimator, TransformerMixin):
    '''
    SMICA transfomer for epochs processing
    '''
    def __init__(self, n_components, freqs, sfreq,
                 transformer='source_powers', avg_noise=False, rng=None):
        self.n_components = n_components
        self.freqs = freqs
        self.sfreq = sfreq
        self.rng = check_random_state(rng)
        self.avg_noise = avg_noise
        self.transformer = transformer

    def fit(self, X, y=None, **kwargs):
        '''
        X is of shape (n, p, T)
        '''
        smica = SMICA(self.n_components, self.freqs, self.sfreq,
                      self.avg_noise, self.rng)
        smica.fit(np.hstack(X), **kwargs)
        self.smica_ = smica
        return self

    def transform(self, X):
        if self.transformer == 'source_powers':
            sources = [self.smica_.compute_sources(x) for x in X]
            powers = [np.mean(s ** 2, axis=1) for s in sources]
            return np.log(np.array(powers))
        elif self.transformer == 'likelihood':
            return np.array([self.smica_.compute_loss(x) for x in X])
