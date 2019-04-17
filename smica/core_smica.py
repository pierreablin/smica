"""
API for the core SMICA algorithm : fit the model to a sequence of covariances
"""
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from ._em import em_algo
from .utils import loss, compute_covariances


class SMICA(BaseEstimator, TransformerMixin):
    '''
    Compute smica decomposition
    '''
    def __init__(self, n_sources, avg_noise=False, rng=None):
        self.rng = check_random_state(rng)
        self.n_sources = n_sources
        self.avg_noise = avg_noise
        self.initialized = False

    def random_init(self, n_components, n_samples):
        rng = self.rng
        self.n_samples_ = n_samples
        self.n_components_ = n_components
        self.A_ = rng.randn(n_components, self.n_sources)
        if self.avg_noise:
            self.sigmas_ = np.abs(rng.randn(n_components))
        else:
            self.sigmas_ = np.abs(rng.randn(n_samples, n_components))
        self.powers_ = np.abs(rng.randn(n_samples, self.n_sources))
        self.initialized = True
        return self

    def fit(self, covs, y=None, **kwargs):
        self.covs_ = covs
        n_samples, n_components, _ = covs.shape
        if not self.initialized:
            self.random_init(n_components, n_samples)
        A, sigmas, powers, x_list = \
            em_algo(self.covs_, self.A_, self.sigmas_, self.powers_,
                    self.avg_noise, **kwargs)
        self.A_ = A
        self.sigmas_ = sigmas
        self.powers_ = powers
        self.x_list_ = x_list
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.powers_

    def copy_params(self, target_smica):
        self.A_ = np.copy(target_smica.A_)
        self.powers_ = np.copy(target_smica.powers_)
        n_samples = len(self.powers_)
        sigmas = target_smica.sigmas_
        if not self.avg_noise:
            if not target_smica.avg_noise:
                self.sigmas_ = np.copy(sigmas)
            else:
                self.sigmas_ = (np.ones(n_samples)[:, None] *
                                sigmas[None, :])
        else:
            if target_ica.avg_noise:
                self.sigmas_ = np.copy(sigmas)
            else:
                1/0
        self.initialized = True
        return self

    def true_loss(self):
        '''
        compute the loss rectified with the log det. >=0, =0 if the model
        holds perfectly.
        '''
        return loss(self.covs_, self.A_, self.sigmas_, self.powers_,
                    self.avg_noise, normalize=True)

    def compute_approx_covs(self):
        '''
        Compute the covariances estimated by the model
        '''
        covs_approx = compute_covariances(self.A_, self.powers_, self.sigmas_,
                                          self.avg_noise)
        return covs_approx
