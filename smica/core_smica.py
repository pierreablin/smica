"""
API for the core SMICA algorithm : fit the model to a sequence of covariances
"""
import numpy as np

from sklearn.utils import check_random_state

from ._em import em_algo
from .utils import loss


class SMICA(object):
    def __init__(self, covs, q, avg_noise=False, rng=None):
        '''
        Compute smica decomposition on covs. q is the number of sources.
        '''
        rng = check_random_state(rng)
        n_epochs, p, _ = covs.shape
        self.covs = covs
        self.n_epochs = n_epochs
        self.p = p
        self.q = q
        self.A = rng.randn(p, q)
        if avg_noise:
            self.sigmas = np.abs(rng.randn(p))
        else:
            self.sigmas = np.abs(rng.randn(n_epochs, p))
        self.powers = np.abs(rng.randn(n_epochs, q))
        self.avg_noise = avg_noise

    def copy_params(self, target_smica):
        self.A = np.copy(target_smica.A)
        self.powers = np.copy(target_smica.powers)
        sigmas = target_smica.sigmas
        if not self.avg_noise:
            if not target_smica.avg_noise:
                self.sigmas = np.copy(sigmas)
            else:
                self.sigmas = np.ones(self.n_epochs)[:, None] * sigmas[None, :]
        else:
            if target_ica.avg_noise:
                self.sigmas = np.copy(sigmas)
            else:
                1/0
        return self

    def fit(self, **kwargs):
        A, sigmas, powers, x_list =\
            em_algo(self.covs, self.A, self.sigmas, self.powers,
                    self.avg_noise, **kwargs)
        self.A = A
        self.sigmas = sigmas
        self.powers = powers
        self.x_list = x_list
        return self

    def true_loss(self):
        '''
        compute the loss rectified with the log det. >=0, =0 if the model
        holds perfectly.
        '''
        return loss(self.covs, self.A, self.sigmas, self.powers,
                    self.avg_noise, normalize=True)

    def compute_approx_covs(self):
        '''
        Compute the covariances estimated by the model
        '''
        covs_approx = np.zeros((self.n_epochs, self.p, self.p))
        A = self.A
        for j, power in enumerate(self.powers):
            covs_approx[j] = A.dot(power[:, None] * A.T) + np.diag(self.sigmas)
        self.covs_approx = covs_approx
        return covs_approx
