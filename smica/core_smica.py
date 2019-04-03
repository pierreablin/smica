"""
API for the core SMICA algorithm : fit the model to a sequence of covariances
"""
import numpy as np

from sklearn.utils import check_random_state

from ._em import loss, em_algo


class SMICA(object):
    def __init__(self, covs, q, avg_noise=True, rng=None):
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

    def _compute_ld_cov(self):
        '''
        compute the sum of logdet of the covariances
        '''
        ld_cov = 0.
        for cov in self.covs:
            ld_cov += np.linalg.slogdet(cov)[1]
        self.ld_cov = ld_cov
        return ld_cov

    def true_loss(self):
        '''
        compute the loss rectified with the log det. >=0, =0 if the model
        holds perfectly.
        '''
        L = loss(self.covs, self.A, self.sigmas, self.powers,
                 self.avg_noise)
        return L - self._compute_ld_cov() - self.p * self.n_epochs

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
