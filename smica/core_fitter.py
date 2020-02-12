"""
API for the core fitter algorithm : fit the model to a sequence of covariances
"""
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from qndiag import qndiag

from ._em import em_algo
from ._lbfgs import lbfgs
from .utils import loss, compute_covariances


class CovarianceFit(BaseEstimator, TransformerMixin):
    '''
    Compute smica decomposition
    '''
    def __init__(self, n_sources, avg_noise=False, corr=False,
                 transformer='power', rng=None):
        self.rng = check_random_state(rng)
        self.n_sources = n_sources
        self.avg_noise = avg_noise
        self.corr = corr
        self.transformer = transformer

    def fit(self, covs, y=None, tol=1e-6, em_it=10000, use_lbfgs=False,
            pgtol=1e-3, verbose=0, n_it_min=10, init='qndiag'):
        self.covs_ = covs
        n_samples, n_components, _ = covs.shape
        # Init
        covs_avg = np.mean(covs, axis=0)
        A = np.linalg.eigh(covs_avg)[1][:, -self.n_sources:]
        if init == 'qndiag':
            covs_pca = np.array([np.dot(A.T, np.dot(C, A)) for C in covs])
            unmix_pca, B, _ = qndiag(covs_pca, return_set=True)
            A = np.dot(A, np.linalg.pinv(B))
            self.A_ = A
            if self.corr:
                self.powers_ = unmix_pca
                covs_expl = np.array([A.dot(P.dot(A.T)) for P in self.powers_])
            else:
                self.powers_ = np.diagonal(unmix_pca, axis1=1, axis2=2)
                covs_expl = np.array([A.dot(P[:, None] * A.T)
                                      for P in self.powers_])
            self.sigmas_ = (np.mean((covs - covs_expl) ** 2) *
                            np.ones(n_components))
        else:
            self.A_ = A
            if self.corr:
                self.powers_ = np.array([A.T.dot(C.dot(A))
                                         for C in covs])
            else:
                self.powers_ = np.array([np.diag(A.T.dot(C.dot(A)))
                                        for C in covs])
            self.sigmas_ = np.mean(covs_avg, axis=1)
        A, sigmas, powers = \
            em_algo(covs, self.A_, self.sigmas_, self.powers_, corr=self.corr,
                    avg_noise=True, tol=tol, max_iter=em_it // 10,
                    verbose=verbose)
        if not self.avg_noise:
            sigmas = sigmas[None, :] * np.ones(n_samples)[:, None]
            A, sigmas, powers = \
                em_algo(covs, A, sigmas, powers, corr=self.corr,
                        avg_noise=False, tol=tol, max_iter=em_it,
                        n_it_min=n_it_min, verbose=verbose)
        if use_lbfgs:
            if verbose:
                print('Running L-BFGS...')
            loss0 = loss(covs, A, sigmas, powers, self.avg_noise, self.corr)
            A, sigmas, powers, f, d = lbfgs(covs, A, sigmas, powers,
                                            self.avg_noise, pgtol=pgtol)
            if verbose:
                print('Done. Loss: {}'.format(loss0 - f))
        self.A_ = A
        self.sigmas_ = sigmas
        self.powers_ = powers
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        if self.transformer == 'powers':
            return self.powers_
        elif self.transformer == 'sigmas':
            return self.sigmas_

    def copy_params(self, target_cov_fit):
        self.A_ = np.copy(target_cov_fit.A_)
        self.powers_ = np.copy(target_cov_fit.powers_)
        n_samples = len(self.powers_)
        sigmas = target_cov_fit.sigmas_
        if not self.avg_noise:
            if not target_cov_fit.avg_noise:
                self.sigmas_ = np.copy(sigmas)
            else:
                self.sigmas_ = (np.ones(n_samples)[:, None] *
                                sigmas[None, :])
        else:
            if target_ica.avg_noise:
                self.sigmas_ = np.copy(sigmas)
            else:
                self.sigmas_ = np.means(sigmas, axis=0)
        self.initialized = True
        return self

    def true_loss(self, covs=None):
        '''
        compute the loss rectified with the log det. >=0, =0 if the model
        holds perfectly.
        '''
        if covs is None:
            covs = self.covs_
        return loss(covs, self.A_, self.sigmas_, self.powers_,
                    avg_noise=self.avg_noise, corr=self.corr, normalize=True)

    def compute_approx_covs(self):
        '''
        Compute the covariances estimated by the model
        '''
        covs_approx = compute_covariances(self.A_, self.powers_, self.sigmas_,
                                          self.avg_noise, self.corr)
        return covs_approx
