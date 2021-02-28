"""
API for the core fitter algorithm : fit the model to a sequence of covariances
"""
import numpy as np

from scipy.linalg import sqrtm
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state

from qndiag import qndiag

from ._em import em_algo
from ._lbfgs import lbfgs
from ._lbfgs_noise import lbfgs_noise
from .utils import loss, compute_covariances


def lbfgs_solver(P, q):
    p, _ = P.shape

    def func(x):
        Px = np.dot(P, x)
        return 0.5 * np.dot(x, Px) + np.dot(q, x), Px + q
    x, _, _ = fmin_l_bfgs_b(func, np.zeros(p), bounds=[(0, None), ] * p)
    return x


def minimize_diag(A, B, C):
    # Minimize |A diag(x)A.T + B diag(y)B.T + C| subject to x, y >=0.
    p, n = A.shape
    _, m = B.shape
    P = np.zeros((n + m, n + m))
    Q = np.zeros(n + m)
    AB = np.dot(A.T, B) ** 2
    P[:n, :n] = np.dot(A.T, A) ** 2
    P[n:, n:] = np.dot(B.T, B) ** 2
    P[:n, n:] = AB
    P[n:, :n] = AB.T
    Q[:n] = np.diag(np.dot(A.T, np.dot(C, A)))
    Q[n:] = np.diag(np.dot(B.T, np.dot(C, B)))
    sol = lbfgs_solver(P, Q)
    return sol[:n], sol[n:]


def fit_powers(A, C):
    p, q = A.shape
    C_si = sqrtm(np.linalg.pinv(C))
    M1 = C_si.dot(A)
    return minimize_diag(M1, C_si, -np.eye(p))


def fit_A_fixed(A, C_list):
    n, _, _ = C_list.shape
    p, q = A.shape
    source_powers = np.zeros((n, q))
    sigmas = np.zeros((n, p))
    for i, C in enumerate(C_list):
        power, sigma = fit_powers(A, C)
        source_powers[i] = power
        sigmas[i] = sigma
    return source_powers, sigmas


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

    def fit(self, covs, y=None, tol=1e-6, em_it=10000, n_it_lbfgs=0,
            verbose=0, n_it_min=10, init='standard'):
        use_lbfgs = n_it_lbfgs > 0
        self.covs_ = covs
        n_samples, n_components, _ = covs.shape
        # Init
        covs_avg = np.mean(covs, axis=0)
        A = np.linalg.eigh(covs_avg)[1][:, -self.n_sources:]
        if init == 'qndiag':
            covs_pca = np.array([np.dot(A.T, np.dot(C, A)) for C in covs])
            B, _ = qndiag(covs_pca)
            A = np.dot(A, np.linalg.pinv(B))
            powers, sigmas = fit_A_fixed(A, covs)
            sigmas += 1e-10
            powers += 1e-10
            if self.avg_noise:
                sigmas = np.mean(sigmas, axis=0)
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
                em_algo(covs, self.A_, self.sigmas_, self.powers_,
                        corr=self.corr, avg_noise=True, tol=tol,
                        max_iter=em_it // 10,
                        verbose=verbose)
            if not self.avg_noise:
                sigmas = sigmas[None, :] * np.ones(n_samples)[:, None]
        A, sigmas, powers = \
            em_algo(covs, A, sigmas, powers, corr=self.corr,
                    avg_noise=self.avg_noise, tol=tol, max_iter=em_it,
                    n_it_min=n_it_min, verbose=verbose)
        if use_lbfgs and not self.avg_noise:
            if verbose:
                print('Running L-BFGS...')
            loss0 = loss(covs, A, sigmas, powers, self.avg_noise, self.corr)
            A, sigmas, powers, f, d = lbfgs(covs, A, sigmas, powers,
                                            max_fun=n_it_lbfgs,
                                            verbose=verbose)
            self.d = d
            if verbose:
                print('Done. Loss gain: %.2f %%' % (100 * (loss0 - f) / f))
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
            1/0  # XXX !!!
            # if target_ica.avg_noise:
            #     self.sigmas_ = np.copy(sigmas)
            # else:
            #     self.sigmas_ = np.means(sigmas, axis=0)
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


class CovarianceFitNoise(BaseEstimator, TransformerMixin):
    '''
    Compute smican decomposition
    '''
    def __init__(self, covs_noise, n_sources, rng=None):
        self.rng = check_random_state(rng)
        self.n_sources = n_sources
        self.covs_noise = covs_noise

    def fit(self, covs, y=None, tol=1e-6, max_iter=10000, init='qndiag',
            verbose=0):
        self.covs_ = covs
        covs_noise = self.covs_noise
        n_s = self.n_sources
        n_samples, n_components, _ = covs.shape
        # Init
        covs_avg = np.mean(covs, axis=0) - np.mean(covs_noise, axis=0)
        A = np.linalg.eigh(covs_avg)[1][:, -n_s:]
        if init == 'qndiag':
            covs_pca = np.array([np.dot(A.T, np.dot(d, A))
                                 for d in covs])
            B, _ = qndiag(covs_pca)
            A = np.dot(A, np.linalg.pinv(B))
            powers = np.diagonal([B.dot(C.dot(B.T))
                                  for C in covs_pca], axis1=1, axis2=2)
            powers = np.maximum(powers, 1e-10)
        else:
            powers = np.array([np.maximum(np.linalg.eigvalsh(d)[-n_s:], 0)
                               for d in covs - covs_noise])
        A, powers, f, d = lbfgs_noise(covs, covs_noise, A, powers,
                                      max_fun=max_iter,
                                      verbose=verbose)
        self.d = d
        self.A_ = A
        self.powers_ = powers
        return self
