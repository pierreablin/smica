import numpy as np

from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering

from .core_fitter import CovarianceFit
from .utils import fourier_sampling, itakura, loss, compute_covariances

eps = 1e-12


def wiener(A, sigmas, powers, corr):
    '''
    The Wiener filter
    '''
    if corr:
        C = np.linalg.pinv(A.T.dot(A / (sigmas[:, None] + eps)) +
                           np.linalg.pinv(powers))
    else:
        C = np.linalg.pinv(A.T.dot(A / (sigmas[:, None] + eps)) +
                           np.diag(1. / (powers + eps)))
    return C.dot(A.T / (sigmas[None, :] + eps))


class SMICA(object):
    '''
    Core smica procedure: transform in the frequency domain, etc..
    '''
    def __init__(self, n_components, freqs, sfreq, avg_noise=False, corr=False,
                 rng=None):
        '''
        n_components : number of sources
        freqs : the frequency intervals
        sfreq : sampling frequency
        '''
        self.n_components = n_components
        self.freqs = freqs
        self.sfreq = sfreq
        self.avg_noise = avg_noise
        self.f_scale = 0.5 * (freqs[1:] + freqs[:-1])
        self.corr = corr
        self.rng = check_random_state(rng)
        self.filtering_method = 'wiener'

    def fit(self, X, y=None, **kwargs):
        '''
        Fits smica to data X (p x n matrix sampled at fs)
        '''
        self.X = X.copy()
        C, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        self.C_ = C
        self.ft_ = ft
        self.freq_idx_ = freq_idx
        covfit = CovarianceFit(self.n_components, self.avg_noise, self.corr,
                               self.rng)
        covfit.fit(C, **kwargs)
        self.A_ = covfit.A_
        self.powers_ = covfit.powers_
        self.sigmas_ = covfit.sigmas_
        return self

    def is_matrix(self):
        IS = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            for j in range(self.n_components):
                IS[i, j] = itakura(self.powers_[:, i], self.powers_[:, j])
        self.IS_ = IS
        return IS

    def compute_approx_covs(self):
        '''
        Compute the covariances estimated by the model
        '''
        covs_approx = compute_covariances(self.A_, self.powers_, self.sigmas_,
                                          self.avg_noise, self.corr)
        return covs_approx

    def compute_f_div(self, halve=False):
        f = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            for j in range(self.n_components):
                p1 = self.powers_[:, i]
                p2 = self.powers_[:, j]
                frac = p1 / p2
                if halve:
                    frac = frac[:len(frac) // 2]
                f[i, j] = np.mean(frac) * np.mean(1. / frac) - 1.
        self.f_div = f
        return f

    def compute_sources(self, X=None, method='wiener'):
        if method == 'wiener':
            if X is None:
                ft = self.ft_
                freq_idx = self.freq_idx_
            else:
                _, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
            p, n = ft.shape
            ft_sources = 1j * np.zeros((self.n_components, n))
            if self.avg_noise:
                sigs = [self.sigmas_, ] * len(self.f_scale)
            else:
                sigs = self.sigmas_
            for j, (sigma, power) in enumerate(zip(sigs, self.powers_)):
                sl = np.arange(freq_idx[j], freq_idx[j + 1])
                W = wiener(self.A_, sigma, power, self.corr)
                transf = np.dot(W, ft[:, sl])
                ft_sources[:, sl] = transf
                ft_sources[:, n - sl] = np.conj(transf)
            return np.real(np.fft.ifft(ft_sources))
        elif method == 'pinv':
            if X is None:
                X = self.X
            return np.linalg.pinv(self.A_).dot(X)

    def filter(self, X=None, bad_sources=[], method=None):
        if method is None:
            method = self.filtering_method
        S = self.compute_sources(X, method=method)
        S[bad_sources] = 0.
        return np.dot(self.A_, S)

    def compute_loss(self, X=None, by_bin=False):

        if X is None:
            covs = self.C_
            freq_idx = self.freq_idx_
        else:
            covs, _, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        self.baseline_loss =\
            loss(covs, np.zeros_like(self.A_),
                 np.diagonal(covs, axis1=1, axis2=2),
                 np.zeros_like(self.powers_), avg_noise=self.avg_noise,
                 corr=self.corr, normalize=True, by_bin=by_bin)
        kls = loss(covs, self.A_, self.sigmas_, self.powers_,
                   avg_noise=self.avg_noise, corr=self.corr,
                   normalize=True, by_bin=True)
        n_modes = np.diff(freq_idx)
        if by_bin:
            return n_modes * kls
        else:
            return np.sum(n_modes * kls)

    def degrees_freedom(self):
        p, m = self.A_.shape
        q = len(self.f_scale)
        return (q * p * (p+1)) // 2, q * p + q * m + m * p

    def cluster(self, mat, n_clusters, **kwargs):
        if 'linkage' not in kwargs:
            kwargs['linkage'] = 'average'
        clustering = AgglomerativeClustering(n_clusters, 'precomputed',
                                             **kwargs)
        clustering.fit(mat + mat.T)
        self.labels_ = clustering.labels_
        return self.labels_

    def compute_filters(self):
        n_mat = len(self.f_scale)
        p = self.A_.shape[0]
        filters = np.zeros((n_mat, self.n_components, p))

        for j, (sigma, power) in enumerate(zip(self.sigmas_, self.powers_)):
            filters[j] = wiener(self.A_, sigma, power, self.corr)
        return filters

    def save_params(self, save_str):
        covs = self.C_
        n_mat, p, _ = covs.shape
        covs_ravel = covs.reshape(n_mat, p ** 2)
        to_save = [self.A_, self.sigmas_, self.powers_, self.f_scale,
                   covs_ravel]
        names = ['mixing_matrix', 'noise_power', 'source_power', 'frequencies',
                 'covariances']
        for array, name in zip(to_save, names):
            np.savetxt(save_str + name + '.csv', array, delimiter=',')
