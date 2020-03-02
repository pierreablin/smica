import numpy as np

from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering

from .core_fitter import CovarianceFitNoise
from .utils import fourier_sampling, itakura, loss

eps = 1e-12


def wiener(A, powers, noise_inv):
    '''
    The Wiener filter
    '''
    C = np.linalg.pinv(A.T.dot(noise_inv.dot(A)) +
                       np.diag(1. / (powers + eps)))
    return C.dot(A.T.dot(noise_inv))


class SMICAN(object):
    '''
    Core smican procedure: transform in the frequency domain, etc..
    '''
    def __init__(self, X_noise, n_components, freqs, sfreq, rng=None):
        '''
        n_components : number of sources
        freqs : the frequency intervals
        sfreq : sampling frequency
        '''
        self.n_components = n_components
        self.freqs = freqs
        self.sfreq = sfreq
        self.f_scale = 0.5 * (freqs[1:] + freqs[:-1])
        if len(X_noise.shape) == 2:
            C, _, _ = fourier_sampling(X_noise, self.sfreq, self.freqs)
        else:
            n_mat, p, _ = X_noise.shape
            C = np.zeros((len(freqs) - 1, p, p))
            for X in X_noise:
                C += fourier_sampling(X, self.sfreq, self.freqs)[0]
            C /= n_mat
        self.C_noise = C
        self.C_noise_inv = np.array([np.linalg.inv(c) for c in C])
        self.rng = check_random_state(rng)

    def fit(self, X, y=None, **kwargs):
        '''
        Fits smica to data X (p x n matrix sampled at fs)
        '''
        self.X = X.copy()
        C, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        self.C_ = C
        self.ft_ = ft
        self.freq_idx_ = freq_idx
        covfit = CovarianceFitNoise(self.C_noise, self.n_components,
                                    self.rng)
        covfit.fit(C, **kwargs)
        self.A_ = covfit.A_
        self.powers_ = covfit.powers_
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
        covs_approx = compute_covariances(self.A_, self.powers_, self.C_noise)
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
            for j, (C_i, power) in enumerate(zip(self.C_noise_inv,
                                                 self.powers_)):
                sl = np.arange(freq_idx[j], freq_idx[j+1])
                W = wiener(self.A_, power, C_i)
                transf = np.dot(W, ft[:, sl])
                ft_sources[:, sl] = transf
                ft_sources[:, n - sl] = np.conj(transf)
            return np.real(np.fft.ifft(ft_sources))
        elif method == 'pinv':
            if X is None:
                X = self.X
            return np.linalg.pinv(self.A_).dot(X)

    def filter(self, X=None, bad_sources=[], method='wiener'):
        S = self.compute_sources(X, method=method)
        S[bad_sources] = 0.
        return np.dot(self.A_, S)
