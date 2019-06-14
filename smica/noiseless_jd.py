import numpy as np

from sklearn.utils import check_random_state
from mne.io import BaseRaw
from joblib import Memory

from qndiag import qndiag

from .core_smica import SMICA
from .mne import ICA, transfer_to_mne
from .utils import fourier_sampling
from .sobi import _transform_set

location = './cachedir'
memory = Memory(location, verbose=0)


class JDIAG(SMICA):
    def __init__(self, n_components,
                 freqs, sfreq, avg_noise=False, rng=None):
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
        self.rng = check_random_state(rng)

    def fit(self, X, y=None, **kwargs):
        '''
        Fits sobi to data X (p x n matrix sampled at fs)
        '''
        kwargs.setdefault('tol', 1e-6)
        kwargs.setdefault('max_iter', 10000)
        self.X = X
        C, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        n_mat, n_sensors, _ = C.shape
        self.C_ = C
        self.ft_ = ft
        self.freq_idx_ = freq_idx
        u, d, _ = np.linalg.svd(np.mean(C, axis=0))
        whitener = (u / d).T[:self.n_components]
        C_ = _transform_set(whitener, C)
        W, _ = qndiag(C_, **kwargs)
        W = W.dot(whitener)
        self.A_ = np.linalg.pinv(W)
        self.powers_ = np.zeros((n_mat, self.n_components))
        for i in range(n_mat):
            self.powers_[i] = np.diag(W.dot(C[i]).dot(W.T))
        scale = np.mean(self.powers_, axis=0, keepdims=True)
        self.A_ = self.A_ * np.sqrt(scale)
        self.powers_ = self.powers_ / scale
        self.sigmas_ = np.zeros((C.shape[0], X.shape[0]))
        return self

    def compute_sources(self, X=None, method='pinv'):
        if method == 'wiener':
            raise ValueError('Only method=pinv is implemented for SOBI')
        return super().compute_sources(X=X, method=method)


class JDIAG_mne(ICA):
    def __init__(self, n_components, freqs, rng=None):
        self.n_components = n_components
        self.freqs = freqs
        self.f_scale = 0.5 * (freqs[1:] + freqs[:-1])
        self.rng = check_random_state(rng)

    def fit(self, inst, picks=None, avg_noise=False, **kwargs):
        '''
        Fits smica to inst (either raw or epochs)
        '''
        self.inst = inst
        self.info = inst.info
        self.sfreq = inst.info['sfreq']
        self.picks = picks
        self.avg_noise = avg_noise
        if isinstance(inst, BaseRaw):
            self.inst_type = 'raw'
            X = inst.get_data(picks=picks)
        else:
            self.inst_type = 'epoch'
            X = inst.get_data()
            n_epochs, _, _ = X.shape
            X = np.hstack(X)
        self.X = X
        X /= np.std(X)
        smica = JDIAG(self.n_components, self.freqs, self.sfreq,
                      self.avg_noise)
        smica.fit(X, **kwargs)
        self.powers = smica.powers_
        self.A = smica.A_
        self.sigmas = smica.sigmas_
        self.smica = smica
        self.ica_mne = transfer_to_mne(self.A, self.inst, self.picks)
        return self
