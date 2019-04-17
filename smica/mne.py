
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA as ICA_
from mne.io.pick import pick_info
from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering
from mne.io import BaseRaw
from mne.epochs import BaseEpochs

from .core_smica import SMICA
from .utils import fourier_sampling, itakura, loss


def wiener(A, sigmas, powers):
    '''
    The Wiener filter
    '''
    C = np.linalg.pinv(A.T.dot(A / sigmas[:, None]) +
                       np.diag(1. / powers))
    return C.dot(A.T / sigmas[None, :])


def transfer_to_mne(A, raw, picks):
    '''
    Hack to use the MNE ICA class providing the estimated mixing matrix A.
    '''
    p, q = A.shape
    ica = ICA_(n_components=q, method='fastica', random_state=0,
               fit_params=dict(max_iter=1))
    ica.info = pick_info(raw.info, picks)
    if ica.info['comps']:
        ica.info['comps'] = []
    ica.ch_names = ica.info['ch_names']
    ica.mixing_matrix_ = A
    ica.pca_components_ = np.eye(p)
    ica.pca_mean_ = None
    ica.unmixing_matrix_ = np.linalg.pinv(A)
    ica.n_components_ = p
    ica._update_ica_names()
    return ica


def transfer_to_ica(raw, picks, freqs, S, A):
    '''
    Transfer the sources S and matrix A to the ica class.
    '''
    n_sensors, p = A.shape
    smica = ICA(n_components=p)
    C, ft, freq_idx = fourier_sampling(S, smica.sfreq, freqs)
    Q, _, _ = C.shape
    smica.powers = np.diagonal(C, axis1=1, axis2=2)
    smica.powers = smica.powers / np.mean(smica.powers, axis=0,
                                          keepdims=True)
    P, N = A.shape
    smica.n_components = N
    smica.avg_noise = False
    smica.A = A
    smica.sigmas = np.zeros((Q, N))
    smica.ica_mne = transfer_to_mne(A, raw, picks,
                                    sort=False)
    return smica


class ICA(object):
    '''
    Mimics some the the mne.preprocessing ICA API.
    '''
    def __init__(self, n_components, freqs, rng=None):
        '''
        n_components : number of sources
        freqs : the frequency intervals
        '''
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
        self._fit(X, **kwargs)
        return self

    def _fit(self, X, **kwargs):
        C, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        self.C = C
        self.ft = ft
        self.freq_idx = freq_idx
        smica_ = SMICA(self.n_components, True, self.rng)
        smica_.fit(self.C, **kwargs)
        if not self.avg_noise:
            smica = SMICA(self.n_components, False, self.rng)
            smica.copy_params(smica_)
            smica.fit(self.C, **kwargs)
        else:
            smica = smica_
        self.A = smica.A_
        self.powers = smica.powers_
        self.sigmas = smica.sigmas_
        self.ica_mne = transfer_to_mne(self.A, self.inst, self.picks)
        return self

    def _plot(self, attr, picks=None, ax=None, **kwargs):
        if ax is None:
            f, ax = plt.subplots()
        to_plot = getattr(self, attr)
        if picks is None:
            picks = np.arange(to_plot.shape[1])
        label = kwargs.get('label')
        if label is not None:
            if type(label) is str:
                line = ax.semilogy(self.f_scale, to_plot[:, picks[0]],
                                   **kwargs)
                kwargs['label'] = None
                if len(picks) > 1:
                    line = ax.semilogy(self.f_scale, to_plot[:, picks[1:]],
                                       **kwargs)
        line = ax.semilogy(self.f_scale, to_plot[:, picks], **kwargs)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        return line

    def plot_noise(self, picks=None, ax=None, **kwargs):
        return self._plot('sigmas', picks=picks, ax=ax, **kwargs)

    def plot_powers(self, picks=None, ax=None, **kwargs):
        return self._plot('powers', picks=picks, ax=ax, **kwargs)

    def plot_components(self, **kwargs):
        self.ica_mne.plot_components(**kwargs)

    def plot_properties(self, picks):
        bandwidth = np.diff(self.freqs).mean()
        self.ica_mne.plot_properties(self.inst, picks=picks,
                                     psd_args=dict(fmin=self.freqs[0],
                                                   fmax=self.freqs[-1],
                                                   bandwidth=bandwidth))

    def is_matrix(self):
        IS = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            for j in range(self.n_components):
                IS[i, j] = itakura(self.powers[:, i], self.powers[:, j])
        self.IS = IS
        return IS

    def compute_f_div(self, halve=False):
        f = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components):
            for j in range(self.n_components):
                p1 = self.powers[:, i]
                p2 = self.powers[:, j]
                frac = p1 / p2
                if halve:
                    frac = frac[:len(frac) // 2]
                f[i, j] = np.mean(frac) * np.mean(1. / frac) - 1.
        self.f_div = f
        return f

    def compute_sources(self, X=None):
        if X is None:
            ft = self.ft
            freq_idx = self.freq_idx
        else:
            _, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        p, n = ft.shape
        ft_sources = 1j * np.zeros((self.n_components, n))
        if self.avg_noise:
            sigs = [self.sigmas, ] * len(self.f_scale)
        else:
            sigs = self.sigmas
        for j, (sigma, power) in enumerate(zip(sigs, self.powers)):
            sl = np.arange(freq_idx[j], freq_idx[j+1])
            W = wiener(self.A, sigma, power)
            transf = np.dot(W, ft[:, sl])
            ft_sources[:, sl] = transf
            ft_sources[:, n - sl] = np.conj(transf)
        return np.real(np.fft.ifft(ft_sources))

    def filter(self, bad_sources=[]):
        S = self.compute_sources()
        S[bad_sources] = 0.
        return np.dot(self.A, S)

    def compute_loss(self, X=None):
        if X is None:
            covs = self.C
        else:
            covs, _, _ = fourier_sampling(X, self.sfreq, self.freqs)
        return loss(covs, self.A, self.sigmas, self.powers,
                    self.avg_noise, normalize=True)

    def degrees_freedom(self):
        p, m = self.A.shape
        q = len(self.f_scale)
        return (q * p * (p+1)) // 2, q * p + q * m + m * p

    def avg_freq(self):
        return np.mean(self.powers * self.f_scale[:, None], axis=0)

    def cluster(self, mat, n_clusters, **kwargs):
        if 'linkage' not in kwargs:
            kwargs['linkage'] = 'average'
        clustering = AgglomerativeClustering(n_clusters, 'precomputed',
                                             **kwargs)
        clustering.fit(mat + mat.T)
        self.labels = clustering.labels_
        return self.labels

    def plot_clusters(self, n_clusters, mode='f_div', order=None):
        if mode == 'f_div':
            mat = self.compute_f_div()
        else:
            mat = self.is_matrix()
        if order is None:
            order = np.arange(n_clusters)
        labels = self.cluster(mat, n_clusters)
        n_rows = 4
        n_cols = (n_clusters - 1) // n_rows + 1
        f, axes = plt.subplots(n_cols, n_rows,
                               figsize=(5 * n_rows, 5 * n_cols))
        for i, axe in zip(range(n_clusters), axes.ravel()):
            axe.grid()
            pick = labels == order[i]
            self.plot_powers(picks=pick, ax=axe)
            axe.set_title('cluster %d | %d sources' % (order[i], sum(pick)))
        return labels

    def band_power(self, f_min, f_max):
        idx = (f_min < self.f_scale) * (self.f_scale < f_max)
        return np.mean(self.powers[idx, :], axis=0)
