import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert

import mne
from mne.preprocessing import ICA as ICA_
from mne.io.pick import pick_info
from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering
from mne.io import BaseRaw
from mne.epochs import BaseEpochs
from mne.viz.topomap import _plot_ica_topomap

from mne.viz import plot_topomap
from .core_smica import SMICA
from .core_smican import SMICAN
from .utils import fourier_sampling, itakura, loss
from .dipolarity import dipolarity_using_sphere_model
from .viz import plot_extended
from .mutual_info import mutual_information_2d


def plot_components(A, raw, picks, **kwargs):
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
    ica.plot_components(**kwargs)


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
    smica = ICA(n_components=p, freqs=freqs)
    sfreq = raw.info['sfreq']
    C, ft, freq_idx = fourier_sampling(S, sfreq, freqs)
    Q, _, _ = C.shape
    smica.powers = np.diagonal(C, axis1=1, axis2=2)
    smica.powers = smica.powers / np.mean(smica.powers, axis=0,
                                          keepdims=True)
    P, N = A.shape
    smica.n_components = N
    smica.avg_noise = False
    smica.A = A
    smica.picks = picks
    smica.sigmas = np.zeros((Q, N))
    smica.ica_mne = transfer_to_mne(A, raw, picks)
    smica_ = SMICA(N, freqs, sfreq)
    smica_.A_ = A
    smica_.sigmas_ = smica.sigmas
    smica_.powers_ = smica.powers
    smica.scaling_ = np.ones(P)
    smica.smica = smica_
    smica.sfreq = sfreq
    smica.smica.X = raw.get_data(picks=picks)
    return smica


class ICA(object):
    '''
    Mimics some the the mne.preprocessing ICA API.
    '''
    def __init__(self, n_components, freqs, room_noise=False, rng=None):
        '''
        n_components : number of sources
        freqs : the frequency intervals
        '''
        self.n_components = n_components
        self.freqs = freqs
        self.f_scale = 0.5 * (freqs[1:] + freqs[:-1])
        self.room_noise = room_noise
        self.rng = check_random_state(rng)

    def fit(self, inst, picks=None, avg_noise=False, corr=False, exclude=None,
            inst_room=None, crop=None, **kwargs):
        '''
        Fits smica to inst (either raw or epochs)
        '''
        self.inst = inst
        self.info = inst.info
        self.sfreq = inst.info['sfreq']
        self.picks = picks
        self.corr = corr
        self.avg_noise = avg_noise
        self.exclude = exclude
        if isinstance(inst, BaseRaw):
            self.inst_type = 'raw'
            X = inst.get_data(picks=picks)
        else:
            self.inst_type = 'epoch'
            X = inst.get_data(picks=picks)
            n_epochs, _, _ = X.shape
            X = np.hstack(X)

        if exclude is not None:
            X = np.delete(X, exclude, axis=0)
        self.X = X
        normalization = np.std(X)
        X /= normalization
        self.normalization = normalization
        scaling = np.std(X, axis=1)
        X /= scaling[:, None]
        self.scaling_ = scaling
        if self.room_noise:
            if isinstance(inst_room, BaseRaw):
                X_room = inst_room.get_data(picks=picks)
            else:
                if crop is not None:
                    inst_room = inst_room.load_data().crop(*crop)
                X_room = inst_room.get_data(picks=picks)
            if exclude is not None:
                X_room = np.delete(X_room, exclude, axis=0)
            X_room /= normalization
            X_room /= scaling[:, None]
            smica = SMICAN(X_room, self.n_components, self.freqs, self.sfreq)
            smica.fit(X, **kwargs)
        else:
            smica = SMICA(self.n_components, self.freqs, self.sfreq,
                          avg_noise=self.avg_noise, corr=self.corr)
            smica.fit(X, **kwargs)
            self.sigmas = smica.sigmas_ * scaling ** 2
        self.powers = smica.powers_
        self.A = smica.A_ * scaling[:, None]
        self.smica = smica
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
        return self.smica.is_matrix()

    def compute_f_div(self, halve=False):
        return self.smica.compute_f_div(halve)

    def compute_sources(self, X=None, **kwargs):
        if X is None:
            X_s = None
        else:
            X_s = X / self.scaling_[:, None]
        return self.smica.compute_sources(X_s, **kwargs)

    def filter(self, X=None, bad_sources=[], method='wiener'):
        if X is None:
            return self.smica.filter(None, bad_sources, method=method)
        else:
            X_s = X / self.scaling_[:, None]
            filtered = self.smica.filter(X_s,
                                         bad_sources, method=method)
            return filtered * self.scaling_[:, None]

    def compute_mutual_information(self, X=None, method='wiener'):
        S = self.compute_sources(X=X, method=method)
        n_sources = S.shape[0]
        MI = np.zeros((n_sources, n_sources))
        for i in range(n_sources):
            for j in range(i):
                mi = mutual_information_2d(S[i], S[j])
                MI[i, j] = mi
                MI[j, i] = mi
        return MI

    def compute_loss(self, X=None, **kwargs):
        return self.smica.compute_loss(X / self.scaling_[:, None], **kwargs)

    def cluster(self, mat, n_clusters, **kwargs):
        labels = self.smica.cluster(mat, n_clusters, **kwargs)
        self.labels = labels
        return labels

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

    def compute_dipolarity(self, inst):
        return dipolarity_using_sphere_model(self.A, inst, self.picks)

    def plot_extended(self, sources=None, **kwargs):
        if sources is None:
            sources = self.compute_sources()
        plot_extended(sources,
                      self.sfreq, self.f_scale,
                      self.powers, self.ica_mne, **kwargs)

    def plot_noise_topo(self, pick_types='mag'):
        picks = mne.pick_types(self.info, meg=pick_types)
        f, axes = plt.subplots(2, 5)
        powers = np.sqrt(self.sigmas) * self.normalization
        for j, ax in enumerate(axes.ravel()):
            f_idx = int(len(self.freqs) / 10)
            power = powers[f_idx, picks]
            plot_topomap()
