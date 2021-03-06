"""
XXX
============================

"""

import numpy as np
from smica import ICA, JDIAG_mne, SOBI_mne, transfer_to_ica
import mne
from mne.datasets import sample
from mne.preprocessing import ICA as ICA_mne

from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)
EPS = 1e-12

# fetch data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)

raw_room = mne.io.read_raw_fif(data_path + '/MEG/sample/ernoise_raw.fif')
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')


def conf_mat(A, B):
    _, n = A.shape
    _, p = B.shape
    A /= np.linalg.norm(A, axis=0)
    B /= np.linalg.norm(B, axis=0)
    return A.T.dot(B)


def matching(corr):
    corr = np.abs(corr)
    n_s, _ = corr.shape
    order = np.arange(n_s)
    sources = set(range(n_s))
    free = set(range(n_s))
    while sources:
        i, j = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        corr[i, j] = - 1
        if i in sources and j in free:
            sources.remove(i)
            free.remove(j)
            order[i] = j
    return order


n_bins = 40
freqs = np.linspace(1, 70, n_bins + 1)
n_comp = 40
smica_room = ICA(10, freqs=freqs, rng=0).fit(raw_room, picks)
jdiag_room = JDIAG_mne(10, freqs=freqs, rng=0).fit(raw_room, picks=picks)
smica = ICA(n_comp, freqs=freqs, rng=0).fit(raw, picks=picks)
to_plot_smica = [37, 38, 39, 20, 4, 3, 19, 8, 12, 2]
raw.filter(1, 70)
jdiag = JDIAG_mne(n_comp, freqs=freqs, rng=0).fit(raw, picks=picks)
to_plot_jdiag = [2, 1, 0, 13, 34, 3, 20, 30, 5, 10]
sobi = SOBI_mne(100, n_comp, freqs).fit(raw, picks=picks)
to_plot_sobi = [2, 0, 8, 12, 20, 31, 3, 17, 9, 10]
ifmx_ = ICA_mne(n_comp, random_state=0).fit(raw, picks=picks)
ifmx = transfer_to_ica(raw, picks, freqs, ifmx_.get_sources(raw).get_data(),
                       ifmx_.get_components())
to_plot_ifmx = [1, 2, 0, 35, 33, 19, 3, 32, 25, 9]
plot_args = dict(number=True, t_min=2, t_max=4)

pow_lims = [(np.min(algo.powers), np.max(algo.powers))
            for algo in [smica, jdiag, sobi, ifmx]]
y_lim = 0.0001, 0.21

for algo, sort, method, name in zip([smica, jdiag, sobi, ifmx],
                                    [to_plot_smica, to_plot_jdiag,
                                     to_plot_sobi, to_plot_ifmx],
                                    ['wiener', 'pinv', 'pinv', 'pinv'],
                                    ['smica', 'jdiag', 'sobi', 'ifmx']):

    algo.plot_extended(sources=algo.compute_sources(method=method),
                       sort=sort, save=name, y_lim=y_lim, **plot_args)
