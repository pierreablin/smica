import numpy as np
import matplotlib.pyplot as plt
from smica import ICA, mutual_information_2d
import mne
from mne.datasets import sample
from picard import picard

from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)
EPS = 1e-12

# fetch data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
# T = 20
# raw.crop(0, T)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

n_bins = 20
freqs = np.linspace(2, 35, n_bins + 1)

raw.filter(2, 35)
n_comp = 20
# smica = ICA(n_components=n_comp, freqs=freqs, rng=0)
# smica.fit(raw, picks=picks, verbose=100, tol=1e-7, em_it=10000, corr=True)
# # P = smica.powers
# # eigs = np.array([np.linalg.eigvalsh(p) for p in P])
# # plt.semilogy(eigs.T[::-1])
# # plt.show()
# S = smica.compute_sources()


def getent2(u, nbins=None):

    """
    This is a python translation of the getent2.m matlab function, which
    is included in the code realeased with the plosone paper
    "EEG independent sources are dipolar"
    """
    nu, Nu = u.shape

    if nbins is None:
        nbins = np.min([100, np.round(np.sqrt(Nu))])

    Hu = np.zeros(nu)
    deltau = np.zeros(nu)

    for i in range(nu):

        umax = np.max(u[i])
        umin = np.min(u[i])

        deltau[i] = (umax - umin) / float(nbins)

        uu = \
            1 + np.round((nbins - 1) * (u[i] - umin) / float(umax - umin))

        temp = np.nonzero(np.diff(np.sort(uu)))[0] + 1

        ##############################################################
        # pmfr = np.diff(np.append(np.append(0, temp), Nu)) / float(Nu)

        temp2 = np.zeros(len(temp) + 2)
        temp2[-1] = Nu
        temp2[1:-1] = temp
        pmfr = np.diff(temp2) / float(Nu)

        ##############################################################

        Hu[i] = -np.sum(pmfr * np.log(pmfr)) + np.log(deltau[i])

    return Hu, deltau


def get_mir(estimated_sources, sfreq=1000):

    h0 = 0
    h, _ = getent2(estimated_sources)

    # Let's transform nats to kbits per second
    # 1 nat = log2(exp(1)) bits
    # EEG.srate is the sampling frequency
    # 1kbit = 1000 bits
    C = np.dot(estimated_sources, estimated_sources.T)
    mir = (- h + .5 * np.linalg.slogdet(C)[1] / len(h)) * \
        np.log2(np.exp(1)) * sfreq / 1000.

    # mir is expressed in kbits / seconds
    return mir


def get_mi(S):
    n_sources = S.shape[0]
    MI = np.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(i):
            mi = mutual_information_2d(S[i], S[j], 1, False)
            MI[i, j] = mi
            MI[j, i] = mi
    return MI


@memory.cache()
def get_sources(n_comp):
    #
    smica = ICA(n_components=n_comp, freqs=freqs, rng=0)
    smica.fit(raw, picks=picks, tol=1e-7, em_it=10000, corr=True)
    # P = smica.powers
    # eigs = np.array([np.linalg.eigvalsh(p) for p in P])
    # plt.semilogy(eigs.T[::-1])
    # plt.show()
    S = smica.compute_sources()
    _, _, S_smica = picard(S, ortho=False, extended=False, max_iter=1000)
    _, _, S_infomax = picard(raw.get_data(picks=picks), n_components=n_comp,
                             ortho=False, extended=False, max_iter=1000)
    S_smica /= np.sqrt(np.mean(S_smica ** 2, axis=1, keepdims=True))
    S_infomax /= np.sqrt(np.mean(S_infomax ** 2, axis=1, keepdims=True))
    return S, S_smica, S_infomax


mi_S = []
mi_I = []
n_comps = [20]
mir_S = []
mir_I = []
for n_comp in n_comps:
    print(n_comp)
    S, S_smica, S_infomax = get_sources(n_comp)
    mir_smica = get_mir(S)
    mi_s = get_mi(S_smica)
    mi_i = get_mi(S_infomax)
    mir_s = get_mir(S_smica)
    mir_i = get_mir(S_infomax)
# plt.plot(n_comps, [np.max(mi) for mi in mi_S], label='smica')
# plt.plot(n_comps, [np.max(mi) for mi in mi_I], label='pca')
# plt.legend()
# plt.show()
