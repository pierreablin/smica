import numpy as np
import os
import matplotlib.pyplot as plt
from smica import ICA, transfer_to_ica, SOBI_mne, JDIAG_mne, dipolarity
import mne

from mne.preprocessing import ICA as ICA_mne

from mne.datasets import sample

from picard import picard
from sklearn.decomposition import fastica

rc = {"pdf.fonttype": 42, 'text.usetex': False, 'font.size': 14,
      'xtick.labelsize': 12, 'ytick.labelsize': 12, 'text.latex.preview': True}

plt.rcParams.update(rc)

colors = ['indianred', 'cornflowerblue', 'k']


def plot_powers(powers, noise_sources, muscle_source, ax, title):
    cols = []
    for i in range(n_components):
        if i in noise_sources:
            cols.append(colors[2])
        elif i in muscle_source:
            cols.append(colors[1])
        else:
            cols.append(colors[0])
    for p, col in zip(powers.T, cols):
        ax.semilogy(freqs[1:], p, color=col)
    ax.semilogy([], [], color=colors[0], label='Brain sources')
    ax.semilogy([], [], color=colors[1], label='Muscle source')
    ax.semilogy([], [], color=colors[2], label='Room noise')
    x_ = ax.set_xlabel('Frequency (Hz.)')
    y_ = ax.set_ylabel('Power')
    t_ = ax.set_title(title)
    ax.set_xlim([0, freqs.max()])
    ax.grid()
    plt.savefig('figures/%s.pdf' % title, bbox_extr_artists=[x_, y_, t_],
                bbox_inches='tight')
    # ax.legend(loc='upper center', ncol=2)


# fetch data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')

n_bins = 40
n_components = 20
freqs = np.linspace(1, 60, n_bins + 1)
#
jdiag = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
jdiag.fit(raw, picks=picks, verbose=True, tol=1e-9, max_iter=1000)

# jdiag2 = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
# jdiag2.fit(raw, pca='other', picks=picks, verbose=True, tol=1e-9, max_iter=1000)


smica = ICA(n_components=n_components, freqs=freqs, rng=0)
smica.fit(raw, picks=picks, verbose=100, tol=1e-10, em_it=100000, corr=True)

# Plot powers

# noise_sources = [6, 8, 9]
# muscle_source = [7]
# f, ax = plt.subplots(figsize=(4, 2))
# plot_powers(smica.powers, noise_sources, muscle_source, ax, 'smica')
# plt.show()
# #
# #
raw.filter(2, 70)
ica = ICA_mne(n_components=n_components, method='fastica', random_state=0)
ica.fit(raw, picks=picks)

ica_mne = transfer_to_ica(raw, picks, freqs,
                          ica.get_sources(raw).get_data(),
                          ica.get_components())

smica.plot_clusters(16)

source_clusters = [0, 2]
idx = np.where(np.logical_or(smica.labels == 0, smica.labels == 2))[0]
brain_sources = smica.compute_sources()[idx]
K, W, _ = picard(brain_sources)
picard_mix = np.linalg.pinv(W @ K)
brain_A = smica.A[:, idx]
fitted_A = brain_A.dot(picard_mix)
Asi = smica.A.copy()
Asi[:, idx] = fitted_A


brain_sources = smica.compute_sources()
K, W, _ = picard(brain_sources)
picard_mix = np.linalg.pinv(W @ K)
fitted_A = smica.A.dot(picard_mix)
brain_sources = smica.compute_sources(raw.get_data(picks=picks), method='pinv')
K, W, _ = picard(brain_sources)
picard_mix = np.linalg.pinv(W @ K)
fitted_A__ = smica.A.dot(picard_mix)


brain_sources = ica_mne.compute_sources(raw.get_data(picks=picks),
                                        method='pinv')
K, W, _ = picard(brain_sources)
picard_mix = np.linalg.pinv(W @ K)
fitted_A_ = smica.A.dot(picard_mix)


brain_sources = ica_mne.compute_sources(raw.get_data(picks=picks),
                                        method='pinv')
K, W, _ = fastica(brain_sources.T)
picard_mix = np.linalg.pinv(W @ K)
fastica_ = smica.A.dot(picard_mix)


gofs = dipolarity(smica.A, raw, picks)[0]
gofj = dipolarity(jdiag.A, raw, picks)[0]
gofi = dipolarity(ica_mne.A, raw, picks)[0]
gofsi = dipolarity(Asi, raw, picks)[0]
gofp = dipolarity(fitted_A_, raw, picks)[0]
gof_ss = dipolarity(fitted_A__, raw, picks)[0]
goff = dipolarity(fastica_, raw, picks)[0]
gof_subspace = dipolarity(fitted_A, raw, picks)[0]
plt.figure()
# plt.plot(np.sort(gofs), label='smica')
plt.plot(np.sort(gofj), label='jdiag')
plt.plot(np.sort(gofp), label='infomax')
# plt.plot(np.sort(gofsi), label='smica + infomax')
plt.plot(np.sort(gof_subspace), label='infomax on smica wiener')
plt.plot(np.sort(gof_ss), label='infomax on smica pinv')
# plt.plot(np.sort(goff), label='fastica')
plt.xlabel('source')
plt.ylabel('dipolarity')
plt.legend()
