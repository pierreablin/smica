import numpy as np
import matplotlib.pyplot as plt
from smica import ICA, transfer_to_ica, SOBI_mne, JDIAG_mne
import mne

from mne.preprocessing import ICA as ICA_mne

from mne.datasets import sample

rc = {"pdf.fonttype": 42, 'text.usetex': True, 'font.size': 14,
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
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

# Compute ICA on raw: chose the frequency decomposition. Here, uniform between
# 2 - 35 Hz.
n_bins = 40
n_components = 10
freqs = np.linspace(1, 60, n_bins + 1)

jdiag = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
jdiag.fit(raw, picks=picks, verbose=True, tol=1e-7, max_iter=10000)

# Plot the powers

noise_sources = [0, 1, 3]
muscle_source = [2]
f, ax = plt.subplots(figsize=(4, 2))
plot_powers(jdiag.powers, noise_sources, muscle_source, ax, 'jdiag')
plt.show()


smica = ICA(n_components=n_components, freqs=freqs, rng=0)
smica.fit(raw, picks=picks, verbose=100, tol=1e-7)

# Plot powers

noise_sources = [6, 8, 9]
muscle_source = [7]
f, ax = plt.subplots(figsize=(4, 2))
plot_powers(smica.powers, noise_sources, muscle_source, ax, 'smica')
plt.show()


ica = ICA_mne(n_components=n_components, method='picard', random_state=0)
ica.fit(raw, picks=picks)

ica_mne = transfer_to_ica(raw, picks, freqs,
                          ica.get_sources(raw).get_data(),
                          ica.mixing_matrix_)

noise_sources = [1, 2]
muscle_source = [4]
f, ax = plt.subplots(figsize=(4, 2))
plot_powers(ica_mne.powers, noise_sources, muscle_source, ax, 'infomax')
plt.show()


sobi = SOBI_mne(p=2000, n_components=n_components, freqs=freqs, rng=0)
sobi.fit(raw, picks=picks, verbose=True, tol=1e-7, max_iter=10000)

# Plot the powers

noise_sources = [1, 8]
muscle_source = [3]
f, ax = plt.subplots(figsize=(4, 2))
plot_powers(sobi.powers, noise_sources, muscle_source, ax, 'sobi')
plt.show()
