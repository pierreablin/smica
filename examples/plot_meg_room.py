import numpy as np
import matplotlib.pyplot as plt
from smica import ICA
import mne
from mne.datasets import sample

# fetch data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/ernoise_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

# Compute ICA on raw: chose the frequency decomposition. Here, uniform between
# 2 - 35 Hz.
n_bins = 40
n_components = 40
freqs = np.linspace(1, 70, n_bins + 1)
smica = ICA(n_components=n_components, freqs=freqs, rng=0)
smica.fit(raw, picks=picks, verbose=100, tol=1e-10, em_it=100000)

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

smica_meg = ICA(n_components=n_components, freqs=freqs, rng=0)
smica_meg.fit(raw, picks=picks, verbose=100, tol=1e-10, em_it=100000)
# Plot the powers

# smica.plot_powers()
# plt.show()

smica.plot_noise()
smica_meg.plot_noise()
