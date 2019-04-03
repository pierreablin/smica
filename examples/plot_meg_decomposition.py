import numpy as np
import matplotlib.pyplot as plt
from smica import ICA
import mne
from mne.datasets import sample

# fetch data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

# Compute ICA on raw: chose the frequency decomposition. Here, uniform between
# 2 - 35 Hz.
n_bins = 30
freqs = np.linspace(2, 35, n_bins + 1)
smica = ICA(n_components=20, freqs=freqs, rng=0)
smica.fit(raw, picks=picks)

# Plot the powers

smica.plot_powers()
plt.show()


# Cluster the spectra

labels = smica.plot_clusters(8)
plt.show()

# Inspect a cluster

idx = np.where(labels == 1)[0]
smica.plot_components(picks=idx)

smica.plot_properties(picks=idx)
