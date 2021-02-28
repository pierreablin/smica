"""
XXX
============================

"""

import numpy as np
import matplotlib.pyplot as plt
from smica import ICA
import mne

from mne.datasets import sample


data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')


n_components = 59
n_bins = 40
freqs = np.linspace(1, 60, n_bins + 1)
loss_list = []

for n_component in np.arange(1, n_components):
    print(n_component)
    smica = ICA(n_components=n_component, freqs=freqs, rng=0)
    smica.fit(raw, picks=picks, verbose=2000, tol=1e-7, em_it=10000)
    loss_list.append(smica.compute_loss())


plt.plot(np.arange(1, n_components), loss_list - np.min(loss_list))
plt.yscale('log')
plt.xlabel('Number of sources')
plt.ylabel('Negative log likelihood')
plt.show()
