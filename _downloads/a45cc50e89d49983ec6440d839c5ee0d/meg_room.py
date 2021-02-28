"""
XXX
============================

"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from smica import ICA, JDIAG_mne, SOBI_mne

# fetch data
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/ernoise_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

# Compute ICA on raw: chose the frequency decomposition. Here, uniform between
# 2 - 35 Hz.
n_bins = 40
n_components = 10
freqs = np.linspace(1, 70, n_bins + 1)
smica = ICA(n_components=n_components, freqs=freqs, rng=0)
smica.fit(raw, picks=picks, verbose=100, tol=1e-15, em_it=100000)

jdiag = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
jdiag.fit(raw, picks=picks, verbose=True, tol=1e-7, max_iter=10000)
raw.filter(1, 70)
sobi = SOBI_mne(p=2000, n_components=n_components, freqs=freqs, rng=0)
sobi.fit(raw, picks=picks, verbose=True, tol=1e-7, max_iter=2000)
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

smica_meg = ICA(n_components=n_components, freqs=freqs, rng=0)
smica_meg.fit(raw, picks=picks, verbose=100, tol=1e-18, em_it=100000)
jdiag_meg = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
jdiag_meg.fit(raw, picks=picks, verbose=True, tol=1e-7, max_iter=10000)
raw.filter(1, 70)
sobi_meg = SOBI_mne(p=2000, n_components=n_components, freqs=freqs, rng=0)
sobi_meg.fit(raw, picks=picks, verbose=True, tol=1e-7, max_iter=2000)

A1 = smica.A
A2 = smica_meg.A
plt.matshow(np.abs(np.dot(np.linalg.pinv(A2), A1)))
A1 = jdiag.A
A2 = jdiag_meg.A
plt.matshow(np.abs(np.dot(np.linalg.pinv(A2), A1)))
A1 = sobi.A
A2 = sobi_meg.A
plt.matshow(np.abs(np.dot(np.linalg.pinv(A2), A1)))
plt.show()
