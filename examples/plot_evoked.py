# We compare smica with other denoising techniques to obtain clain erps
import os
import numpy as np
import matplotlib.pyplot as plt
from smica import ICA
import mne
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')
raw.set_eeg_reference('average', projection=True)  # set EEG average reference
events = mne.find_events(raw)
tmin, tmax = -0.2, 0.5
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4}


# Fit smica

smica = ICA(n_components=30,
            freqs=np.linspace(1, 40, 35), rng=0).fit(raw, picks)


# Plot the clusters

labels = smica.plot_clusters(10)
plt.show()

# We identify that clusters 6, 7, 8, 9 correspond to noise
bad_clusters = [6, ]
bad_sources = np.concatenate([np.where(labels == cluster)[0]
                              for cluster in bad_clusters])

X_filtered = smica.filter(bad_sources=bad_sources)
raw_filtered = raw.copy()
raw_filtered._data[picks] = X_filtered

# Epochs the signal and observe the average auditory response
evokeds = []
for raw_ in [raw, raw_filtered]:
    epochs = mne.Epochs(raw_, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=(None, 0.), proj=False)
    evokeds.append(epochs['Visual/Right'].average(picks=picks))

# Plot average

sfreq = smica.sfreq
n_samples = evokeds[0].data.shape[1]
t_large = np.linspace(-.2, n_samples / sfreq - .2, n_samples)


f, ax = plt.subplots(1, 2, figsize=(15, 5))
for name, evoked in zip(['raw', 'filtered'], evokeds):
    ax[0].plot(t_large, np.mean(evoked.data ** 2, axis=0))
    sl = range(160, 200)
    ax[1].plot(t_large[sl], np.mean(evoked.data ** 2, axis=0)[sl], label=name)
f.legend()
plt.show()
