"""
XXX
============================

"""
import numpy as np

import matplotlib.pyplot as plt

from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from smica import ICA

tmin, tmax = -1., 4.

subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))

# Apply band-pass filter
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events, _ = events_from_annotations(raw)

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')


event_id = dict(hands=2, feet=3)
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
labels = epochs.events[:, -1] - 2
epochs_data = epochs.get_data()

l_list = []
for event_id in [dict(hands=2), dict(feet=3)]:
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    n_components = 20
    freqs = np.linspace(7, 30, 31)
    ica = ICA(n_components=n_components, freqs=freqs, rng=0)
    ica.fit(epochs, max_iter=5000, verbose=100)
    l_list.append([ica.compute_loss(x)
                   for x in epochs_data])
plt.figure()
likelihoods = np.array(l_list)
for i, label in enumerate(['hand', 'feet']):
    plt.scatter(*(likelihoods[j, labels == i] for j in [0, 1]), label=label)

plt.xlabel('negative loglik of model hand')
plt.ylabel('negative loglik of model feet')
l_min = np.min(likelihoods)
l_max = np.max(likelihoods)
t = np.linspace(l_min, l_max)
plt.title(subject)
plt.plot(t, t, color='k', label='y=x')
plt.legend()
plt.show()
