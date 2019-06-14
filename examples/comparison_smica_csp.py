# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectKBest

import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from smica import SMICATransformer, SMICAPredictor

mne.set_log_level('CRITICAL')
# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.

acc_smica = []
acc_csp = []
n_comp = 1
for subject in [2, 7, 8, 10, 13, 15, 16, 29]:
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames],
                           verbose='CRITICAL')

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    # raw.filter(2., 30., fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy()
    labels = epochs.events[:, -1] - 2

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(9, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    n_bins = 12
    freqs = np.linspace(2, 30, n_bins+1)
    sfreq = raw.info['sfreq']
    st = SMICATransformer(n_components=20, freqs=freqs, sfreq=sfreq,
                          transformer='source_powers')
    kbest = SelectKBest(k=n_comp)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([('SMICA', st), ('kbest', kbest), ('lda', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=3)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("SMI Classification accuracy: %f / Chance level: %f" %
          (np.mean(scores), class_balance))
    acc_smica.append(scores)

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    raw.filter(7, 30)
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy()
    labels = epochs.events[:, -1] - 2

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(9, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=1, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=3)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("CSP Classification accuracy: %f / Chance level: %f \n" %
          (np.mean(scores), class_balance))
    acc_csp.append(scores)

# print([np.mean(score) for score in acc_csp])
# print([np.mean(score) for score in acc_smica])
# print([np.std(score) for score in acc_csp])
# print([np.std(score) for score in acc_smica])
f, ax = plt.subplots(figsize=(4, 3))

bp = ax.boxplot(acc_csp, 0, 's', 0, positions=np.arange(1, 9) - .3)
bp2 = ax.boxplot(acc_smica, 0, 's', 0, positions=np.arange(1, 9) + .3)
ax.set_xlabel('Accuracy')
ax.set_ylabel('Subject')

ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set_ylim([0, 9])
plt.legend()
plt.yticks(np.arange(1, 9))
plt.show()
