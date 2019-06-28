import os.path as op

import numpy as np

import mne


data_path = mne.datasets.brainstorm.bst_resting.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'bst_resting'
trans = op.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
src = op.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')
fname_bem = op.join(subjects_dir, subject, 'bem',
                    subject + '-5120-bem-sol.fif')
raw_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_spontaneous_20111102_01_AUX.ds')
raw = mne.io.read_raw_ctf(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', eeg=False)
n_channels = raw.info['nchan']
ch_names = raw.info['ch_names']
cov = mne.Covariance(np.eye(n_channels),
                     ch_names, bads=[],
                     projs=[], nfree=1)
info = mne.pick_info(raw.info, sel=picks)
evoked = mne.EvokedArray(np.ones((info['nchan'], 1)), info, tmin=0)
mne.fit_dipole(evoked, cov, fname_bem, min_dist=5)
