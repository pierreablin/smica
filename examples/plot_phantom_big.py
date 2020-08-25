import os.path as op
import numpy as np
import socket

import mne
from mne import find_events, fit_dipole
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif
from smica import ICA, transfer_to_ica, JDIAG_mne
from mne.preprocessing import maxwell_filter
from mne.preprocessing import ICA as ICA_mne
from joblib import Parallel, delayed, Memory


location = './cachedir'
memory = Memory(location, verbose=0)


def get_root_dir():
    hostname = socket.gethostname()
    if 'drago' in hostname:
        return '/storage/store/data/phantom_aston/'
    else:
        return '/home/pierre/work/smica/dataset/eric/'


def get_raw_filtered_smica(bad_sources, smica, raw, picks, method='wiener'):
    X_filtered = smica.filter(raw._data[picks], bad_sources=bad_sources,
                              method=method)
    raw_filtered = raw.copy()
    raw_filtered._data[picks] = X_filtered
    return raw_filtered


def get_error(epochs, cov):
    t_peak = 60e-3  # ~60 MS at largest peak
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)
    data = []
    for ii in [dipole_no]:
        evoked = epochs[str(ii)].average().crop(t_peak, t_peak)
        data.append(evoked.data[:, 0])
    evoked = mne.EvokedArray(np.array(data).T, evoked.info, tmin=0.)
    dip, res = fit_dipole(evoked, cov, sphere, n_jobs=1)
    actual_pos = mne.dipole.get_phantom_dipoles()[0][dipole_no - 1]
    diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
    return diffs[0], np.sum(res.data ** 2) / np.sum(evoked.data ** 2)


def get_error_A(a, cov, info):
    t_peak = 60e-3  # ~60 MS at largest peak
    sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)
    if a.ndim == 1:
        data = [a, ]
    else:
        data = a.T
    evoked = mne.EvokedArray(np.array(data).T, info, tmin=0.)
    dip, res = fit_dipole(evoked, cov, sphere, n_jobs=1)
    actual_pos = mne.dipole.get_phantom_dipoles()[0][dipole_no - 1]
    diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
    print(res.data.shape, evoked.data.shape)
    if a.ndim == 1:
        return diffs, np.sum(res.data ** 2) / np.sum(evoked.data ** 2)
    return diffs, np.sum(res.data ** 2, axis=0) / np.sum(evoked.data ** 2, axis=0)


@memory.cache()
def process_one(dipole_no, amp, n_components, crop_idx, crop_len=None):
    raw_fname = '%sAmp%d_Dip%d_IASoff.fif' % (get_root_dir(), amp, dipole_no)
    raw = read_raw_fif(raw_fname)
    raw.load_data()
    raw.crop(5, None)
    if crop_len is not None:
        raw.crop(crop_idx * crop_len, (crop_idx + 1) * crop_len)
    raw.fix_mag_coil_types()
    events = find_events(raw, 'SYS201')
    picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                           stim=False, exclude='bads')
    n_bins = 40

    freqs = np.linspace(1, 70, n_bins + 1)
    smica = ICA(n_components, freqs, rng=0).fit(raw, picks=picks,
                                                em_it=50000, n_it_lbfgs=200,
                                                tol=1e-8,
                                                verbose=100, n_it_min=10000)

    ica = ICA_mne(n_components, method='picard').fit(raw.copy().filter(1, 70),
                                                     picks=picks)
    jdiag = JDIAG_mne(n_components, freqs, rng=0).fit(raw, picks=picks)

    tmin, tmax = -0.1, 0.1
    event_id = [dipole_no]
    # pows = np.linalg.norm(smica.A, axis=0) * np.linalg.norm(smica.powers, axis=0) ** .5
    #
    raw_max = raw.copy().filter(1, 40)
    raw_max = maxwell_filter(raw_max, origin=(0., 0., 0.))
    epochs = mne.Epochs(raw_max, events, event_id, tmin, tmax, baseline=(None, -0.01),
                        preload=True, picks=picks)
    cov = mne.compute_covariance(epochs, tmax=0)
    e1 = get_error(epochs, cov)
    e_2 = get_error_A(smica.A, cov, epochs.info)
    good = np.argmin(e_2[0])
    e2 = [e_2[0][good], e_2[1][good]]

    e_3 = get_error_A(jdiag.A, cov, epochs.info)
    good = np.argmin(e_3[0])
    e3 = [e_3[0][good], e_3[1][good]]

    e_4 = get_error_A(ica.get_components(), cov, epochs.info)
    good = np.argmin(e_4[0])
    e4 = [e_4[0][good], e_4[1][good]]
    crop_str = crop_len if crop_len is not None else 0
    np.save('results/phantom_%d_%d_%d_%d_%d.npy' %
            (dipole_no, amp, n_components, crop_idx, crop_str),
            np.array([e1, e2, e3, e4]))
    print(e1, e2, e3, e4)
    return e1, e2, e3, e4


n_components = 30
dipoles = [5, 8]
amp = 200
dipole_no = 8
crop_idx = 0
e1, e2, e3, e4 = process_one(dipole_no, amp, n_components, crop_idx)
