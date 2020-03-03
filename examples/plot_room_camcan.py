# We compare smica with other denoising techniques to obtain clain erps
import os
import socket
import numpy as np
import matplotlib.pyplot as plt
from smica import ICA, transfer_to_ica
from sklearn.decomposition import PCA
import mne
from mne.preprocessing import ICA as ICA_mne
from mne.filter import filter_data
from mne.preprocessing import maxwell_filter
from joblib import Memory

import pandas as pd
import seaborn as sns


def get_camcan_subject_ids(root_dir, start=None, end=None):
    """
    Args:
        root_dir (str):

    Keyword Args:
        start (None or str):
        end (None or str):

    Returns:
        (list):
    """
    subj_ids = [os.path.basename(f.path) for f in os.scandir(root_dir)
                if f.is_dir()]
    all_subj_ids = [subj for subj in subj_ids if subj.startswith('CC')]

    if start is not None:
        all_subj_ids = [i for i in all_subj_ids if int(i[2:]) >= int(start[2:])]
    if end is not None:
        all_subj_ids = [i for i in all_subj_ids if int(i[2:]) <= int(end[2:])]

    return all_subj_ids


def get_camcam_recording_fname(root_dir, subject=None, record_type='passive'):
    """
    Args:
        root_dir:

    Keyword Args:
        subject (str, list or None):
        record_type (str):

    Returns:
        (list)
    """
    if subject is None:  # Find all existing subject names in `root_dir`
        subject = get_camcan_subject_ids(root_dir)

    subject = [subject] if not isinstance(subject, list) else subject
    if "drago" in hostname:
        fnames = [os.path.join(root_dir, subj, record_type, record_type + '_raw.fif')
                  for subj in subject]
    else:
        fnames = [os.path.join(root_dir, subj, record_type + '_raw.fif')
                  for subj in subject]
    if len(fnames) != len(subject):
        raise ValueError('Found {} files for {} subjects.'.format(
            len(fnames), len(subject)))

    return fnames


hostname = socket.gethostname()


def get_data_paths():
    """Get data paths so the analysis works on different computers.
    """
    if 'drago' in hostname:
        root_dir = '/storage/store/data/camcan/camcan47/cc700/meg/pipeline/release004/data/aamod_meg_get_fif_00001'
        sss_dir = '/storage/store/work/hjacobba/data/CAMCAN/sss_params'
        # save_dir = '/storage/store/work/hjacobba/data/CAMCAN/results'
        # save_dir = '/storage/inria/hjacobba/mne_data/camcan/results'
        # coreg_files = '/storage/store/data/camcan-mne/freesurfer/CC110033'
    else:
        root_dir = './'
        sss_dir = './'

    cal_fname = os.path.join(sss_dir, 'sss_cal.dat')
    ctc_fname = os.path.join(sss_dir, 'ct_sparse.fif')

    return root_dir, cal_fname, ctc_fname


root_dir, _, _ = get_data_paths()

subjects = get_camcan_subject_ids(root_dir, start=None, end=None)
memory = Memory(location='.', verbose=0)


event_id = {'Auditory 300Hz': 6,  # See trigger_codes.txt
            'Auditory 600Hz': 7,
            'Auditory 1200Hz': 8,
            # 'Visual Checkerboard': 9
            }


@memory.cache()
def get_explained(subject_no, n_comp, n_bads):
    subject = subjects[subject_no]
    raw_fname = root_dir + '%s/passive_raw.fif' % subject
    room_fname = root_dir + '%s/emptyroom_%s.fif' % (subject, subject)
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw_room = mne.io.read_raw_fif(room_fname, preload=True)
    raw.filter(0.5, 70)
    raw_room.filter(1, 70)
    mne.channels.fix_mag_coil_types(raw.info)
    mne.channels.fix_mag_coil_types(raw_room.info)
    picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                           stim=False, exclude='bads')
    events = mne.find_events(raw)
    tmin, tmax = -0.1, 0.3

    expes = list(event_id.keys())
    expe = expes[2]
    # reject = dict(grad=4000e-13, mag=1e-12, eog=150e-6)
    reject = None

    # Fit smica
    freqs = np.linspace(1, 70, 41)
    smica = ICA(n_components=n_comp,
                freqs=freqs, rng=0).fit(raw, picks, tol=1e-9,
                                        verbose=100,
                                        n_it_min=3000, em_it=3000,
                                        n_it_lbfgs=3000)
    ica = ICA_mne(n_comp).fit(raw, picks)

    X_room = raw_room.get_data(picks=picks)
    X_room /= np.linalg.norm(X_room)
    Ss = smica.compute_sources(X_room)
    As = smica.A
    del smica
    Ai = ica.get_components()
    Si = np.linalg.pinv(Ai).dot(X_room)

    def greedy_comp(X_room, A, S, n_comp=5):
        bads = []
        vals = []
        Xc = X_room.copy()
        idxs = np.arange(S.shape[0])
        for i in range(n_comp):
            print(i)
            diffs = [np.linalg.norm(Xc - a[:, None] * s)
                     for a, s in zip(A.T[idxs], S[idxs])]
            j = idxs[np.argmin(diffs)]
            bads.append(j)
            Xc -= A[:, j][:, None] * S[j]
            vals.append(np.linalg.norm(Xc))
            idxs = np.delete(idxs, np.where(idxs == j))
        return bads, vals

    bads, vals = greedy_comp(X_room, As, Ss, n_bads)
    badi, vali = greedy_comp(X_room, Ai, Si, n_bads)
    np.save('results/expl_%d_%d_%d.npy' % (subject_no, n_comp, n_bads),
            np.array((vals, vali)))
    return vals, vali


n_subjs = np.arange(2)
n_bads = 5
E = np.array([get_explained(subject, 30, 5) for subject in range(2)])

dfs = pd.DataFrame(E[:, 0, :], columns=np.arange(1, n_bads + 1))
dfs = dfs.stack().reset_index()
dfs['method'] = 'smica'
dfi = pd.DataFrame(E[:, 1, :], columns=np.arange(1, n_bads + 1))
dfi = dfi.stack().reset_index()
dfi['method'] = 'infomax'
df = pd.concat([dfs, dfi], axis=0)
df.columns = ['Subject', '# removed components', 'Residual variance', 'method']
sns.swarmplot(x='# removed components', y='Residual variance',
              hue='method', dodge=True, data=df)
# df = pd.concat((dfs, dfi), keys=['smica', 'infomax'])
# n_s, _, n_b = E.shape
# f, ax = plt.subplots(2, n_s // 2)
# for j, (axe, e) in enumerate(zip(ax.ravel(), E)):
#     axe.plot(e[0], label='smica')
#     # axe.plot(E[1], label='pca')
#     axe.plot(e[1], label='infomax')
#     axe.legend()
#     axe.set_xlabel('components')
#     axe.set_ylim([0, 1])
#     axe.grid(True)
#     if j % 5 == 0:
#         axe.set_ylabel('residual variance')
#
# plt.show()
# plt.figure()
# plt.plot(vals, label='smica')
# plt.plot(vali, label='infomax')
# plt.plot(valp, label='pca')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(np.cumsum(1 - diffs[0]), label='smica')
# plt.plot(np.cumsum(1 - diffs[2]), label='pca')
# plt.plot(np.cumsum(1 - diffs[1]), label='infomax')
# plt.legend()
# plt.show()
