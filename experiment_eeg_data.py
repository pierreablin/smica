import numpy as np

import mne
from itertools import product
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from joblib import Parallel, delayed, Memory

from picard import picard
from smica import dipolarity, JDIAG_mne, SOBI_mne, ICA

location = './cachedir'
memory = Memory(location, verbose=0)
SUBJECTS = ['ap82', 'cj82', 'cz84', 'ds76', 'ds80', 'gm84', 'gv84']


def standard_mixing(epochs, picks, algorithm, algo_args):
    algorithm.fit(epochs, picks, **algo_args)
    return algorithm.A


@memory.cache()
def compute_decomposition(algo_name, subject, n_components, decomp_num,
                          n_decomp):
    runs = [6, 10, 14]  # motor imagery: hands vs feet
    filename = '/home/pierre/work/smica/dataset/%s.set' % SUBJECTS[subject]
    epochs = mne.io.read_epochs_eeglab(filename, verbose='CRITICAL')
    epochs.set_channel_types({'LEYE': 'eog', 'REYE': 'eog'})
    epochs.filter(2, 60)
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                           stim=False, exclude='bads')
    n_bins = 40
    freqs = np.linspace(2, 60, n_bins+1)
    if algo_name == 'smica':
        algorithm = ICA(n_components=n_components, freqs=freqs, rng=0)
        algo_args = dict(em_it=100000, tol=1e-8, n_it_min=100000)
        mixing = standard_mixing(epochs, picks, algorithm, algo_args)
    if algo_name == 'jdiag':
        algorithm = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
        algo_args = dict(max_iter=1000, tol=1e-10)
        mixing = standard_mixing(epochs, picks, algorithm, algo_args)
    if algo_name == 'sobi':
        algorithm = SOBI_mne(1000, n_components=n_components, freqs=freqs,
                             rng=0)
        algo_args = dict()
        mixing = standard_mixing(epochs, picks, algorithm, algo_args)
    if algo_name == 'infomax':
        jdiag = JDIAG_mne(n_components=n_components, freqs=freqs, rng=0)
        jdiag.fit(epochs, picks, max_iter=10)
        sources = jdiag.compute_sources(method='pinv')
        K, W, _ = picard(sources, max_iter=1000, ortho=False)
        picard_mix = np.linalg.pinv(np.dot(W, K))
        mixing = np.dot(jdiag.A, picard_mix)
    if algo_name in ['pinv_infomax', 'wiener_infomax']:
        smica = ICA(n_components=n_components, freqs=freqs, rng=0)
        algo_args = dict(em_it=100000, tol=1e-8, n_it_min=100000)
        smica.fit(epochs, picks, **algo_args)
        method = {'pinv_infomax': 'pinv',
                  'wiener_infomax': 'wiener'}[algo_name]
        sources = smica.compute_sources(method=method)
        K, W, _ = picard(sources, max_iter=1000, ortho=False)
        picard_mix = np.linalg.pinv(np.dot(W, K))
        mixing = np.dot(smica.A, picard_mix)
    gof, _, _ = dipolarity(mixing, epochs, picks)
    print(decomp_num, n_decomp)
    return gof


if __name__ == '__main__':
    algo_names = ['smica', 'jdiag', 'sobi', 'infomax', 'pinv_infomax',
                  'wiener_infomax']
    n_components = [2]
    subjects = [1]
    iter_list = list(product(algo_names, subjects, n_components))
    k = len(iter_list)
    Parallel(n_jobs=2)(
        delayed(compute_decomposition)(algo_name, subject, n_components, i, k)
        for i, (algo_name, subject, n_components) in enumerate(iter_list)
    )
    # [compute_decomposition(algo_name, subject, n_component, i, k)
    #  for i, (algo_name, subject, n_component) in enumerate(iter_list)]
