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
T = 20
raw.crop(0, T)
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads')

# Compute ICA on raw: chose the frequency decomposition. Here, uniform between
# 2 - 35 Hz.
n_bins = 10
freqs = np.linspace(2, 35, n_bins + 1)
# n_comp = 20
# smica = ICA(n_components=n_comp, freqs=freqs, rng=0)
# smica.fit(raw, picks=picks, corr=True, verbose=100, tol=1e-7, em_it=10000)
# print(smica.compute_loss() / n_bins)
# losses = []
n_comps = list(range(1, 50)) + [60, 70, 80, 90, 100]

# n_comp = 10
# smica = ICA(n_components=n_comp, freqs=freqs, rng=0)
# smica.fit(raw, picks=picks, verbose=100, tol=1e-7, em_it=10000)
# smica2 = ICA(n_components=n_comp, freqs=freqs, rng=0)
# smica2.fit(raw, picks=picks, corr=True, verbose=1, tol=1e-7, em_it=10000)
losses = []
for n_comp in n_comps:
    # print('SMICA uncorr')
    # smica = ICA(n_components=n_comp, freqs=freqs, rng=0)
    # smica.fit(raw, picks=picks, verbose=100, tol=1e-7, em_it=10000)

    print(n_comp)

    smica2 = ICA(n_components=n_comp, freqs=freqs, rng=0)
    smica2.fit(raw, picks=picks, corr=True, verbose=100, tol=1e-7, em_it=10000)
    losses.append(smica2.compute_loss())
# baseline = smica2.smica.baseline_loss
# losses.insert(0, (baseline, baseline))
f, ax = plt.subplots()
plt.plot(n_comps, losses)
# plt.plot(n_comps, [loss[1] for loss in losses])
plt.yscale('log')
plt.show()
# C = smica.smica.C_
# Cs = smica.smica.compute_approx_covs()
# Cs2 = smica2.smica.compute_approx_covs()
# smica_loss = smica.compute_loss(by_bin=True)
# smica2_loss = smica2.compute_loss(by_bin=True)
# f, ax = plt.subplots(3, n_bins, figsize=(30, 7))
# for i in range(n_bins):
#     for j, c in enumerate([C[i], Cs2[i], Cs[i]]):
#         axe = ax[j, i]
#         axe.matshow(c)
#         axe.set_xticks([])
#         axe.set_yticks([])
#         if j == 0:
#             if i == 0:
#                 x_ = axe.set_ylabel('Covs')
#             axe.set_title('%d - %d Hz' % (freqs[i], freqs[i+1]))
#         if j == 1:
#             if i == 0:
#                 axe.set_ylabel('Subspace')
#             axe.set_title('loss=%.1f' % smica2_loss[i])
#         if j == 2:
#             if i == 0:
#                 axe.set_ylabel('SMICA')
#         print(n_comp)        axe.set_title('loss=%.1f' % smica_loss[i])
# t_ = f.suptitle('%d components' % n_comp)
# # f.tight_layout()
# plt.savefig('covs_%d.png' % n_comp, bbox_extra_artists=[t_, x_],
#             bbox_inches='tight')
# plt.show()

#
# f, ax = plt.subplots(2, n_bins, figsize=(30, 7))
# for i in range(n_bins):
#     for j, c in enumerate([Cs2[i], Cs[i]]):
#         axe = ax[j, i]
#         axe.matshow(C[i].dot(np.linalg.pinv(c)))
#         axe.set_xticks([])
#         axe.set_yticks([])
#         if j == 0:
#             if i == 0:
#                 x_ = axe.set_ylabel('Subspace')
#             axe.set_title('KL=%.1f' % smica2_loss[i])
#         if j == 1:
#             if i == 0:
#                 axe.set_ylabel('SMICA')
#             axe.set_title('KL=%.1f' % smica_loss[i])
# t_ = f.suptitle('%d components' % n_comp)
# # f.tight_layout()
# plt.savefig('covs_invs_%d.png' % n_comp, bbox_extra_artists=[t_, x_],
#             bbox_inches='tight')
# plt.show()


def extract(C, i, j):
    op = [[C[:, i, i], C[:, i, j]], [C[:, j, i], C[:, j, j]]]
    op = np.array(op).T
    return op


def compute_dist_sensors(smica):
    C = smica.smica.C_
    C_s = smica.smica.compute_approx_covs()
    _, p, _ = C.shape
    D = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            d = np.sum([KL(A, B)
                        for A, B in zip(extract(C, i, j), extract(C_s, i, j))])
            D[i, j] = d
            D[j, i] = d
    return D


def KL(A, B):
    p, _ = A.shape
    D = A.dot(np.linalg.pinv(B))
    return np.trace(D) - np.linalg.slogdet(D)[1] - p
# # Plot the powers
#
# smica.plot_powers()
# plt.show()
#
#
# # Cluster the spectra
#
# labels = smica.plot_clusters(8)
# plt.show()
#
# # Inspect a cluster
#
# idx = np.where(labels == 1)[0]
# smica.plot_components(picks=idx)
#
# smica.plot_properties(picks=idx)
