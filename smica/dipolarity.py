from __future__ import division
import os
import mne
from mne.transforms import (read_trans, apply_trans, _get_trans)
from mne.utils import get_subjects_dir
from mne.surface import read_surface
import numpy as np


def dip_depth_sphere(dip, sphere):

    inner_sphere = sphere['layers'][0]['rad']
    depth = inner_sphere - np.sqrt(np.sum(dip.pos ** 2, axis=1))

    direction = dip.pos.copy()
    direction /= np.sqrt(np.sum(dip.pos ** 2, axis=1))[:, None]

    ori = dip.ori.copy()
    ori /= np.sqrt(np.sum(ori ** 2, axis=1))[:, None]

    radiality = np.abs(np.sum(ori * direction, axis=1))

    return depth, radiality


# def dipolarity_using_sphere_model(A, info, head_radius=0.085):
def dipolarity_using_sphere_model(A, inst, picks, head_radius=0.085, n_jobs=1):

    # head_radius must be in meters !

    n_channels = inst.info['nchan']
    ch_names = inst.info['ch_names']
    cov = mne.Covariance(np.eye(n_channels),
                         ch_names, bads=[],
                         projs=[], nfree=1)

    # project the electrodes to the outer sphere
    pos = np.array([c['loc'][:3] for c in inst.info['chs']])
    pos /= np.sqrt(np.sum(pos ** 2, axis=1))[:, None]
    pos *= head_radius
    kind = "eeglab"
    selection = np.arange(n_channels)
    montage = mne.channels.Montage(pos, ch_names, kind, selection)
    inst.set_montage(montage)
    #
    # # Specify the eog channels
    # inst.set_channel_types({'LEYE': 'eog', 'REYE': 'eog'})

    # eeglab:
    # EEG.dipfit.vol
    #   r: [71 72 79 85]
    #   c: [0.3300 1 0.0042 0.3300]
    #   o: [0 0 0]

    # # XXX : try auto mode
    # sphere = mne.make_sphere_model(r0='auto',
    #                                head_radius='auto',
    #                                relative_radii=[0.8353, 0.847, 0.93, 1],
    #                                sigmas=(0.33, 1.0, 0.0042, 0.33),
    #                                info=info)

    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0),
                                   head_radius=head_radius,
                                   relative_radii=[0.8353, 0.847, 0.93, 1],
                                   sigmas=(0.33, 1.0, 0.0042, 0.33))
    new_info = mne.pick_info(inst.info, sel=picks)
    gof, dip, evoked = dipolarity(A, new_info, cov,
                                  sphere, fname_trans=None,
                                  min_dist=1, verbose=False,
                                  n_jobs=n_jobs)

    return gof, dip, evoked, sphere


def dipolarity(mixing, inst, picks, fname_bem=None, fname_trans=None,
               min_dist=5, n_jobs=1, verbose=None):
    n_channels = inst.info['nchan']
    ch_names = inst.info['ch_names']
    cov = mne.Covariance(np.eye(n_channels),
                         ch_names, bads=[],
                         projs=[], nfree=1)
    if fname_bem is None:
        head_radius = 0.085
        pos = np.array([c['loc'][:3] for c in inst.info['chs']])
        pos /= np.sqrt(np.sum(pos ** 2, axis=1))[:, None]
        pos *= head_radius
        kind = "eeglab"
        selection = np.arange(n_channels)
        montage = mne.channels.Montage(pos, ch_names, kind, selection)
        inst.set_montage(montage)
        inst.set_channel_types({'LEYE': 'eog', 'REYE': 'eog'})
        fname_bem =\
            mne.make_sphere_model(r0=(0.0, 0.0, 0.0),
                                  head_radius=head_radius,
                                  relative_radii=[0.8353, 0.847, 0.93, 1],
                                  sigmas=(0.33, 1.0, 0.0042, 0.33))
        picks = mne.pick_types(inst.info, eeg=True, eog=False)
    info = mne.pick_info(inst.info, sel=picks)
    evoked = mne.EvokedArray(mixing, info, tmin=0,
                             comment='ICA components')

    info['projs'] = []  # get rid of SSP projections
    if 'eeg' in evoked:
        evoked = evoked.copy()
        avg_proj = mne.io.make_eeg_average_ref_proj(evoked.info)
        evoked.add_proj([avg_proj])
        evoked.apply_proj()

    dip, residual = mne.fit_dipole(evoked, cov, fname_bem, fname_trans,
                                   min_dist=min_dist, verbose=verbose,
                                   n_jobs=n_jobs)

    gof = (1. - np.sum(residual.data ** 2, axis=0) /
           np.sum(evoked.data ** 2, axis=0))

    gof = 100. * gof  # Scale the gof between 0 and 100

    # if some gof is NaN, raise an Error
    if np.sum(np.isnan(gof)) > 0:
        raise ValueError('ERROR!, some gof are NaN. This usually happens '
                         'because some dipoles position resulting to be too '
                         'close to the inner sphere. To solve this issue, '
                         'consider using  a different value for the min_dist '
                         'parameter.')

    # Do not sort the columns of A, neither the gof, this way is easier to
    # mantain the one-to-one correspondence between the columns of A and the
    # the measures associated to them (gof, depth, radiality, etc)

    # idx_sort_desc = np.argsort(gof)[::-1]
    # evoked.data = evoked.data[:, idx_sort_desc]
    # gof = gof[idx_sort_desc]

    return gof, dip, evoked


def dip_depth(dip, fname_trans, subject, subjects_dir):
    trans = read_trans(fname_trans)
    trans = _get_trans(trans)[0]
    subjects_dir = get_subjects_dir(subjects_dir=subjects_dir)
    fname = os.path.join(subjects_dir, subject, 'bem',
                         'inner_skull.surf')
    points, faces = read_surface(fname)
    points = apply_trans(trans['trans'], points * 1e-3)

    pos = dip.pos
    ori = dip.ori

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors()
    nn.fit(points)
    depth, idx = nn.kneighbors(pos, 1, return_distance=True)
    idx = np.ravel(idx)

    direction = pos - points[idx]
    direction /= np.sqrt(np.sum(direction ** 2, axis=1))[:, None]
    ori /= np.sqrt(np.sum(ori ** 2, axis=1))[:, None]

    radiality = np.abs(np.sum(ori * direction, axis=1))
    return np.ravel(depth), radiality
