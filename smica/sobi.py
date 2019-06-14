import numpy as np

from sklearn.utils import check_random_state
from mne.io import BaseRaw
from joblib import Memory

from .core_smica import SMICA
from .mne import ICA, transfer_to_mne
from .utils import fourier_sampling

location = './cachedir'
memory = Memory(location, verbose=0)


def _transform_set(M, D):
    '''Moves the matrices D using the matrix M

    Parameters
    ----------
    M : array-like, shape (N, N)
        Movement

    D : array-like, shape (K, N, N)
        Array of current estimated matrices. K is the number of matrices

    Returns
    -------
    op : array-like, shape (K, N, N)
        Array of the moved matrices. op[i] = M.D[i].M.T
    '''
    K, _, _ = D.shape
    N, _ = M.shape
    op = np.zeros((K, N, N))
    for i, d in enumerate(D):
        op[i] = M.dot(d.dot(M.T))
    return op


def _move(epsilon, D):
    '''Moves the matrices D by a perturbation epsilon

    Parameters
    ----------
    epsilon : array-like, shape (N, N)
        Perturbation

    D : array-like, shape (K, N, N)
        Array of current estimated matrices. K is the number of matrices

    Returns
    -------
    M : array-like, shape (N, N)
        Displacement matrix

    op : array-like, shape (K, N, N)
        Array of the moved matrices. op[i] = M.D[i].M.T
    '''
    _, N, _ = D.shape
    M = np.eye(N) + epsilon
    return M, _transform_set(M, D)


def _joint_diag(C, max_iter, tol=1e-7, theta=0.5, verbose=False):
    if verbose:
        print(C.shape)
    K, N, _ = C.shape
    D = C.copy()
    W = np.eye(N)
    for n in range(max_iter):
        # Isolate the diagonals
        diagonals = np.diagonal(D, axis1=1, axis2=2)
        # Compute the z_ij
        z = np.dot(diagonals.T, diagonals)
        # Compute the y_ij
        y = np.sum(D * diagonals[:, None, :], axis=0)
        # Compute the new W
        z_diag = np.diagonal(z)
        det = (z_diag[:, None] * z_diag[None, :] - z ** 2) + np.eye(N)

        eps = (z * y.T - z_diag[:, None] * y) / det
        # np.fill_diagonal(W, 0.)
        # Stopping criterion
        norm = np.sqrt(np.mean(eps ** 2))
        if verbose:
            print(n, norm)
        if norm < tol:
            break
        # Scale
        if norm > theta:
            eps *= theta / norm
        # Move
        M, D = _move(eps, D)
        W = M @ W
    return W


@memory.cache(ignore=['verbose'])
def sobi(X, p, n_components=None, lag0=True,
         tol=1e-7, max_iter=1000, verbose=False):
    """
    Use sobi for source estimation
    X :  data matrix
    p :  number of time lags to use
    """
    n_sensors, n_samples = X.shape
    if n_components is None:
        n_components = n_sensors
    u, d, _ = np.linalg.svd(X, full_matrices=False)
    del _
    whitener = (u / d).T[:n_components]
    del u, d
    Y = whitener.dot(X)
    C = np.zeros((p, n_components, n_components))
    if lag0:
        lags = range(p)
    else:
        lags = range(1, p)
    for i in lags:
        t = n_samples - i
        C[i] = np.dot(Y[:, -t:], Y[:, i:].T) / t
    W = _joint_diag(C, max_iter=max_iter, tol=tol, verbose=verbose)
    return W.dot(whitener)


class SOBI(SMICA):
    def __init__(self, p, n_components,
                 freqs, sfreq, avg_noise=False, rng=None):
        '''
        n_components : number of sources
        freqs : the frequency intervals
        sfreq : sampling frequency
        '''
        self.p = p
        self.n_components = n_components
        self.freqs = freqs
        self.sfreq = sfreq
        self.avg_noise = avg_noise
        self.f_scale = 0.5 * (freqs[1:] + freqs[:-1])
        self.rng = check_random_state(rng)

    def fit(self, X, y=None, **kwargs):
        '''
        Fits sobi to data X (p x n matrix sampled at fs)
        '''
        self.X = X
        C, ft, freq_idx = fourier_sampling(X, self.sfreq, self.freqs)
        n_mat, n_sensors, _ = C.shape
        self.C_ = C
        self.ft_ = ft
        self.freq_idx_ = freq_idx
        W = sobi(X, self.p, self.n_components, **kwargs)
        self.A_ = np.linalg.pinv(W)
        self.powers_ = np.zeros((n_mat, self.n_components))
        for i in range(n_mat):
            self.powers_[i] = np.diag(W.dot(C[i]).dot(W.T))
        scale = np.mean(self.powers_, axis=0, keepdims=True)
        self.A_ = self.A_ * np.sqrt(scale)
        self.powers_ = self.powers_ / scale
        self.sigmas_ = np.zeros((C.shape[0], X.shape[0]))
        return self

    def compute_sources(self, X=None, method='pinv'):
        if method == 'wiener':
            raise ValueError('Only method=pinv is implemented for SOBI')
        return super().compute_sources(X=X, method=method)


class SOBI_mne(ICA):
    def __init__(self, p, n_components, freqs, rng=None):
        self.p = p
        self.n_components = n_components
        self.freqs = freqs
        self.f_scale = 0.5 * (freqs[1:] + freqs[:-1])
        self.rng = check_random_state(rng)

    def fit(self, inst, picks=None, avg_noise=False, **kwargs):
        '''
        Fits smica to inst (either raw or epochs)
        '''
        self.inst = inst
        self.info = inst.info
        self.sfreq = inst.info['sfreq']
        self.picks = picks
        self.avg_noise = avg_noise
        if isinstance(inst, BaseRaw):
            self.inst_type = 'raw'
            X = inst.get_data(picks=picks)
        else:
            self.inst_type = 'epoch'
            X = inst.get_data()
            n_epochs, _, _ = X.shape
            X = np.hstack(X)
        self.X = X
        X /= np.std(X)
        smica = SOBI(self.p, self.n_components, self.freqs, self.sfreq,
                     self.avg_noise)
        smica.fit(X, **kwargs)
        self.powers = smica.powers_
        self.A = smica.A_
        self.sigmas = smica.sigmas_
        self.smica = smica
        self.ica_mne = transfer_to_mne(self.A, self.inst, self.picks)
        return self
