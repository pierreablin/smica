import warnings

import numpy as np
from numba import njit
from numpy.linalg import norm

from joblib import Memory

from .utils import loss, compute_covariances

location = './cachedir'
memory = Memory(location, verbose=0)
EPS = 1e-12


def one_update(C_hat, C, a):
    '''
    exact update for the noise/powers
    '''
    Rna = np.linalg.solve(C, a)
    diff = C_hat - C
    return np.dot(np.dot(diff, Rna), Rna) / (np.dot(a, Rna)) ** 2


def invert(weighted_cys, sigmas, c_ss, n_jobs=1):
    '''
    Used to compute A in the m step
    '''
    M = np.einsum('it, ijk->tkj', sigmas, c_ss)
    inv = np.linalg.solve(M, weighted_cys)
    return inv


@njit
def compute_w_cys(sigmas, cov_signal_source):
    n, p, q = cov_signal_source.shape
    w_s = np.zeros((q, p))
    for j in range(n):
        w_s += cov_signal_source[j].T / sigmas[j, :]
    return w_s.T


@njit
def compute_sigmas(A, cov_signal_source, cov_signal_signal,
                   cov_source_source_s):
    n_epochs, p, q = cov_signal_source.shape
    sigmas_square = np.zeros((n_epochs, p))
    AR_d = np.zeros(p)
    AcA_d = np.zeros(p)
    # update sigma
    for j in range(n_epochs):
        c_si_so = cov_signal_source[j]
        Acss = A.dot(cov_source_source_s[j].T)
        for i in range(p):
            AR_d[i] = np.dot(A[i], c_si_so[i])
            AcA_d[i] = np.dot(A[i], Acss[i])
        sigmas_square[j] = np.diag(cov_signal_signal[j]) - 2 * AR_d
        sigmas_square[j] += AcA_d
    return sigmas_square


def m_step(cov_source_source_s, cov_signal_source, cov_signal_signal, corr,
           avg_noise, A=None):
    if avg_noise:
        cov_source_source = np.mean(cov_source_source_s, axis=0)
        cov_source_source_inv = np.linalg.inv(cov_source_source)
        A = cov_signal_source.dot(cov_source_source_inv)
        sigmas_square = np.diag(cov_signal_signal - A.dot(cov_signal_source.T))
    else:
        sigmas_square = compute_sigmas(A, cov_signal_source, cov_signal_signal,
                                       cov_source_source_s)
        # update A
        weighted_cys = compute_w_cys(sigmas_square, cov_signal_source)
        A = invert(weighted_cys, 1. / (sigmas_square + EPS),
                   cov_source_source_s)
    if corr:
        source_powers = cov_source_source_s
    else:
        source_powers = np.diagonal(cov_source_source_s, axis1=1, axis2=2)
    return A, sigmas_square, source_powers


@njit
def pairwise_dots1(A, B, op):
    n, a, _ = A.shape
    _, b, _ = B.shape
    for i in range(n):
        op[i] = np.dot(A[i], B[i].T)
    return op


@njit
def pairwise_dots2(A, B, op):
    n, a, _ = A.shape
    _, _, b = B.shape
    for i in range(n):
        op[i] = np.dot(A[i], B[i])
    return op


@njit
def one_dots(A, B, op):
    n, a, _ = A.shape
    _, b = B.shape
    for i in range(n):
        op[i] = np.dot(A[i], B)
    return op


@njit
def compute_matrices_uncorr(A, sigmas, source_powers):
    p, q = A.shape
    n, _ = sigmas.shape
    op = np.zeros((n, q, q))
    for i in range(n):
        op[i] = np.dot(A.T / sigmas[i, :], A)
        for j in range(q):
            op[i, j, j] += 1 / source_powers[i, j]
    return op


@njit
def compute_matrices_corr(A, sigmas, source_powers):
    p, q = A.shape
    n, _ = sigmas.shape
    op = np.zeros((n, q, q))
    for i in range(n):
        op[i] = np.dot(A.T / sigmas[i, :], A)
        op[i] += np.linalg.pinv(source_powers[i])
    return op


def compute_matrices(A, sigmas, source_powers, corr):
    if corr:
        return compute_matrices_corr(A, sigmas, source_powers)
    else:
        return compute_matrices_uncorr(A, sigmas, source_powers)


# @profile
def e_step(covs, covs_inv, A, sigmas_square, source_powers, corr, avg_noise):
    n_epochs, p, _ = covs.shape
    p, q = A.shape
    cov_source_source_s = np.zeros((n_epochs, q, q))
    cov_signal_source = np.zeros((n_epochs, p, q))
    wiener = np.zeros((n_epochs, q, p))
    if avg_noise:
        cov_signal_signal = np.mean(covs, axis=0)
    else:
        cov_signal_signal = covs
    if avg_noise:
        At_sig_A = np.zeros((n_epochs, q, q))
        At_sig_A_ = A.T.dot(A / (sigmas_square[:, None] + EPS))
        proj = A.T / (sigmas_square[None, :] + EPS)
    if not avg_noise:
        At_sig_A = compute_matrices(A, sigmas_square, source_powers, corr)
        proj = A.T / sigmas_square[:, None, :]
    else:
        for epoch, source_power in enumerate(source_powers):
            if avg_noise:
                if corr:
                    At_sig_A[epoch] = At_sig_A_ + np.linalg.pinv(source_power)
                else:
                    At_sig_A[epoch] = At_sig_A_ + np.diag(1. / source_power)
    expected_cov = np.linalg.inv(At_sig_A)
    if avg_noise:
        wiener = one_dots(expected_cov, proj, wiener)
    else:
        pairwise_dots2(expected_cov, proj, wiener)
    pairwise_dots1(covs, wiener, cov_signal_source)
    pairwise_dots2(wiener, cov_signal_source, cov_source_source_s)
    cov_source_source_s += expected_cov
    if avg_noise:
        cov_signal_source = np.mean(cov_signal_source, axis=0)

    return cov_source_source_s, cov_signal_source, cov_signal_signal


# @profile
@memory.cache(ignore=['verbose'])
def em_algo(covs, A, sigmas_square, source_powers, corr, avg_noise,
            max_iter=10000, verbose=False, tol=1e-7, n_it_min=10,
            cd_every=0, n_jobs=1):
    '''
    EM algorithm to fit the SMICA model on the covariances matrices covs
    '''
    use_joblib = n_jobs > 1
    n_sensors, n_sources = A.shape
    n_mat, _, _ = covs.shape
    covs_inv = np.array([np.linalg.inv(cov) for cov in covs])
    loss_init = loss(covs, A, sigmas_square, source_powers, avg_noise, corr)
    loss_old = loss_init
    criterion = 0
    do_cd = cd_every != 0
    for it in range(max_iter):
        cov_source_source_s, cov_signal_source, cov_signal_signal =\
            e_step(covs, covs_inv, A, sigmas_square, source_powers, corr,
                   avg_noise)

        A, sigmas_square, source_powers =\
            m_step(cov_source_source_s, cov_signal_source, cov_signal_signal,
                   corr, avg_noise, A)
        # CD updates
        if do_cd:
            if it % cd_every == 0 and not avg_noise:
                covs_estimates = compute_covariances(A, source_powers,
                                                     sigmas_square)
                # Noise updates
                for mat in range(n_mat):
                    for sensor in range(n_sensors):
                        e_i = np.zeros(n_sensors)
                        e_i[sensor] = 1.
                        update = one_update(covs[mat], covs_estimates[mat],
                                            e_i)
                        new_coef = np.maximum(update + sigmas_square[mat,
                                                                     sensor],
                                              EPS)
                        diff = new_coef - sigmas_square[mat, sensor]
                        sigmas_square[mat, sensor] = new_coef
                        covs_estimates[mat, sensor, sensor] += diff
                # Powers updates
                source_powers.setflags(write=1)
                for source in range(n_sources):
                    a = A[:, source]
                    aaT = np.outer(a, a)
                    for mat in range(n_mat):
                        update = one_update(covs[mat], covs_estimates[mat], a)
                        new_coef = np.maximum(update + source_powers[mat,
                                                                     source],
                                              EPS)
                        diff = new_coef - source_powers[mat, source]
                        source_powers[mat, source] = new_coef
                        covs_estimates[mat] += diff * aaT
        # Rescale

        if corr:
            scale = np.mean(np.diagonal(source_powers, axis1=1, axis2=2),
                            axis=0)
            source_powers /= np.sqrt(np.outer(scale, scale))[None, :, :]
        else:
            scale = np.mean(source_powers, axis=0, keepdims=True)
            source_powers = source_powers / scale
        A = A * np.sqrt(scale)
        if it > 0 and it % 50 == 0:
            loss_value = loss(covs, A, sigmas_square, source_powers, avg_noise,
                              corr)
            criterion = (loss_old - loss_value)
            criterion /= (np.abs(loss_old) + np.abs(loss_value))
            loss_old = loss_value
            if criterion < tol and it > n_it_min:
                break
        if verbose:
            if (it - 1) % verbose == 0 and it > 0:
                loss_print = loss(covs, A, sigmas_square, source_powers,
                                  avg_noise,
                                  corr)
                print('it {:5d}, loss: {:10.5e}, crit: {:04.2e}'.format(
                        it, loss_print, criterion))
        A_old = A.copy()
        sigmas_old = sigmas_square.copy()
        powers_old = source_powers.copy()
    else:
        warnings.warn('Warning, em algorithm did not converge: '
                      'criterion %.2e' % criterion)
    return A, sigmas_square, source_powers
