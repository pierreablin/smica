import numpy as np
from numpy.linalg import norm

from joblib import Memory

from .utils import loss

location = './cachedir'
memory = Memory(location, verbose=0)


def invert(weighted_cys, sigmas, c_ss):
    '''Used to compute A in the m step
    '''
    p, q = weighted_cys.shape
    inv = np.zeros((p, q))
    for j in range(p):
        M = np.sum(sigmas[:, j][:, None, None] * c_ss, axis=0)
        inv[j] = np.linalg.solve(M.T, weighted_cys[j])
    return inv


def m_step(cov_source_source_s, cov_signal_source, cov_signal_signal,
           avg_noise, A=None):
    if avg_noise:
        cov_source_source = np.mean(cov_source_source_s, axis=0)
        cov_source_source_inv = np.linalg.inv(cov_source_source)
        A = cov_signal_source.dot(cov_source_source_inv)
        sigmas_square = np.diag(cov_signal_signal - A.dot(cov_signal_source.T))
    else:
        n_epochs, p, q = cov_signal_source.shape
        sigmas_square = np.zeros((n_epochs, p))
        # update sigma
        for j in range(n_epochs):
            c_si_so = cov_signal_source[j]
            AR_ys = A.dot(c_si_so.T)
            sigmas_square[j] =\
                np.diag(cov_signal_signal[j] - AR_ys - AR_ys.T
                        + A.dot(cov_source_source_s[j]).dot(A.T))
        # update A
        weighted_cys = np.zeros((p, q))
        for j in range(n_epochs):
            sigma_inv = 1. / sigmas_square[j]
            weighted_cys += sigma_inv[:, None] * cov_signal_source[j]
        A = invert(weighted_cys, 1. / sigmas_square, cov_source_source_s)

    source_powers = np.diagonal(cov_source_source_s, axis1=1, axis2=2)
    return A, sigmas_square, source_powers


def e_step(covs, covs_inv, A, sigmas_square, source_powers, avg_noise):
    n_epochs, p, _ = covs.shape
    p, q = A.shape
    cov_source_source_s = np.zeros((n_epochs, q, q))
    if avg_noise:
        cov_signal_source = np.zeros((p, q))
        cov_signal_signal = np.mean(covs, axis=0)
    else:
        cov_signal_source = np.zeros((n_epochs, p, q))
        cov_signal_signal = covs
    if avg_noise:
        At_sig_A = A.T.dot(A / sigmas_square[:, None])
        proj = A.T / sigmas_square[None, :]
    for epoch, (cov, cov_inv) in enumerate(zip(covs, covs_inv)):
        if not avg_noise:
            At_sig_A = A.T.dot(A / sigmas_square[epoch, :, None])
            proj = A.T / sigmas_square[epoch, None, :]
        expected_cov = np.linalg.inv(At_sig_A +
                                     np.diag(1. / source_powers[epoch]))
        wiener = expected_cov.dot(proj)
        cov_signal_source_inst = cov.dot(wiener.T)
        if avg_noise:
            cov_signal_source += cov_signal_source_inst
        else:
            cov_signal_source[epoch] = cov_signal_source_inst
        cov_source_source_s[epoch] = wiener.dot(cov_signal_source_inst) +\
            expected_cov
    if avg_noise:
        cov_signal_source /= n_epochs
    return cov_source_source_s, cov_signal_source, cov_signal_signal


@memory.cache(ignore=['verbose', 'return_iterates'])
def em_algo(covs, A, sigmas_square, source_powers, avg_noise,
            max_iter=1000, verbose=False, return_iterates=False,
            dA_thresh=1e-4, dS_thresh=1e-6):
    '''
    EM algorithm to fit the SMICA model on the covariances matrices covs
    '''
    covs_inv = np.array([np.linalg.inv(cov) for cov in covs])
    x_list = []
    los_old = np.inf
    for it in range(max_iter):

        if return_iterates:
            x_list.append(numpy_to_vect(A, sigmas_square, source_powers))

        cov_source_source_s, cov_signal_source, cov_signal_signal =\
            e_step(covs, covs_inv, A, sigmas_square, source_powers, avg_noise)

        A, sigmas_square, source_powers =\
            m_step(cov_source_source_s, cov_signal_source, cov_signal_signal,
                   avg_noise, A)
        scale = np.mean(source_powers, axis=0, keepdims=True)
        A = A * np.sqrt(scale)
        source_powers = source_powers / scale
        if it > 0:
            dA = np.linalg.norm(A - A_old) / np.linalg.norm(A)
            dS = (np.linalg.norm(sigmas_square - sigmas_old)
                  / np.linalg.norm(sigmas_square))
            if dA < dA_thresh and dS < dS_thresh:
                break
        if verbose:
            if (it - 1) % verbose == 0:
                los = loss(covs, A, sigmas_square, source_powers, avg_noise)
                print('it {:5d}, loss: {:10.5e}, dloss: {:04.2e}, dA: {:04.2e}'
                      ', dSigma: {:04.2e}, dPowers: {:04.2e}'.format(
                        it,
                        los,
                        los_old - los,
                        dA,
                        dS,
                        np.linalg.norm(source_powers - powers_old)))
                los_old = los
        A_old = A.copy()
        sigmas_old = sigmas_square.copy()
        powers_old = source_powers.copy()
    return A, sigmas_square, source_powers, x_list
