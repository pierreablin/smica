import numpy as np
import warnings

import numpy as np
from numpy.linalg import norm

from joblib import Memory

from .utils import loss, compute_covariances


def one_update(C_hat, C, a):
    Rna = np.linalg.solve(C, a)
    diff = C_hat - C
    return np.dot(np.dot(diff, Rna), Rna) / (np.dot(a, Rna)) ** 2


def cd_algo(covs, A, sigmas_square, source_powers, avg_noise=False,
            max_iter=10000, verbose=False, tol=1e-8, n_it_min=10):
    '''
    CD algorithm to fit the powers and variances
    '''
    loss_init = loss(covs, A, sigmas_square, source_powers, False)
    loss_old = loss_init
    criterion = 0
    n_sensors, n_sources = A.shape
    n_mat, _, _ = covs.shape
    covs_estimates = compute_covariances(A, source_powers, sigmas_square)
    for it in range(max_iter):
        # Noise updates
        for mat in range(n_mat):
            for sensor in range(n_sensors):
                e_i = np.zeros(n_sensors)
                e_i[sensor] = 1.
                update = one_update(covs[mat], covs_estimates[mat], e_i)
                new_coef = np.maximum(update + sigmas_square[mat, sensor], 0)
                diff = new_coef - sigmas_square[mat, sensor]
                sigmas_square[mat, sensor] = new_coef
                covs_estimates[mat, sensor, sensor] += diff
        # Powers updates
        for source in range(n_sources):
            a = A[:, source]
            aaT = np.outer(a, a)
            for mat in range(n_mat):
                update = one_update(covs[mat], covs_estimates[mat], a)
                new_coef = np.maximum(update + source_powers[mat, source], 0)
                diff = new_coef - source_powers[mat, source]
                source_powers[mat, source] = new_coef
                covs_estimates[mat] += diff * aaT
        scale = np.mean(source_powers, axis=0, keepdims=True)
        A = A * np.sqrt(scale)
        source_powers = source_powers / scale
        if it > 0:
            loss_value = loss(covs, A, sigmas_square, source_powers, False)
            criterion = (loss_old - loss_value)
            criterion /= (np.abs(loss_old) + np.abs(loss_value))
            loss_old = loss_value
            if criterion < tol and it > n_it_min:
                break
        if verbose:
            if (it - 1) % verbose == 0 and it > 0:
                print('it {:5d}, loss: {:10.5e}, crit: {:04.2e}'.format(
                        it, loss_old, criterion))
        A_old = A.copy()
        sigmas_old = sigmas_square.copy()
        powers_old = source_powers.copy()
    else:
        warnings.warn('Warning, cd algorithm did not converge: '
                      'critertion %.2e' % criterion)
    return A, sigmas_square, source_powers


if __name__ == '__main__':
    n_mat = 10
    n_sources = 3
    n_sensors = 5
    A = np.random.randn(n_sensors, n_sources)
    powers = np.random.rand(n_mat, n_sources)
    sigmas = np.random.rand(n_mat, n_sensors)
    noise = 0.1 * np.random.randn(n_sensors, n_sensors)
    noise = noise.dot(noise.T)
    covs = np.array([np.dot(A, power[:, None] * A.T) + np.diag(sig) + noise
                     for power, sig in zip(powers, sigmas)])
    cd_algo(covs, A, sigmas, powers, False,
            max_iter=20, verbose=1, tol=-100, n_it_min=10)
