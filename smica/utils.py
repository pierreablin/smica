import numpy as np


def loss(covs, A, sigma, source_powers, avg_noise=True, corr=False,
         normalize=False, by_bin=False):
    '''
    Compute the loss
    '''
    n_epochs, p, _ = covs.shape
    loss_values = np.zeros(n_epochs)
    for j, (cov, power) in enumerate(zip(covs, source_powers)):
        if corr:
            R = A.dot(power.dot(A.T))
        else:
            R = A.dot(power[:, None] * A.T)
        if avg_noise:
            R += np.diag(sigma)
        else:
            R += np.diag(sigma[j])
        CR = cov.dot(np.linalg.inv(R))
        loss_value = CR.trace() - np.linalg.slogdet(CR)[1] - p
        loss_values[j] = loss_value
    if by_bin:
        return loss_values
    return np.sum(loss_values)


def fourier_sampling(X, sfreq, freqs, window=False):
    '''
    Computes spectral covariances from the matrix X. sfreq is the
    sampling frequency of X, the covariances will be computed between the
    frequencies specified by freqs.
    '''
    p, n = X.shape
    # f_max = freqs[-1]
    # f_min = freqs[0]
    freq_idx = n * freqs / sfreq
    freq_idx = freq_idx.astype(int)
    # n_f_max = int(n * f_max / sfreq)
    # n_f_min = int(n * f_min / sfreq)
    # n_f = n_f_max - n_f_min
    n_bins = len(freqs) - 1
    if window:
        win = np.hanning(n)
        X *= win
    fourier_transform = np.fft.fft(X, axis=1)
    C = np.zeros((n_bins, p, p))
    scale = n_bins / sfreq / n
    for i in range(n_bins):
        sl = np.arange(freq_idx[i], freq_idx[i+1])
        ft = fourier_transform[:, sl]
        C[i] = np.real(np.dot(ft, ft.conj().T)) / len(sl) * scale
    return C, fourier_transform, freq_idx


def compute_covariances(A, powers, sigmas, avg_noise=False, corr=False):
    if corr:
        if avg_noise:
            covs = np.array([np.dot(A, power.dot(A.T)) + np.diag(sigmas)
                             for power in powers])
        else:
            covs = np.array([np.dot(A, power.dot(A.T)) + np.diag(sigma)
                             for power, sigma in zip(powers, sigmas)])
    else:
        if avg_noise:
            covs = np.array([np.dot(A, power[:, None] * A.T) + np.diag(sigmas)
                             for power in powers])
        else:
            covs = np.array([np.dot(A, power[:, None] * A.T) + np.diag(sigma)
                             for power, sigma in zip(powers, sigmas)])
    return covs


def itakura(p1, p2):
    frac = p1 / p2
    return np.mean(frac - np.log(frac) - 1)
