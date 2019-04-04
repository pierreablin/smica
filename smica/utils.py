import numpy as np


def loss(covs, A, sigma, source_powers, avg_noise=True,
         normalize=False):
    '''
    Compute the loss
    '''
    loss_value = 0.
    n_epochs, p, _ = covs.shape
    for j, (cov, power) in enumerate(zip(covs, source_powers)):
        if avg_noise:
            R = A.dot(power[:, None] * A.T) + np.diag(sigma)
        else:
            R = A.dot(power[:, None] * A.T) + np.diag(sigma[j])
        loss_value += cov.dot(np.linalg.inv(R)).trace()
        loss_value += np.linalg.slogdet(R)[1]
        if normalize:
            loss_value -= np.linalg.slogdet(cov)[1]
    if normalize:
        loss_value -= p * n_epochs
    return loss_value


def fourier_sampling(X, sfreq, freqs):
    '''
    Computes spectral covariances from the matrix X. sfreq is the
    sampling frequency of X, the covariances will be computed between the
    frequencies specified by freqs.
    '''
    p, n = X.shape
    f_max = freqs[-1]
    f_min = freqs[0]
    freq_idx = n * freqs / sfreq
    freq_idx = freq_idx.astype(int)
    n_f_max = int(n * f_max / sfreq)
    n_f_min = int(n * f_min / sfreq)
    n_f = n_f_max - n_f_min
    n_bins = len(freqs) - 1
    fourier_transform = np.fft.fft(X, axis=1)
    C = np.zeros((n_bins, p, p))
    for i in range(n_bins):
        sl = np.arange(freq_idx[i], freq_idx[i+1])
        ft = fourier_transform[:, sl]
        C[i] = np.real(np.dot(ft, ft.conj().T)) / len(sl)
    return C, fourier_transform, freq_idx


def itakura(p1, p2):
    frac = p1 / p2
    return np.mean(frac - np.log(frac) - 1)
