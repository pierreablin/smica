import autograd.numpy as np
from autograd import grad

from scipy.optimize import fmin_l_bfgs_b


def loss(variables, covs, avg_noise=True):
    A, sigma, source_powers = variables
    loss_value = 0.
    n_epochs, p, _ = covs.shape
    for j, (cov, power) in enumerate(zip(covs, source_powers)):
        if avg_noise:
            R = np.dot(A, power[:, None] * A.T) + np.diag(sigma)
        else:
            R = np.dot(A, power[:, None] * A.T) + np.diag(sigma[j])
        loss_value = loss_value + np.trace(np.dot(cov, np.linalg.inv(R)))
        loss_value = loss_value + np.linalg.slogdet(R)[1]
    return loss_value


def ravelize(A, sigma, source_powers):
    return np.concatenate((A.ravel(), sigma.ravel(), source_powers.ravel()))


def vectorize(x, n_epochs, n_sensors, n_sources, avg_noise):
    if avg_noise:
        sigma_dim = (n_sensors, )
    else:
        sigma_dim = (n_epochs, n_sensors)
    dims = [(n_sensors, n_sources), sigma_dim, (n_epochs, n_sources)]
    idxs = np.cumsum([np.prod(dim) for dim in dims])
    A = x[:idxs[0]].reshape(dims[0])
    sigma = x[idxs[0]:idxs[1]].reshape(dims[1])
    source_powers = x[idxs[1]:].reshape(dims[2])
    return A, sigma, source_powers


def lbfgs(covs, A, sigma, source_powers, avg_noise, **kwargs):
    x0 = ravelize(A, sigma, source_powers)
    gradient_fn = grad(loss)
    n_sensors, n_sources = A.shape
    n_epochs, _ = source_powers.shape
    bounds = [(None, None), ] * (n_sensors * n_sources)
    if avg_noise:
        n_vars = n_epochs * n_sources + n_sensors
    else:
        n_vars = n_epochs * (n_sources + n_sensors)
    bounds += [(0, None), ] * n_vars

    def loss_and_grad(x):
        variables = vectorize(x, n_epochs, n_sensors, n_sources,
                              avg_noise)
        l_value = loss(variables, covs, avg_noise)
        g_value = gradient_fn(variables, covs, avg_noise)

        return l_value, ravelize(*g_value)

    x, _, d = fmin_l_bfgs_b(loss_and_grad, x0, bounds=bounds, **kwargs)
    A, sigma, source_powers = vectorize(x, n_epochs, n_sensors, n_sources,
                                        avg_noise)
    l_final = loss((A, sigma, source_powers), covs, avg_noise)
    # Rescale
    scale = np.mean(source_powers, axis=0, keepdims=True)
    A = A * np.sqrt(scale)
    source_powers = source_powers / scale
    return A, sigma, source_powers, l_final, d
