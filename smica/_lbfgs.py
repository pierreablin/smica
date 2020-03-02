import numpy as np
from joblib import Memory

from scipy.optimize import fmin_l_bfgs_b


location = './cachedir'
memory = Memory(location, verbose=0)


def x_to_params(x, p, q, n):
    A = x[:p * q].reshape(p, q)
    P = x[p * q: p * q + q * n].reshape(n, q)
    N = x[p * q + q * n:].reshape(n, p)
    return A, P, N


def params_to_x(A, P, N):
    p, q = A.shape
    n, _ = P.shape
    x = np.zeros(p * q + n * q + n * p)
    x[:p * q] = A.ravel()
    x[p * q: p * q + q * n] = P.ravel()
    x[p * q + q * n:] = N.ravel()
    return x


def kl(A, B):
    p, _ = A.shape
    C = np.dot(A, np.linalg.inv(B))
    return np.trace(C) - np.linalg.slogdet(C)[1] - p


def kl_i(A, B):
    p, _ = A.shape
    C = np.dot(A, B)
    return np.trace(C) - np.linalg.slogdet(C)[1] - p


def loss(C_list, A, powers, sigmas):
    op = 0.
    for C, P, S in zip(C_list, powers, sigmas):
        B = np.dot(A, P[:, None] * A.T) + np.diag(S)
        op += kl(C, B)
    return op


def loss_and_gradients(C_list, A, powers, sigmas):
    loss = 0.
    grad_A = np.zeros_like(A)
    grad_powers = np.zeros_like(powers)
    grad_sigmas = np.zeros_like(sigmas)
    for i, (C_hat, P, S) in enumerate(zip(C_list, powers, sigmas)):
        C = np.dot(A, P[:, None] * A.T) + np.diag(S)
        C_inv = np.linalg.inv(C)
        loss += kl_i(C_hat, C_inv)
        ker = np.dot(C_inv, np.dot(C - C_hat, C_inv))
        grad_sigmas[i] = np.diag(ker)
        KA = np.dot(ker, A)
        grad_powers[i] = np.diag(np.dot(A.T, KA))
        grad_A += 2 * KA * P
    return loss, grad_A, grad_powers, grad_sigmas


def func_grad(x, p, q, n, C_list):
    A, pows, sigmas = x_to_params(x, p, q, n)
    val, gA, gP, gS = loss_and_gradients(C_list, A, pows, sigmas)
    return val, params_to_x(gA, gP, gS)


@memory.cache(ignore=['verbose'])
def lbfgs(C_list, A0, N0, P0, max_fun=15000, verbose=False):
    p, q = A0.shape
    n, _, _ = C_list.shape
    bounds = [(None, None), ] * (p * q)
    bounds += [(0, None), ] * (p * n + q * n)
    x0 = params_to_x(A0, P0, N0)
    if verbose:
        iprint = 50
    else:
        iprint = -1
    x, f, d = fmin_l_bfgs_b(func_grad, x0, args=(p, q, n, C_list),
                            bounds=bounds, factr=10, m=30, maxfun=max_fun,
                            iprint=iprint)
    A, source_powers, sigma = x_to_params(x, p, q, n)
    scale = np.mean(source_powers, axis=0, keepdims=True)
    A = A * np.sqrt(scale)
    source_powers = source_powers / scale
    return A, sigma, source_powers, f, d
