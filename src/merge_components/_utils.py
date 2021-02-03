"""Useful functions.
"""

import numpy as np


def min_ij(f, K):
    """Find i and j (0 <= i < j < K) that minimize f(i, j).

    Args:
        func (Callable): Function of i and j.
        K (int): Range of i and j.
    Returns:
        Tuple[int, int, float]: (i_best, j_best, value_min).
    """
    value_min = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            value_tmp = f(i, j)
            if value_tmp < value_min:
                i_best, j_best = i, j
                value_min = value_tmp
    return i_best, j_best, value_min


def max_ij(f, K):
    """Find i and j (0 <= i < j < K) that maximize f(i, j).

    Args:
        func (Callable): Function of i and j.
        K (int): Range of i and j.
    Returns:
        Tuple[int, int]: (i_best j_best, value_max).
    """
    i_best, j_best, m_value_min = min_ij(lambda i, j: (-1) * f(i, j), K)
    return i_best, j_best, - m_value_min


def mc(
    prob,
    weights=None
):
    """
    Calculate MC from prob.

    Args:
        prob (ndarray): Probabilities (shape = (N, K)).
        weights (Optional[ndarray]): Data weights (shape = (N, K)).
    Returns:
        float: MC.
    """
    N = len(prob)
    if weights is None:
        weights = np.ones(N)

    rho = np.dot(weights, prob) / np.sum(weights)
    H_Z = - np.dot(rho, np.log(rho + 1e-50))
    H_ZbarX = (
        - np.sum(np.dot(weights, prob * np.log(prob + 1e-50)))
        / np.sum(weights)
    )
    return H_Z - H_ZbarX


def nmc(
    prob,
    weights=None
):
    """
    Calculate NMC from prob.

    Args:
        prob (ndarray): Probabilities (shape = (N, K)).
        weights (Optional[ndarray]): Data weights (shape = (N, K)).
    Returns:
        float: NMC.
    """
    N = len(prob)
    if weights is None:
        weights = np.ones(N)

    rho = np.dot(weights, prob) / np.sum(weights)
    H_Z = - np.dot(rho, np.log(rho + 1e-50))
    H_ZbarX = (
        - np.sum(np.dot(weights, prob * np.log(prob + 1e-50)))
        / np.sum(weights)
    )

    if H_Z < 1e-15:
        return 0.0
    else:
        return (H_Z - H_ZbarX) / H_Z
