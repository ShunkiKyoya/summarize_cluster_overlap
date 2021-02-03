"""Unsupervised Clustering Evaluation Functions.
"""

import numpy as np


def max_intra_distance(
    X,
    prob_merged
):
    """Calculate Maximum Intra Cluster Distance.

    Args:
        X (ndarray): Data (shape = (N, D)).
        prob_merged (ndarray): Merged post probabilities
        (shape = (N, L)).
    Returns:
        float: Maximum intra cluster distance.
    """
    L = prob_merged.shape[1]

    # means_merged
    means_merged = (
        np.dot(prob_merged.T, X) /
        np.sum(prob_merged, axis=0).reshape([-1, 1])
    )

    # squared distance
    squared_distances = np.zeros(L)
    for l_ in range(L):
        diffs = np.sum((X - means_merged[l_])**2, axis=1)
        squared_distances[l_] = (
            np.dot(prob_merged[:, l_], diffs) /
            sum(prob_merged[:, l_])
        )

    return max(squared_distances)


def min_inter_distance(
    X,
    prob_merged
):
    """Calculate Minimum Inter Cluster Distance.

    Args:
        X (ndarray): Data (shape = (N, D)).
        prob_merged (ndarray): Merged post probabilities
        (shape = (N, L)).
    Returns:
        float: Minimum inter cluster distance.
    """
    D = X.shape[1]
    L = prob_merged.shape[1]

    # means_merged
    means_merged = (
        np.dot(prob_merged.T, X) /
        np.sum(prob_merged, axis=0).reshape([-1, 1])
    )

    # distance matrix
    dists = np.zeros([L, L])
    for d in range(D):
        dists += (
            means_merged[:, d]
            - means_merged[:, d].reshape([-1, 1])
        )**2
    dists += np.eye(L) * 1e+300

    return np.min(dists)
