"""Supervised Clustering Evaluation Functions.

.. References:
    * Lawrence Hubert and Phipps Arabie. Comparing Partitions. Journal of
    Classification, 2, 193--218, 1985.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def f_measure(Z_true, Z_pred):
    """F-measure.

    Args:
        Z_true (ndarray): True labels (shape = (N,)).
        Z_pred (ndarray): Predicted labels (shape = (N,)).
    Returns:
        float: Score.
    """
    mat = contingency_matrix(Z_true, Z_pred)
    purity = np.sum(np.max(mat, axis=0)) / np.sum(mat)
    i_purity = np.sum(np.max(mat, axis=1)) / np.sum(mat)
    return 2 / (1 / (purity) + 1 / (i_purity))


def ari(Z_true, Z_pred):
    """Adjusted Rand Index.

    Args:
        Z_true (ndarray): True labels (shape = (N,)).
        Z_pred (ndarray): Predicted labels (shape = (N,)).
    Returns:
        float: Score.
    """
    return adjusted_rand_score(Z_true, Z_pred)
