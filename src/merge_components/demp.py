"""Merge Components using DEMP.

..References:
    Christian Hennig: Methods for Merging Gaussian Miture Components.
    Advances in Data Analysis and Classification, 4, 3--34, 2010.
"""
import numpy as np

from ._base import BaseMergeComponents


class DEMPMergeComponents(BaseMergeComponents):
    """
    Cluster merging based on DEMP.
    """

    def __init__(self):
        super().__init__(search_mode='max')

    def criterion(self, i, j, prob_tmp):
        Z = np.argmax(prob_tmp, axis=1)
        prob_protect = prob_tmp + 1e-50

        def missprob(k_, l_):
            mp = (
                np.dot(prob_protect[:, k_], (Z == l_).astype(float)) /
                np.sum(prob_protect[:, k_])
            )
            return mp

        return max(missprob(i, j), missprob(j, i))


class DEMP2MergeComponents(BaseMergeComponents):
    """
    Cluster merging based on DEMP (All).
    """

    def __init__(self):
        super().__init__(search_mode='max')

    def criterion(self, i, j, prob_tmp):
        prob_ij = prob_tmp[:, [i, j]] + 1e-50
        Z = np.argmax(prob_ij, axis=1)

        def missprob(k_, l_):
            mp = (
                np.dot(prob_ij[:, k_], (Z == l_).astype(float)) /
                np.sum(prob_ij[:, k_])
            )
            return mp

        return max(missprob(0, 1), missprob(1, 0))
