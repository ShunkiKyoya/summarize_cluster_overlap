"""Merge Components using MC and NMC.

.. References:
    Shunki Kyoya and Kenji Yamanishi. Mixture Complexity and Its Application
    to Gradual Clustering Change Detection. arXiv: 2007.07467.
"""

import numpy as np

from ._base import BaseMergeComponents
from ._utils import mc, nmc


class MCMergeComponents(BaseMergeComponents):
    """
    Cluster merging based on MC.
    """

    def __init__(self):
        super().__init__(search_mode='min')

    def criterion(self, i, j, prob_tmp):
        probs_ij = prob_tmp[:, [i, j]]
        crit = mc(
            prob=probs_ij / np.sum(probs_ij + 1e-50, axis=1).reshape([-1, 1]),
            weights=np.sum(probs_ij, axis=1)
        )
        return crit


class NMCMergeComponents(BaseMergeComponents):
    """
    Cluster merging based on NMC.
    """

    def __init__(self):
        super().__init__(search_mode='min')

    def criterion(self, i, j, prob_tmp):
        probs_ij = prob_tmp[:, [i, j]]
        crit = nmc(
            prob=probs_ij / np.sum(probs_ij + 1e-50, axis=1).reshape([-1, 1]),
            weights=np.sum(probs_ij, axis=1)
        )
        return crit
