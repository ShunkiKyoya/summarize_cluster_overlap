"""Merge Components using Latent Entropy.

.. References:
    Jean-Patrick Baudry, Adrian E. Raftery, Gilles Celeux, Kenneth LO, and
    Raphael Gottardo. Combining Mixture Components for Clustering.
    Journal of Computational and Graphical Statistics, 9(2), 332--353, 2010.
"""

import numpy as np

from ._base import BaseMergeComponents


class EntMergeComponents(BaseMergeComponents):
    """Merge Components using Latent Entropy.
    """
    def __init__(self):
        super().__init__(search_mode='max')

    def criterion(self, i, j, prob_tmp):

        def ent(prob):
            return - np.dot(prob, np.log(prob + 1e-50))

        ent_after = ent(prob_tmp[:, i] + prob_tmp[:, j])
        ent_before = ent(prob_tmp[:, i]) + ent(prob_tmp[:, j])

        return (-1) * (ent_after - ent_before)


class NEnt1MergeComponents(BaseMergeComponents):
    """Merge Components using Normalized Latent Entropy (NEnt1).
    """
    def __init__(self):
        super().__init__(search_mode='max')

    def criterion(self, i, j, prob_tmp):

        def ent(prob):
            return - np.dot(prob, np.log(prob + 1e-50))

        ent_after = ent(prob_tmp[:, i] + prob_tmp[:, j])
        ent_before = ent(prob_tmp[:, i]) + ent(prob_tmp[:, j])
        weights = np.sum(prob_tmp[:, [i, j]])

        return (-1) * (ent_after - ent_before) / weights
