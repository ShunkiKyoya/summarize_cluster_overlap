from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from ._utils import max_ij, min_ij, mc, nmc


class BaseMergeComponents(metaclass=ABCMeta):
    """
    Base class for merging clusters.
    """

    def __init__(self, search_mode):
        """
        Args:
            search_mode (str):
                Assign 'max' or 'min' according to whether the algorithm wants
                to maximize or minimize the criterion.
        """
        self.search_mode = search_mode

    @abstractmethod
    def criterion(self, i, j, prob_tmp):
        """
        Cost function to merge components i and j.

        Args:
            i (int): Index of the first component.
            j (int): Index of the second cluster.
            prob_tmp (ndarray): Tmp. probabilities (shape = (N, K_tmp)).
        Returns:
            float: Cost to merge i and j.
        """
        pass

    def fit(self, prob_0):
        """Fit.

        Args:
            prob_0 (ndarray): Initial probabilities (shape = (N, K)).
        """
        K_0 = prob_0.shape[1]
        K_tmp = K_0
        prob_tmp = deepcopy(prob_0)
        members_tmp = [[i] for i in range(K_0)]
        self.members_list_ = [deepcopy(members_tmp)]
        self.nmc_0_ = nmc(prob_0)
        self.nmc_2_list_ = []
        self.criterion_list_ = []

        for _ in range(K_0 - 1):
            # search
            if self.search_mode == 'max':
                i, j, crit = max_ij(
                    lambda i, j: self.criterion(i, j, prob_tmp),
                    K_tmp
                )
            elif self.search_mode == 'min':
                i, j, crit = min_ij(
                    lambda i, j: self.criterion(i, j, prob_tmp),
                    K_tmp
                )
            # update
            self.criterion_list_.append(crit)
            prob_ij = prob_tmp[:, [i, j]] + 1e-50
            self.nmc_2_list_.append(nmc(
                prob=prob_ij / np.sum(prob_ij, axis=1).reshape((-1, 1)),
                weights=np.sum(prob_ij, axis=1)
            ))

            members_tmp[i].extend(members_tmp[j])
            members_tmp.pop(j)
            self.members_list_.append(deepcopy(members_tmp))

            prob_tmp[:, i] += prob_tmp[:, j]
            prob_tmp = np.delete(prob_tmp, j, axis=1)

            K_tmp -= 1

        # determine K_nmc_
        self.K_nmc_ = K_0
        for nmc_2_ in self.nmc_2_list_:
            if nmc_2_ < self.nmc_0_:
                self.K_nmc_ -= 1
            else:
                break

    def prob_merged(self, prob_0, K_merged):
        """Merged probabilities.

        Args:
            prob_0 (ndarray): Initial probabilities (shape = (N, K)).
            K_merged (ndarray): The number of components after merging.
        Returns:
            ndarray: Merged probabilities (shape = (N, K_merged)).
        """
        prob_merged_ = np.zeros((prob_0.shape[0], K_merged))
        for i, member in enumerate(
            self.members_list_[prob_0.shape[1] - K_merged]
        ):
            prob_merged_[:, i] = np.sum(prob_0[:, member], axis=1)
        return prob_merged_

    def clustering_summarization(self, prob_0, K_merged):
        """Create clustering summarization.

        Args:
            prob_0 (ndarray): Initial probabilities (shape = (N, K)).
            K_merged (ndarray): The number of components after merging.
        Returns:
            dict: clustering summary.
        """
        summary = {}
        prob_merged_ = self.prob_merged(prob_0, K_merged)
        members_ = self.members_list_[prob_0.shape[1] - K_merged]

        # upper
        summary['Upper-components'] = {
            'MC': mc(prob_merged_),
            'NMC': nmc(prob_merged_)
        }

        # lower
        for k in range(K_merged):
            weights = np.sum(prob_0[:, members_[k]] + 1e-50, axis=1)
            prob_k = (
                (prob_0[:, members_[k]] + 1e-50) / weights.reshape([-1, 1])
            )
            summary[f'Component {k}'] = {
                'Weight': np.sum(weights) / len(prob_0),
                'MC': mc(prob_k, weights=weights),
                'exp(MC)': np.exp(mc(prob_k, weights=weights)),
                'NMC': nmc(prob_k, weights=weights)
            }
        return summary
