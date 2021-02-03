"""Utilize Mixture of Mixture Model.

R programs are copied from the following reference.

References:
    Gertraud Malsiner-Walli, Sylvia Fruhwirth-Schnatter and Bettina Grun.
    Identifying Mixtures of Mixtures Using Bayesian Estimation. Journal of
    Computational and Graphical Statics, 26(2), 285--295, 2017.
"""

import os

import pyper


class MixtureOfMixture():
    """Call mixture of mixture model from python.
    """
    def __init__(
        self,
        K=10,
        L=4,
        M=4000,
        burnin=4000,
        random_state=0
    ):
        """
        Args:
            K (int): The initial number of upper components.
            L (int): The initial number of lower components.
            M (int): The number of iterations in MCMC.
            burnin (int): The number of burnin.
            random_state (int): random state.
        """
        self.K = K
        self.L = L
        self.M = M
        self.burnin = burnin
        self.random_state = random_state

    def fit(self, X):
        """Fit.

        Args:
            X (ndarray): Data (shape = (N, K))
        """
        N, D = X.shape
        r = pyper.R()
        r.assign("y", X)
        r.assign("N", N)
        r.assign("r", D)
        r.assign("sim", 1)
        r.assign("K", self.K)
        r.assign("L", self.L)
        r.assign("M", self.M)
        r.assign("burnin", self.burnin)
        r.assign("my_seed", self.random_state)
        path_here = os.path.dirname(__file__)
        r("source(\"{}/rmvnormMix.R\")".format(path_here))
        r("source(\"{}/MixOfMix_estimation.R\")".format(path_here))
        r("source(\"{}/MixOfMix_identification.R\")".format(path_here))
        self.log_ = r("source(\"{}/Analysis.R\")".format(path_here))
        self.K_pred_ = r.get("K0_sim")
        self.Z_pred_ = r.get("ass_k")
        if self.Z_pred_ is not None:
            self.Z_pred_ -= 1
