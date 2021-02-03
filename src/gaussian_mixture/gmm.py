"""Functions for Gaussian Mixture Model.
"""

import math

import numpy as np
from numpy.random import RandomState
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from .dnml_gmm import log_pc_gmm, _pc_multinomial


class GMMUtils():
    """Useful Functions for Gaussian Mixture Model.
    """

    def __init__(self, rho, means, covariances):
        """
        Args:
            rho (ndarray): Mixture proportion (shape = (K,)).
            means (ndarray): Mean vectors (shape = (K, D)).
            covariances (ndarray): Covariance matrices (shape = (K, D, D)).
        """
        self.rho = rho
        self.means = means
        self.covariances = covariances
        self.K = len(rho)

    def sample(self, N=100, random_state=None):
        """Sample from GMM.

        Args:
            N (int): Number of the points to sample.
            random_state (Optional[int]): Random state.
        Returns:
            ndarray: Sampled points (shape = (N, K)).
        """
        random = RandomState(random_state)
        nk = random.multinomial(N, self.rho)
        X = []
        for mean, cov, size in zip(self.means, self.covariances, nk):
            X_new = multivariate_normal.rvs(
                mean=mean,
                cov=cov,
                size=size,
                random_state=random
            )
            if size == 0:
                pass
            elif size == 1:
                X.append(X_new)
            else:
                X.extend(X_new)
        return np.array(X)

    def logpdf(self, X):
        """Calculate log pdf.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: Matrix of log pdf (shape = (N, K)).
        """
        N = len(X)
        log_pdf = np.zeros([N, self.K])
        for k in range(self.K):
            log_pdf[:, k] = multivariate_normal.logpdf(
                X,
                self.means[k],
                self.covariances[k],
                allow_singular=True
            )
        return log_pdf

    def prob_latent(self, X):
        """Probability of the latent variables.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: Matrix of latent probabilities (shape = (N, K)).
        """
        log_pdf = self.logpdf(X)
        log_rho_pdf = np.log(self.rho + 1e-50) + log_pdf
        log_prob = (
            log_rho_pdf -
            logsumexp(log_rho_pdf, axis=1).reshape((-1, 1))
        )
        return np.exp(log_prob)


def _comp_loglike(*, X, Z, rho, means, covariances):
    """complete log-likelihood

    Args:
        X (ndarray): Data (shape = (N, K)).
        Z (ndarray): Latent variables (shape = (N,)).
        rho (ndarray): Mixture proportion (shape = (K,)).
        means (ndarray): Mean vectors (shape = (K, D)).
        covariances (ndarray): Covariance matrices (shape = (K, D, D)).
    Returns:
        float: Complete log likelihood.
    """
    _, D = X.shape
    K = len(means)
    nk = np.bincount(Z, minlength=K)

    if min(nk) <= 0:
        return np.nan
    else:
        c_loglike = 0
        for k in range(K):
            c_loglike += nk[k] * np.log(rho[k])
            c_loglike -= 0.5 * nk[k] * D * np.log(2 * math.pi * math.e)
            c_loglike -= 0.5 * nk[k] * np.log(np.linalg.det(covariances[k]))
        return c_loglike


class GMMModelSelection():
    """Model Selection of Gaussian Mixture Model.
    """
    def __init__(
        self,
        K_max=20,
        reg_covar=1e-3,
        random_state=None,
        mode='GMM_BIC',
        weight_concentration_prior=1.0,
        tol=1e-3,
        degrees_of_freedom_prior=None,
    ):
        """
        Args:
            K_max (int): Maximum number of the components.
            reg_covar (float): Reguralization for covariance.
            random_state (Optional[int]): Random state.
            mode (str): Estimation mode. Choose from the following:
                'GMM_BIC' (EM algorithm + BIC)
                'GMM_DNML' (EM algorithm + DNML)
                'BGMM' (Variational Bayes based on Dirichlet distribution).
            weight_concentration_prior (float): Weight concentration prior
                for BGMM.
            tol (float): Tolerance for GMM convergence.
        """
        self.K_max = K_max
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.mode = mode
        self.weight_concentration_prior = weight_concentration_prior
        self.tol = tol
        self.degrees_of_freedom_prior = degrees_of_freedom_prior

    def fit(self, X):
        """Select the best model.

        Args:
            X (ndarray): Data (shape = (N, K)).
        """
        if self.mode == 'GMM_DNML':
            log_pc_array = log_pc_gmm(
                K_max=self.K_max,
                N_max=X.shape[0],
                D=X.shape[1]
            )

        if self.mode in ['GMM_BIC', 'GMM_DNML']:
            model_list = []
            criterion_list = []
            for K in range(1, self.K_max + 1):
                # Fit
                model_new = GaussianMixture(
                    n_components=K,
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                    n_init=10,
                    tol=self.tol,
                    max_iter=10000
                )
                model_new.fit(X)
                model_list.append(model_new)
                # Calculate information criterion
                if self.mode == 'GMM_BIC':
                    criterion_list.append(model_new.bic(X))
                elif self.mode == 'GMM_DNML':
                    Z = model_new.predict(X)
                    loglike = _comp_loglike(
                        X=X,
                        Z=Z,
                        rho=model_new.weights_,
                        means=model_new.means_,
                        covariances=model_new.covariances_
                    )
                    complexity = np.log(_pc_multinomial(len(X), K))
                    for k in range(K):
                        Z_k = sum(Z == k)
                        if log_pc_array[1, Z_k] != - np.inf:
                            complexity += log_pc_array[1, Z_k]
                    criterion_list.append(- loglike + complexity)
            idx_best = np.nanargmin(criterion_list)
            self.model_best_ = model_list[idx_best]

        elif self.mode == 'BGMM':
            self.model_best_ = BayesianGaussianMixture(
                n_components=self.K_max,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
                weight_concentration_prior=self.weight_concentration_prior,
                weight_concentration_prior_type='dirichlet_distribution',
                max_iter=10000,
                n_init=10,
                tol=self.tol,
                degrees_of_freedom_prior=self.degrees_of_freedom_prior
            )
            self.model_best_.fit(X)
        else:
            raise ValueError('methods should be GMM_BIC, GMM_DNML or BGMM.')

        self.K_ = self.model_best_.n_components
        self.rho_ = self.model_best_.weights_
        self.means_ = self.model_best_.means_
        self.covariances_ = self.model_best_.covariances_

    def prob_latent(self, X):
        """Probability of the latent variables.

        Args:
            X (ndarray): Data (shape = (N, D)).
        Returns:
            ndarray: Matrix of latent probabilities (shape = (N, K)).
        """
        analysis = GMMUtils(
            rho=self.rho_,
            means=self.means_,
            covariances=self.covariances_
        )
        return analysis.prob_latent(X)
