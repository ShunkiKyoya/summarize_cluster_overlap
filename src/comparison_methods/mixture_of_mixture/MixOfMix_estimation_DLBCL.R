#################################################################################################
#### Gibbs Sampling for estimating a sparse finite mixture of  mixtures model 
#################################################################################################

MixOfMix_estimation <- function(y, S_0, I_0, mu_0, sigma_0, eta_0, e0, 
                               d0, c0k, C0k_0, g0, G0, b0k_0, invB0k_0, lam_0, M, burnin, R_0, c_proposal, nu_1, nu_2, random, 
                               L, M0_inv, m0) {
  
  K <- length(mu_0[1, , 1])  # number of components 
  N <- nrow(y)  # number of observations
  r <- ncol(y)  # number of dimensions
  
  ## defing current parameters:
  sigma_j <- array(0, dim = c(r, r, K, L))
  invsigma_j <- array(0, dim = c(r, r, K, L))
  mu_j <- array(0, dim = c(r, K, L))
  mu_k <- matrix(0, r, K)
  invB0k_j <- array(0, dim = c(r, r, K))
  B0k_j <- array(0, dim = c(r, r, K))
  eta_j <- rep(0, K)
  S_j <- rep(0, N)
  I_j <- rep(0, N)
  lam_j <- matrix(0, r, K)  # lam_j is now a matrix with dim=c(r,K)
  C0k_j <- array(0, dim = c(r, r, K))
  b0k_j <- matrix(0, r, K)
  det_invsigma_j <- matrix(0, K, L)
  
  
  ## initializing current parameter values:
  eta_j <- eta_0
  w_j <- matrix(1/(L), K, L)
  sigma_j <- sigma_0
  invsigma_j <- sigma_j
  mu_j <- mu_0
  S_j <- S_0
  lam_j <- lam_0
  C0k_j <- C0k_0
  invB0k_j <- invB0k_0
  B0k_j[, , 1:K] <- solve(invB0k_0[, , 1])
  b0k_j <- b0k_0
  Nk_j <- tabulate(S_j, K)
  cat("\n", " Initial partition: ", "\n", Nk_j, "\n")
  M0 <- solve(M0_inv)
  
  ## generating matrices for storing the draws:
  result <- list(Eta = matrix(0, M, K), Mu = array(0, dim = c(M, r, K, L)), 
                 Mu_k = array(0, dim = c(M, r, K)), S_alt_matrix = matrix(0L, M, N), 
                 S_neu_matrix = matrix(0L, M, N), I_neu_matrix = matrix(0L, M, N), 
                 Nk_matrix_alt = matrix(0L, M, K), Nk_matrix_neu = matrix(0L, M, K), 
                 Nk_view = matrix(0L,(M + burnin)/100 + 1, K), 
                 Nl_matrix_neu = matrix(0L, M + burnin, K * L), 
                 lam_matrixKj = array(0,dim = c(M, r, K)), 
                 bkl_matrix = array(0, dim = c(M + burnin, r, K, L)), 
                 Bkl_matrix = array(0,dim = c(M + burnin, r, r, K, L)), 
                 mixprior = rep(0, M), mixlik = rep(0, M), nonnorm_post = rep(0, M), 
                 mode_list = vector("list", K), mixlikmode_list = vector("list", K))
  
  ## Initialising the storing matrices:
  result$Mu[1, , , ] <- mu_0
  result$Eta[1, ] <- eta_0
  result$S_alt_matrix[1, ] <- S_0
  result$lam_matrixKj[1, , ] <- lam_0
  for (k in 1:K) {
    result$mode_list[[k]] <- list(nonnorm_post = -(10)^18)
  }
  for (k in 1:K) {
    result$mixlikmode_list[[k]] <- list(mixlik = -(10)^18)
  }
  
  ## constant parameters for every iteration:
  p_gig <- nu_1 - L/2
  a_gig <- 2 * nu_2
  gn <- g0 + L * c0k
  B0k_0 <- array(0, dim = c(r, r, K))
  B0k_0[, , 1:K] <- solve(invB0k_0[, , 1])
  
  
  s <- 1
  result$Nk_view[1, ] <- Nk_j
  
  ################################################ algorithm starts:
  m <- 2
  while (m <= M | m <= burnin) {
    
    if (m == burnin) {
      m <- 1
      burnin <- 0
    }
    
    Nk_j <- tabulate(S_j, K)
    S_alt_j <- S_j
    Nk_alt_j <- Nk_j
    K0_j <- sum(Nk_j != 0)
    
    if (!(m%%100)) {
      cat("\n", m, " ", Nk_j)
      s <- s + 1
      result$Nk_view[s, ] <- Nk_j
    }
    
    
    #################### first step: sample mixture weights eta (conditional on classification S_j): 
    ## sample eta_j:
    ek <- e0 + Nk_j
    eta_j <- bayesm::rdirichlet(ek)
    
    
    #################### second step: classification of observations (conditional on knowing the parameters)
    ## sample S_j:
    matk <- array(0, dim = c(N, K, L))
    for (k in 1:K) {
      matk[, k, ] <- sapply(1:L, function(l) w_j[k, l] * dmvnorm(y, mu_j[, k, l], sigma_j[, , 
                                                                                          k, l]))
    }
    mat <- sapply(1:K, function(k) eta_j[k] * rowSums(matk[, k, , drop = F]))
    S_j <- apply(mat, 1, function(x) sample(1:K, 1, prob = x))
    Nk_j <- tabulate(S_j, K)
    Nk_neu_j <- Nk_j
    
    
    #################### third step: For cluster k=1,...K:
    I_j <- rep(0, N)
    for (k in 1:K) {
      if (Nk_neu_j[k] != 0) {
        # if a cluster is non-empty
        yk <- y[S_j == k, , drop = FALSE]
        
        #### (3a): Classification of the observations within a cluster to the subcomponents:
        matl <- matk[S_j == k, k, ]
        if (is.matrix(matl)) {
          I_l <- apply(matl, 1, function(x) sample(1:L, 1, prob = x))
        } else {
          I_l <- sample(1:L, 1, prob = matl)
        }
        I_j[S_j == k] <- I_l
        
        #### (3b): parameter simulation conditional on classification I_j:
        
        ## (3b_i): sample subcomponent weights w_j:
        Nl_j <- tabulate(I_l, L)
        result$Nl_matrix_neu[m, (L * (k - 1) + 1):(L * (k - 1) + L)] <- Nl_j
        dl <- d0 + Nl_j
        w_j[k, ] <- bayesm::rdirichlet(dl)
        
        ## (3b_ii): sample Sigma_kl^{-1}:
        Ckl <- array(0, dim = c(r, r, L))
        ckl <- c0k + Nl_j/2
        for (l in 1:L) {
          if (Nl_j[l] != 0) {
            Ckl[, , l] <- C0k_j[, , k] + 0.5 * crossprod(sweep(yk[I_l == l, , drop = FALSE], 
                                                               2, mu_j[, k, l], FUN = "-"))
          } else {
            Ckl[, , l] <- C0k_j[, , k]
          }
          sig <- bayesm::rwishart(2 * ckl[l], 0.5 * chol2inv(chol(Ckl[, , l])))
          sigma_j[, , k, l] <- chol2inv(chol(sig$W))
          invsigma_j[, , k, l] <- sig$W
          det_invsigma_j[k, l] <- det(invsigma_j[, , k, l])
        }
        
        ## (3b_iii): sample subcomponent means mu_kl:
        mean_ykl <- matrix(0, r, L)
        mean_ykl <- sapply(1:L, function(l) colMeans(yk[I_l == l, , drop = FALSE]))
        if (sum(is.na(mean_ykl)) > 0) {
          # to catch the case if a group is empty: NA values are substituted by zeros
          mean_ykl[is.na(mean_ykl)] <- 0
        }
        Bkl <- array(0, dim = c(r, r, L))
        bkl <- matrix(0, r, L)
        invB0k_j[, , k] <- invB0k_0[, , k]/lam_j[, k]
        for (l in 1:L) {
          Bkl[, , l] <- chol2inv(chol(invB0k_j[, , k] + invsigma_j[, , k, l] * Nl_j[l]))
          bkl[, l] <- Bkl[, , l] %*% (invB0k_j[, , k] %*% b0k_j[, k] + invsigma_j[, , k, l] %*% 
                                        mean_ykl[, l] * Nl_j[l])
          mu_j[, k, l] <- t(chol(Bkl[, , l])) %*% rnorm(r) + bkl[, l]
        }
        
      } else {
        # if the cluster is empty, sample from the priors
        
        dl <- d0
        w_j[k, ] <- bayesm::rdirichlet(rep(dl, L))
        B0k_j[, , k] <- B0k_0[, , k] * lam_j[, k]
        invB0k_j[, , k] <- invB0k_0[, , k]/lam_j[, k]
        for (l in 1:L) {
          sig <- bayesm::rwishart(2 * c0k, 0.5 * chol2inv(chol(C0k_j[, , k])))
          sigma_j[, , k, l] <- chol2inv(chol(sig$W))
          invsigma_j[, , k, l] <- sig$W
          det_invsigma_j[k, l] <- det(invsigma_j[, , k, l])
          mu_j[, k, l] <- t(chol(B0k_j[, , k])) %*% rnorm(r) + b0k_j[, k]
        }
      }
      mu_k[, k] <- mu_j[, k, ] %*% w_j[k, ]  #these are the weighted cluster centers, they are clusterd in the ppr to obtain a unique labeling
    }
    
    #################### fourth step: For cluster k=1,...K sample hyperparameters:
    for (k in 1:K) {
      
      ## (4a): sample cluster- and dimension-specifc lambda_kj:
      for (j in 1:r) {
        b_gig_k <- rep(0, r)
        b_gig_k <- rowSums(((mu_j[, k, , drop = F] - b0k_j[, k])^2)/diag(B0k_0[, , 1]))
        ## NEWEST GIGrvg genator:
        b_gig_k <- rep(0, r)
        b_gig_k <- rowSums(((mu_j[, k, , drop = F] - b0k_j[, k])^2)/diag(B0k_0[, , 1]))
        if (b_gig_k[j] == 0) 
          b_gig_k[j] <- 1e-07  #to catch the case if mu_j[j,k,]~~b0k_j[j]
        ran <- GIGrvg::rgig(1, lambda = p_gig, chi = b_gig_k[j], psi = a_gig)
        # lam_j[j, k] <- ran  # DLBCL ???
        lam_j[j, k] <- 1  # DLBCL ???
      }
      
      ## (4b): sample hyperparameter C0k conditionally on sigma_kl:
      # C0k_j[, , k] <- bayesm::rwishart(2 * gn, 0.5 * chol2inv(chol(G0 + rowSums(invsigma_j[, , 
                                                                                          #  k, , drop = F], dims = 2))))$W  #from package 'bayesm'
      C0k_j[, , k] <- gn * chol2inv(chol(G0 + rowSums(invsigma_j[, , k, , drop = F], dims = 2)))  # DLBCL ???

      ## (4c): sample b0k from N(1/L*sum(mu_j[,k,]);1/L*B0 )k
      invB0k_j[, , k] <- invB0k_0[, , k]/lam_j[, k]
      Mtilde <- solve(M0_inv + L * invB0k_j[, , k])
      mtilde <- Mtilde %*% (M0_inv %*% m0 + invB0k_j[, , k] %*% (rowSums(mu_j[, k, , drop = F])))
      b0k_j[, k] <- MASS::mvrnorm(1, mtilde, Mtilde)
    }
    
    
    #################### fifth step: evaluating the posterior distribution
    ## (5a) evaluating  the mixture likelihood:
    matk <- array(0, dim = c(N, K, L))
    for (k in 1:K) {
      matk[, k, ] <- sapply(1:L, function(l) w_j[k, l] * dmvnorm(y, mu_j[, k, l], sigma_j[, , 
                                                                                          k, l]))
    }
    mat <- sapply(1:K, function(k) eta_j[k] * rowSums(matk[, k, , drop = F]))
    mixlik_j <- sum(log(rowSums(mat)))
    
    ## (5b) evaluating the mixture prior:
    sig <- 0
    for (k in 1:K) {
      sig <- c(sig, sum(sapply(1:L, function(l) bayesm::lndIWishart(2 * c0k, 0.5 * C0k_j[, , k], 
                                                                    sigma_j[, , k, l]))))
    }
    mixprior_j <- log(MCMCpack::ddirichlet(as.vector(eta_j), rep(e0, K))) +   # modified 
      sum(log(MCMCpack::ddirichlet(w_j,rep(d0, L)))) + 
      sum(rowSums(sapply(1:K, function(k) dmvnorm(t(mu_j[, k, ]), b0k_j[, k],B0k_j[, , k]*lam_j[, k], log = TRUE)))) + 
      sum(sig) + sum(sapply(1:K, function(k) bayesm::lndIWishart(2 *g0, 0.5 * G0, C0k_j[, , k]))) + 
      sum(sapply(1:K, function(k) dmvnorm(t(b0k_j[, k]), m0, M0,log = TRUE))) +
  		sum(mapply(function(x) dgamma(x,shape=nu_1,scale=1/nu_2),lam_j))
    
    ## (5c) storing the nonnormalized posterior:
    if (burnin == 0) {
      result$nonnorm_post[m] <- mixlik_j + mixprior_j
    }
    
    
    #################### sixth step: storing the draws
    if (burnin == 0) {
      result$Mu[m, , , ] <- mu_j
      result$Mu_k[m, , ] <- mu_k
      result$Eta[m, ] <- eta_j
      result$S_alt_matrix[m, ] <- S_alt_j
      result$S_neu_matrix[m, ] <- S_j
      result$I_neu_matrix[m, ] <- I_j
      result$Nk_matrix_alt[m, ] <- Nk_alt_j
      result$Nk_matrix_neu[m, ] <- Nk_j
      result$lam_matrixKj[m, , ] <- lam_j
      result$bkl_matrix[m, , , ] <- bkl
      result$Bkl_matrix[m, , , , ] <- Bkl
      result$mixlik[m] <- mixlik_j
    }
    
    if ((burnin == 0) & (result$nonnorm_post[m] > result$mode_list[[K0_j]]$nonnorm_post)) {
      result$mode_list[[K0_j]] <- list(nonnorm_post = result$nonnorm_post[m], mu = mu_j[, Nk_alt_j != 0, ],
                                       bkl = result$bkl_matrix[m, , Nk_alt_j != 0, ], 
                                       Bkl = result$Bkl_matrix[m, , , Nk_alt_j != 0, ], 
                                       mu_k = mu_k[, Nk_alt_j != 0],
      																 Sigma_kl=sigma_j[,,Nk_alt_j!=0,],w=w_j[Nk_alt_j!=0,])
    }
    
    m <- m + 1
  }
  return(result)
}



####################################################
## Copied from package mvtnorm and slightly modified
dmvnorm <- function(x, mean, sigma, log = FALSE) {
  if (is.vector(x)) {
    x <- matrix(x, ncol = length(x))
  }
  if (missing(mean)) {
    mean <- rep(0, length = ncol(x))
  }
  if (missing(sigma)) {
    sigma <- diag(ncol(x))
  }
  if (NCOL(x) != NCOL(sigma)) {
    stop("x and sigma have non-conforming size")
  }
  if (!isSymmetric(sigma, tol = sqrt(.Machine$double.eps), check.attributes = FALSE)) {
    stop("sigma must be a symmetric matrix")
  }
  if (length(mean) != NROW(sigma)) {
    stop("mean and sigma have non-conforming size")
  }
  distval <- mahalanobis(x, center = mean, cov = chol2inv(chol(sigma)), inverted = TRUE)
  logdet <- sum(log(eigen(sigma, symmetric = TRUE, only.values = TRUE)$values))
  logretval <- -(ncol(x) * log(2 * pi) + logdet + distval)/2
  if (log) 
    return(logretval)
  if (is.na(sum(logretval))) 
    return(rep(0, length(x[, 1])))
  exp(logretval)
} 
