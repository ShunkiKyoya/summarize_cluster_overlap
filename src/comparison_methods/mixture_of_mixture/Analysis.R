######################## reading sources:

######################## simulation characteristics:
## number of specified mixture clusters
# K <- 10
## number of subcomponents for each cluster
# L <- 3
## number of iterations M (without burnin)
# M <- 50
# burnin <- 50
Mmax <- M + burnin

## prior hyperparameters:
e0 <- 0.001
d0 <- ceiling((r * (r + 1)/2 + r)/2 + 1) + 2  

phiB <- 0.5
phiW <- 0.1

c0k <- 2.5 + (r - 1)/2
g0 <- 0.5 + (r - 1)/2

nu_1 <- 10
nu_2 <- 10

## vectors, matrices and arrays for storing the simulation results
y_sim <- array(0, c(sim, N, r))
K0_sim <- rep(0L, sim)
p_K0_sim <- matrix(0, sim, K)
M0_sim <- rep(0L, sim)
non_perm_rate_sim <- rep(0, sim)
classError_sim <- rep(0, sim)
adjustRand_sim <- rep(0, sim)
ass_k_matrix <- matrix(0, sim, N)
# z_k_matrix <- matrix(0, sim, N)
lam_matrixKj_sim <- array(0, dim = c(sim, M, r, K))
table_list <- list()
mode_list_sim <- list()



######################## simulations are starting:
for (i in 1:sim) {
  
  cat("\n", "\n", " Data set: ", i)
  
  ###### step 0: generating data
  ## set seed
  sese <- i * 10000 + my_seed
  set.seed(sese)
  
  ###### step 1: define prior parameters and initial values:
  
  ## prior parameters:
  b0k_0 <- t(apply(y, 2, function(x) quantile(x, probs = seq(1/(2 * K), 1 - 1/(2 * K), 1/K), names = F)))  #initial locations of the K cluster centers
  R_0 <- phiW * (1 - phiB) * diag(diag(cov(y)))
  
  m0 <- apply(y, 2, function(x) sum(range(x)/2))  #midpoint!
  M0_inv <- solve(10 * cov(y))
  
  lam_0 <- matrix(1, r, K)  #lam_0 is matrix with 1's!
  
  C0k <- (c0k - (r + 1)/2) * (1 - phiW) * (1 - phiB) * cov(y)
  G0 <- solve(diag(diag(C0k))) * g0
  
  ## initial values for parameters to be estimated:
  eta_0 <- rep(1/K, K)
  sigma_0 <- array(0, dim = c(r, r, K, L))
  for (k in 1:K) {
    for (l in 1:L) {
      sigma_0[, , k, l] <- C0k
    }
  }
  C0k_0 <- array(0, dim = c(r, r, K))
  for (k in 1:K) {
    C0k_0[, , k] <- C0k
  }
  invB0k_0 <- array(0, dim = c(r, r, K))
  for (k in 1:K) {
    invB0k_0[, , k] <- solve(R_0)
  }
  
  ## setting initial values for the classification of observations by clustering the data through
  # k_means
  cluster <- K
  subcluster <- L
  mu_0 <- array(0, dim = c(r, K, L))
  cl_y <- kmeans(y, centers = cluster, nstart = 30)
  S_0 <- cl_y[["cluster"]]
  I_0 <- rep(0, N)
  for (k in 1:K) {
    if (sum(cl_y[["cluster"]] == k) > L) {
      cl_gr <- kmeans(y[cl_y[["cluster"]] == k, ], centers = subcluster)
      I_0[S_0 == k] <- cl_gr[["cluster"]]
      mu_0[, k, ] <- cbind(t(cl_gr$centers))
    } else {
      I_0[S_0 == k] <- 1
    }
  }
  
  
  
  ###### step 2:call MCMC procedures
  estGibbs <- MixOfMix_estimation(y, S_0, I_0, mu_0, sigma_0, eta_0, 
                                  e0, d0, c0k, C0k_0, g0, G0, b0k_0, invB0k_0, lam_0, M, burnin, R_0, c_proposal, nu_1, nu_2, 
                                  random = FALSE, L, M0_inv, m0)
  Mu <- estGibbs$Mu
  Mu_k <- estGibbs$Mu_k
  Eta <- estGibbs$Eta
  S_alt_matrix <- estGibbs$S_alt_matrix
  S_neu_matrix <- estGibbs$S_neu_matrix
  Nk_matrix_alt <- estGibbs$Nk_matrix_alt
  lam_matrixKj <- estGibbs$lam_matrixKj
  mode_list <- estGibbs$mode_list
  
  
  
  
  ###### step 3: estimating the number of non-empty clusters through the mode of the emirical frequency of
  # the non-empty clusters during MCMC:
  
  K0_vector <- rowSums(Nk_matrix_alt != 0)  #vector with number of non-empty clusters of each iteration
  p_K0 <- tabulate(K0_vector, K)
  # par(mfrow = c(1, 1))  #empirical distribution of the number of non-empty clusters during MCMC
  K0 <- which.max(p_K0)
  K0  #mode K0 is the estimator for K_true
  M0 <- sum(K0_vector == K0)
  M0  #M0 draws have exactly K0 non-empty clusters
  
  ### selecting those draws where the number of non-empty clusters was exactly K0:
  Nk_matrix_K0 <- (Nk_matrix_alt[K0_vector == K0, ] != 0)
  M0_Nk_1 <- sum(rowSums((Nk_matrix_alt[K0_vector == K0, ]) == 1) == 1)
  M0_Nk_1  ##number of iterations with  components consisting of one observation only
  
  ### extracting those draws which are sampled from exactly K0 non-empty clusters i)Mu:
  Muk_inter <- array(0, dim = c(M0, r, K))
  Muk_inter <- Mu_k[K0_vector == K0, , ]  #matrix with draws only from the K0 interesting clusters
  Muk_K0 <- array(0, dim = c(M0, r, K0))
  for (j in 1:r) {
    Muk_K0[, j, ] <- matrix(t(Muk_inter[, j, ])[t(Nk_matrix_K0)], ncol = K0, byrow = T)
  }
  
  ### iii)Eta:
  Eta_inter <- matrix(0, M0, K)  #matrix with draws only from the K0 interesting clusters
  Eta_inter <- Eta[K0_vector == K0, ]
  Eta_K0 <- matrix(t(Eta_inter)[t(Nk_matrix_K0)], byrow = T, ncol = K0)
  
  ### iv)lamda_kj:
  lam_matrixKj_inter <- array(0, dim = c(M0, r, K))  #matrix with draws only from the K0 interesting clusters
  lam_matrixKj_inter <- lam_matrixKj[K0_vector == K0, , ]
  lam_matrixKj_K0 <- array(0, dim = c(M0, r, K0))
  for (j in 1:r) {
    lam_matrixKj_K0[, j, ] <- matrix(t(lam_matrixKj_inter[, j, ])[t(Nk_matrix_K0)], byrow = T, ncol = K0)
  }
  
  ### v)S_matrix:
  S_matrix_inter <- matrix(0, M, N)
  for (m in 1:M) {
    if (K0_vector[m] == K0) {
      perm_S <- rep(0, K)
      perm_S[Nk_matrix_alt[m, ] != 0] <- 1:K0
      S_matrix_inter[m, ] <- perm_S[S_alt_matrix[m, ]]
    }
  }
  S_matrix_K0 <- S_matrix_inter[K0_vector == K0, ]
  
  
  ###### step 4: identifying the estimated mixture of mixtures model:
  
  ### clustering the draws in the ppr and relabeling of the draws:
  map_muk <- mode_list[[K0]]$mu_k
  clust_FS_K0 <- MixOfMix_identification(Muk_K0, Eta_K0, lam_matrixKj_K0, S_matrix_K0, map_muk)
  
  Muk_only_perm <- clust_FS_K0$Mu_only_perm
  Eta_only_perm <- clust_FS_K0$Eta_only_perm
  lam_matrixKj_only_perm <- clust_FS_K0$lam_matrixKj_only_perm
  S_matrix_only_perm <- clust_FS_K0$S_matrix_only_perm
  non_perm_rate <- clust_FS_K0$non_perm_rate
  non_perm_rate
  Muk_Matrix_FS <- colMeans(Muk_only_perm, dims = 1)
  Eta_Matrix_FS <- colMeans(Eta_only_perm, dims = 1)
  
  
  ###### step 5: calculation of the posterior classification of the observations for each observation the
  # frequency of the assignment to the clusters is calculated:
  
  Ass_k <- matrix(0, N, K0)
  for (n in 1:N) {
    Ass_k[n, ] <- tabulate(S_matrix_only_perm[, n], K0)
  }
  ass_k <- apply(Ass_k, 1, which.max)
  # ClassError <- mclust::classError(ass_k, z_k)$errorRate
  # AdjustRand <- mclust::adjustedRandIndex(ass_k, z_k)
  # table_zk_ass_k <- table(z_k, ass_k)
  # table_zk_ass_k
  
  
  ###### step 6: storing the results:
  
  y_sim[i, , ] <- y
  K0_sim[i] <- K0
  p_K0_sim[i, ] <- p_K0
  M0_sim[i] <- M0
  non_perm_rate_sim[i] <- non_perm_rate
  # classError_sim[i] <- ClassError
  # adjustRand_sim[i] <- AdjustRand
  ass_k_matrix[i, ] <- ass_k
  # table_list[[i]] <- table_zk_ass_k
  mode_list_sim[[i]] <- mode_list
  # z_matrix[i,]=z
  # z_k_matrix[i, ] <- z_k
  lam_matrixKj_sim[i, , , ] <- lam_matrixKj
  
}



######################## Investestigating the simulation results:

cat("\n", "\n", "Estimated number of clusters: ", K0_sim)
# cat("\n", "misclassification rate: ", classError_sim)
cat("\n", "adjusted Rand index:", adjustRand_sim, "\n")


######################## plotting the estimated cluster distributions  
# or plotting the estimated classification:

# if (r == 2) {
#   i <- 1  # select the number of data set to be shown
#   K0 <- K0_sim[i]
#   List <- mode_list_sim[[i]][[K0]]
#   k <- 1
#   pro <- List$w[k, ]
#   # pro
#   mean <- List$mu[, k, ]
#   sigma <- List$Sigma_kl[, , k, ]
#   cholsigma <- sigma
#   for (l in 1:L) {
#     cholsigma[, , l] <- chol(List$Sigma_kl[, , k, l])
#   }
#   variance <- list(modelName = "VVV", d = r, G = L, cholsigma = cholsigma, sigma = sigma)
#   parameters <- list(pro = pro, mean = mean, variance = variance)
#   par(mfrow = c(1, 1))
#   mclust::surfacePlot(data = y, parameters = parameters, type = "contour")
#   for (k in seq_len(K0)[-1]) {
#     pro <- List$w[k, ]
#     mean <- List$mu[, k, ]
#     sigma <- List$Sigma_kl[, , k, ]
#     cholsigma <- sigma
#     for (l in 1:L) {
#       cholsigma[, , l] <- chol(List$Sigma_kl[, , k, l])
#     }
#     variance <- list(modelName = "VVV", d = r, G = L, cholsigma = cholsigma, sigma = sigma)
#     parameters <- list(pro = pro, mean = mean, variance = variance)
#     mclust::surfacePlot(data = y, parameters = parameters, add = TRUE, type = "contour")
#   }
# } else {
#   if (i == 1) 
#     pairs(y, pch = ass_k + 15, col = ass_k, main = "estimated classification", cex = 1.2, lty = 2)
# }










