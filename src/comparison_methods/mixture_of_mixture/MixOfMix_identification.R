########################################################################################################
# Clustering the MCMC Mu-draws in the point process representation by applying k-means. 
# The  obtained (unique) labeling is used 
#   for reordering the Eta draws and the indicator-matrix S. 
# Only 'true' permutations  are used for parameter estimation.
########################################################################################################


MixOfMix_identification <- function(Mu, Eta, lam_matrixKj, S_matrix, map_mu) {
  K <- length(Mu[1, 1, ])
  M <- length(Mu[, 1, 1])
  r <- length(Mu[1, , 1])
  N <- length(S_matrix[1, ])
  
  #################### step 0: create a matrix of size M*K, 
  # formed from all MCMC-draws of Mu, for applying  k-means clustering algorithm in the next step
  KM <- (Mu[, , 1])
  if (K > 1) {
    for (k in 2:K) {
      KM <- rbind(KM, Mu[, , k])
    }
  }
  names_Mu <- c()
  for (j in 1:r) {
    names_Mu <- c(names_Mu, paste("mu", j, sep = ""))
  }
  colnames(KM) <- c(names_Mu)
  
  #################### step 1: cluster the Mu-draws in the point process representation 
  # by applying k-means cluster algorithm
  x <- KM
  colnames(x) <- NULL
  cent <- t(map_mu)   #starting classification is obtained by k-means clustering with known cluster means(=mu_map)
  cl_y <- kmeans(x, centers = cent, iter.max = 1000)
  class <- cl_y$cluster
  # classification matrix is constructed:
  Rho_m <- NULL
  for (l in 0:(K - 1)) {
    Rho_m <- cbind(Rho_m, class[(l * M + 1):((l + 1) * M)])
  }
  
  
  #################### step 2: identifying the number of iterations where the classification of the draws
  # results in a non-permutation:
  m_rho <- NULL
  for (m in 1:M) {
    if (any(sort(Rho_m[m, ]) != 1:K)) 
      m_rho <- c(m_rho, m)
  }
  non_perm_rate <- length(m_rho)/M
  non_perm_rate  ###rate of non-permutations iterations
  M0 <- M - length(m_rho)
  
  
  #################### step 3: relabel draws of Mu, Eta and S: unique labeling is achieved 
  # by reordering the draws trough Rho_m where simultaneuously  
  # the draws belonging to iterations which are not permutations are removed
  Mu_only_perm <- array(0, dim = c(M0, r, K))
  Eta_only_perm <- matrix(0, M0, K)
  lam_matrixKj_only_perm <- array(0, dim = c(M0, r, K))
  S_matrix_only_perm <- matrix(0, M0, N)
  
  for (m in seq_len(M0)) {
    j <- setdiff(1:M, m_rho)[m]
    Mu_only_perm[m, , Rho_m[j, ]] <- Mu[j, , ]
    Eta_only_perm[m, Rho_m[j, ]] <- Eta[j, ]
    lam_matrixKj_only_perm[m, , Rho_m[j, ]] <- lam_matrixKj[j, , ]
    S_matrix_only_perm[m, ] <- Rho_m[j, ][S_matrix[j, ]]
  }
  
  
  return(list(S_matrix_only_perm = S_matrix_only_perm, Mu_only_perm = Mu_only_perm, Eta_only_perm = Eta_only_perm, 
              lam_matrixKj_only_perm = lam_matrixKj_only_perm, non_perm_rate = non_perm_rate, Rho_m = Rho_m, 
              m_rho = m_rho))
}






