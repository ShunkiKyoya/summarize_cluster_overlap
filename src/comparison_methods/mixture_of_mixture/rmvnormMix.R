##############################################################
## generating observations from a normal mixture distribution:
#############################################################

rmvnormMix <- function(n, eta, mu, sigma, z) {
  K <- ncol(mu)
  r <- nrow(mu)
  y <- matrix(0,n,r)
  if(missing(z)){  
    z <- sample(1:K, n, replace = TRUE, prob = eta)
  }
  for (k in 1:K) {
    if (sum(z == k)) {
      y[z==k, ] <- MASS::mvrnorm(sum(z==k), mu[, k], sigma[, , k])
    }
  }
  return(list(y = y, z = z, eta = eta, mu = mu, sigma = sigma))
}
