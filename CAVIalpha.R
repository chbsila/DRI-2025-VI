# ------------------------------------------------------------
# Coordinate Ascent Variational Inference (CAVI) Algorithm
# for Laplace Spike-and-Slab in High-Dimensional Linear Regression
# with Rényi Divergence
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

### Simulate a linear regression problem of size n times p, with sparsity level s ###
n <- 10
p <- 2
s <- 0

#' ### Generate toy data ###
X <- matrix(rnorm(n*p), n, p) 
theta <- numeric(p) 
theta[sample.int(p, s)] <- runif(s, -3, 3) #sample non-zero coefficients uar 
pos_TR <- as.numeric(theta != 0) 
Y <- X %*% theta + rnorm(n)

### Rényi VB CAVI Implementation ###

# Helper Functions 

A_i_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p) {
  term <- - YtX[i] + mu_i * XtX[i,i]
  sum_j <- 0
  for (j in setdiff(1:p, i)) {
    sum_j <- sum_j + gamma[j] * mu[j] * XtX[j,i]
  }
  term <- term + 0.5 * sum_j + lambda * sign(gamma[i] * mu_i)
  return( (term^2) * sigma1[i]^2 )
}

A_i_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p) {
  term <- - YtX[i] + mu[i] * XtX[i,i]
  sum_j <- 0
  for (j in setdiff(1:p, i)) {
    sum_j <- sum_j + gamma[j] * mu[j] * XtX[j,i]
  }
  term <- term + 0.5 * sum_j + lambda * sign(gamma[i] * mu[i])
  return( (term^2) * sigma_i^2 )
}

B_i_mu <- function(mu_i, X, sigma1, i) {
  return( XtX[i,i] * sigma1[i]^2 - 1 )
}

B_i_sigma <- function(sigma_i, X, sigma1, i) {
  return( XtX[i,i] * sigma_i^2 - 1 )
}

C_i_mu <- function(mu_i, X, mu, sigma1, gamma, i, a, p) {
  sum_k <- 0
  for (k in setdiff(1:p, i)) {
    t1 <- (a/2) * mu_i * XtX[k,i]
    t2 <- (a^2 /4) * (mu_i * XtX[k,i] * gamma[k] * mu[k])^2 
    t3 <- gamma[k] * (1 - gamma[k]) * mu[k]^2 + gamma[k] * sigma1[k]^2
    sum_k <- sum_k + t1 + t2 * t3
  }
  return(sum_k)
}

C_i_sigma <- function(sigma_i, X, mu, sigma1, gamma, i, a, p) {
  sum_k <- 0
  for (k in setdiff(1:p, i)) {
    t1 <- (a/2) * mu[i] * XtX[k,i]
    t2 <- (a^2 /4) * (mu[i] * XtX[k,i] * gamma[k] * mu[k])^2 
    t3 <- gamma[k] * (1 - gamma[k]) * mu[k]^2 + gamma[k] * sigma1[k]^2
    sum_k <- sum_k + t1 + t2 * t3
  }
  return(sum_k)
}

#-------------------------------------------------------------------------------------------------------------------

g_func_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, a, p, i) {
  sum_j <- 0
  for (j in setdiff(1:p, i)) {
    sum_j <- sum_j + gamma[j] * mu[j] * XtX[j, i]
  }
  exponent <- a * (
    - YtX[i] * mu_i + 0.5 * mu_i^2 * XtX[i, i] + mu_i * sum_j +
    lambda * abs(mu_i) - log(sigma1[i])
  )
  return(exp(exponent))
}

#-------------------------------------------------------------------------------------------------------------------
g_func_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, a, p, i) {
  sum_j <- 0
  for (j in setdiff(1:p, i)) {
    sum_j <- sum_j + gamma[j] * mu[j] * XtX[j, i]
  }
  exponent <- a * (
    - YtX[i] * mu[i] + 0.5 * mu[i]^2 * XtX[i, i] + mu[i] * sum_j +
    lambda * abs(mu[i]) - log(sigma_i)
  )
  return(exp(exponent))
}
#-------------------------------------------------------------------------------------------------------------------

# Objective functions for given i

kappa_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, a, p, i) {
  g_val <- g_func_mu(mu_i, X, Y, mu, sigma1, gamma, lambda, a, p, i)  # scalar for index i
  Ai <- A_i_mu(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p)
  Bi <- B_i_mu(mu_i, X, sigma1, i)
  Ci <- C_i_mu(mu_i, X, mu, sigma1, gamma, i, a, p)
  return( g_val * (1 + (a^2 / 2) * Ai + (a / 2) * Bi + 0.5 * Ci) )
}

kappa_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, a, p, i) {
  g_val <- g_func_sigma(sigma_i, X, Y, mu, sigma1, gamma, lambda, a, p, i)  # scalar for index i
  Ai <- A_i_sigma(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p)
  Bi <- B_i_sigma(sigma_i, X, sigma1, i)
  Ci <- C_i_sigma(sigma_i, X, mu, sigma1, gamma, i, a, p)
  return( g_val * (1 + (a^2 / 2) * Ai + (a / 2) * Bi + 0.5 * Ci) )
}

#-------------------------------------------------------------------------------------------------------------------
#Helper for gamma_i

# Base function h evaluated at E[theta, z_i]
h_func <- function(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, lambda, a, p) {
  diff <- gamma_i * mu[i] - mu[i]
  term1 <- log(sqrt(2) / (sqrt(pi) * sigma1[i] * lambda))
  term2 <- - (diff^2) / (2 * sigma1[i]^2)
  term3 <- lambda * abs(gamma_i * mu[i])
  term4 <- log(gamma_i) - log(w_bar)
  term5 <- log(1 - gamma_i) - log(1 - w_bar)
  
  sum_j <- sum(XtX[-i, i] * gamma[-i] * mu[-i])
  
  exponent <- a * (
    gamma_i * (term1 + term2 + term3 + term4) +
    (1 - gamma_i) * term5 -
    YtX[i] * gamma_i * mu[i] +
    0.5 * (gamma_i * mu[i])^2 * XtX[i, i] +
    (gamma_i * mu[i]) * sum_j
  )
  exp(exponent)
}

# ∂²h/∂θ_k² for k != i
d2h_theta_k2 <- function(gamma_i, gamma, mu, sigma1, XtX, h_val, i, k, a, p) {
  (a^2) * (gamma_i^2) * (mu[i]^2) * (XtX[k, i]^2) * h_val
}

# ∂²h/∂θ_i²
d2h_theta_i2 <- function(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, h_val, a, i) {
  diff <- (gamma_i * mu[i] - mu[i]) / sigma1[i]^2
  sign_term <- lambda * sign(gamma_i * mu[i])
  
  inner_sum <- sum(mu[-i] * gamma[-i] * XtX[-i, i])
  
  square_term <- gamma_i * (-diff + sign_term) - YtX[i] + gamma_i * mu[i] * XtX[i, i] + inner_sum
  
  a * (
    - gamma_i / sigma1[i]^2 + XtX[i, i] + a * (square_term^2)
  ) * h_val
}

# ∂²h/∂θ_i ∂z_i
d2h_theta_i_zi <- function(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, w_bar, h_val, a, i, p) {
  log_term <- log(sqrt(2) / (sqrt(pi) * sigma1[i] * lambda)) - ((gamma_i * mu[i] - mu[i])^2) / (2 * sigma1[i]^2) +
    lambda * abs(gamma_i * mu[i]) + log(gamma_i) - log(w_bar) - log(1 - gamma_i) + log(1 - w_bar)
  
  diff <- (gamma_i * mu[i] - mu[i]) / sigma1[i]^2
  sign_term <- lambda * sign(gamma_i * mu[i])
  inner_sum <- sum(mu[-i] * gamma[-i])
  
  first_bracket <- gamma_i * (-diff + sign_term) - YtX[i] + gamma_i * mu[i] * XtX[i, i] + inner_sum
  
  a^2 * (log_term * first_bracket + a * (-diff + sign_term)) * h_val
}

# ∂²h/∂z_i²
d2h_zi2 <- function(gamma_i, mu, sigma1, lambda, w_bar, a, i) {
  log_term <- log(sqrt(2) / (sqrt(pi) * sigma1[i] * lambda)) - ((gamma_i * mu[i] - mu[i])^2) / (2 * sigma1[i]^2) +
    lambda * abs(gamma_i * mu[i]) + log(gamma_i) - log(w_bar) - log(1 - gamma_i) + log(1 - w_bar)
  
  (a^2) * (log_term^2)
}

# Ψ_i(gamma_i) combining everything
psi_i <- function(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, lambda, a, p) {
    
  h_val <- h_func(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, lambda, a, p)
  
  # Φ_i
  Phi_i <- 0
  for (j in seq_along(mu)) {
    if (j != i) {
      d2h_j <- d2h_theta_k2(gamma_i, gamma, mu, sigma1, XtX, h_val, i, j, a)
      cov_jj <- gamma[j] * (1 - gamma[j]) * mu[j]^2 + gamma[j] * sigma1[j]^2
      Phi_i <- Phi_i + d2h_j * cov_jj
    }
  }
  
  # Ω_i
  cov_i_p1 <- gamma_i * (1 - gamma_i) * mu[i]
  Omega_i <- d2h_theta_i_zi(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, w_bar, h_val, a, i) * cov_i_p1
  
  # Γ_i
  var_zi <- gamma_i * (1 - gamma_i)
  Gamma_i <- d2h_zi2(gamma_i, mu, sigma1, lambda, w_bar, a, i) * var_zi
  
  # Λ_i
  var_theta_i <- gamma_i * (1 - gamma_i) * mu[i]^2 + gamma[i] * sigma1[i]^2
  Lambda_i <- d2h_theta_i2(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, h_val, a, i) * var_theta_i
  
  h_val + 0.5 * Phi_i + Omega_i + 0.5 * Gamma_i + 0.5 * Lambda_i
} 

#-------------------------------------------------------------------------------------------------------------------

# binary entropy 
entropy <- function(z){
    z[z == 0] <- 1e-10    # Replace 0 with a tiny positive number
    z[z == 1] <- 1 - 1e-10  # Replace 1 with just less than 1
    return (-z*log2(z) - (1-z)*log2(1-z))
}

# change in entropy 
delta <- function(gamma_old, gamma_new){
    return(max(abs(entropy(gamma_old) - entropy(gamma_new))))
}

# stopping criteria
eps <- 1e-7
max_iterations <- 100

# Extract dimensions
n <- nrow(X)
p <- ncol(X)

#Initialize a
a <- 0.5

# Initial estimator for mu using OLS MLE (replaced it since the other package doesn't work)
ols_fit <- lm(Y ~ X - 1)  
mu <- coef(ols_fit)

# Fix any NAs in mu (e.g., due to collinearity)
mu[is.na(mu)] <- 0

# Prioritize update order by absolute value of mu
abs_mu <- abs(mu)

# Use na.rm=TRUE just in case
threshold <- quantile(abs_mu, probs = 0.5, na.rm = TRUE)
update_order <- order(abs_mu, decreasing = TRUE)

# Initialize gamma vector: for example, set to 1 where mu is nonzero, else 0
gamma <- ifelse(mu != 0, 1, 0)
sigma1 = rep(1, p) 

# Initialize alpha and beta based on gamma counts
alpha <- sum(gamma)
beta <- (p - alpha +1) #I added 1 since there is a chance that it is equal to 1 which is bad!

# Compute XtX and YtX outside helpers for efficiency
XtX <- t(X) %*% X
YtX <- t(Y) %*% X

# Compute w_bar (prior weight parameter)
w_bar <- alpha / (alpha + beta)


rvi.fit <- function(X,
                    Y,
                    mu,
                    sigma1,
                    gamma,
                    alpha,
                    beta,
                    a,
                    prior_scale = 1) {
  k <- 1
  deltav <- 1
  gamma_old <- gamma
  
  while (k < max_iterations && deltav > eps) {
    gamma_old <- gamma  # update gamma_old before the loop
    
    for (i in update_order) {
      mu[i] <- optimize(f = function(mu_i) unname(kappa_mu(mu_i, X, Y, mu, sigma1, gamma, prior_scale, a, p, i)),interval = c(-10, 10))$minimum
      sigma1[i] <- optimize(f = function(sigma_i) unname(kappa_sigma(sigma_i, X, Y, mu, sigma1, gamma, prior_scale, a, p, i)),interval = c(1e-7, 10))$minimum
      gamma[i] <- optimize(f = function(gamma_i) unname(psi_i(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, prior_scale, a)),interval = c(1e-6, 0.99999))$minimum
    }
    k <- k + 1
    deltav <- delta(gamma_old, gamma)
  }
                           
  return(list(mu = mu, sigma1 = sigma1, gamma = gamma))
}

rvi.fit(X, Y, mu, sigma1, gamma, alpha, beta, 0.5, prior_scale = 1)

