# ------------------------------------------------------------
# Coordinate Ascent Variational Inference (CAVI) Algorithm
# for Laplace Spike-and-Slab in High-Dimensional Linear Regression
# with Rényi Divergence
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

library(sparsevb)
library(glmnet)

### Simulate a linear regression problem ###
n <- 100
p <- 200
s <- 10

X <- matrix(rnorm(n*p), n, p) 
theta <- numeric(p) 
theta[sample.int(p, s)] <- runif(s, -3, 3) # non-zero coefficients
pos_TR <- as.numeric(theta != 0) 
Y <- X %*% theta + rnorm(n)

eps_safe <- 1e-7 

A_i_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  smoothed_grad <- mu_i / sqrt(mu_i^2 + eps_safe)
  term <- -YtX[i] + mu_i * XtX[i, i] + sum_j + lambda * smoothed_grad
  return((term^2) * sigma1[i]^2)
}

A_i_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  smoothed_grad <- mu[i] / sqrt(mu[i]^2 + eps_safe)
  term <- -YtX[i] + mu[i] * XtX[i, i] + sum_j + lambda * smoothed_grad
  return((term^2) * sigma_i^2)
}

B_i_mu <- function(mu_i, sigma1, lambda, i, XtX) {
  denom <- sqrt(mu_i^2 + eps_safe)
  second_deriv_term <- (1 / denom) - (mu_i^2) / (denom^3)
  return(XtX[i, i] * sigma1[i]^2 - 1 + lambda * sigma1[i]^2 * second_deriv_term)
}

B_i_sigma <- function(sigma_i, mu, lambda, i, XtX) {
  denom <- sqrt(mu[i]^2 + eps_safe)
  second_deriv_term <- (1 / denom) - (mu[i]^2) / (denom^3)
  return(XtX[i, i] * sigma_i^2 - 1 + lambda * sigma_i^2 * second_deriv_term)
}

C_i_mu <- function(mu_i, mu, sigma1, gamma, i, XtX) {
  sum_k <- sum((XtX[-i, i]^2) * (gamma[-i] * (1 - gamma[-i]) * mu[-i]^2 + gamma[-i] * sigma1[-i]^2))
  return(mu_i^2 * sum_k)
}

C_i_sigma <- function(mu, sigma1, gamma, i, XtX) {
  sum_k <- sum((XtX[-i, i]^2) * (gamma[-i] * (1 - gamma[-i]) * mu[-i]^2 + gamma[-i] * sigma1[-i]^2))
  return(mu[i]^2 * sum_k)
}

g_func_mu <- function(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  exponent <- (a-1) * (-YtX[i] * mu_i + 0.5 * mu_i^2 * XtX[i, i] + mu_i * sum_j +
                     lambda * sqrt(mu_i^2 + eps_safe) - log(sigma1[i]))
  return(exp(exponent))
}

g_func_sigma <- function(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  exponent <- (a-1) * (-YtX[i] * mu[i] + 0.5 * mu[i]^2 * XtX[i, i] + mu[i] * sum_j +
                     lambda * sqrt(mu[i]^2 + eps_safe) - log(sigma_i))
  return(exp(exponent))
}

kappa_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, a, p, i, XtX, YtX) {
  g_val <- g_func_mu(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX)
  Ai <- A_i_mu(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX)
  Bi <- B_i_mu(mu_i, sigma1, lambda, i, XtX)
  Ci <- C_i_mu(mu_i, mu, sigma1, gamma, i, XtX)
  return(g_val * (1 + ((a-1)^2 / 2) * Ai + ((a-1) / 2) * Bi + ((a-1)^2 / 2) * Ci))
}

kappa_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, a, p, i, XtX, YtX) {
  g_val <- g_func_sigma(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX)
  Ai <- A_i_sigma(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX)
  Bi <- B_i_sigma(sigma_i, mu, lambda, i, XtX)
  Ci <- C_i_sigma(mu, sigma1, gamma, i, XtX)
  return(g_val * (1 + ((a-1)^2 / 2) * Ai + ((a-1) / 2) * Bi + ((a-1)^2 / 2) * Ci))
}

# Entropy and delta ------------------------------------------------------
entropy <- function(z) { z <- pmin(pmax(z, 1e-10), 1-1e-10); -z*log2(z)-(1-z)*log2(1-z) }
delta <- function(g_old, g_new) max(abs(entropy(g_old)-entropy(g_new)))

# Initialization ---------------------------------------------------------
XtX <- t(X) %*% X
YtX <- t(Y) %*% X

# Ridge regression
ridge_fit <- glmnet(X, Y, alpha = 0, lambda = 0.1, intercept = FALSE) 
mu <- as.vector(coef(ridge_fit))[-1]
mu[is.na(mu)] <- 0

# Gamma and Sigma Initialization
gamma <- ifelse(abs(mu) > 1, 1, 0)
sigma1 <- rep(1, p)

#Hyperprior
alpha <- sum(gamma)             
beta <- p - alpha          
w_bar <- alpha / (alpha + beta)

# Prioritized update order
update_order <- order(abs(mu), decreasing=TRUE)

# CAVI Iteration ---------------------------------------------------------
rvi.fit <- function(X, Y, mu, sigma1, gamma, alpha, beta, a, prior_scale=1) {
  XtX <- t(X)%*%X; YtX <- t(Y)%*%X; w_bar <- alpha/(alpha+beta)
  eps <- 1e-7; max_iterations <- 1000
  k <- 1; deltav <- 1; gamma_old <- gamma
  while (k<max_iterations && deltav>eps) {
    gamma_old <- gamma
    for (i in update_order) {
      mu[i] <- optimize(f=function(m) kappa_mu(m,X,Y,mu,sigma1,gamma,prior_scale,a,p,i,XtX,YtX),
                        interval=c(-3,3))$minimum
      sigma1[i] <- optimize(f=function(s) kappa_sigma(s,X,Y,mu,sigma1,gamma,prior_scale,a,p,i,XtX,YtX),
                            interval=c(1e-7,100))$minimum
      Gamma_i <- log(alpha / beta) + log(sqrt(pi) * sigma1[i] * prior_scale / sqrt(2)) + YtX[i] * mu[i] - mu[i] * sum((XtX[i, -i]) * gamma[-i] * mu[-i]) -0.5 * XtX[i, i] * (sigma1[i]^2 + mu[i]^2) -
      prior_scale * sigma1[i] * sqrt(2 / pi) * exp(-mu[i]^2 / (2 * sigma1[i]^2)) - prior_scale * mu[i] * (1 - 2 * pnorm(-mu[i] / sigma1[i])) +0.5
      gamma[i] <- 1 / (1 + exp(-Gamma_i))  # logit inverse
    }
    alpha <- sum(gamma) # Empirical Bayes
    beta  <- p - alpha
    k <- k+1
    deltav <- delta(gamma_old,gamma)
  }
  return(list(mu=mu, sigma1=sigma1, gamma=gamma))
}

# Assessing Performance ---------------------------------------------------------
posterior_mean <- test$mu * test$gamma
pos <- as.numeric(test$gamma > 0.5) 
TPR <- sum((pos == 1) & (pos_TR == 1)) / sum(pos_TR)                 # True Positive Rate
FDR <- sum((pos == 1) & (pos_TR == 0)) / max(sum(pos), 1)            # False Discovery Rate
L2 <- sqrt(sum((posterior_mean - theta)^2))                          # L2 error
MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2) / n)                  # Prediction error
