# ------------------------------------------------------------
# Coordinate Ascent Variational Inference (CAVI) Algorithm
# for Laplace Spike-and-Slab in High-Dimensional Linear Regression
# with Rényi Divergence
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

library(sparsevb)
library(pROC)

set.seed(123) 

### Simulate a linear regression problem ###
n <- 100
p <- 300
s <- 20

X <- matrix(rnorm(n*p), n, p) 
theta <- numeric(p) 
theta[sample.int(p, s)] <- runif(s, -3, 3) # non-zero coefficients
pos_TR <- as.numeric(theta != 0) 
Y <- X %*% theta + rnorm(n)

eps_safe <- 1e-7 

# Helper Functions for A, B, C terms ------------------------------------------------------

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

# g(·) and κ(·) functions ------------------------------------------------------

g_func_mu <- function(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  exponent <- a * (-YtX[i] * mu_i + 0.5 * mu_i^2 * XtX[i, i] + mu_i * sum_j +
                     lambda * sqrt(mu_i^2 + eps_safe) - log(sigma1[i]))
  return(exp(exponent))
}

g_func_sigma <- function(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  exponent <- a * (-YtX[i] * mu[i] + 0.5 * mu[i]^2 * XtX[i, i] + mu[i] * sum_j +
                     lambda * sqrt(mu[i]^2 + eps_safe) - log(sigma_i))
  return(exp(exponent))
}

kappa_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, a, p, i, XtX, YtX) {
  g_val <- g_func_mu(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX)
  Ai <- A_i_mu(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX)
  Bi <- B_i_mu(mu_i, sigma1, lambda, i, XtX)
  Ci <- C_i_mu(mu_i, mu, sigma1, gamma, i, XtX)
  return(g_val * (1 + (a^2 / 2) * Ai + (a / 2) * Bi + (a^2 / 2) * Ci))
}

kappa_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, a, p, i, XtX, YtX) {
  g_val <- g_func_sigma(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX)
  Ai <- A_i_sigma(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX)
  Bi <- B_i_sigma(sigma_i, mu, lambda, i, XtX)
  Ci <- C_i_sigma(mu, sigma1, gamma, i, XtX)
  return(g_val * (1 + (a^2 / 2) * Ai + (a / 2) * Bi + (a^2 / 2) * Ci))
}

# h(·), Ψ(·) and derivatives ------------------------------------------------------

h_func <- function(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, lambda, a) {
  diff <- gamma_i * mu[i] - mu[i]
  term1 <- log(sqrt(2) / (sqrt(pi) * sigma1[i] * lambda))
  term2 <- - (diff^2) / (2 * sigma1[i]^2)
  term3 <- lambda * sqrt((gamma_i * mu[i])^2 + eps_safe)
  term4 <- log(gamma_i) - log(w_bar)
  term5 <- log(1 - gamma_i) - log(1 - w_bar)
  sum_j <- sum(XtX[-i, i] * gamma[-i] * mu[-i])
  exponent <- a * (gamma_i * (term1 + term2 + term3 + term4) + 
                   (1 - gamma_i) * term5 -
                   YtX[i] * gamma_i * mu[i] +
                   0.5 * (gamma_i * mu[i])^2 * XtX[i, i] +
                   (gamma_i * mu[i]) * sum_j)
  return(exp(exponent))
}

# Derivatives for Ψ_i
d2h_theta_k2 <- function(gamma_i, mu, XtX, h_val, i, k, a) {
  (a^2) * (gamma_i^2) * (mu[i]^2) * (XtX[k, i]^2) * h_val
}

d2h_theta_i2 <- function(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, h_val, a, i) {
  diff <- (gamma_i * mu[i] - mu[i]) / sigma1[i]^2
  smooth_term <- lambda * (1/sqrt(mu[i]^2 + eps_safe) - mu[i]^2/(mu[i]^2 + eps_safe)^(3/2))
  inner_sum <- sum(mu[-i] * gamma[-i] * XtX[-i, i])
  square_term <- gamma_i * (-diff + lambda * mu[i]/sqrt(mu[i]^2 + eps_safe)) - YtX[i] +
                 gamma_i * mu[i] * XtX[i, i] + inner_sum
  return((a * (-gamma_i / sigma1[i]^2 + XtX[i, i] + smooth_term) + (a^2) * square_term^2) * h_val)
}

d2h_theta_i_zi <- function(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, w_bar, h_val, a, i) {
  log_term <- log(sqrt(2)/(sqrt(pi)*sigma1[i]*lambda)) -
    ((gamma_i * mu[i] - mu[i])^2)/(2*sigma1[i]^2) + lambda*sqrt((gamma_i*mu[i])^2+eps_safe) +
    log(gamma_i) - log(w_bar) - log(1-gamma_i) + log(1-w_bar)
  diff <- (gamma_i*mu[i]-mu[i]) / sigma1[i]^2
  inner_sum <- sum(mu[-i]*gamma[-i]*XtX[-i,i])
  first_bracket <- gamma_i*(-diff + lambda*mu[i]/sqrt(mu[i]^2+eps_safe)) - YtX[i] +
                   gamma_i*mu[i]*XtX[i,i] + inner_sum
  return(a^2 * (log_term * first_bracket + (-diff + lambda*mu[i]/sqrt(mu[i]^2+eps_safe))) * h_val)
}

d2h_zi2 <- function(gamma_i, mu, sigma1, lambda, w_bar, a, i) {
  log_term <- log(sqrt(2)/(sqrt(pi)*sigma1[i]*lambda)) -
    ((gamma_i * mu[i]-mu[i])^2)/(2*sigma1[i]^2) + lambda*sqrt((gamma_i*mu[i])^2+eps_safe) +
    log(gamma_i) - log(w_bar) - log(1-gamma_i) + log(1-w_bar)
  return((a^2) * (log_term^2))
}

psi_i <- function(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, lambda, a) {
  h_val <- h_func(gamma_i, gamma, mu, sigma1, w_bar, XtX, YtX, i, lambda, a)
  Phi_i <- sum(sapply(setdiff(1:length(mu), i), function(j) {
    d2h_theta_k2(gamma_i, mu, XtX, h_val, i, j, a) *
      (gamma[j]*(1-gamma[j])*mu[j]^2 + gamma[j]*sigma1[j]^2)
  }))
  Omega_i <- d2h_theta_i_zi(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, w_bar, h_val, a, i) *
    (gamma_i * (1-gamma_i) * mu[i])
  Gamma_i <- d2h_zi2(gamma_i, mu, sigma1, lambda, w_bar, a, i) * (gamma_i * (1-gamma_i))
  Lambda_i <- d2h_theta_i2(gamma_i, gamma, mu, sigma1, lambda, XtX, YtX, h_val, a, i) *
    (gamma_i*(1-gamma_i)*mu[i]^2 + gamma_i*sigma1[i]^2)
  return(h_val + 0.5*Phi_i + Omega_i + 0.5*Gamma_i + 0.5*Lambda_i)
}

# Entropy and delta ------------------------------------------------------
entropy <- function(z) { z <- pmin(pmax(z, 1e-10), 1-1e-10); -z*log2(z)-(1-z)*log2(1-z) }
delta <- function(g_old, g_new) max(abs(entropy(g_old)-entropy(g_new)))

# Initialization ---------------------------------------------------------
XtX <- t(X) %*% X
YtX <- t(Y) %*% X
ols_fit <- lm(Y ~ X - 1)
mu <- coef(ols_fit); mu[is.na(mu)] <- 0
gamma <- ifelse(mu!=0, 1, 0)
sigma1 <- rep(1, p)
alpha <- sum(gamma); beta <- (p - alpha + 1)
w_bar <- alpha / (alpha + beta)
update_order <- order(abs(mu), decreasing=TRUE)

# CAVI Iteration ---------------------------------------------------------
rvi.fit <- function(X, Y, mu, sigma1, gamma, alpha, beta, a, prior_scale=1) {
  XtX <- t(X)%*%X; YtX <- t(Y)%*%X; w_bar <- alpha/(alpha+beta)
  eps <- 1e-7; max_iterations <- 100
  k <- 1; deltav <- 1; gamma_old <- gamma
  while (k<max_iterations && deltav>eps) {
    gamma_old <- gamma
    for (i in update_order) {
      mu[i] <- optimize(f=function(m) -log(kappa_mu(m,X,Y,mu,sigma1,gamma,prior_scale,a,p,i,XtX,YtX)),
                        interval=c(-10,10))$minimum
      sigma1[i] <- optimize(f=function(s) -log(kappa_sigma(s,X,Y,mu,sigma1,gamma,prior_scale,a,p,i,XtX,YtX)),
                            interval=c(1e-7,10))$minimum
      gamma[i] <- optimize(f=function(g) -log(psi_i(g,gamma,mu,sigma1,w_bar,XtX,YtX,i,prior_scale,a)),
                           interval=c(1e-6,0.99999))$minimum
    }
    k <- k+1
    deltav <- delta(gamma_old,gamma)
  }
  return(list(mu=mu, sigma1=sigma1, gamma=gamma))
}

#Implementing Ray and Szabo CAVI

### Run the algorithm in linear mode with Laplace prior and prioritized initialization ### 

test <- svb.fit( X,
      Y,
      family = "linear",
      slab = "laplace",
      max_iter = 10,
      tol = 1e-05
)
posterior_mean <- test$mu * test$gamma #approximate posterior mean
pos <- as.numeric(test$gamma > 0.5) #signals
TPR <- sum(pos[which(pos_TR == 1)])/sum(pos_TR) #True positive rate
print(TPR)
FDR <- sum(pos[which(pos_TR != 1)])/max(sum(pos), 1) #False discovery rate
print(FDR)
L2 <- sqrt(sum((posterior_mean - theta)^2)) #L_2-error
print(L2)
MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2)/n) #Mean squared prediction error
print(MSPE)

### Assess the quality of the posterior estimates for different alpha values ###
 
a_values <- c(1.5, 2, 3, 5, 100)
results <- data.frame(a = a_values, TPR = NA, FDR = NA, L2 = NA, MSPE = NA)

for (j in seq_along(a_values)) {
  a_val <- a_values[j]
  test <- suppressWarnings(rvi.fit(X, Y, mu, sigma1, gamma, alpha, beta, a_val, prior_scale = 1))
  
  posterior_mean <- test$mu * test$gamma
  pos <- as.numeric(test$gamma > 0.5)  # 0.5 threshold: we might change this, depending on ROC
  TPR <- sum(pos[pos_TR == 1]) / sum(pos_TR)  # True Positive Rate
  FDR <- sum(pos[pos_TR != 1]) / max(sum(pos), 1)  # False Discovery Rate
  L2 <- sqrt(sum((posterior_mean - theta)^2))  # L2 error
  MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2) / n)  # Prediction error
  results[j, ] <- c(a_val, TPR, FDR, L2, MSPE)
}
print(results)

### Ranking the alpha values ###

rank_results <- results
rank_results$TPR_rank  <- rank(-rank_results$TPR)       # descending because higher TPR = Good
rank_results$FDR_rank  <- rank(rank_results$FDR)        # ascending because lower FDR = Good
rank_results$L2_rank   <- rank(rank_results$L2)         # ascending because lower L2 = Good
rank_results$MSPE_rank <- rank(rank_results$MSPE)       # ascending because lower MSPE = Good

# Average of ranks
rank_results$Composite <- rowMeans(rank_results[, c("TPR_rank", "FDR_rank", "L2_rank", "MSPE_rank")])
ranked <- rank_results[order(rank_results$Composite), ]

print(ranked)
cat("Best alpha value based on our weird ranking is:", ranked$a[1])

