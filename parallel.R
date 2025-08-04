# ------------------------------------------------------------
# Coordinate Ascent Variational Inference (CAVI) Algorithm
# for Laplace Spike-and-Slab in High-Dimensional Linear Regression
# with Rényi Divergence
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

# ============================================================
# Libraries
# ============================================================
library(ggplot2)
library(foreach)
library(glmnet)
library(tidyr)
library(dplyr)
library(doMC)
registerDoMC(cores = parallel::detectCores() - 1)

# ============================================================
# Global Constants
# ============================================================
eps_safe <- 1e-7       # For numerical stability in sqrt/log
number_of_simulations <- 100

# ============================================================
# Simulation Function
# ============================================================
simulate <- function(n, p, s) {
  X <- matrix(rnorm(n * p), n, p)
  theta <- numeric(p)
  theta[sample.int(p, s)] <- runif(s, -3, 3)  # Non-zero coefficients
  Y <- X %*% theta + rnorm(n)
  list(X = X, Y = Y, theta = theta)
}

# ============================================================
# Helper Functions (Core Components)
# ============================================================
A_i_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  smoothed_grad <- mu_i / sqrt(mu_i^2 + eps_safe)
  term <- -YtX[i] + mu_i * XtX[i, i] + sum_j + lambda * smoothed_grad
  (term^2) * sigma1[i]^2
}

A_i_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  smoothed_grad <- mu[i] / sqrt(mu[i]^2 + eps_safe)
  term <- -YtX[i] + mu[i] * XtX[i, i] + sum_j + lambda * smoothed_grad
  (term^2) * sigma_i^2
}

B_i_mu <- function(mu_i, sigma1, lambda, i, XtX) {
  denom <- sqrt(mu_i^2 + eps_safe)
  second_deriv <- (1 / denom) - (mu_i^2) / (denom^3)
  XtX[i, i] * sigma1[i]^2 - 1 + lambda * sigma1[i]^2 * second_deriv
}

B_i_sigma <- function(sigma_i, mu, lambda, i, XtX) {
  denom <- sqrt(mu[i]^2 + eps_safe)
  second_deriv <- (1 / denom) - (mu[i]^2) / (denom^3)
  XtX[i, i] * sigma_i^2 - 1 + lambda * sigma_i^2 * second_deriv
}

C_i_mu <- function(mu_i, mu, sigma1, gamma, i, XtX) {
  sum_k <- sum((XtX[-i, i]^2) *
                 (gamma[-i] * (1 - gamma[-i]) * mu[-i]^2 +
                    gamma[-i] * sigma1[-i]^2))
  mu_i^2 * sum_k
}

C_i_sigma <- function(mu, sigma1, gamma, i, XtX) {
  sum_k <- sum((XtX[-i, i]^2) *
                 (gamma[-i] * (1 - gamma[-i]) * mu[-i]^2 +
                    gamma[-i] * sigma1[-i]^2))
  mu[i]^2 * sum_k
}

g_func_mu <- function(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  exponent <- (a - 1) * (-YtX[i] * mu_i + 0.5 * mu_i^2 * XtX[i, i] +
                           mu_i * sum_j + lambda * sqrt(mu_i^2 + eps_safe) - log(sigma1[i]))
  exp(exponent)
}

g_func_sigma <- function(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- sum(gamma[-i] * mu[-i] * XtX[-i, i])
  exponent <- (a - 1) * (-YtX[i] * mu[i] + 0.5 * mu[i]^2 * XtX[i, i] +
                           mu[i] * sum_j + lambda * sqrt(mu[i]^2 + eps_safe) - log(sigma_i))
  exp(exponent)
}

kappa_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, a, p, i, XtX, YtX) {
  g_val <- g_func_mu(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX)
  Ai <- A_i_mu(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX)
  Bi <- B_i_mu(mu_i, sigma1, lambda, i, XtX)
  Ci <- C_i_mu(mu_i, mu, sigma1, gamma, i, XtX)
  g_val * (1 + ((a - 1)^2 / 2) * Ai + ((a - 1) / 2) * Bi + ((a - 1)^2 / 2) * Ci)
}

kappa_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, a, p, i, XtX, YtX) {
  g_val <- g_func_sigma(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX)
  Ai <- A_i_sigma(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX)
  Bi <- B_i_sigma(sigma_i, mu, lambda, i, XtX)
  Ci <- C_i_sigma(mu, sigma1, gamma, i, XtX)
  g_val * (1 + ((a - 1)^2 / 2) * Ai + ((a - 1) / 2) * Bi + ((a - 1)^2 / 2) * Ci)
}

# ============================================================
# Entropy and Delta
# ============================================================
entropy <- function(z) {
  z <- pmin(pmax(z, 1e-10), 1 - 1e-10)  # Clamp to avoid log(0)
  -z * log2(z) - (1 - z) * log2(1 - z)
}

delta <- function(g_old, g_new) {
  max(abs(entropy(g_old) - entropy(g_new)))
}

# ============================================================
# CAVI Algorithm
# ============================================================
rvi.fit <- function(X, Y, a, prior_scale = 1) {
  n <- nrow(X); p <- ncol(X)
  XtX <- t(X) %*% X
  YtX <- t(Y) %*% X
  eps <- 1e-7
  max_iterations <- 100
  
  # Initialization: Ridge regression
  ridge_fit <- glmnet(X, Y, alpha = 0, lambda = 0.1, intercept = FALSE)
  mu <- as.vector(coef(ridge_fit))[-1]; mu[is.na(mu)] <- 0
  gamma <- ifelse(abs(mu) > 1, 1, 0)
  sigma1 <- rep(1, p)
  
  alpha_h <- sum(gamma)
  beta_h <- p - alpha_h
  
  update_order <- order(abs(mu), decreasing = TRUE)
  k <- 1; deltav <- 1
  
  while (k < max_iterations && deltav > eps) {
    gamma_old <- gamma
    
    for (i in update_order) {
      mu[i] <- optimize(
        f = function(m) kappa_mu(m, X, Y, mu, sigma1, gamma, prior_scale, a, p, i, XtX, YtX),
        interval = c(-3, 3))$minimum
      
      sigma1[i] <- optimize(
        f = function(s) kappa_sigma(s, X, Y, mu, sigma1, gamma, prior_scale, a, p, i, XtX, YtX),
        interval = c(1e-7, 100))$minimum
      
      Gamma_i <- log(alpha_h / beta_h) + log(sqrt(pi) * sigma1[i] * prior_scale / sqrt(2)) +
        YtX[i] * mu[i] - mu[i] * sum((XtX[i, -i]) * gamma[-i] * mu[-i]) -
        0.5 * XtX[i, i] * (sigma1[i]^2 + mu[i]^2) -
        prior_scale * sigma1[i] * sqrt(2 / pi) * exp(-mu[i]^2 / (2 * sigma1[i]^2)) -
        prior_scale * mu[i] * (1 - 2 * pnorm(-mu[i] / sigma1[i])) + 0.5
      
      gamma[i] <- 1 / (1 + exp(-Gamma_i))
    }
    
    alpha_h <- sum(gamma); beta_h <- p - alpha_h
    k <- k + 1
    deltav <- delta(gamma_old, gamma)
  }
  
  list(mu = mu, sigma1 = sigma1, gamma = gamma)
}

# ============================================================
# Metrics Computation
# ============================================================
compute_metrics <- function(mu, sigma1, gamma, theta, X, Y) {
  n <- nrow(X)
  posterior_mean <- mu * gamma
  pos_TR <- as.numeric(theta != 0)
  pos <- as.numeric(gamma > 0.5)
  
  TPR <- sum((pos == 1) & (pos_TR == 1)) / sum(pos_TR)
  FDR <- sum((pos == 1) & (pos_TR == 0)) / max(sum(pos), 1)
  L2  <- sqrt(sum((posterior_mean - theta)^2))
  MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2) / n)
  
  data.frame(TPR, FDR, L2, MSPE)
}

# ============================================================
# Experiment Configurations
# ============================================================
a_values <- c(1.1, 1.2, 1.3, 1.5, 2, 3, 5, 10)
configurations <- list(
  list(name = "(i)",   n = 100, p = 200,  s = 10),
  list(name = "(ii)",  n = 400, p = 1000, s = 40),
  list(name = "(iii)", n = 200, p = 800,  s = 5)
)

# ============================================================
# Run Simulations
# ============================================================
results <- data.frame()

for (config in configurations) {
  for (a in a_values) {
    
    metrics_list <- foreach(sim_idx = 1:number_of_simulations, .combine = bind_rows,
                            .packages = c("dplyr", "glmnet")) %dopar% {
      sim <- simulate(config$n, config$p, config$s)
      fit <- suppressWarnings(rvi.fit(sim$X, sim$Y, a, prior_scale = 1))
      compute_metrics(fit$mu, fit$sigma1, fit$gamma, sim$theta, sim$X, sim$Y)
    }
    
    metrics_summary <- metrics_list %>%
      summarise(
        TPR_mean = mean(TPR),  TPR_sd = sd(TPR),
        FDR_mean = mean(FDR),  FDR_sd = sd(FDR),
        L2_mean  = mean(L2),   L2_sd  = sd(L2),
        MSPE_mean= mean(MSPE), MSPE_sd= sd(MSPE)
      ) %>%
      mutate(config = config$name, a = a,
             number_of_simulations = number_of_simulations)
    
    write.csv(metrics_summary, paste0("results_", a, config, ".csv"))
    results <- bind_rows(results, metrics_summary)
  }
}

write.csv(results, "DRI_results.csv")

# ============================================================
# Plotting: Mean ± SD
# ============================================================
results_long <- results %>%
  pivot_longer(cols = c("TPR_mean", "FDR_mean", "L2_mean", "MSPE_mean",
                        "TPR_sd", "FDR_sd", "L2_sd", "MSPE_sd"),
               names_to = c("Metric", "Stat"), names_sep = "_") %>%
  pivot_wider(names_from = Stat, values_from = value)

ggplot(results_long, aes(x = a, y = mean, color = config, group = config)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.1) +
  facet_wrap(~ Metric, scales = "free_y") +
  labs(
    title = paste("Performance Metrics by Configuration (", number_of_simulations, " sims)", sep = ""),
    x = "Alpha (a)",
    y = "Metric Value",
    color = "Configuration"
  ) +
  theme_minimal(base_size = 14)
