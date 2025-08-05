# ============================================================
# Set Working Directory to Existing DRI Folder
# ============================================================
save_dir <- "~/Desktop/DRI"   # <-- Change path if DRI is in another location
setwd(save_dir)
cat("Saving all results to:", getwd(), "\n")

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
eps_safe <- 1e-7
number_of_simulations <- 100

# ============================================================
# Simulation Function
# ============================================================
simulate <- function(n, p, s) {
  X <- matrix(rnorm(n * p), n, p)
  theta <- numeric(p)
  theta[sample.int(p, s)] <- runif(s, -3, 3)
  Y <- X %*% theta + rnorm(n)
  list(X = X, Y = Y, theta = theta)
}

# ============================================================
# Safe helper for sums (avoids empty indexing issues)
# ============================================================
safe_sum <- function(x) if (length(x) == 0) 0 else sum(x)

# ============================================================
# Helper Functions (Core Components)
# ============================================================
A_i_mu <- function(mu_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX) {
  sum_j <- if (p > 1) safe_sum(gamma[-i] * mu[-i] * XtX[-i, i]) else 0
  smoothed_grad <- mu_i / sqrt(mu_i^2 + eps_safe)
  term <- -YtX[i] + mu_i * XtX[i, i] + sum_j + lambda * smoothed_grad
  (term^2) * sigma1[i]^2
}

A_i_sigma <- function(sigma_i, X, Y, mu, sigma1, gamma, lambda, i, p, XtX, YtX) {
  sum_j <- if (p > 1) safe_sum(gamma[-i] * mu[-i] * XtX[-i, i]) else 0
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
  sum_k <- if (length(mu) > 1) safe_sum((XtX[-i, i]^2) *
                 (gamma[-i] * (1 - gamma[-i]) * mu[-i]^2 +
                  gamma[-i] * sigma1[-i]^2)) else 0
  mu_i^2 * sum_k
}

C_i_sigma <- function(mu, sigma1, gamma, i, XtX) {
  sum_k <- if (length(mu) > 1) safe_sum((XtX[-i, i]^2) *
                 (gamma[-i] * (1 - gamma[-i]) * mu[-i]^2 +
                  gamma[-i] * sigma1[-i]^2)) else 0
  mu[i]^2 * sum_k
}

g_func_mu <- function(mu_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- if (length(mu) > 1) safe_sum(gamma[-i] * mu[-i] * XtX[-i, i]) else 0
  exponent <- (a - 1) * (-YtX[i] * mu_i + 0.5 * mu_i^2 * XtX[i, i] +
                           mu_i * sum_j + lambda * sqrt(mu_i^2 + eps_safe) - log(sigma1[i]))
  exp(exponent)
}

g_func_sigma <- function(sigma_i, mu, sigma1, gamma, lambda, a, i, XtX, YtX) {
  sum_j <- if (length(mu) > 1) safe_sum(gamma[-i] * mu[-i] * XtX[-i, i]) else 0
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
  z <- pmin(pmax(z, 1e-10), 1 - 1e-10)
  -z * log2(z) - (1 - z) * log2(1 - z)
}

delta <- function(g_old, g_new) {
  max(abs(entropy(g_old) - entropy(g_new)))
}

# ============================================================
# CAVI Algorithm with Safe Checks
# ============================================================
rvi.fit <- function(X, Y, a, prior_scale = 1) {
  n <- nrow(X); p <- ncol(X)
  XtX <- t(X) %*% X
  YtX <- t(Y) %*% X
  eps <- 1e-7
  max_iterations <- 100

  # Initializations
  ridge_fit <- glmnet(X, Y, alpha = 0, lambda = 0.1, intercept = FALSE)
  mu <- as.vector(coef(ridge_fit))[-1]; mu[is.na(mu)] <- 0
  gamma <- ifelse(abs(mu) > 1, 1, 0)
  sigma1 <- rep(1, p)
  
  alpha_h <- sum(gamma)
  beta_h <- max(p - alpha_h, eps)
  
  update_order <- order(abs(mu), decreasing = TRUE)
  k <- 1; deltav <- 1
  
  while (k < max_iterations && deltav > eps) {
    gamma_old <- gamma
    
    for (i in update_order) {
      mu[i] <- tryCatch(
        optimize(f = function(m) kappa_mu(m, X, Y, mu, sigma1, gamma, prior_scale, a, p, i, XtX, YtX),
                 interval = c(-3, 3))$minimum, error = function(e) mu[i])
      
      sigma1[i] <- tryCatch(
        optimize(f = function(s) kappa_sigma(s, X, Y, mu, sigma1, gamma, prior_scale, a, p, i, XtX, YtX),
                 interval = c(1e-7, 100))$minimum, error = function(e) sigma1[i])
      
      # Safe Gamma update
      Gamma_i <- log((alpha_h + eps) / (beta_h + eps)) + log(sqrt(pi) * sigma1[i] * prior_scale / sqrt(2)) +
        YtX[i] * mu[i] - safe_sum((XtX[i, -i]) * gamma[-i] * mu[-i]) -
        0.5 * XtX[i, i] * (sigma1[i]^2 + mu[i]^2) -
        prior_scale * sigma1[i] * sqrt(2 / pi) * exp(-mu[i]^2 / (2 * sigma1[i]^2)) -
        prior_scale * mu[i] * (1 - 2 * pnorm(-mu[i] / sigma1[i])) + 0.5
      
      gamma[i] <- 1 / (1 + exp(-Gamma_i))
      if (!is.finite(gamma[i])) gamma[i] <- 0.5  # fallback
    }
    
    alpha_h <- sum(gamma); beta_h <- max(p - alpha_h, eps)
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
  
  TPR <- if (sum(pos_TR) > 0) sum((pos == 1) & (pos_TR == 1)) / sum(pos_TR) else 0
  FDR <- if (sum(pos) > 0) sum((pos == 1) & (pos_TR == 0)) / sum(pos) else 0
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
# Run Simulations with foreach for configs and a_values
# ============================================================
results <- foreach(config = configurations, .combine = bind_rows) %:%
  foreach(a = a_values, .combine = bind_rows) %dopar% {
    metrics_list <- foreach(sim_idx = 1:number_of_simulations, .combine = bind_rows,
                            .packages = c("dplyr", "glmnet")) %dopar% {
      tryCatch({
        sim <- simulate(config$n, config$p, config$s)
        fit <- suppressWarnings(rvi.fit(sim$X, sim$Y, a, prior_scale = 1))
        compute_metrics(fit$mu, fit$sigma1, fit$gamma, sim$theta, sim$X, sim$Y)
      }, error = function(e) {
        message(paste("Error in sim", sim_idx, "for config", config$name, "a=", a, ":", e$message))
        return(NULL)
      })
    }
    
    metrics_summary <- metrics_list %>%
      summarise(
        TPR_mean = mean(TPR, na.rm = TRUE),  TPR_sd = sd(TPR, na.rm = TRUE),
        FDR_mean = mean(FDR, na.rm = TRUE),  FDR_sd = sd(FDR, na.rm = TRUE),
        L2_mean  = mean(L2, na.rm = TRUE),   L2_sd  = sd(L2, na.rm = TRUE),
        MSPE_mean= mean(MSPE, na.rm = TRUE), MSPE_sd= sd(MSPE, na.rm = TRUE)
      ) %>%
      mutate(config = config$name, a = a,
             number_of_simulations = number_of_simulations)
    
    write.csv(metrics_summary, paste0("results_", a, config$name, ".csv"))
    metrics_summary
  }

write.csv(results, "DRI_results.csv")
