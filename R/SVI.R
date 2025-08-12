# ------------------------------------------------------------
# Stochastic Variational Inference (SVI) Algorithm
# for Laplace Spike-and-Slab in High-Dimensional Linear Regression
# with Rényi Divergence and Monte Carlo Rényi Lower Bound
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

# ============================================================
# Set Working Directory
# ============================================================
save_dir <- "~/Desktop/Other Methods"   
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
library(progress)   # Progress bars
library(tictoc)     # Run-time profiling
library(stats)      # For dnorm, rnorm, dbinom

# ============================================================
# Global Constants
# ============================================================
number_of_simulations <- 100

# ============================================================
# Simulation Function
# ============================================================
simulate <- function(n, p, s) {
  X <- matrix(rnorm(n * p, mean = 0), n, p)
  theta <- numeric(p)
  theta[sample.int(p, s)] <- runif(s, -3, 3)
  Y <- X %*% theta + rnorm(n)
  list(X = X, Y = Y, theta = theta)
}

entropy <- function(z) {
  z <- pmin(pmax(z, 1e-10), 1 - 1e-10)
  -z * log2(z) - (1 - z) * log2(1 - z)
}

delta <- function(g_old, g_new) {
  max(abs(entropy(g_old) - entropy(g_new)))
}

# ============================================================
# Laplace Density
# ============================================================
dlaplace <- function(x, lambda) {
  (lambda / 2) * exp(-lambda * abs(x))
}

# ============================================================
# Log Joint Density
# ============================================================
log_joint <- function(theta, z, Y, X, sigma2, lambda, w) {
  n <- length(Y)
  resid <- Y - X %*% theta
  log_lik <- -0.5 * n * (log(2 * pi) + log(sigma2^2)) -
    (t(resid) %*% resid) / (2 * sigma2^2)
  log_prior_theta <- sum(ifelse(z == 1, log(dlaplace(theta[z == 1], lambda)), 0))
  log_prior_z <- sum(z * log(w) + (1 - z) * log(1 - w))
  as.numeric(log_lik + log_prior_theta + log_prior_z)
}

# ============================================================
# Log Variational Density
# ============================================================
log_variational <- function(theta, z, mu, sigma1, gamma) {
  s1 <- pmax(sigma1, 1e-12)
  g  <- pmin(pmax(gamma, 1e-12), 1 - 1e-12)

  # impossible: z=0 but theta!=0 (delta spike)
  if (any(z == 0 & theta != 0)) return(-Inf)

  # z=0 contributes log(1-g)
  term0 <- sum((z == 0) * log1p(-g))

  # z=1 contributes log g + log N
  idx1 <- which(z == 1L)
  term1 <- if (length(idx1)) {
    sum(log(g[idx1]) + dnorm(theta[idx1], mu[idx1], s1[idx1], log = TRUE))
  } else 0

  term0 + term1
}


# ============================================================
# SVI Algorithm
# ============================================================
svi.fit <- function(X, Y, a, prior_scale = 50, sigma2 = 1.0,
                    alpha_h = 1.0, beta_h = 1.0, eta_mu = 0.05, eta_sigma = 0.05,
                    eta_tau = 0.2, K = 100, max_iter = 500,
                    eps = 1e-6, verbose = TRUE, grad_clip = 5) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Ridge initialization
  ridge_fit <- glmnet(X, Y, alpha = 0, lambda = 0.1, intercept = FALSE)
  mu <- as.vector(coef(ridge_fit))[-1]; mu[is.na(mu)] <- 0
  
  gamma <- pmin(pmax(abs(mu) / max(abs(mu) + 1e-8), 0.05), 0.95)
  
  tau <- qlogis(gamma)
  log_sigma1 <- rep(0, p)
  
  alpha_h <- sum(gamma)
  beta_h  <- p - alpha_h
  w <- alpha_h / (alpha_h + beta_h)
  
  vr_bound_prev <- -Inf
  
  for (iter in 1:max_iter) {
    grad_mu <- grad_log_sigma <- grad_tau <- rep(0, p)
    log_ratios <- numeric(K)
    theta_samples <- vector("list", K)
    z_samples <- vector("list", K)
    
    gamma  <- plogis(tau)
    sigma1 <- exp(log_sigma1)
    
    for (k in 1:K) {
      z_k <- rbinom(p, size = 1, prob = gamma)
      theta_k <- ifelse(z_k == 1,
                        rnorm(p, mu, sigma1),
                        0)
      lp <- log_joint(theta_k, z_k, Y, X, sigma2, prior_scale, w)
      lq <- log_variational(theta_k, z_k, mu, sigma1, gamma)
      log_ratios[k] <- (lp - lq) * (1 - a)
      
      theta_samples[[k]] <- theta_k
      z_samples[[k]]     <- z_k
    }
    
    finite_mask <- is.finite(log_ratios)
    if (!any(finite_mask)) {
      warning("All log_ratios are non-finite; try smaller lr or K.")
      break
    }
    
    lr_fin <- log_ratios[finite_mask]
    m <- max(lr_fin)
    weights_fin <- exp(lr_fin - m)
    weights_fin <- weights_fin / sum(weights_fin)
    k_idx <- which(finite_mask)
    
    for (t in seq_along(k_idx)) {
      k <- k_idx[t]
      w_k <- weights_fin[t]
      theta_k <- theta_samples[[k]]
      z_k     <- z_samples[[k]]
      
      for (i in 1:p) {
        g_i <- pmin(pmax(gamma[i], 1e-12), 1 - 1e-12)
        if (!is.na(z_k[i]) && z_k[i] == 1) {
          diff <- theta_k[i] - mu[i]
          s1   <- sigma1[i]
          grad_mu[i]        <- grad_mu[i]        + w_k * (diff / (s1^2))
          grad_log_sigma[i] <- grad_log_sigma[i] + w_k * ((diff^2 - s1^2) / (s1^3)) * s1
        }
        grad_tau[i] <- grad_tau[i] + w_k *
          ((z_k[i] / g_i) - ((1 - z_k[i]) / (1 - g_i))) * g_i * (1 - g_i)
      }
    }
    
    gamma_old <- gamma
    
    mu          <- mu + eta_mu * grad_mu
    log_sigma1  <- log_sigma1 + eta_sigma * grad_log_sigma
    tau         <- tau + eta_tau * grad_tau
    
    vr_bound <- (1 / (1 - a)) * (log(mean(exp(lr_fin - m))) + m)
    if (!is.finite(vr_bound)) vr_bound <- vr_bound_prev
    
    gamma <- plogis(tau)
    gamma_diff <- max(abs(gamma - gamma_old))
    vr_change <- abs(vr_bound - vr_bound_prev)
    
    if ((gamma_diff < eps) || (vr_change < 1e-6)) break
    vr_bound_prev <- vr_bound
  }
  
  list(mu = mu, sigma1 = exp(log_sigma1), gamma = gamma)
}

# ============================================================
# Metrics Computation
# ============================================================
compute_metrics <- function(mu, sigma1, gamma, theta, X, Y) {
  n <- nrow(X)
  posterior_mean <- mu * gamma
  pos_TR <- as.numeric(theta != 0)
  pos <- as.numeric(gamma > 0.8)
  TPR <- sum((pos == 1) & (pos_TR == 1)) / sum(pos_TR)
  FDR <- sum((pos == 1) & (pos_TR == 0)) / max(sum(pos), 1)
  L2 <- sqrt(sum((posterior_mean - theta)^2))
  MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2) / n)
  data.frame(TPR, FDR, L2, MSPE)
}

# ============================================================
# Experiment Configurations
# ============================================================
a_values <- c(0.01, 0.1, 0.25, 0.5, 0.9, 1.01,
              1.1, 1.2, 1.3, 1.5, 2, 3, 5, 100)
configurations <- list(
  list(name = "(i)",  n = 100,  p = 200,   s = 10),
  list(name = "(ii)", n = 400,  p = 1000,  s = 40),
  list(name = "(iii)", n = 200, p = 800,   s = 5),
  list(name = "(iv)",  n = 300, p = 450,   s = 20)
)

# ============================================================
# Optimized Run
# ============================================================
results <- list()
tic("Total runtime")

for (config in configurations) {
  cat("\n=== Running configuration:", config$name, "===\n")
  
  sims <- vector("list", number_of_simulations)
  for (i in 1:number_of_simulations) {
    sims[[i]] <- simulate(config$n, config$p, config$s)
  }
  
  for (a in a_values) {
    cat("\nAlpha:", a, "\n")
    pb <- progress_bar$new(total = number_of_simulations,
                           format = "[:bar] :percent ETA: :eta")
    
    metrics_list <- foreach(sim_idx = 1:number_of_simulations,
                            .combine = bind_rows) %do% {
      pb$tick()
      sim <- sims[[sim_idx]]
      fit <- suppressWarnings(svi.fit(sim$X, sim$Y, a, prior_scale = 1))
      compute_metrics(fit$mu, fit$sigma1, fit$gamma, sim$theta, sim$X, sim$Y)
    }
    
    metrics_summary <- metrics_list %>%
      summarise(
        TPR_mean = mean(TPR, na.rm = TRUE), TPR_SD = sd(TPR, na.rm = TRUE),
        FDR_mean = mean(FDR, na.rm = TRUE), FDR_SD = sd(FDR, na.rm = TRUE),
        L2_mean  = mean(L2, na.rm = TRUE),  L2_SD  = sd(L2, na.rm = TRUE),
        MSPE_mean= mean(MSPE, na.rm = TRUE), MSPE_SD= sd(MSPE, na.rm = TRUE)
      ) %>%
      mutate(config = config$name, a = a,
             number_of_simulations = number_of_simulations)
    
    write.csv(metrics_summary,
              paste0("SVI_results_", a, "_", config$name, ".csv"))
    results <- append(results, list(metrics_summary))
  }
}

results <- bind_rows(results)
write.csv(results, "SVI_DRI_results.csv")
toc()














