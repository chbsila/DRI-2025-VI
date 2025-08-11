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
  return((lambda / 2) * exp(-lambda * abs(x)))
}

# ============================================================
# Log Joint Density: log p(theta, z, Y)
# ============================================================
log_joint <- function(theta, z, Y, X, sigma2, lambda, w) {
  n <- length(Y)
  resid <- Y - X %*% theta
  log_lik <- -0.5 * n * (log(2 * pi) + log(sigma2^2)) - (t(resid) %*% resid) / (2 * sigma2^2)
  log_prior_theta <- sum(ifelse(z == 1, log(dlaplace(theta[z == 1], lambda)), 0))
  log_prior_z <- sum(z * log(w) + (1 - z) * log(1 - w))
  return(as.numeric(log_lik + log_prior_theta + log_prior_z))
}

# ============================================================
# Log Variational Density: log P_{\mu, \sigma, \gamma}(theta, z)
# ============================================================
log_variational <- function(theta, mu, sigma1, gamma) {
  log_q <- 0
  for (i in seq_along(theta)) {
    if (theta[i] == 0) {
      log_q <- log_q + log((1 - gamma[i]))
    } else {
      log_q <- log_q + log(gamma[i] * dnorm(theta[i], mu[i], sigma1[i]))
    }
  }
  return(log_q)
}

# ============================================================
# SVI Algorithm 
# ============================================================
svi.fit <- function(X, Y, a, prior_scale = 1.0, sigma2 = 1.0,
                    alpha_h = 1.0, beta_h = 1.0,
                    lr = 0.01, K = 100, max_iter = 100,
                    eps = 1e-7, verbose = TRUE) {
  n <- nrow(X)
  p <- ncol(X)

  # Ridge initialization
  ridge_fit <- glmnet(X, Y, alpha = 0, lambda = 0.1, intercept = FALSE)
  mu <- as.vector(coef(ridge_fit))[-1]; mu[is.na(mu)] <- 0
  gamma <- ifelse(abs(mu) > 0, 0.9, 0.1) 

  # Reparameterization
  tau <- qlogis(gamma)
  
  sigma1 <- rep(1, p)
  alpha_h <- sum(gamma)
  beta_h <- p - alpha_h
  w <- alpha_h / (alpha_h + beta_h)

  vr_bound_prev <- 0

  for (iter in 1:max_iter) {

    grad_mu <- grad_sigma <- grad_tau <- rep(0, p)
    log_ratios <- numeric(K)
    theta_samples <- vector("list", K)
    z_samples <- vector("list", K)

    gamma  <- plogis(tau)

    for (k in 1:K) {
      z_k <- rbinom(p, size = 1, prob = gamma)
      theta_k <- sapply(1:p, function(i) if (z_k[i] == 1) rnorm(1, mu[i], sigma1[i]) else 0)

      log_p <- log_joint(theta_k, z_k, Y, X, sigma2, prior_scale, w)
      log_q <- log_variational(theta_k, mu, sigma1, gamma)
      log_ratios[k] <- (log_p - log_q) * (1 - a)

      theta_samples[[k]] <- theta_k
      z_samples[[k]] <- z_k
    }

    # Normalized weights by dividing since exp(large number) = NaN
    log_ratios_normalized <- (log_ratios - max(log_ratios, na.rm = TRUE))
    weights <- exp(log_ratios_normalized)
    weights <- weights / sum(weights, na.rm = TRUE)

    # Gradients
    for (k in 1:K) {
      theta_k <- theta_samples[[k]]
      z_k <- z_samples[[k]]
      for (i in 1:p) {
        if (z_k[i] == 1) {
          diff <- theta_k[i] - mu[i]
          s1   <- sigma1[i]
          grad_mu[i]    <- grad_mu[i]    + weights[k] * (diff / (s1^2))
          grad_sigma[i] <- grad_sigma[i] + weights[k] * (((diff^2) - s1^2) / (s1^3))
        }
        grad_tau[i] <- grad_tau[i] + weights[k] *
        ((z_k[i] / gamma[i]) - ((1 - z_k[i]) / (1 - gamma[i]))) * gamma[i] * (1 - gamma[i])
      }
    }

    gamma_old <- gamma
                        
    # Updates
    mu     <- mu     + lr * grad_mu
    sigma1 <- sigma1 + lr * grad_sigma
    tau <- tau + lr * grad_tau

    # Bound
    vr_bound <- (1 / (1 - a)) * (log(mean(exp(log_ratios_normalized))) + max(log_ratios, na.rm = TRUE))
    if (!is.finite(vr_bound)) vr_bound <- vr_bound_prev

    gamma  <- plogis(tau)

    gamma_new <- gamma
                        
    # Entropy as stopping criterion
    if (delta(gamma_new, gamma_old) < eps) break
    vr_bound_prev <- vr_bound
 
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
  L2 <- sqrt(sum((posterior_mean - theta)^2))
  MSPE <- sqrt(sum((X %*% posterior_mean - Y)^2) / n)
  data.frame(TPR, FDR, L2, MSPE)
}

# ============================================================
# Experiment Configurations
# ============================================================
a_values <- c(0.01, 0.1, 0.25, 0.5, 0.77, 0.9, 0.99, 1.01, 1.1, 1.2, 1.3, 1.5, 2, 3, 5, 100)
configurations <- list(
  list(name = "(i)",  n = 100,  p = 200,   s = 10),
  list(name = "(ii)", n = 400,  p = 1000,  s = 40),
  list(name = "(iii)", n = 200, p = 800,   s = 5),
  list(name = "(iv)",  n = 300, p = 450,   s = 20)
)

# ============================================================
# Optimized Run: Shared Simulations Per Config
# ============================================================
results <- list()
tic("Total runtime")

for (config in configurations) {
  cat("\n=== Running configuration:", config$name, "===\n")
  
  # Generate simulations ONCE
  sims <- vector("list", number_of_simulations)
  for (i in 1:number_of_simulations) {
    sims[[i]] <- simulate(config$n, config$p, config$s)
  }
  
  for (a in a_values) {
    cat("\nAlpha:", a, "\n")
    pb <- progress_bar$new(total = number_of_simulations, format = "[:bar] :percent ETA: :eta")
    
    metrics_list <- foreach(sim_idx = 1:number_of_simulations, .combine = bind_rows) %do% {
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
      mutate(config = config$name, a = a, number_of_simulations = number_of_simulations)
    
    write.csv(metrics_summary, paste0("SVI_results_", a, "_", config$name, ".csv"))
    results <- append(results, list(metrics_summary))
  }
}

results <- bind_rows(results)
write.csv(results, "SVI_DRI_results.csv")
toc()











