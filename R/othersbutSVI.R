# ------------------------------------------------------------
# Metrics Summary for other Bayesian Methods (Excluding SVI): 
# SSLASSO, ebreg, sparsevb
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

# ============================================================
# Libraries
# ============================================================
library(sparsevb)
library(SSLASSO)
library(glmnet)
library(tidyr)
library(dplyr)
library(ebreg)
library(progress)   # For progress bars
library(tictoc)     # For profiling runtime

# ============================================================
# Global Constants
# ============================================================
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
a_values <- c(0.1, 0.2, 0.25, 0.5, 0.9, 1.1, 1.2, 1.3, 1.5, 2, 3, 5, 10)
configurations <- list(
  list(name = "(i)",    n = 100,  p = 200,   s = 10),
  list(name = "(ii)",   n = 400,  p = 1000,  s = 40),
  list(name = "(iii)",  n = 200,  p = 800,   s = 5),
  list(name = "(iv)",   n = 300,  p = 2000,  s = 20)
)

# ============================================================
# Parallel Run with Progress Bar + Median Metrics
# ============================================================
results <- list()
tic("Total runtime")  # Start profiling total time

m <- 0

for (config in configurations) {
  while (m < number_of_simulations){
    sim <- simulate(config$n, config$p, config$s)
    for (a in a_values) {
      cat("\nRunning:", config$name, "| Alpha:", a, "\n")
      pb <- progress_bar$new(total = number_of_simulations, format = "[:bar] :percent ETA: :eta")
      metrics_list <- foreach(sim_idx = 1:number_of_simulations, .combine = bind_rows) %do% {
      pb$tick()
      sim <- simulate(config$n, config$p, config$s)
      fit <- suppressWarnings(svb.fit(sim$X, sim$Y, family="linear", slab = "laplace", sigma= rep(1,ncol(sim$X)), prior_scale = 1)) #sparsevb fit
      # try fit with other methods
      # run compute metrics on each method fit and this simulation
      compute_metrics(fit$mu, fit$sigma1, fit$gamma, sim$theta, sim$X, sim$Y)
      }
      #concatenate things that go together
    
  }
    
    metrics_summary <- metrics_list %>%
      summarise(
        TPR_median = median(TPR, na.rm = TRUE), TPR_IQR = IQR(TPR, na.rm = TRUE),
        FDR_median = median(FDR, na.rm = TRUE), FDR_IQR = IQR(FDR, na.rm = TRUE),
        L2_median  = median(L2, na.rm = TRUE),  L2_IQR  = IQR(L2, na.rm = TRUE),
        MSPE_median= median(MSPE, na.rm = TRUE), MSPE_IQR= IQR(MSPE, na.rm = TRUE)
      ) %>%
      mutate(config = config$name, a = a, number_of_simulations = number_of_simulations)
    
    write.csv(metrics_summary, paste0("results_", a, "_", config$name, ".csv"))
    results <- append(results, list(metrics_summary))
  }
}

results <- bind_rows(results)
write.csv(results, "DRI_results.csv")
toc()  # End profiling total time
