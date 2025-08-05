
# ------------------------------------------------------------
# Metrics Summary for other Bayesian Methods (Excluding SVI): 
# SSLASSO, ebreg, sparsevb
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

# ============================================================
# Set Working Directory to Existing DRI Folder
# ============================================================
save_dir <- "~/Desktop/Other Methods"   
setwd(save_dir)
cat("Saving all results to:", getwd(), "\n")

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

# ============================================================
# Metrics Computation
# ============================================================
compute_metrics <- function(mu, gamma, theta, X, Y) {
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
configurations <- list(
  list(name = "(i)",    n = 100,  p = 200,   s = 10),
  list(name = "(ii)",   n = 400,  p = 1000,  s = 40),
  list(name = "(iii)",  n = 200,  p = 800,   s = 5),
  list(name = "(iv)",   n = 300,  p = 2000,  s = 20)
)

methods <- c("sparsevb", "SSLASSO", "ebreg")
results <- list()

# ============================================================
# Optimized Run: Shared Simulations Per Config
# ============================================================
tic("Total runtime")

for (config in configurations) {
  cat("\n=== Running configuration:", config$name, "===\n")
  
  # Generate simulations ONCE for this configuration
  sims <- vector("list", number_of_simulations)
  for (i in 1:number_of_simulations) {
    sims[[i]] <- simulate(config$n, config$p, config$s)
  }
  
  for (method in methods) {
    cat("\nMethod:", method, "\n")
    pb <- progress_bar$new(total = number_of_simulations, format = paste(method, " [:bar] :percent ETA: :eta"))
    
    metrics_list <- vector("list", number_of_simulations)
    
    for (sim_idx in 1:number_of_simulations) {
      pb$tick()
      sim <- sims[[sim_idx]]  # Reuse pre-generated simulation
      
      # ------------------------
      # Fit each method
      # ------------------------
      if (method == "sparsevb") {
        fit <- svb.fit(sim$X, sim$Y, family = "linear", slab = "laplace",
                       sigma = rep(1, ncol(sim$X)), prior_scale = 1)
        mu <- fit$mu
        gamma <- fit$gamma
        
      } else if (method == "SSLASSO") {
        fit <- SSLASSO(X = sim$X, y = sim$Y, family = "gaussian")
        mu <- as.vector(coef(fit)[-1])
        gamma <- ifelse(abs(mu) > 1e-6, 1, 0)
        
      } else if (method == "ebreg") {
        fit <- ebreg(X = sim$X, y = sim$Y)
        mu <- fit$PosteriorMean
        gamma <- ifelse(abs(mu) > 1e-6, 1, 0)
      }
      
      # ------------------------
      # Compute metrics
      # ------------------------
      metrics_list[[sim_idx]] <- compute_metrics(mu, gamma, sim$theta, sim$X, sim$Y)
    }
    
    # Combine metrics
    metrics_df <- bind_rows(metrics_list)
    
    # Median + IQR summary
    metrics_summary <- metrics_df %>%
      summarise(
        TPR_median = median(TPR, na.rm = TRUE), TPR_IQR = IQR(TPR, na.rm = TRUE),
        FDR_median = median(FDR, na.rm = TRUE), FDR_IQR = IQR(FDR, na.rm = TRUE),
        L2_median  = median(L2, na.rm = TRUE),  L2_IQR  = IQR(L2, na.rm = TRUE),
        MSPE_median= median(MSPE, na.rm = TRUE), MSPE_IQR= IQR(MSPE, na.rm = TRUE)
      ) %>%
      mutate(config = config$name, method = method,
             number_of_simulations = number_of_simulations)
    
    # Save individual method result
    write.csv(metrics_summary, paste0("results_", method, "_", config$name, ".csv"), row.names = FALSE)
    results <- append(results, list(metrics_summary))
  }
}

# Combine all summaries
results <- bind_rows(results)
write.csv(results, "DRI_results_other_methods_optimized.csv", row.names = FALSE)

toc()
