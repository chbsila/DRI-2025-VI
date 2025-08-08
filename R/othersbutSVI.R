# ------------------------------------------------------------
# Metrics Summary for other Bayesian Methods (Excluding SVI): 
# spikeslab, varbvs, sparsevb
#
# Authors: Chadi Bsila, Kevin Wang, Annie Tang
# Supported by: DRI 2025
# ------------------------------------------------------------

# =========================
# Working Directory
# =========================
save_dir <- "~/Desktop/Other Methods"
setwd(save_dir)
cat("Saving all results to:", getwd(), "\n")

# =========================
# Libraries
# =========================
library(sparsevb)
library(spikeslab)
library(glmnet)
library(tidyr)
library(dplyr)
library(varbvs)
library(progress)
library(tictoc)

# =========================
# Constants
# =========================
number_of_simulations <- 100

# =========================
# Simulation
# =========================
simulate <- function(n, p, s) {
  X <- matrix(rnorm(n * p), n, p)
  theta <- numeric(p); theta[sample.int(p, s)] <- runif(s, -3, 3)
  Y <- as.numeric(X %*% theta + rnorm(n))
  list(X = X, Y = Y, theta = theta)
}

# =========================
# Metrics
# =========================
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

# =========================
# Configs
# =========================
configurations <- list(
  list(name = "(i)",    n = 100,  p = 200,   s = 10),
  list(name = "(ii)",   n = 400,  p = 1000,  s = 40),
  list(name = "(iii)",  n = 200,  p = 800,   s = 5),
  list(name = "(iv)",   n = 300,  p = 450,   s = 20)
)

methods <- c("sparsevb", "spikeslab", "varbvs")
results <- list()

# =========================
# Run
# =========================
tic("Total runtime")

for (config in configurations) {
  cat("\n=== Running configuration:", config$name, "===\n")
  
  sims <- vector("list", number_of_simulations)
  for (i in 1:number_of_simulations) {
    sims[[i]] <- simulate(config$n, config$p, config$s)
  }
  
  for (method in methods) {
    cat("\nMethod:", method, "\n")
    pb <- progress_bar$new(total = number_of_simulations,
                           format = paste(method, " [:bar] :percent ETA: :eta"))
    
    metrics_list <- vector("list", number_of_simulations)
    
    for (sim_idx in 1:number_of_simulations) {
      pb$tick()
      sim <- sims[[sim_idx]]
      p <- ncol(sim$X); xnames <- colnames(sim$X)
      
      if (method == "sparsevb") {
        fit <- svb.fit(sim$X, sim$Y,
                       family = "linear", slab = "laplace",
                       sigma = rep(1, p), prior_scale = 1)
        
        mu <- as.numeric(fit$mu)
        gamma <- as.numeric(fit$gamma)
        
      } else if (method == "spikeslab") {
        
        fit <- spikeslab(x = sim$X,
                         y = sim$Y,
                         verbose = FALSE)
        mu <- fit$bma # Bayesian Model Averaging (BMA) 
        gamma <- as.numeric(abs(mu) > 1e-3)
      } 
      
      else if (method == "varbvs") {
        fit <- varbvs(X = sim$X, Z = NULL,
                         y = sim$Y, family = "gaussian", verbose = FALSE)
        mu <- fit$beta #Beta: "Averaged" posterior mean regression coefficients
        gamma <- fit$pip #Pip: "Averaged" posterior inclusion probabilities
        
      } 
      
      metrics_list[[sim_idx]] <- compute_metrics(mu, gamma, sim$theta, sim$X, sim$Y)
    }
    
    metrics_df <- bind_rows(metrics_list)
    
    metrics_summary <- metrics_df %>%
      summarise(
        TPR_mean = mean(TPR, na.rm = TRUE), TPR_SD = sd(TPR, na.rm = TRUE),
        FDR_mean = mean(FDR, na.rm = TRUE), FDR_SD = sd(FDR, na.rm = TRUE),
        L2_mean  = mean(L2, na.rm = TRUE),  L2_SD  = sd(L2, na.rm = TRUE),
        MSPE_mean= mean(MSPE, na.rm = TRUE), MSPE_SD= sd(MSPE, na.rm = TRUE)
      ) %>%
      mutate(config = config$name, method = method,
             number_of_simulations = number_of_simulations)
    
    write.csv(metrics_summary,
              paste0("results_", method, "_", config$name, ".csv"),
              row.names = FALSE)
    results <- append(results, list(metrics_summary))
  }
}

results <- bind_rows(results)
write.csv(results, "DRI_results_other_methods.csv", row.names = FALSE)

toc()
