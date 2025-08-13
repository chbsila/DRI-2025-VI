# Bayesian Variable Selection with Rényi Divergence

This repository contains the implementation and evaluation of **Bayesian variable selection methods** for **sparse high-dimensional linear regression**.  
The focus is on **Laplace spike-and-slab priors** and **Rényi divergence–based variational inference**, with comparisons to other state-of-the-art Bayesian methods.

The work supports the [DRI 2025] project by Chadi Bsila, Kevin Wang, and Annie Tang.

## Implemented Methods

### 1. Stochastic Variational Inference (SVI)
- File: `R/SVI.R`
- Implements **mean-field variational Bayes** with **Laplace spike-and-slab priors**.
- Uses **Rényi’s α-divergence** instead of the standard KL divergence.
- Supports:
  - Monte Carlo Rényi lower bound
  - Multiple α values
  - Parallel execution

### 2. Coordinate Ascent Variational Inference (CAVI)
- File: `R/alphasparsevb.R`
- Closed-form coordinate updates using **second-order delta method** approximations.
- Efficient computation with **pre-computed XtX/YtX** matrices.
- Parallelized over simulation replications.

### 3. Other Bayesian Methods
- File: `R/othersbutSVI.R`
- Benchmarks against:
  - [`sparsevb`](https://cran.r-project.org/web/packages/sparsevb/index.html)
  - [`spikeslab`](https://cran.r-project.org/web/packages/spikeslab/index.html)
  - [`varbvs`](https://cran.r-project.org/web/packages/varbvs/index.html)
- Produces comparable metrics to SVI and CAVI runs.


## Output Metrics

Each method produces `.csv` files summarizing performance over multiple simulation replications.  
Metrics include:

- **TPR** – True Positive Rate 
- **FDR** – False Discovery Rate
- **L2** – \ell_2 error between estimated and true coefficients
- **MSPE** – Mean Squared Prediction Error
