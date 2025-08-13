# Bayesian Variable Selection with Rényi Divergence

This repository contains the implementation and evaluation of **Bayesian variable selection methods** for **sparse high-dimensional linear regression**. The focus is on **Laplace spike-and-slab priors** and **Rényi divergence–based variational inference**, with comparisons to other state-of-the-art Bayesian methods.

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

## Experimental Configurations

We evaluate methods across **four sparse high-dimensional regression setups**:

| Config ID | (n, p, s)        | Description |
|-----------|-----------------|-------------|
| (i)       | (100, 200, 10)  | Small sample, moderate dimension, 10 true signals |
| (ii)      | (400, 1000, 40) | Larger sample, high dimension, 40 true signals |
| (iii)     | (200, 800, 5)   | Moderate sample, high dimension, extremely sparse |
| (iv)      | (300, 450, 20)  | Moderate sample & dimension, 20 true signals |

Where:
- **n** = number of observations
- **p** = number of predictors
- **s** = number of nonzero coefficients

## α Values

We explore both **mass-covering** and **zero-forcing** regimes:

- **Mass-covering** (`α < 1`): `0.01, 0.10, 0.25, 0.50, 0.90`
- **Zero-forcing / standard KL limit** (`α ≥ 1`):
  - For **CAVI**: `1.01, 1.10, 1.20, 1.30, 1.50, 2, 3, 5, 100`
  - For **SVI**: same as CAVI, plus mass-covering set above

## Output Metrics

Each method produces `.csv` files summarizing performance over multiple simulation replications.  
Metrics include:

- **TPR** – True Positive Rate 
- **FDR** – False Discovery Rate
- **L2** – \(\ell_2\) error between estimated and true coefficients
- **MSPE** – Mean Squared Prediction Error
