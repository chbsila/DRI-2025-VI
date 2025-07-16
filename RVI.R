{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "a8741471-0ff3-4b0c-a804-3d2ea798ef7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ------------------------------------------------------------\n",
    "# Coordinate Ascent Variational Inference (CAVI) Algorithm\n",
    "# for Laplace Spike-and-Slab in High-Dimensional Linear Regression\n",
    "# with Rényi Divergence\n",
    "#\n",
    "# Authors: Chadi Bsila, Kevin Wang, Annie Tang, Laurie Heyer\n",
    "# Supported by: DRI 2025\n",
    "# ------------------------------------------------------------\n",
    "   \n",
    "### Simulate a linear regression problem of size n times p, with sparsity level s ###\n",
    "n <- 2\n",
    "p <- 5\n",
    "s <- 2\n",
    "\n",
    "#' ### Generate toy data ###\n",
    "X <- matrix(rnorm(n*p), n, p) \n",
    "theta <- numeric(p) \n",
    "theta[sample.int(p, s)] <- runif(s, -3, 3) #sample non-zero coefficients uar \n",
    "pos_TR <- as.numeric(theta != 0) \n",
    "Y <- X %*% theta + rnorm(n) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "37ad1603-82f0-4e90-980a-8bbca6495408",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "           [,1]      [,2]       [,3]       [,4]      [,5]\n",
      "[1,]  0.6469761 0.6372051 -0.1434706 -0.5986700 -2.114045\n",
      "[2,] -0.2330486 0.1159547 -1.5389011  0.3589094  0.634924\n",
      "[1] 0.9362627 0.0000000 0.7209698 0.0000000 0.0000000\n",
      "[1] 1 0 1 0 0\n",
      "          [,1]\n",
      "[1,]  2.702830\n",
      "[2,] -2.483392\n"
     ]
    }
   ],
   "source": [
    "print(X) #standard Gaussian design matrix\n",
    "print(theta) #vector of zeroes of length p\n",
    "print(pos_TR) #true positives, binary vector \n",
    "print(Y) #add standard Gaussian noise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "fee13da9-bc7f-46ca-b262-2f2580207b6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#' ### CAVI Implementation ###\n",
    "\n",
    "# binary entropy \n",
    "entropy <- function(z){\n",
    "    return (-z*log2(z) - (1-z)*log2(1-z))\n",
    "    }\n",
    "\n",
    "# change in entropy \n",
    "delta <- function(gamma_old, gamma_new){\n",
    "    return(max(abs(entropy(gamma_old) - entropy(gamma_new))))\n",
    "}\n",
    "\n",
    "# stopping criteria\n",
    "eps <- 1e-5\n",
    "max_iterations <- 10\n",
    "\n",
    "rvi.fit <- function(X,\n",
    "                    Y,\n",
    "                    mu,\n",
    "                    sigma = rep(1, ncol(X)),\n",
    "                    gamma,\n",
    "                    alpha,\n",
    "                    beta,\n",
    "                    prior_scale = 1) {\n",
    "\n",
    " # extracting dimensions\n",
    " n = nrow(X)\n",
    " p = ncol(X)\n",
    "\n",
    "#compute initial estimator for mu using MLE \n",
    "cvfit <- cv.glmnet(X, Y, family = \"gaussian\", alpha = 0)\n",
    "mu = as.numeric(coef(cvfit, s = \"lambda.min\"))\n",
    "\n",
    "#generate prioritized updating order\n",
    "update_order = order(abs(mu[1:p]), decreasing = TRUE)\n",
    "update_order = update_order - 1\n",
    "\n",
    "#compute initial estimators for alpha, beta, and gamma using LASSO regression\n",
    "cvfit <- cv.glmnet(X, Y, family = \"gaussian\", alpha = 1)\n",
    "s_hat = length(which(coef(cvfit, s = \"lambda.1se\")[-1] != 0))\n",
    "alpha = s_hat\n",
    "beta = p - s_hat\n",
    "gamma = rep(s_hat/p, p)\n",
    "gamma[which(coef(cvfit, s = \"lambda.1se\")[-1] != 0)] = 1\n",
    "\n",
    "# iteration loop\n",
    "k <- 1\n",
    "while (k < max_iterations && entropy(0.5) > eps){\n",
    "    while (i < p+1){\n",
    "        }\n",
    "    k <- k+1\n",
    "    }\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "869fb109-466d-434e-86b1-b1e1faa1ce4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#' ### Fitting Implementation ###\n",
    "#rvi.fit <- function(X,\n",
    "#                    Y,\n",
    "#                    mu,\n",
    "#                    sigma = rep(1, ncol(X)),\n",
    "#                   gamma,\n",
    "#                   alpha,\n",
    "#                   beta,\n",
    "#                   prior_scale = 1,\n",
    "#                   noise_sd,\n",
    "#                   max_iter = 1000,\n",
    "#                  tol = 1e-5) {\n",
    "#   n = nrow(X)\n",
    "#   p = ncol(X)\n",
    "#   cvfit = cv.glmnet(X, Y, family = \"linear\", \"gaussian\", \"binomial\"), intercept = intercept, alpha = 0) #\n",
    "#   mu = as.numeric(coef(cvfit, s = \"lambda.min\"))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "R 4.4",
   "language": "R",
   "name": "ir-4.4.2"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "4.4.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
