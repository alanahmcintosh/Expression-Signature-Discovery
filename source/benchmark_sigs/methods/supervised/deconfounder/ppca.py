from __future__ import annotations

import numpy as np
import numpy.random as npr
from scipy import sparse, stats


# Define class for PPCA
class ppca:
    def __init__(self, factors=2, sigma2=None):
        """
        Principal Component Analysis with Probabilistic Principal Component Analysis (PPCA) class constructor.

        Args:
            factors (int, optional): Number of latent factors. Default is 2.
            sigma2 (float, optional): Noise variance. Default is None.

        Attributes:
            factors (int): Number of latent factors.
            ran (bool): Flag indicating whether the parameters have been generated.
            sigma2 (float): Noise variance.
            x (numpy.ndarray): Observed data matrix.
            z_mu (numpy.ndarray): Mean of the latent variables z.
            z_cov (numpy.ndarray): Covariance matrix of the latent variables z.
            W (numpy.ndarray): Weight matrix.
            means (numpy.ndarray): Mean of the observed data.
            n (int): Number of data points.
            D (int): Dimensionality of data.
            cov (numpy.ndarray): Covariance matrix of the observed data.
            cov_inv (numpy.ndarray): Inverse of the covariance matrix of the observed data.
        """
        self.factors = factors  # Number of latent factors
        self.ran = False  # Flag to indicate whether the parameters have been generated
        self.sigma2 = sigma2  # Noise Variance
        self.x = None
        self.z_mu = None
        self.z_cov = None
        self.W = None
        self.means = None
        self.n = None
        self.D = None
        self.cov = None
        self.cov_inv = None

    def generate(self, n, standardise=True):
        """
        Generate synthetic data based on the PPCA model.

        Args:
            n (int): Number of data points to generate.
            standardise (bool, optional): Flag indicating whether to standardize generated data. Default is True.

        Returns:
            tuple: A tuple containing generated data (gen_x) and latent variables (z).
        """
        if self.ran:  # Check if PPCA parameters have been learned using max_likelihood().
            # Initialize arrays for generated data (gen_x) and latent variables (z).
            gen_x = np.zeros((n * self.n, self.D))  # Placeholder for generated data
            z = np.zeros((n * self.n, self.factors))  # Placeholder for latent variables (z)
            jf = 0  # Counter to keep track of array positions

            # Loop through each data point to generate latent variables (z).
            for i in range(n):
                for j in range(self.n):
                    # Generate latent variables (z) from multivariate normal distribution
                    z[j + jf, :] = npr.multivariate_normal(self.z_mu[:, j + jf], self.z_cov)
                jf += self.n  # Increment counter

            # Calculate generated data (gen_x) by transforming latent variables using weights and adding means.
            gen_x = np.matmul(z, np.transpose(self.W)) + self.means

            # Return the generated data (gen_x) and the latent variables (z) as a tuple.
            return (gen_x, z)
        else:
            # Print a message indicating that max_likelihood() needs to be run before generating data.
            print("Use max_likelihood() to learn parameters first.")

    def max_likelihood(self, x, factors=None, standardise=True, mask=None):
        """
        Estimate PPCA model parameters using maximum likelihood.

        Args:
            x (numpy.ndarray): Observed data matrix.
            standardise (bool, optional): Flag indicating whether to standardize data. Default is True.
            mask (numpy.ndarray, optional): Mask for data. Default is None.

        Returns:
            None
        """
        x = np.array(x)
        self.n = x.shape[0]
        self.D = x.shape[1]
        self.x = x + 0.0

        # standardise x by subtracting the mean and dividing by the standard deviation
        if standardise:
            means = np.zeros((x.shape[1],))
            for i in range(x.shape[1]):
                means[i] = np.mean(x[:, i])
                self.x[:, i] = (x[:, i] - means[i]) / (np.var(x[:, i]) ** 0.5)
        else:
            # keep means and delete from x.
            means = np.zeros((x.shape[1],))
            for i in range(x.shape[1]):
                means[i] = np.mean(x[:, i])
                self.x[:, i] = x[:, i] - means[i]

        # Calculate S, the covariance matrix of the observed data
        if mask is None:
            S = 1 / self.n * np.matmul(np.transpose(self.x), self.x)
        else:
            mask = np.array(mask)  # ensure mask is a numpy array
            if mask.shape != x.shape:
                raise ValueError("Mask shape must match data shape.")
            # apply the mask
            masked_x = np.where(mask, x, 0)
            # calculate S using the masked data
            S = 1 / self.n * np.matmul(np.transpose(masked_x), masked_x)
            # restore the original x array
            self.x = x

        # do eigenvalue decomposition
        L, U = np.linalg.eigh(S)
        # sort eigenvalues in decreasing order
        idx_l = np.argsort(-L)
        U = U[:, idx_l]
        L = L[idx_l]

        if self.factors >= self.D:
            raise ValueError(
                f"PPCA requires factors < number of features; got factors={self.factors}, D={self.D}"
            )

        if self.sigma2 is None:
            # get sigma2 maximums likelihood estimation.
            self.sigma2 = 1 / (self.D - self.factors) * np.sum(L[self.factors :])

        # get W using the first factors eigenvectors of S
        U = U[:, 0 : self.factors]
        L_diag = np.diag(L[0 : self.factors])
        self.W = np.matmul(U, (L_diag - self.sigma2 * np.eye(self.factors)) ** 0.5)

        # calculate M, an auxiliary variable useful for the rest of calculations.
        self.M = np.matmul(np.transpose(self.W), self.W) + self.sigma2 * np.eye(self.factors)
        try:
            self.M_inv = np.linalg.pinv(self.M, rcond=1e-3)  # increase rcond value to improve convergence
        except Exception:
            print("SVD did not converge")
            return

        self.C = np.matmul(self.W, np.transpose(self.W)) + self.sigma2 * np.eye(self.D)  # covariance matrix
        self.C_inv = (
            self.sigma2 ** -1 * np.eye(self.D)
            - self.sigma2 * np.matmul(np.matmul(self.W, self.M_inv), np.transpose(self.W))
        )  # inverse of C
        self.z_mu = np.matmul(np.matmul(self.M_inv, np.transpose(self.W)), np.transpose(self.x))  # mean of z
        self.z_cov = self.sigma2 * self.M  # covariance matrix of z
        self.ran = True
        self.U = U  # eigenvectors sorted by eigenvalues
        self.L = L_diag
        self.means = means
        self.S = S  # covariance matrix of observed data

    def get_W_cov(self):
        """
        Compute and return the covariance matrix of the weight matrix W.

        Returns:
            W_cov (numpy.ndarray): The covariance matrix of the weight matrix W.
        """
        M_inv = self.M_inv  # Inverse of the auxiliary variable M
        sigma2 = self.sigma2  # Variance parameter sigma^2
        W_cov = sigma2 * M_inv  # Compute the covariance matrix of W
        return W_cov  # Return the computed covariance matrix

    def holdout(self, x, holdout_portion=0.2, n_rep=100, seed=None):
        """
        Generate holdout data and create a holdout mask for training and validation sets.

        Args:
            x (numpy.ndarray): Input data matrix.
            holdout_portion (float, optional): Proportion of data to be held out. Default is 0.2.
            n_rep (int, optional): Number of replicates for generating holdout data. Default is 100.
            seed (int, optional): Seed for random number generation. Default is None.

        Returns:
            None
        """
        if seed is not None:
            np.random.seed(seed)

        n, D = x.shape  # Get the dimensions of the input data
        n_holdout = int(holdout_portion * n * D)  # Calculate the number of holdout elements

        # Randomly censor some parts of x to create a holdout mask
        self.holdout_row = np.random.randint(n, size=n_holdout)  # Randomly select rows for holdout
        self.holdout_col = np.random.randint(D, size=n_holdout)  # Randomly select columns for holdout
        self.holdout_mask = (
            sparse.coo_matrix((np.ones(n_holdout), (self.holdout_row, self.holdout_col)), shape=x.shape)
        ).toarray()  # Generate a holdout mask

        holdout_subjects = np.unique(self.holdout_row)  # Get unique row indices for holdout
        x_train = np.multiply(1 - self.holdout_mask, x)  # Create training set by masking holdout elements

        # Store the training set without NaN values in x_train attribute
        self.x_train = x_train[np.logical_not(np.isnan(x_train))]

        a, b = x_train.shape  # Get the dimensions of the training set (kept for parity with original code)
        self.x_val = np.multiply(self.holdout_mask, x)  # Create validation set by applying the holdout mask

        # Initialize an array for generating holdout data during the predictive check
        self.holdout_gen = np.zeros((n_rep, *(x_train.shape)))


def predicitve_check(func, factors, holdout_portion=0.2, n_rep=100):
    """
    Perform a predictive check on the specified function.

    Args:
        func: An instance of the ppca class or a similar class with generate and holdout methods.
        factors (int): Number of latent factors.
        holdout_portion (float, optional): Proportion of data to be held out. Default is 0.2.
        n_rep (int, optional): Number of replicates for generating holdout data. Default is 100.

    Returns:
        overall_pval (float): Overall predictive check p-value.
    """
    gen_x, unk = func.generate(1)  # Generate a single data point using the specified function

    # Generate holdout data for each factor and store it in holdout_gen
    for i in range(factors):
        func.holdout_gen[i] = np.multiply(gen_x, func.holdout_mask)

    w_mean = np.mean(func.W.flatten())  # Mean of the weight matrix W
    z_mean = np.mean(func.z_mu.flatten())  # Mean of the latent variable z_mu

    W_cov = func.get_W_cov()  # Covariance matrix of the weight matrix W
    W_std = np.sqrt(np.diag(W_cov))  # Standard deviation of the weight matrix W

    z_covariance = func.z_cov  # Covariance matrix of the latent variable z_cov
    z_standard_deviation = np.sqrt(np.diag(z_covariance))  # Standard deviation of the latent variable z_cov

    n_eval = 100  # Number of samples drawn from inferred Z and W

    obs_ll = []  # List to store log-likelihood values for observed data
    rep_ll = []  # List to store log-likelihood values for replicated data

    # Generate log-likelihood values for observed and replicated data
    for j in range(n_eval):
        w_sample = npr.normal(w_mean, W_std)  # Sample from W distribution
        z_sample = npr.normal(z_mean, z_standard_deviation)  # Sample from latent variable z distribution

        # Compute holdoutmean_sample and log-likelihood for observed and replicated data
        holdoutmean_sample = np.multiply(z_sample.dot(w_sample), func.holdout_mask)
        obs_ll.append(np.mean(stats.norm(holdoutmean_sample, 0.1).logpdf(func.x_val), axis=1))
        rep_ll.append(np.mean(stats.norm(holdoutmean_sample, 0.1).logpdf(func.holdout_gen), axis=2))

    obs_ll_per_zi = np.mean(np.array(obs_ll), axis=0)  # Mean log-likelihood for observed data
    rep_ll_per_zi = np.mean(np.array(rep_ll), axis=0)  # Mean log-likelihood for replicated data

    num_datapoints, data_dim = func.x_train.shape  # Dimensions of the training data

    # Calculate p-values for each datapoint and compute overall p-value
    pvals = np.array([np.mean(rep_ll_per_zi[:, i] < obs_ll_per_zi[i]) for i in range(num_datapoints)])
    holdout_subjects = np.unique(func.holdout_row)  # Unique holdout row indices
    overall_pval = np.mean(pvals[holdout_subjects])  # Overall predictive check p-value
    return overall_pval
    # print(f'Predictive check using {factors}: p-value {overall_pval}')


def choose_latent_dim_ppca(
    X,
    k_range=range(2, 99),
    holdout_portion=0.2,
    pval_cutoff=0.1,
    seed=44,
):
    D=X.shape[1]

    valid_k = [k for k in k_range if 1 <= k < D]

    if len(valid_k) == 0:
        return 1

    for k in valid_k:
        model = ppca(factors=k)
        model.holdout(X, holdout_portion=holdout_portion, seed=seed)
        model.max_likelihood(model.x_train, standardise=False)
        pval = predicitve_check(model, k, holdout_portion=holdout_portion)

        if pval >= pval_cutoff:
            return k  # first valid k found

    # If none satisfy the cutoff, return max tested value
    return valid_k[-1]
