import numpy as np
import pandas as pd

def sim_mut(mutation_probabilities, n_samples):
    """
    Simulate binary mutation matrix from per-gene mutation probabilities.

    Parameters:
    -----------
    mutation_probabilities : list or np.ndarray
        Length-G array, where each value is the probability of mutation for one gene.
        (G = number of genes/alterations)
    n_samples : int
        Number of synthetic samples to generate.

    Returns:
    --------
    sim_mutations : np.ndarray
        Binary mutation matrix of shape (n_samples × G),
        where 1 = mutated, 0 = wild-type.
    """
    sim_mutations = np.zeros((n_samples, len(mutation_probabilities)))

    # For each gene, simulate mutation presence across samples
    for i in range(len(mutation_probabilities)):
        sim_mutations[:, i] = np.random.binomial(
            n=1,
            p=mutation_probabilities[i],
            size=n_samples
        )

    return sim_mutations


def sim_cna(cna_lambdas, n_samples):
    """
    Simulate CNA-like values from Poisson distributions.

    Parameters:
    -----------
    cna_lambdas : list or np.ndarray
        Length-G array where each value is the expected CNA count (lambda) for one gene.
        These represent average CNA signal magnitude across samples.
    n_samples : int
        Number of synthetic samples to generate.

    Returns:
    --------
    sim_cna : np.ndarray
        Matrix of simulated CNA values (n_samples × G),
        where values are integers >= 0 (e.g., copy number intensity).
    """
    sim_cna = np.zeros((n_samples, len(cna_lambdas)), dtype=int)

    # For each gene, simulate CNA values using a Poisson distribution
    for i in range(len(cna_lambdas)):
        sim_cna[:, i] = np.random.poisson(
            lam=cna_lambdas[i],
            size=n_samples
        )

    return sim_cna

def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=None):
    """
    Estimate DESeq2-style parameters from real RNA-seq data.

    Parameters:
    -----------
    rna_df : pd.DataFrame
        RNA-seq count matrix (samples × genes), untransformed.
    size_factor_sd : float
        Standard deviation for log-normal sample size factors (library sizes).
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    gene_means : pd.Series
        Mean expression for each gene.
    gene_vars : pd.Series
        Variance of expression for each gene.
    D : pd.Series
        DESeq2-style dispersion estimates for each gene.
    s : np.ndarray
        Per-sample library size scaling factors.
    mu_matrix : pd.DataFrame
        Expected counts (samples × genes) for input to NB simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    # Mean and variance per gene
    gene_means = rna_df.mean()
    gene_vars = rna_df.var()

    # Estimate dispersions: α = (var - mean) / mean^2
    D = (gene_vars - gene_means) / (gene_means ** 2)
    D = D.clip(lower=1e-6)  # avoid zero or negative values

    # Simulate sample-specific size factors from log-normal distribution
    n_samples = rna_df.shape[0]
    s = np.random.lognormal(mean=0, sigma=size_factor_sd, size=n_samples)

    # Build q_ij: replicate gene means across all samples
    q_matrix = np.tile(gene_means.values, (n_samples, 1))  # shape: (samples, genes)

    # Apply size factors to get μ_ij = s_j · q_ij
    mu_matrix = q_matrix * s[:, np.newaxis]
    mu_matrix = pd.DataFrame(mu_matrix, index=rna_df.index, columns=rna_df.columns)

    return gene_means, gene_vars, D, s, mu_matrix

def sim_mod_exp(x_mut, B, C, D, s, a):
    """
    Simulate gene expression means (mu_ij) from:
        μ_ij = s_j * (x_j·B + a_i) * c_ij where
            s_j = sequencing depth for sample j, 
            a_i = basal expression for gene i, 
            x_j = binary matrix for sample j, 1 = alteration present,
            c_ij = copy number for gene i in sample j


    Parameters:
    -----------
    x_mut : np.ndarray
        Binary matrix (samples × alterations), 1 = alteration present.
    B : np.ndarray
        Effect matrix (alterations × genes), beta_ki = effect of alteration k on gene i.
    C : np.ndarray
        Copy number matrix (samples × genes), c_ij = copy number factor for gene i in sample j.
    D : np.ndarray
        Dispersion vector (genes,) — not used in mu calculation, for NB sampling later.
    s : np.ndarray
        Size factor vector (samples,) — models sequencing depth.
    a : np.ndarray
        Basal expression vector (genes,) — baseline expression for each gene.

    Returns:
    --------
    mu : np.ndarray
        Matrix of expected expression values (samples × genes)
    """

    #  mu = s*(np.matmul(x_mut, B) + D)*(C)

    # Linear component from mutations: (samples × alterations) · (alterations × genes) = (samples × genes)
    xB = np.matmul(x_mut, B)  # shape: (n_samples, n_genes)

    # Add basal expression (broadcasting over rows)
    base_expr = xB + a  # shape: (n_samples, n_genes)

    # Multiply by copy number modulation (elementwise)
    expr_cna = base_expr * C  # shape: (n_samples, n_genes)

    # Multiply by size factors per sample (broadcasting over columns)
    mu = expr_cna * s[:, np.newaxis]  # shape: (n_samples, n_genes)

    return mu

def sample_nb(mu, dispersions):
    """
    Sample from NB(mu, dispersion) for each gene/sample.
    """
    r = 1 / dispersions  # shape: (genes,)
    r = np.clip(r, 1e-6, 1e6)

    # p_ij = r / (r + mu_ij) → broadcasting
    p = r / (r + mu)
    
    # Negative Binomial sampling
    counts = np.random.negative_binomial(n=r, p=p)
    return counts
