"""
Negative binomial sampling utilities for RNA simulation.

Exports:
- sample_nb: vectorized NB sampling given mean (mu) and per-gene dispersions
- sample_nb_for_signature_genes: only NB-sample a subset of genes (e.g., signature genes),
  while rounding the rest to integer baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_nb(mu, dispersions, rng=None):
    """
    Sample RNA-seq counts using a Negative Binomial distribution.
    variance = mu + mu^2 * dispersion
        where
            mu = expected expression level
            dispersion = gene specfic noise

    Handles edge cases and clips invalid values.

    Parameters
    ----------
    mu : array-like
        Mean expression matrix (samples x genes) OR vector (genes,).
    dispersions : array-like or pd.Series
        Per-gene dispersions of length n_genes.
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    np.ndarray
        Sampled counts with same shape as mu.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = np.asarray(mu)

    if isinstance(dispersions, pd.Series):
        dispersions = dispersions.values
    dispersions = np.asarray(dispersions)

    # Promote 1D mu -> 2D (1 x genes) for consistent broadcasting
    if mu.ndim == 1:
        mu = mu[np.newaxis, :]

    # Check for NaNs or infs in inputs
    if np.any(~np.isfinite(mu)):
        raise ValueError("mu contains non-finite values")
    if np.any(~np.isfinite(dispersions)):
        raise ValueError("dispersions contain non-finite values")

    mu = np.clip(mu, 1e-8, 1e6)
    dispersions = np.clip(dispersions, 1e-8, 1e6)

    # NB "size" parameter r = 1/alpha
    r = 1.0 / dispersions
    r = np.clip(r, 1e-6, 1e6)

    if mu.shape[1] != r.shape[0]:
        raise ValueError(f"Shape mismatch: mu {mu.shape}, r {r.shape}")

    r = r[np.newaxis, :]  # broadcast across samples

    # Convert (mu, r) to NB parameter p such that E[Y] = mu
    # For numpy negative_binomial(n=r, p=p): mean = n*(1-p)/p
    # => p = r/(r+mu)
    p = r / (r + mu)

    if np.any(np.isnan(p)) or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError(
            f"Invalid p in NB sampling: min={np.nanmin(p)}, max={np.nanmax(p)}, NaNs={np.isnan(p).sum()}"
        )

    return rng.negative_binomial(n=r, p=p)


def sample_nb_for_signature_genes(
    mu_df,
    dispersions,
    sig_genes,
    seed=44,
):
    """
    Sample NB counts only for signature genes using sample_nb().
    Non-signature genes are rounded.

    Parameters
    ----------
    mu_df : pd.DataFrame
        Mean expression matrix (samples x genes).
    dispersions : pd.Series
        Per-gene dispersions indexed by gene.
    sig_genes : list[str]
        Genes to NB-sample. Others will be rounded.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Count matrix (samples x genes) as integers.
    """
    if mu_df is None or mu_df.empty:
        raise ValueError("[sample_nb_for_signature_genes] mu_df is empty.")
    if dispersions is None or len(dispersions) == 0:
        raise ValueError("[sample_nb_for_signature_genes] dispersions is empty.")

    rng = np.random.default_rng(seed)
    counts = mu_df.copy()

    # Keep only genes present in both
    sig_genes = [g for g in sig_genes if g in mu_df.columns and g in dispersions.index]
    if len(sig_genes) == 0:
        counts = counts.round().astype(int)
        counts[counts < 0] = 0
        return counts

    # --- sample NB only for signature genes ---
    mu_sub = mu_df[sig_genes].values
    disp_sub = dispersions.loc[sig_genes].values

    sampled = sample_nb(mu_sub, disp_sub, rng=rng)
    counts.loc[:, sig_genes] = sampled

    # --- non-signature genes → integer baseline ---
    counts = counts.round().astype(int)
    counts[counts < 0] = 0

    return counts
