from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_counts_log_cpm(
    Y_counts,
    pseudo = 1.0,
    zscore = True,
):
    """
    DESeq2-style size-factor normalization (median-of-ratios),
    then log transform and (optionally) z-score per gene.

    Parameters
    ----------
    Y_counts
        Raw count matrix with shape (n_samples, n_genes). Rows are samples, columns are genes.
    pseudo
        Pseudocount used for log transform. Default 1.0 (log1p behavior).
    zscore
        If True, z-score per gene after log transform.

    Returns
    -------
    pd.DataFrame
        Normalized (and optionally z-scored) expression matrix.

    Notes
    -----
    - If samples differ only by depth and have identical composition,
      normalized expression will be identical across samples.
    """

    if not isinstance(Y_counts, pd.DataFrame):
        raise TypeError("Y_counts must be a pandas DataFrame (samples x genes).")
    if Y_counts.shape[0] == 0 or Y_counts.shape[1] == 0:
        return Y_counts.copy()

    Y = Y_counts.copy()

    # Total counts per sample
    s = Y.sum(axis=1)

    # Drop all-zero samples
    keep = s > 0
    Y = Y.loc[keep]
    s = s.loc[keep]
    if Y.shape[0] == 0:
        return Y  # empty

    # -----
    # Median-of-ratios size factors
    # -----
    # Geometric mean per gene (ignoring zeros by treating them as NaN in logs)
    Y_float = Y.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logY = np.log(Y_float)
    logY[~np.isfinite(logY)] = np.nan  # log(0) -> nan

    gmean_log = np.nanmean(logY.values, axis=0)  # per gene
    gmean = np.exp(gmean_log)

    # genes with gmean==0 or nan are unusable
    valid_genes = np.isfinite(gmean) & (gmean > 0)

    if valid_genes.sum() == 0:
        # No usable genes to compute size factors
        # Fall back to simple library-size scaling (rare edge case)
        size_factors = (s / np.median(s)).astype(float)
    else:
        gmean_valid = gmean[valid_genes]
        ratios = Y_float.iloc[:, valid_genes] / gmean_valid  # sample x gene

        # ignore zeros / invalid in ratios (where count==0)
        ratios = ratios.mask(~np.isfinite(ratios) | (ratios <= 0), np.nan)

        size_factors = pd.Series(
            np.nanmedian(ratios.values, axis=1),
            index=Y.index,
            dtype=float,
        )

        # If a sample has all zeros for valid genes, median becomes nan; fall back
        if size_factors.isna().any():
            fallback = (s / np.median(s)).astype(float)
            size_factors = size_factors.fillna(fallback)

    # Normalize counts
    Y_norm = Y.div(size_factors, axis=0)

    # Log transform
    # pseudo=1.0 -> log1p(Y_norm)
    Y_norm = np.log1p(Y_norm + pseudo - 1.0)

    if not zscore:
        return Y_norm

    # Z-score per gene (safe for 0-variance genes: sklearn outputs 0s)
    Z = pd.DataFrame(
        StandardScaler(with_mean=True, with_std=True).fit_transform(Y_norm),
        index=Y_norm.index,
        columns=Y_norm.columns,
    )
    return Z
