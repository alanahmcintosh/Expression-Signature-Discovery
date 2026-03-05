"""
DESeq2-style parameter estimation utilities for RNA simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pydeseq2.dds import DeseqDataSet


def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=44, condition="A"):
    """
    Estimate DESeq2-style parameters (mean, variance, dispersion, size factors)
    from a real RNA count matrix.

    Parameters
    ----------
    rna_df : pd.DataFrame
        RNA counts matrix (samples x genes). Will be rounded to int.
    size_factor_sd : float
        Currently unused (kept for signature compatibility / future extension).
    seed : int or None
        Random seed (PyDESeq2 is mostly deterministic here, but retained for API stability).
    condition : str
        Reference level used in the fake design.

    Returns
    -------
    (pd.Series, pd.Series, pd.Series, pd.Series)
        gene_means, gene_vars, dispersions, size_factors
    """
    if rna_df is None or rna_df.empty:
        raise ValueError("[estimate_deseq2_parameters] rna_df is empty.")
    if seed is not None:
        _ = np.random.default_rng(seed)  # keep for API parity

    # Round to integers — DESeq2 expects counts
    rna_df = rna_df.round().astype(int)

    # Fake two-group design to trigger DESeq2 estimation
    fake_conditions = (
        ["A"] * (rna_df.shape[0] // 2)
        + ["B"] * (rna_df.shape[0] - rna_df.shape[0] // 2)
    )
    metadata = pd.DataFrame({"condition": fake_conditions}, index=rna_df.index)

    # Run DESeq2
    dds = DeseqDataSet(
        counts=rna_df,
        metadata=metadata,
        design_factors="condition",
        ref_level=condition,
    )
    dds.deseq2()

    # Retrieve means, variances, dispersions, and size factors
    gene_means = rna_df.mean()
    gene_vars = rna_df.var()

    dispersions = pd.Series(dds.varm["dispersions"], index=rna_df.columns).fillna(0.1)
    dispersions = dispersions.clip(lower=1e-6, upper=1e6)

    size_factors = pd.Series(dds.obsm["size_factors"], index=rna_df.index, name="size_factor")

    return gene_means, gene_vars, dispersions, size_factors


def draw_size_factors_from_deseq(size_factors, n_samples, subtype=None, rng=44):
    """
    Resample DESeq2 size factors (sample-level normalization coefficients)
    to assign realistic per-sample scaling in the simulated data.

    Optionally stratified by subtype proportions.

    Parameters
    ----------
    size_factors : pd.Series
        DESeq2 size factors indexed by sample.
    n_samples : int
        Number of size factors to draw.
    subtype : pd.Series or None
        Subtype labels indexed by sample (must align to size_factors index).
        If provided, draws size factors proportionally by subtype.
    rng : int or np.random.Generator
        Seed or Generator.

    Returns
    -------
    np.ndarray
        Array of length n_samples of resampled size factors.
    """
    if size_factors is None or len(size_factors) == 0:
        raise ValueError("[draw_size_factors_from_deseq] size_factors is empty.")
    if n_samples <= 0:
        return np.array([], dtype=float)

    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    sf = pd.Series(size_factors).dropna()

    # No subtypes: resample directly
    if subtype is None:
        return rng.choice(sf.values, size=int(n_samples), replace=True)

    # Subtype-aware sampling: preserve subtype proportions
    subtype = pd.Series(subtype).loc[sf.index]
    counts = subtype.value_counts()
    props = counts / counts.sum()
    alloc = (props * n_samples).round().astype(int)
    if len(alloc) > 0:
        alloc.iloc[-1] += n_samples - alloc.sum()  # ensure sum == n_samples

    out = []
    for s, n in alloc.items():
        pool = sf.loc[subtype[subtype == s].index]
        if len(pool) == 0:
            pool = sf  # fallback
        out.append(rng.choice(pool.values, size=int(n), replace=True))

    return np.concatenate(out)[: int(n_samples)]
