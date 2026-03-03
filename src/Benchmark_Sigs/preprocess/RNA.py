"""
RNA preprocessing utilities used prior to RNA simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# 10. PRE-RNA SIMULATION PROCESSING
# =============================================================================

def preprocess_rna_for_simulation(rna_df, strategy="auto", user_scale=None, verbose=True):
    """
    Preprocess RNA-seq expression data to convert from normalized form (e.g., CPM, RPKM, RSEM)
    into pseudo-counts for negative binomial simulation.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Expression matrix (samples x genes).
    strategy : str
        "auto" chooses a scale factor heuristically based on the median 75th percentile
        across samples. "manual" uses user_scale.
    user_scale : float or None
        Scale factor if strategy="manual".
    verbose : bool
        If True, print applied scale factor.

    Returns
    -------
    (pd.DataFrame, float)
        (rna_scaled_counts, scale_factor)
    """
    if rna_df is None or rna_df.empty:
        raise ValueError("[preprocess_rna_for_simulation] rna_df is empty.")

    if strategy == "manual":
        if user_scale is None:
            raise ValueError("You must provide user_scale if using manual strategy.")
        scale_factor = user_scale

    elif strategy == "auto":
        # Heuristic based on typical expression magnitudes
        q75 = rna_df.quantile(0.75).median()
        if q75 < 1:
            scale_factor = 100
        elif q75 < 10:
            scale_factor = 50
        elif q75 < 100:
            scale_factor = 10
        else:
            scale_factor = 1

    else:
        raise ValueError("strategy must be 'auto' or 'manual'")

    # Multiply expression and round to int
    rna_scaled = (rna_df * scale_factor).round().astype(int)

    # NB counts cannot be negative
    rna_scaled[rna_scaled < 0] = 0

    if verbose:
        print(f"[preprocess_rna] Applied scale factor: {scale_factor}")

    return rna_scaled, scale_factor


def select_genes_with_expr_filter(
    rna_df,
    alterations_df,
    target_total=10_000,
    min_cpm=1.0,
    min_prop_samples=0.20,   # keep gene if expressed in ≥20% samples
    use_mad=False,           # rank by MAD of log1p-CPM instead of variance
    verbose=False,
):
    """
    Select a set of genes for simulation while guaranteeing that all genes present in
    `alterations_df`are retained if present in RNA.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Samples x genes expression matrix.
    alterations_df : pd.DataFrame
        Samples x alteration features (columns like TP53_LOF, MYC_AMP, etc.).
    target_total : int
        Total number of genes to keep.
    min_cpm : float
        Minimum CPM threshold for counts-based expression filter.
    min_prop_samples : float
        Minimum fraction of samples meeting expression threshold.
    use_mad : bool
        If True, rank by MAD; else rank by variance.
    verbose : bool
        If True, print selection summary.

    Returns
    -------
    dict
        {
          "genes_to_keep": list[str],
          "altered_genes_kept": list[str],
          "low_expr_altered_flagged": list[str],
          "genes_dropped_low_expr": list[str]
        }
    """
    if rna_df is None or rna_df.empty:
        raise ValueError("[select_genes_with_expr_filter] rna_df is empty.")
    if alterations_df is None or alterations_df.empty:
        raise ValueError("[select_genes_with_expr_filter] alterations_df is empty.")

    # ---------- 0) Clean inputs ----------
    rna_df = rna_df.copy()
    rna_df.columns = rna_df.columns.astype(str)

    alterations_df = alterations_df.copy()
    alterations_df.columns = alterations_df.columns.astype(str)

    # ---------- 1) Identify altered genes present in RNA ----------
    altered_genes = list({c.split("_")[0] for c in alterations_df.columns})
    altered_in_expr = [g for g in altered_genes if g in rna_df.columns]

    # ---------- 2) Decide if rna_df looks like raw counts ----------
    values = rna_df.to_numpy()
    nonneg = (values >= 0).mean() > 0.999
    int_like = (np.isclose(values, np.round(values)).mean() > 0.98)
    looks_like_counts = bool(nonneg and int_like and np.nanmax(values) >= 50)

    # ---------- 3) Library-size normalization & log1p-CPM ----------
    if looks_like_counts:
        lib_sizes = rna_df.sum(axis=1).replace(0, np.nan)  # avoid div by 0
        cpm = (rna_df.div(lib_sizes, axis=0)) * 1e6
        log_cpm = np.log1p(cpm)
        expressed_mask = (cpm >= min_cpm).mean(axis=0) >= min_prop_samples
    else:
        # Already TPM / FPKM / log2(TPM+1) style. Use a small floor for expression.
        log_cpm = np.log1p(rna_df)
        expressed_mask = (rna_df > 0.1).mean(axis=0) >= min_prop_samples

    # ---------- 4) Keep all altered genes regardless of expression ----------
    low_expr_altered = [g for g in altered_in_expr if not bool(expressed_mask.get(g, False))]

    # Genes eligible for variability ranking = non-altered & pass expression filter
    non_altered = [g for g in rna_df.columns if g not in altered_in_expr]
    non_alt_keep = [g for g in non_altered if bool(expressed_mask.get(g, False))]

    # ---------- 5) Rank by variability among eligible non-altered genes ----------
    if len(non_alt_keep) > 0:
        X = log_cpm[non_alt_keep]
        if use_mad:
            med = X.median(axis=0)
            variability = (X.sub(med, axis=1)).abs().median(axis=0)
        else:
            variability = X.var(axis=0, ddof=1)

        variability = variability.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked_non_altered = variability.sort_values(ascending=False).index.tolist()
    else:
        ranked_non_altered = []

    # ---------- 6) Decide how many non-altered to take ----------
    n_altered = len(altered_in_expr)
    n_remaining = max(0, int(target_total) - n_altered)

    top_non_altered = ranked_non_altered[:n_remaining] if n_remaining > 0 else []

    # ---------- 7) Compose final set ----------
    genes_to_keep = sorted(set(altered_in_expr + top_non_altered))

    # ---------- 8) Book-keeping / reporting ----------
    low_expr_non_altered = [g for g in non_altered if not bool(expressed_mask.get(g, False))]
    genes_dropped_low_expr = sorted(low_expr_non_altered)

    if verbose:
        total_genes = rna_df.shape[1]
        kept_non_altered = len(top_non_altered)
        dropped_low_expr = len(genes_dropped_low_expr)
        print("[Gene selection]")
        print(f"  Total RNA genes: {total_genes}")
        print(f"  Altered genes present in RNA: {n_altered}")
        print(f"  Target total: {target_total} -> selecting {kept_non_altered} non-altered to fill")
        print(f"  Low-expression (non-altered) genes dropped: {dropped_low_expr}")
        if low_expr_altered:
            preview = ", ".join(low_expr_altered[:10])
            suffix = " ..." if len(low_expr_altered) > 10 else ""
            print(
                f"  NOTE: {len(low_expr_altered)} altered gene(s) are lowly expressed but KEPT: "
                f"{preview}{suffix}"
            )

    return {
        "genes_to_keep": genes_to_keep,
        "altered_genes_kept": sorted(altered_in_expr),
        "low_expr_altered_flagged": sorted(low_expr_altered),
        "genes_dropped_low_expr": genes_dropped_low_expr,
    }
