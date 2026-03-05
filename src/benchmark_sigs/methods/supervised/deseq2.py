
from __future__ import annotations

import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def get_deseq2_signature_binary(
    X,
    Y,
    gof,
    n_cpus=4,
    alpha=0.05,
    min_group_n=1,
):
    """
    PyDESeq2 v0.4.12-compatible binary DE signature.
    Uses design_factors (NOT design=...).

    Parameters
    ----------
    X : pd.DataFrame
        Alteration/predictor matrix (samples x features).
    Y : pd.DataFrame
        Raw count expression matrix (samples x genes).
    gof : str
        Column name in X for the binary predictor (0/1).
    n_cpus : int
        Number of CPUs for PyDESeq2.
    alpha : float
        FDR threshold for significance.
    min_group_n : int
        Minimum samples required in each group (0 and 1).

    Returns
    -------
    list[str]
        Significant genes (index of results_df) with padj < alpha.
    """

    # raw counts
    counts = Y.astype(int)

    # predictor
    x = pd.to_numeric(X[gof], errors="coerce").dropna()
    if x.shape[0] < 2 * min_group_n:
        return []

    counts = counts.loc[x.index]
    x = x.astype(int)

    # group size guard
    vc = x.value_counts()
    if (0 not in vc) or (1 not in vc) or vc[0] < min_group_n or vc[1] < min_group_n:
        return []

    # metadata table
    metadata = pd.DataFrame(index=counts.index)
    metadata["condition"] = x.astype(str)

    # Make it categorical and set reference level explicitly
    metadata["condition"] = pd.Categorical(metadata["condition"], categories=["0", "1"])

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors="condition",
        ref_level=["condition", "0"],
        refit_cooks=False,
        n_cpus=n_cpus,
    )
    dds.deseq2()

    stat_res = DeseqStats(
        dds,
        contrast=["condition", "1", "0"],
        alpha=alpha,
    )
    stat_res.summary()

    res = stat_res.results_df
    sig = res[res["padj"] < alpha]
    return list(sig.index)
