
from __future__ import annotations

import pandas as pd

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def get_deseq2_signature_binary(
    X,
    Y,
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

    metadata = X.copy()

    for col in metadata.columns:
        metadata[col] = pd.Categorical(metadata[col].astype(str), categories=['0', '1'])

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors=list(metadata.columns),
        ref_level=[metadata.columns[0], "0"],
        refit_cooks=False,
        n_cpus=n_cpus,
    )
    dds.deseq2()

    results = {}
    sig_genes = {}

    for gene in metadata.columns:
 
        deseq_gene = gene.replace("_", "-")

        stat_res = DeseqStats(
            dds,
            contrast=[deseq_gene, "1", "0"],
            alpha=alpha,
            quiet=True,
        )
        stat_res.summary()

        res = stat_res.results_df.copy()
        sig = res.loc[res["padj"].notna() & (res["padj"] < alpha)]

        results[gene] = res
        sig_genes[gene] = sig.index.tolist()

    return results, sig_genes

def precompute_deseq2_results(
    X_alt_df,
    Y_count_df,
    n_cpus=4,
    alpha=0.05,
):
    """
    Fit DESeq2 once across all alterations in X_alt_df.

    Returns
    -------
    tuple
        (results_dict, sig_genes_dict)
    """
    return get_deseq2_signature_binary(
        X_alt_df,
        Y_count_df,
        n_cpus=n_cpus,
        alpha=alpha,
    )