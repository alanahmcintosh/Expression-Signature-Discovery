from __future__ import annotations

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def get_deseq2_signature_binary(
    x_binary,
    y_counts,
    alt_name,
    n_cpus=4,
    alpha=0.05,
    min_group_n=3,
):
    """
    Run DESeq2 for one binary alteration only.
    x_binary: pd.Series indexed by sample, values 0/1
    y_counts: pd.DataFrame samples x genes, raw integer counts
    alt_name: name of the alteration
    """
    x_binary = x_binary.loc[y_counts.index].astype(int)
    counts = y_counts.astype(int)

    n0 = (x_binary == 0).sum()
    n1 = (x_binary == 1).sum()
    if n0 < min_group_n or n1 < min_group_n:
        return None, []

    metadata = pd.DataFrame(
        {alt_name: pd.Categorical(x_binary.astype(str), categories=["0", "1"])},
        index=counts.index,
    )

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors=[alt_name],
        ref_level=[alt_name, "0"],
        refit_cooks=False,
        n_cpus=n_cpus,
    )
    dds.deseq2()

    deseq_alt = alt_name.replace("_", "-")
    stat_res = DeseqStats(
        dds,
        contrast=[deseq_alt, "1", "0"],
        alpha=alpha,
        quiet=True,
    )
    stat_res.summary()

    res = stat_res.results_df.copy()
    sig = res.loc[res["padj"].notna() & (res["padj"] < alpha)]
    return res, sig.index.tolist()


def precompute_deseq2_results(
    X_alt_df,
    Y_count_df,
    n_cpus=4,
    alpha=0.05,
    min_group_n=5,
):
    """
    Fit DESeq2 separately for each alteration.
    """
    X_alt_df = X_alt_df.loc[Y_count_df.index]

    results = {}
    sig_genes = {}

    for alt in X_alt_df.columns:
        try:
            res, sig = get_deseq2_signature_binary(
                x_binary=X_alt_df[alt],
                y_counts=Y_count_df,
                alt_name=alt,
                n_cpus=n_cpus,
                alpha=alpha,
                min_group_n=min_group_n,
            )
            results[alt] = res
            sig_genes[alt] = sig
        except Exception as e:
            print(f"Failed DESeq2 for {alt}: {e}")
            results[alt] = None
            sig_genes[alt] = []

    return results, sig_genes