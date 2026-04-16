from __future__ import annotations

import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import importr, isinstalled

def check_limma_dependencies():
    required = ["limma", "edgeR"]

    missing = [p for p in required if not isinstalled(p)]

    if missing:
        raise RuntimeError(
            f"Missing R packages: {missing}. "
            "Run in R: BiocManager::install(c('limma','edgeR'))"
        )

check_limma_dependencies()
edgeR = importr("edgeR")
limma = importr("limma")


def _pandas_to_r_df(df: pd.DataFrame):
    """Convert pandas DataFrame to R data.frame."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _r_to_pandas_df(r_df):
    """Convert R data.frame to pandas DataFrame."""
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)


def get_limma_voom_signature_binary(
    x_binary: pd.Series,
    y_counts: pd.DataFrame,
    alt_name: str,
    alpha: float = 0.05,
    min_group_n: int = 3,
    normalize_method: str = "TMM",
):
    """
    Run limma-voom for one binary alteration only.

    Parameters
    ----------
    x_binary : pd.Series
        Binary alteration vector indexed by sample, values 0/1.
    y_counts : pd.DataFrame
        Raw integer counts, shape = samples x genes.
    alt_name : str
        Name of the alteration.
    alpha : float
        Adjusted p-value threshold.
    min_group_n : int
        Minimum number of samples required in each group.
    normalize_method : str
        edgeR normalization method, usually "TMM".

    Returns
    -------
    res : pd.DataFrame | None
        Full limma result table indexed by gene.
    sig_genes : list[str]
        Significant genes at FDR < alpha.
    """
    x_binary = x_binary.loc[y_counts.index].astype(int)
    counts = y_counts.loc[x_binary.index].copy().astype(int)

    n0 = int((x_binary == 0).sum())
    n1 = int((x_binary == 1).sum())
    if n0 < min_group_n or n1 < min_group_n:
        return None, []

    # limma/voom expects genes x samples
    counts_t = counts.T.copy()
    counts_t.index = counts_t.index.astype(str)
    counts_t.columns = counts_t.columns.astype(str)

    metadata = pd.DataFrame(
        {
            "sample": counts.index.astype(str),
            "group": pd.Categorical(
                x_binary.astype(str).values,
                categories=["0", "1"],
            ),
        }
    )

    # Convert to R
    counts_r = _pandas_to_r_df(counts_t)
    meta_r = _pandas_to_r_df(metadata)

    ro.globalenv["counts_py"] = counts_r
    ro.globalenv["meta_py"] = meta_r
    ro.globalenv["norm_method_py"] = ro.StrVector([normalize_method])

    # Run R workflow
    ro.r(
        """

        suppressPackageStartupMessages({
            library(edgeR)
            library(limma)
        })

        counts_mat <- as.matrix(counts_py)
        storage.mode(counts_mat) <- "integer"

        meta_df <- as.data.frame(meta_py)
        rownames(meta_df) <- meta_df$sample
        meta_df$sample <- NULL
        meta_df$group <- factor(meta_df$group, levels = c("0", "1"))

        # Ensure sample order matches
        counts_mat <- counts_mat[, rownames(meta_df), drop = FALSE]

        # edgeR normalization
        dge <- DGEList(counts = counts_mat)
        dge <- calcNormFactors(dge, method = norm_method_py[[1]])

        # Design
        design <- model.matrix(~ group, data = meta_df)

        # voom + limma
        v <- voom(dge, design = design, plot = FALSE)
        fit <- lmFit(v, design)
        fit <- eBayes(fit)

        tt <- topTable(
            fit,
            coef = "group1",
            number = Inf,
            sort.by = "none",
            adjust.method = "BH"
        )

        tt$gene <- rownames(tt)
        tt
        """
    )

    res = _r_to_pandas_df(ro.r("tt"))

    if "gene" in res.columns:
        res = res.set_index("gene")

    # Standardize column names to resemble pyDESeq2 output
    res = res.rename(
        columns={
            "P.Value": "pvalue",
            "adj.P.Val": "padj",
        }
    )

    sig = res.loc[res["padj"].notna() & (res["padj"] < alpha)]
    return res, sig.index.tolist()


def precompute_limma_voom_results(
    X_alt_df: pd.DataFrame,
    Y_count_df: pd.DataFrame,
    alpha: float = 0.05,
    min_group_n: int = 5,
    normalize_method: str = "TMM",
):
    """
    Fit limma-voom separately for each alteration.

    Parameters
    ----------
    X_alt_df : pd.DataFrame
        Binary alteration matrix, shape = samples x alterations.
    Y_count_df : pd.DataFrame
        Raw integer counts, shape = samples x genes.
    alpha : float
        Adjusted p-value threshold.
    min_group_n : int
        Minimum group size required for each binary contrast.
    normalize_method : str
        edgeR normalization method.

    Returns
    -------
    results : dict
        Mapping alt -> full limma result DataFrame or None.
    sig_genes : dict
        Mapping alt -> list of significant genes.
    """
    X_alt_df = X_alt_df.loc[Y_count_df.index]

    results = {}
    sig_genes = {}

    for alt in X_alt_df.columns:
        try:
            res, sig = get_limma_voom_signature_binary(
                x_binary=X_alt_df[alt],
                y_counts=Y_count_df,
                alt_name=alt,
                alpha=alpha,
                min_group_n=min_group_n,
                normalize_method=normalize_method,
            )
            results[alt] = res
            sig_genes[alt] = sig
        except Exception as e:
            print(f"Failed limma-voom for {alt}: {e}")
            results[alt] = None
            sig_genes[alt] = []

    return results, sig_genes

