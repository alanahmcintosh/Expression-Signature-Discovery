from __future__ import annotations

import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr, isinstalled
from .limma import _r_to_pandas_df, _pandas_to_r_df

def check_edger_dependencies():
    required = ["edgeR"]

    missing = [p for p in required if not isinstalled(p)]

    if missing:
        raise RuntimeError(
            f"Missing R packages: {missing}. "
            "Run in R: BiocManager::install(c('edgeR'))"
        )

check_edger_dependencies()
edgeR = importr("edgeR")


def get_edger_signature_binary(
    x_binary: pd.Series,
    y_counts: pd.DataFrame,
    alt_name: str,
    alpha: float = 0.05,
    min_group_n: int = 3,
    normalize_method: str = "TMM",
    filter_low_expr: bool = True,
    robust: bool = True,
):
    """
    Run edgeR quasi-likelihood DE for one binary alteration only.

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
    filter_low_expr : bool
        Whether to apply edgeR::filterByExpr before fitting.
    robust : bool
        Whether to use robust dispersion / QL fitting.

    Returns
    -------
    res : pd.DataFrame | None
        Full edgeR result table indexed by gene.
    sig_genes : list[str]
        Significant genes at FDR < alpha.
    """
    x_binary = x_binary.loc[y_counts.index].astype(int)
    counts = y_counts.loc[x_binary.index].copy().astype(int)

    n0 = int((x_binary == 0).sum())
    n1 = int((x_binary == 1).sum())
    if n0 < min_group_n or n1 < min_group_n:
        return None, []

    # edgeR expects genes x samples
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
    ro.globalenv["filter_low_expr_py"] = ro.BoolVector([filter_low_expr])
    ro.globalenv["robust_py"] = ro.BoolVector([robust])

    ro.r(
        """
        suppressPackageStartupMessages({
            library(edgeR)
        })

        counts_mat <- as.matrix(counts_py)
        storage.mode(counts_mat) <- "integer"

        meta_df <- as.data.frame(meta_py)
        rownames(meta_df) <- meta_df$sample
        meta_df$sample <- NULL
        meta_df$group <- factor(meta_df$group, levels = c("0", "1"))

        # Ensure sample order matches
        counts_mat <- counts_mat[, rownames(meta_df), drop = FALSE]

        # Create DGEList and normalize
        y <- DGEList(counts = counts_mat, samples = meta_df)
        y <- calcNormFactors(y, method = norm_method_py[[1]])

        # Simple two-group design: coef "group1" = group1 vs group0
        design <- model.matrix(~ group, data = meta_df)

        # Filter low-expression genes in a design-aware way
        if (filter_low_expr_py[[1]]) {
            keep <- filterByExpr(y, design = design)
            y <- y[keep, , keep.lib.sizes = FALSE]
        }

        # If nothing remains after filtering, return empty result
        if (nrow(y) == 0) {
            tt <- data.frame()
        } else {
            # NB dispersion + quasi-likelihood fit
            y <- estimateDisp(y, design = design, robust = robust_py[[1]])
            fit <- glmQLFit(y, design = design, robust = robust_py[[1]])
            qlf <- glmQLFTest(fit, coef = "group1")

            tt <- topTags(
                qlf,
                n = Inf,
                sort.by = "none",
                adjust.method = "BH"
            )$table

            tt$gene <- rownames(tt)
        }
        """
    )

    res = _r_to_pandas_df(ro.r("tt"))

    if res is None or len(res) == 0:
        return None, []

    if "gene" in res.columns:
        res = res.set_index("gene")

    # Standardize column names to resemble pyDESeq2 output
    # edgeR topTags typically returns: logFC, logCPM, F, PValue, FDR
    res = res.rename(
        columns={
            "PValue": "pvalue",
            "FDR": "padj",
        }
    )

    sig = res.loc[res["padj"].notna() & (res["padj"] < alpha)]
    return res, sig.index.tolist()


def precompute_edger_results(
    X_alt_df: pd.DataFrame,
    Y_count_df: pd.DataFrame,
    alpha: float = 0.05,
    min_group_n: int = 5,
    normalize_method: str = "TMM",
    filter_low_expr: bool = True,
    robust: bool = True,
):
    """
    Fit edgeR separately for each alteration.

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
    filter_low_expr : bool
        Whether to apply edgeR::filterByExpr before fitting.
    robust : bool
        Whether to use robust dispersion / QL fitting.

    Returns
    -------
    results : dict
        Mapping alt -> full edgeR result DataFrame or None.
    sig_genes : dict
        Mapping alt -> list of significant genes.
    """
    X_alt_df = X_alt_df.loc[Y_count_df.index]

    results = {}
    sig_genes = {}

    for alt in X_alt_df.columns:
        try:
            res, sig = get_edger_signature_binary(
                x_binary=X_alt_df[alt],
                y_counts=Y_count_df,
                alt_name=alt,
                alpha=alpha,
                min_group_n=min_group_n,
                normalize_method=normalize_method,
                filter_low_expr=filter_low_expr,
                robust=robust,
            )
            results[alt] = res
            sig_genes[alt] = sig
        except Exception as e:
            print(f"Failed edgeR for {alt}: {e}")
            results[alt] = None
            sig_genes[alt] = []

    return results, sig_genes