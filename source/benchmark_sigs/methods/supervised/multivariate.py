from __future__ import annotations
import re

import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from rpy2.robjects.packages import importr, isinstalled
from .limma import check_limma_dependencies, _r_to_pandas_df, _pandas_to_r_df
from .edgeR import check_edger_dependencies



check_limma_dependencies()
edgeR = importr("edgeR")
limma = importr("limma")



def precompute_limma_voom_results_multivariate(
    X_alt_df: pd.DataFrame,
    Y_count_df: pd.DataFrame,
    alpha: float = 0.05,
    min_group_n: int = 5,
    normalize_method: str = "TMM",
    add_intercept: bool = True,
):
    """
    Fit one multivariable limma-voom model:
        expression ~ alt1 + alt2 + alt3 + ...

    and extract a result table for each alteration coefficient.

    Parameters
    ----------
    X_alt_df : pd.DataFrame
        Binary alteration matrix, shape = samples x alterations.
    Y_count_df : pd.DataFrame
        Raw integer counts, shape = samples x genes.
    alpha : float
        Adjusted p-value threshold.
    min_group_n : int
        Minimum number of positive and negative samples required
        for an alteration to be retained before modeling.
    normalize_method : str
        edgeR normalization method, usually "TMM".
    add_intercept : bool
        Whether to include an intercept in the design.

    Returns
    -------
    results : dict[str, pd.DataFrame | None]
        Mapping alteration -> full limma table.
    sig_genes : dict[str, list[str]]
        Mapping alteration -> significant genes.
    kept_alts : list[str]
        Alterations retained in the design.
    dropped_alts : dict[str, str]
        Alterations excluded before fitting, with reason.
    """
    # Align samples
    common_samples = X_alt_df.index.intersection(Y_count_df.index)
    X_alt_df = X_alt_df.loc[common_samples].copy()
    Y_count_df = Y_count_df.loc[common_samples].copy()

    # Ensure integer counts
    Y_count_df = Y_count_df.astype(int)

    # Basic binary cleanup + minimum group filter
    kept_cols = []
    dropped_alts = {}

    for alt in X_alt_df.columns:
        x = pd.to_numeric(X_alt_df[alt], errors="coerce").fillna(0).astype(int)

        uniq = sorted(x.unique())
        if not set(uniq).issubset({0, 1}):
            dropped_alts[alt] = f"non-binary values: {uniq}"
            continue

        n0 = int((x == 0).sum())
        n1 = int((x == 1).sum())

        if n0 < min_group_n or n1 < min_group_n:
            dropped_alts[alt] = f"insufficient group sizes (n0={n0}, n1={n1})"
            continue

        if x.nunique() < 2:
            dropped_alts[alt] = "constant column"
            continue

        kept_cols.append(alt)
        X_alt_df[alt] = x

    if not kept_cols:
        return {}, {}, [], dropped_alts

    X_model = X_alt_df[kept_cols].copy()

    # limma/voom expects genes x samples
    counts_t = Y_count_df.T.copy()
    counts_t.index = counts_t.index.astype(str)
    counts_t.columns = counts_t.columns.astype(str)

    # Build metadata/design dataframe for R
    # Make syntactically safe names for R
    design_df = X_model.copy()
    design_df.index = design_df.index.astype(str)
    design_df.columns = [
        c.replace("-", "_").replace(" ", "_").replace(":", "_").replace("/", "_")
        for c in design_df.columns
    ]

    # Keep mapping from safe names back to original names
    safe_to_orig = dict(zip(design_df.columns, kept_cols))

    design_df = design_df.reset_index().rename(columns={"index": "sample"})

    counts_r = _pandas_to_r_df(counts_t)
    design_r = _pandas_to_r_df(design_df)

    ro.globalenv["counts_py"] = counts_r
    ro.globalenv["design_py"] = design_r
    ro.globalenv["norm_method_py"] = ro.StrVector([normalize_method])
    ro.globalenv["add_intercept_py"] = ro.BoolVector([add_intercept])

    ro.r(
        """
        suppressPackageStartupMessages({
            library(edgeR)
            library(limma)
        })

        counts_mat <- as.matrix(counts_py)
        storage.mode(counts_mat) <- "integer"

        design_df <- as.data.frame(design_py)
        rownames(design_df) <- design_df$sample
        design_df$sample <- NULL

        # Ensure numeric predictors
        for (j in seq_len(ncol(design_df))) {
            design_df[[j]] <- as.numeric(design_df[[j]])
        }

        # Match sample order
        counts_mat <- counts_mat[, rownames(design_df), drop = FALSE]

        dge <- DGEList(counts = counts_mat)
        dge <- calcNormFactors(dge, method = norm_method_py[[1]])

        if (isTRUE(add_intercept_py[[1]])) {
            design <- model.matrix(~ ., data = design_df)
        } else {
            design <- model.matrix(~ 0 + ., data = design_df)
        }

        # Drop aliased / non-estimable coefficients if needed
        qrobj <- qr(design)
        if (qrobj$rank < ncol(design)) {
            keep <- qrobj$pivot[seq_len(qrobj$rank)]
            design <- design[, keep, drop = FALSE]
        }

        v <- voom(dge, design = design, plot = FALSE)
        fit <- lmFit(v, design)
        fit <- eBayes(fit)

        design_colnames <- colnames(design)
        """
    )

    design_cols = list(ro.r("design_colnames"))

    results = {}
    sig_genes = {}

    for safe_alt, orig_alt in safe_to_orig.items():
        coef_name = safe_alt if not add_intercept else safe_alt

        if coef_name not in design_cols:
            # Could have been dropped due to rank deficiency
            results[orig_alt] = None
            sig_genes[orig_alt] = []
            dropped_alts[orig_alt] = "dropped from design due to rank deficiency / aliasing"
            continue

        ro.globalenv["coef_name_py"] = ro.StrVector([coef_name])

        try:
            ro.r(
                """
                tt <- topTable(
                    fit,
                    coef = coef_name_py[[1]],
                    number = Inf,
                    sort.by = "none",
                    adjust.method = "BH"
                )
                tt$gene <- rownames(tt)
                """
            )

            res = _r_to_pandas_df(ro.r("tt"))

            if "gene" in res.columns:
                res = res.set_index("gene")

            res = res.rename(
                columns={
                    "P.Value": "pvalue",
                    "adj.P.Val": "padj",
                }
            )

            sig = res.loc[res["padj"].notna() & (res["padj"] < alpha)]

            results[orig_alt] = res
            sig_genes[orig_alt] = sig.index.tolist()

        except Exception as e:
            print(f"Failed limma-voom extraction for {orig_alt}: {e}")
            results[orig_alt] = None
            sig_genes[orig_alt] = []

    return results, sig_genes, kept_cols, dropped_alts



check_edger_dependencies()



def _make_safe_r_names(columns: list[str]) -> list[str]:
    """
    Make column names safe and unique for use in an R model matrix.

    This is intentionally simple and deterministic so we can map back later.
    """
    safe = []
    seen = {}

    for col in columns:
        x = str(col)
        x = re.sub(r"[^0-9A-Za-z_]", "_", x)
        if re.match(r"^[0-9]", x):
            x = f"X_{x}"
        if x == "":
            x = "X"

        if x in seen:
            seen[x] += 1
            x = f"{x}__{seen[x]}"
        else:
            seen[x] = 0

        safe.append(x)

    return safe


def get_edger_signature_binary_multivariate(
    x_binary: pd.Series,
    X_covariates: pd.DataFrame,
    y_counts: pd.DataFrame,
    alt_name: str,
    alpha: float = 0.05,
    min_group_n: int = 3,
    normalize_method: str = "TMM",
    filter_low_expr: bool = True,
    robust: bool = True,
    drop_constant_covariates: bool = True,
):
    """
    Run edgeR quasi-likelihood DE for one binary alteration while adjusting
    for other covariates.

    Parameters
    ----------
    x_binary : pd.Series
        Binary alteration vector indexed by sample, values 0/1.
        This is the variable of interest.
    X_covariates : pd.DataFrame
        Covariate matrix indexed by sample. Can include other binary features
        or continuous covariates. The target alteration column does not need
        to be included; it will be added as the first predictor automatically.
    y_counts : pd.DataFrame
        Raw integer counts, shape = samples x genes.
    alt_name : str
        Name of the alteration of interest.
    alpha : float
        Adjusted p-value threshold.
    min_group_n : int
        Minimum number of samples required in each target group.
    normalize_method : str
        edgeR normalization method, usually "TMM".
    filter_low_expr : bool
        Whether to apply edgeR::filterByExpr before fitting.
    robust : bool
        Whether to use robust dispersion / QL fitting.
    drop_constant_covariates : bool
        Whether to remove covariate columns with only one unique value.

    Returns
    -------
    res : pd.DataFrame | None
        Full edgeR result table indexed by gene.
    sig_genes : list[str]
        Significant genes at FDR < alpha.
    """
    common_idx = y_counts.index.intersection(x_binary.index).intersection(X_covariates.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping samples among x_binary, X_covariates, and y_counts.")

    x_binary = x_binary.loc[common_idx].astype(int)
    covars = X_covariates.loc[common_idx].copy()
    counts = y_counts.loc[common_idx].copy().astype(int)

    n0 = int((x_binary == 0).sum())
    n1 = int((x_binary == 1).sum())
    if n0 < min_group_n or n1 < min_group_n:
        return None, []

    # Remove any duplicated target column from covariates if present
    covars = covars.drop(columns=[c for c in covars.columns if c == alt_name], errors="ignore")

    # Force numeric if possible
    for c in covars.columns:
        covars[c] = pd.to_numeric(covars[c], errors="coerce")

    # Fill NA conservatively; for your alteration matrices this usually should not happen
    if covars.isna().any().any():
        covars = covars.fillna(0)

    if drop_constant_covariates and covars.shape[1] > 0:
        keep_cols = [c for c in covars.columns if covars[c].nunique(dropna=False) > 1]
        covars = covars[keep_cols]

    # Build model frame with target first so coefficient naming is easy
    model_df = pd.concat(
        [x_binary.rename(alt_name), covars],
        axis=1,
    )

    # Make column names R-safe and keep map
    original_cols = list(model_df.columns)
    safe_cols = _make_safe_r_names(original_cols)
    col_map = dict(zip(original_cols, safe_cols))
    reverse_map = dict(zip(safe_cols, original_cols))
    model_df.columns = safe_cols

    target_safe = col_map[alt_name]

    # edgeR expects genes x samples
    counts_t = counts.T.copy()
    counts_t.index = counts_t.index.astype(str)
    counts_t.columns = counts_t.columns.astype(str)

    metadata = model_df.copy()
    metadata.index = metadata.index.astype(str)
    metadata["sample"] = metadata.index

    counts_r = _pandas_to_r_df(counts_t)
    meta_r = _pandas_to_r_df(metadata)

    ro.globalenv["counts_py"] = counts_r
    ro.globalenv["meta_py"] = meta_r
    ro.globalenv["norm_method_py"] = ro.StrVector([normalize_method])
    ro.globalenv["filter_low_expr_py"] = ro.BoolVector([filter_low_expr])
    ro.globalenv["robust_py"] = ro.BoolVector([robust])
    ro.globalenv["target_coef_py"] = ro.StrVector([target_safe])

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

        # Ensure sample order matches
        counts_mat <- counts_mat[, rownames(meta_df), drop = FALSE]

        # Build DGEList
        y <- DGEList(counts = counts_mat)
        y <- calcNormFactors(y, method = norm_method_py[[1]])

        # Multivariate additive design:
        # target alteration + all supplied covariates
        design <- model.matrix(~ ., data = meta_df)

        # Optional design-aware filtering
        if (filter_low_expr_py[[1]]) {
            keep <- filterByExpr(y, design = design)
            y <- y[keep, , keep.lib.sizes = FALSE]
        }

        if (nrow(y) == 0) {
            tt <- data.frame()
        } else {
            y <- estimateDisp(y, design = design, robust = robust_py[[1]])
            fit <- glmQLFit(y, design = design, robust = robust_py[[1]])

            coef_name <- target_coef_py[[1]]

            if (!(coef_name %in% colnames(design))) {
                stop(
                    paste0(
                        "Target coefficient '", coef_name,
                        "' not found in design matrix. Available: ",
                        paste(colnames(design), collapse = ", ")
                    )
                )
            }

            qlf <- glmQLFTest(fit, coef = which(colnames(design) == coef_name))

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

    res = res.rename(
        columns={
            "PValue": "pvalue",
            "FDR": "padj",
        }
    )

    sig = res.loc[res["padj"].notna() & (res["padj"] < alpha)]
    return res, sig.index.tolist()


def precompute_edger_results_multivariate(
    X_alt_df: pd.DataFrame,
    Y_count_df: pd.DataFrame,
    covariates_df: pd.DataFrame | None = None,
    alpha: float = 0.05,
    min_group_n: int = 5,
    normalize_method: str = "TMM",
    filter_low_expr: bool = True,
    robust: bool = True,
    include_other_alterations_as_covariates: bool = True,
):
    """
    Fit multivariate edgeR separately for each alteration, adjusting for
    other supplied covariates and optionally all other alteration features.

    Parameters
    ----------
    X_alt_df : pd.DataFrame
        Binary alteration matrix, shape = samples x alterations.
    Y_count_df : pd.DataFrame
        Raw integer counts, shape = samples x genes.
    covariates_df : pd.DataFrame | None
        Additional covariates indexed by sample, e.g. subtype, purity,
        batch, PCs, etc. Must be numeric as written here.
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
    include_other_alterations_as_covariates : bool
        If True, use all other alteration columns in X_alt_df as covariates
        when testing each target alteration.

    Returns
    -------
    results : dict
        Mapping alt -> full edgeR result DataFrame or None.
    sig_genes : dict
        Mapping alt -> list of significant genes.
    """
    X_alt_df = X_alt_df.loc[Y_count_df.index]

    if covariates_df is None:
        covariates_df = pd.DataFrame(index=X_alt_df.index)
    else:
        covariates_df = covariates_df.loc[X_alt_df.index]

    results = {}
    sig_genes = {}

    for alt in X_alt_df.columns:
        try:
            if include_other_alterations_as_covariates:
                other_alts = X_alt_df.drop(columns=[alt], errors="ignore")
                model_covars = pd.concat([covariates_df, other_alts], axis=1)
            else:
                model_covars = covariates_df.copy()

            res, sig = get_edger_signature_binary_multivariate(
                x_binary=X_alt_df[alt],
                X_covariates=model_covars,
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
            print(f"Failed multivariate edgeR for {alt}: {e}")
            results[alt] = None
            sig_genes[alt] = []

    return results, sig_genes

