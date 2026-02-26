import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, ElasticNet, RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

from Deconfounder import *



'''
Helper Functions
'''

def select_features_elbow(feature_names, weights, tol=1e-8):
    """
    Elbow-based feature selector.

    - Takes absolute weights/importances.
    - Finds the 'knee' of the sorted curve (rank vs |weight|).
    - Returns all features up to the knee index.

    """
    weights = np.asarray(weights).ravel()
    abs_w = np.abs(weights)

    # Only consider clearly non-zero weights
    mask = abs_w > tol
    if mask.sum() == 0:
        return []

    abs_w_nz = abs_w[mask]
    names_nz = np.asarray(feature_names)[mask]

    n = len(abs_w_nz)
    if n <= 1:
        return names_nz.tolist()

    # Sort descending by |weight|
    order = np.argsort(abs_w_nz)[::-1]
    w_sorted = abs_w_nz[order]
    names_sorted = names_nz[order]

    # Coordinates (rank, value)
    x = np.arange(n)
    y = w_sorted

    # Line from first to last point
    x0, y0 = 0, y[0]
    x1, y1 = n - 1, y[-1]
    dx = x1 - x0
    dy = y1 - y0
    denom = np.sqrt(dx**2 + dy**2)

    if denom == 0:
        # Flat line — all weights equal: keep all non-zero
        return names_sorted.tolist()

    # Per-point distance to the straight line
    # |dy*x_i - dx*y_i + x1*y0 - y1*x0| / sqrt(dx^2+dy^2)
    dist = np.abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom
    knee_idx = int(np.argmax(dist))

    # Keep all features up to and including the knee
    return names_sorted[: knee_idx + 1].tolist()


def normalize_counts_log_cpm(Y_counts, pseudo=1.0, zscore=True):
    """
    DESeq2-style size-factor normalization (median-of-ratios),
    then log transform and (optionally) z-score per gene.

    Notes:
    - If samples differ only by depth and have identical composition,
      normalized expression will be identical across samples.
    """

    Y = Y_counts.copy()

    # Total counts per sample
    s = Y.sum(axis=1)

    # Drop all-zero samples
    keep = s > 0
    Y = Y.loc[keep]
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
        size_factors = s / np.median(s)
    else:
        gmean_valid = gmean[valid_genes]
        ratios = Y_float.iloc[:, valid_genes] / gmean_valid  # sample x gene
        # ignore zeros in ratios (where count==0)
        ratios = ratios.mask(~np.isfinite(ratios) | (ratios <= 0), np.nan)
        size_factors = pd.Series(np.nanmedian(ratios.values, axis=1), index=Y.index)

        # If a sample has all zeros for valid genes, median becomes nan; fall back
        if size_factors.isna().any():
            fallback = (s / np.median(s)).astype(float)
            size_factors = size_factors.fillna(fallback)

    # Normalize counts
    Y_norm = Y.div(size_factors, axis=0)

    # Log transform
    Y_norm = np.log1p(Y_norm + pseudo - 1.0)  # pseudo=1.0 -> log1p

    if not zscore:
        return Y_norm

    # Z-score per gene (safe for 0-variance genes: sklearn outputs 0s)
    Z = pd.DataFrame(
        StandardScaler(with_mean=True, with_std=True).fit_transform(Y_norm),
        index=Y_norm.index,
        columns=Y_norm.columns,
    )
    return Z


def align_XY(X, Y):
    common = X.index.intersection(Y.index)
    X2 = X.loc[common]
    Y2 = Y.loc[common]
    return X2, Y2

##################################
# 2. Supervised Methods
##################################


def fit_alt_to_expr_weights_lasso(
    X_alt, Y_expr,
    alpha_range=np.logspace(-3, 0, 25),
    n_splits=3, n_repeats=1, random_state=44,
    n_jobs=8, max_iter=5000, tol=1e-3,
):
    """
    Per-gene LassoCV on: expr_gene ~ X_alt
    Returns weights: predictors x genes (NaN where coef==0).
    """
    X, Y = align_XY(X_alt, Y_expr)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values

        lcv = LassoCV(
            alphas=alpha_range, cv=cv, random_state=random_state,
            n_jobs=n_jobs, max_iter=max_iter, tol=tol
        )
        lcv.fit(X, y)

        model = Lasso(alpha=lcv.alpha_, max_iter=max_iter, tol=tol, random_state=random_state)
        model.fit(X, y)

        coef = model.coef_
        nz = np.flatnonzero(coef)
        if nz.size:
            W.iloc[nz, W.columns.get_loc(gene)] = coef[nz]

    return W


def fit_alt_to_expr_weights_elasticnet(
    X_alt, Y_expr,
    alpha_range=np.logspace(-3, 0, 25), l1_ratios=(0.5,),
    n_splits=3, n_repeats=1, random_state=44,
    n_jobs=8, max_iter=5000, tol=1e-3,
):
    """
    Per-gene ElasticNetCV on: expr_gene ~ X_alt
    Returns weights: predictors x genes (NaN where coef==0).
    """
    X, Y = align_XY(X_alt, Y_expr)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values

        encv = ElasticNetCV(
            alphas=alpha_range, l1_ratio=list(l1_ratios),
            cv=cv, random_state=random_state, n_jobs=n_jobs,
            max_iter=max_iter, tol=tol
        )
        encv.fit(X, y)

        model = ElasticNet(
            alpha=encv.alpha_, l1_ratio=encv.l1_ratio_,
            max_iter=max_iter, tol=tol, random_state=random_state
        )
        model.fit(X, y)

        coef = model.coef_
        nz = np.flatnonzero(coef)
        if nz.size:
            W.iloc[nz, W.columns.get_loc(gene)] = coef[nz]

    return W


def fit_alt_to_expr_weights_ridge(X_alt, Y_expr, alphas=np.logspace(-3, 3, 13)):
    """
    RidgeCV multi-output regression: Y_expr ~ X_alt (fits ALL genes at once, fast).
    Returns dense weights: predictors x genes.
    """
    X, Y = align_XY(X_alt, Y_expr)
    rcv = RidgeCV(alphas=alphas)
    rcv.fit(X.values, Y.values)
    return pd.DataFrame(rcv.coef_.T, index=X.columns, columns=Y.columns, dtype=float)


def fit_alt_to_expr_weights_svm(X_alt, Y_expr, C=1.0, epsilon=0.1, max_iter=5000, tol=1e-3, random_state=44):
    """
    LinearSVR per gene: expr_gene ~ X_alt
    Returns weights: predictors x genes (NaN where coef==0).
    """
    X, Y = align_XY(X_alt, Y_expr)
    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values
        svr = LinearSVR(C=C, epsilon=epsilon, max_iter=max_iter, tol=tol, random_state=random_state)
        svr.fit(X.values, y)
        coef = svr.coef_
        nz = np.flatnonzero(coef)
        if nz.size:
            W.iloc[nz, W.columns.get_loc(gene)] = coef[nz]

    return W


def fit_alt_to_expr_importances_rf(X_alt, Y_expr, n_estimators=200, max_depth=12, random_state=44, n_jobs=8):
    """
    RF regressor per gene: expr_gene ~ X_alt
    Returns importances: predictors x genes (dense).
    """
    X, Y = align_XY(X_alt, Y_expr)
    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=n_jobs
        )
        rf.fit(X.values, y)
        W[gene] = rf.feature_importances_

    return W

###############################
# 3. Get signatures from methods
###############################


def signature_from_weights_for_alt(
    W,
    alt,
    mode = "nonzero",   # "nonzero" or "elbow"
    coef_tol = 1e-6,
    elbow_tol = 1e-8,
):
    """
    W: predictors x genes (weights or importances)
    alt: predictor name (e.g., TP53_LOF, MYC_AMP)
    mode:
      - "nonzero": return genes with |weight| > coef_tol (good for Lasso/ElasticNet)
      - "elbow": use select_features_elbow on |weights| (good for Ridge/RF/SVR)
    """
    if alt not in W.index:
        return []

    row = W.loc[alt].copy()
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if mode == "nonzero":
        return row.index[np.abs(row.values) > coef_tol].tolist()

    if mode == "elbow":
        return select_features_elbow(row.index, row.values, tol=elbow_tol)

    raise ValueError("mode must be 'nonzero' or 'elbow'")


###############################
# 4. DESEQ2
###############################

def get_deseq2_signature_binary(
    X, Y, gof,
    n_cpus=4,
    alpha=0.05,
    min_group_n=1,
):
    """
    PyDESeq2 v0.4.12-compatible binary DE signature.
    Uses design_factors (NOT design=...).
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



#####################################
# 5. Causal (Deconfounder) Signature
####################################

def get_deconfounder_signature(gof, global_results):
    coefs_tr = global_results['Deconfounder']
    if isinstance(gof, list):
        gof = gof[0]
    gene_signature = coefs_tr[[gof]].dropna()
    return list(gene_signature.index)


###############################
# 7. Wrappers
###############################

def class_supervised_signatures(W_dict, gof):
    """
    Extract signatures for one alteration from precomputed weights/importances.
    """
    signatures = {}
    signatures["Lasso"] = signature_from_weights_for_alt(W_dict["Lasso"], gof, mode="nonzero")
    signatures["ElasticNet"] = signature_from_weights_for_alt(W_dict["ElasticNet"], gof, mode="nonzero")
    signatures["Ridge"] = signature_from_weights_for_alt(W_dict["Ridge"], gof, mode="elbow")
    signatures["SVM"] = signature_from_weights_for_alt(W_dict["SVM"], gof, mode="elbow") 
    signatures["Random Forest"] = signature_from_weights_for_alt(W_dict["Random Forest"], gof, mode="elbow")
    return signatures


def precompute_supervised_weights(X_alt_df, Y_norm_df):
    """
    Fit effect models ONCE: expr ~ alterations.
    Returns dict of method -> weights matrix (predictors x genes).
    """
    W = {}
    W["Lasso"]         = fit_alt_to_expr_weights_lasso(X_alt_df, Y_norm_df)
    W["ElasticNet"]    = fit_alt_to_expr_weights_elasticnet(X_alt_df, Y_norm_df)
    W["Ridge"]         = fit_alt_to_expr_weights_ridge(X_alt_df, Y_norm_df)
    W["SVM"]           = fit_alt_to_expr_weights_svm(X_alt_df, Y_norm_df)
    W["Random Forest"] = fit_alt_to_expr_importances_rf(X_alt_df, Y_norm_df)
    return W


def create_supervised_signatures(
    X,
    Y,
    gof,
    global_results=None,
    W_dict=None,
    min_unique_x=2,
    min_std_x=1e-8,
    min_group_n=1,
):
    # ---- align samples once ----
    common = X.index.intersection(Y.index)
    X = X.loc[common]
    Y = Y.loc[common]

    if gof not in X.columns:
        raise KeyError(f"{gof} not found in X columns.")

    x = pd.to_numeric(X[gof], errors="coerce")
    valid_idx = x.dropna().index
    x = x.loc[valid_idx].astype(int)

    Y_sub = Y.loc[valid_idx]

    nunq = x.nunique(dropna=True)
    std = float(x.std())
    if nunq < min_unique_x or not (std > min_std_x):
        return {"SKIPPED": f"{gof}: predictor constant/low-var (n_unique={nunq}, std={std})"}

    vc = x.value_counts()
    if (0 not in vc) or (1 not in vc) or (vc[0] < min_group_n) or (vc[1] < min_group_n):
        return {"SKIPPED": f"{gof}: insufficient group sizes {vc.to_dict()} (min_group_n={min_group_n})"}

    signatures = {}

    # ---- effect-style ML models: extract from precomputed weights ----
    if W_dict is not None:
        signatures.update(class_supervised_signatures(W_dict, gof))


    # ---- Deconfounder ----
    if global_results is not None:
        signatures["Deconfounder"] = get_deconfounder_signature(gof, global_results)

    
    # ---- DESeq2 ----
    try:
        if (Y_sub < 0).any().any():
            raise ValueError(f"{gof}: Y contains negative values; DESeq2 requires non-negative counts.")

        X_design = pd.DataFrame({gof: x}, index=x.index)
        signatures["DESeq2"] = get_deseq2_signature_binary(X_design, Y_sub, gof)

    except Exception as e:
        signatures["DESeq2_ERROR"] = repr(e)

    return signatures

