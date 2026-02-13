# ========================
# Libraries
# ========================
import numpy as np
import pandas as pd

# Machine learning models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LassoCV, ElasticNetCV, Ridge
from sklearn.svm import SVC, SVR
from sklearn.svm import LinearSVC


# Dimensionality reduction / clustering
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# Model evaluation
from sklearn.model_selection import RepeatedKFold

# Statistics
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Hierarchical clustering
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

# RNA-seq differential expression
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference


from pandas.api.types import CategoricalDtype

# Custom functions
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

    No min/max, no manual thresholds.
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


def normalize_counts_log_cpm(Y_counts, libsize_target=1e4):
    """
    CPM-like scaling + log1p, then z-score per gene.
    Returns a DataFrame with the same index/columns (minus any all-zero samples).

    Steps:
    1. Compute library size per sample.
    2. Scale counts so total = libsize_target (CPM-like).
    3. Log1p transform for variance stabilization.
    4. Z-score per gene so ML models behave well.
    """
    # Total counts per sample
    s = Y_counts.sum(axis=1)

    # Drop all-zero samples (DESeq2 would drop them too)
    keep = s > 0
    Y_counts = Y_counts.loc[keep]
    s = s[keep]

    # Scale factor to reach target library size
    p = libsize_target / s
    Y_norm = Y_counts.mul(p, axis=0)

    # Log1p transform
    Y_norm = np.log1p(Y_norm)

    # Z-score per gene
    Z = pd.DataFrame(
        StandardScaler(with_mean=True, with_std=True).fit_transform(Y_norm),
        index=Y_norm.index,
        columns=Y_norm.columns,
    )

    return Z


'''
For Binary Features (Mutations/Fusions) use classification algorithims
'''

def get_lasso_class_signature(X, y, tol=1e-6, cv=3, Cs=10, max_iter=3000, n_jobs=4):
    if len(np.unique(y.dropna())) < 2:
        return []

    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",              # faster/more scalable than liblinear for many features
        random_state=44,
        cv=cv,
        Cs=Cs,
        max_iter=max_iter,
        n_jobs=n_jobs
    )
    model.fit(X, y)
    coefs = model.coef_[0]
    return X.columns[np.abs(coefs) > tol].tolist()


def get_elasticnet_class_signature(X, y, tol=1e-6, cv=3, Cs=10, max_iter=3000, n_jobs=4):
    if len(np.unique(y.dropna())) < 2:
        return []

    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.5],
        random_state=44,
        cv=cv,
        Cs=Cs,
        max_iter=max_iter,
        n_jobs=n_jobs
    )
    model.fit(X, y)
    coefs = model.coef_[0]
    return X.columns[np.abs(coefs) > tol].tolist()


def get_svm_class_signature(X, y):
    if len(np.unique(y.dropna())) < 2:
        return []

    model = LinearSVC(
        random_state=44,
        max_iter=5000,
        tol=1e-3,
    )
    model.fit(X, y)

    coefs = model.coef_[0]
    return select_features_elbow(X.columns, coefs)



def get_ridgereg_class_signature(X, y, tol=1e-3):
    """Logistic regression with L2: elbow on |weights|."""
    if len(np.unique(y.dropna())) < 2:
        return []

    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        random_state=44,
        max_iter=3000,
        tol=tol
    )

    model.fit(X, y)
    coefs = model.coef_[0]
    return select_features_elbow(X.columns, coefs)


def get_rf_class_signature(X, y):
    """Random forest classifier: elbow on feature_importances."""
    if len(np.unique(y.dropna())) < 2:
        return []

    model = RandomForestClassifier(
        random_state=44,
        n_estimators=200,
        max_depth=12,
    )
    model.fit(X, y)
    importances = model.feature_importances_
    return select_features_elbow(X.columns, importances)

'''
DESEQ2
'''

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

    # metadata table (v0.4.x uses metadata=)
    metadata = pd.DataFrame(index=counts.index)
    metadata["condition"] = x.astype(str)

    # Make it categorical and set reference level explicitly
    metadata["condition"] = pd.Categorical(metadata["condition"], categories=["0", "1"])

    # v0.4.12 API: design_factors, ref_level
    # (DefaultInference exists in 0.4.12, but inference= is optional here)
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




'''
Causal (Deconfounder) Signature
'''
def get_deconfounder_signature(gof, global_results):
    coefs_tr = global_results['Deconfounder']
    if isinstance(gof, list):
        gof = gof[0]
    gene_signature = coefs_tr[[gof]].dropna()
    return list(gene_signature.index)

def class_supervised_signatures(Y_norm_df, x):
    """
    Supervised signatures for binary alterations (mutations, fusions, binarised CNAs).
    X = normalized RNA, y = binary alteration status.
    """
    signatures = {}
    signatures['Random Forest']       = get_rf_class_signature(Y_norm_df, x)
    signatures['Lasso']               = get_lasso_class_signature(Y_norm_df, x)
    signatures['ElasticNet']          = get_elasticnet_class_signature(Y_norm_df, x)
    signatures['SVM']                 = get_svm_class_signature(Y_norm_df, x)
    signatures['Ridge']               = get_ridgereg_class_signature(Y_norm_df, x)
    return signatures


def create_supervised_signatures(
    X,
    Y,
    gof,
    global_results=None,
    Y_norm_df=None,
    min_unique_x=2,
    min_std_x=1e-8,
    min_group_n=1,   # NEW: prevent tiny groups
):
    """
    Binary predictor signatures only (mut/fusion/AMP/DEL).

    - ML models use normalized RNA
    - DESeq2 uses raw counts
    """

    # ---- align samples once ----
    common_idx = X.index.intersection(Y.index)
    X = X.loc[common_idx]
    Y = Y.loc[common_idx]

    if gof not in X.columns:
        raise KeyError(f"{gof} not found in X columns.")

    # predictor
    x = pd.to_numeric(X[gof], errors="coerce")

    # ---- drop NaNs consistently ----
    valid_idx = x.dropna().index
    x = x.loc[valid_idx]
    Y_sub = Y.loc[valid_idx]

    # ---- enforce 0/1 (important for robustness) ----
    # if x is already 0/1 this is a no-op; if it’s float, it will coerce cleanly
    x = x.astype(int)

    # ---- guard: constant / near-constant predictor ----
    nunq = x.nunique(dropna=True)
    std = float(x.std())
    if nunq < min_unique_x or not (std > min_std_x):
        return {"SKIPPED": f"{gof}: predictor constant/low-var (n_unique={nunq}, std={std})"}

    # ---- group-size guard ----
    vc = x.value_counts()
    if (0 not in vc) or (1 not in vc) or (vc[0] < min_group_n) or (vc[1] < min_group_n):
        return {"SKIPPED": f"{gof}: insufficient group sizes {vc.to_dict()} (min_group_n={min_group_n})"}

    # ---- normalize RNA for ML models (NOT for DESeq2) ----
    if Y_norm_df is None:
        Y_norm_df = normalize_counts_log_cpm(Y)

    common2 = x.index.intersection(Y_norm_df.index)
    x_ml = x.loc[common2]
    Y_ml = Y_norm_df.loc[common2]

    signatures = {}

    # ---- classification ONLY ----
    base_sigs = class_supervised_signatures(Y_ml, x_ml)
    signatures.update(dict(base_sigs))

    # ---- Deconfounder ----
    if global_results is not None:
        try:
            signatures["Deconfounder"] = get_deconfounder_signature(gof, global_results)
        except Exception as e:
            signatures["Deconfounder_ERROR"] = repr(e)

    # ---- DESeq2 ----
    try:
        if (Y_sub < 0).any().any():
            raise ValueError(f"{gof}: Y contains negative values; DESeq2 requires non-negative counts.")

        # NEW: pass only the needed design column
        X_design = pd.DataFrame({gof: x}, index=x.index)

        signatures["DESeq2"] = get_deseq2_signature_binary(X_design, Y_sub, gof)

    except Exception as e:
        signatures["DESeq2_ERROR"] = repr(e)

    return signatures



