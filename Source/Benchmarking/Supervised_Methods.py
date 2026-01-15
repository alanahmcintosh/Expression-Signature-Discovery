# ========================
# Libraries
# ========================
import numpy as np
import pandas as pd

# Machine learning models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LassoCV, ElasticNetCV, Ridge
from sklearn.svm import SVC, SVR


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
try:
    # Newer API (if they ever move it here)
    from pydeseq2.inference import DefaultInference
except ImportError:
    # Your installed version (0.4.12) puts it here
    from pydeseq2.default_inference import DefaultInference


from pandas.api.types import CategoricalDtype

# Custom functions
from Deconfounder import *




'''
Helper Functions
'''

def _select_features_elbow(feature_names, weights, tol=1e-8):
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
        class_weight="balanced",
        random_state=44,
        cv=cv,
        Cs=Cs,
        max_iter=max_iter,
        n_jobs=n_jobs,
        scoring="balanced_accuracy",
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
        class_weight="balanced",
        l1_ratios=[0.5],
        random_state=44,
        cv=cv,
        Cs=Cs,
        max_iter=max_iter,
        n_jobs=n_jobs,
        scoring="balanced_accuracy",
    )
    model.fit(X, y)
    coefs = model.coef_[0]
    return X.columns[np.abs(coefs) > tol].tolist()


from sklearn.svm import LinearSVC

def get_svm_class_signature(X, y):
    if len(np.unique(y.dropna())) < 2:
        return []

    model = LinearSVC(
        class_weight="balanced",
        random_state=44,
        max_iter=5000,
        tol=1e-3,
    )
    model.fit(X, y)

    coefs = model.coef_[0]
    return _select_features_elbow(X.columns, coefs)



def get_ridgereg_class_signature(X, y):
    """Logistic regression with L2: elbow on |weights|."""
    if len(np.unique(y.dropna())) < 2:
        return []

    model = LogisticRegression(
        penalty="l2",
        solver="saga",
        class_weight="balanced",
        random_state=44,
        max_iter=3000,
        tol=1e-3,
        n_jobs=4,   # saga supports this
    )

    model.fit(X, y)
    coefs = model.coef_[0]
    return _select_features_elbow(X.columns, coefs)


def get_rf_class_signature(X, y):
    """Random forest classifier: elbow on feature_importances."""
    if len(np.unique(y.dropna())) < 2:
        return []

    model = RandomForestClassifier(
        class_weight='balanced',
        random_state=44,
        n_estimators=200,
        max_depth=12,
        n_jobs=4,
    )
    model.fit(X, y)
    importances = model.feature_importances_
    return _select_features_elbow(X.columns, importances)


'''
For Continuous Features (CNAs) use regression algorithims
'''

def get_lasso_reg_signature(X, y, tol=1e-6, cv=3, max_iter=3000, n_jobs=4):
    model = LassoCV(cv=cv, random_state=44, n_jobs=n_jobs, max_iter=max_iter, tol=1e-3)
    model.fit(X, y)
    coefs = model.coef_
    return X.columns[np.abs(coefs) > tol].tolist()


def get_elasticnet_reg_signature(X, y, tol=1e-6, cv=3, max_iter=3000, n_jobs=4):
    model = ElasticNetCV(
        cv=cv,
        random_state=44,
        n_jobs=n_jobs,
        max_iter=max_iter,
        tol=1e-3,
        l1_ratio=[0.5],
    )
    model.fit(X, y)
    coefs = model.coef_
    return X.columns[np.abs(coefs) > tol].tolist()


def get_ridge_reg_signature(X, y):
    model = Ridge(solver="sag", random_state=44, tol=1e-3)
    model.fit(X, y)
    coefs = model.coef_
    return _select_features_elbow(X.columns, coefs)

from sklearn.svm import LinearSVR

def get_svr_reg_signature(X, y):
    model = LinearSVR(
        random_state=44,
        max_iter=5000,
        tol=1e-3,
    )
    model.fit(X, y)
    coefs = model.coef_
    return _select_features_elbow(X.columns, coefs)


def get_rf_reg_signature(X, y):
    model = RandomForestRegressor(
        random_state=44,
        n_estimators=200,
        n_jobs=4,
        max_depth=12
    )
    model.fit(X, y)
    importances = model.feature_importances_
    return _select_features_elbow(X.columns, importances)

'''
DESEQ2
'''


def get_deseq2_signature_binary(X, Y, gof):
    """
    X_train: mutation/CNA/fusion matrix
    Y_train: gene expression count matrix (int)
    gof: alteration of interest
    """
    # Only keep the alteration of interest in metadata
    metadata = X.copy()
    counts_df = Y.astype(int)
    inference = DefaultInference(n_cpus=4)

    # Make sure it's binary categorical
    # Make sure it's binary categorical (robust to 0.0/1.0, True/False, strings)
    s = pd.to_numeric(metadata[gof], errors="coerce").fillna(0).astype(int)
    metadata[gof] = s.astype(str).astype("category")
    metadata[gof] = metadata[gof].cat.reorder_categories(["0", "1"], ordered=True)


    dds = DeseqDataSet(
        counts=counts_df,
        metadata=metadata,
        design_factors=[gof],
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()

    contrast = [gof, "1", "0"]
    stat_res = DeseqStats(dds, contrast=contrast, inference=inference)
    stat_res.summary()

    de = stat_res.results_df
    significant_genes = de[de['padj'] < 0.05]
    return list(significant_genes.index)

import numpy as np

def get_deseq2_signature_cont(X, Y, gof, min_std=1e-6):
    """
    DESeq2 signature for continuous 0–25 integer-valued GOFs/CNAs.
    Uses a numeric contrast vector (required in recent PyDESeq2 versions).
    """
    # Only keep the alteration of interest in metadata
    metadata = X.copy()
    counts_df = Y.astype(int)

    # Force numeric and check variance
    metadata[gof] = pd.to_numeric(metadata[gof], errors="coerce")
    col = metadata[gof]

    # Drop rows with NaN in this predictor (if any)
    valid_idx = col.dropna().index
    if len(valid_idx) < 3:  # not enough samples to fit anything sensible
        print(f"[DESeq2 cont] {gof}: <3 non-NaN samples. Skipping.")
        return []

    col = col.loc[valid_idx]
    std = col.std()

    if std < min_std:
        print(f"[DESeq2 cont] {gof}: ~zero variance (std={std:.2e}). Skipping.")
        return []

    metadata = metadata.loc[valid_idx]
    counts_df = counts_df.loc[valid_idx]

    inference = DefaultInference(n_cpus=4)

    try:
        # Simple design with intercept + continuous term
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design=f"~ {gof}",      # use formula syntax; continuous auto-detected
            refit_cooks=True,
            inference=inference,
        )
        dds.deseq2()

        # --- Build numeric contrast vector for the continuous term ---
        # Get design matrix (can be an attribute or in obsm, depending on version)
        design = getattr(dds, "design_matrix", None)
        if design is None:
            design = dds.obsm["design_matrix"]

        cols = design.columns

        # Column name may be exactly gof, or something like 'gof[]' in some versions
        if gof in cols:
            target_col = gof
        else:
            matches = [c for c in cols if c.startswith(gof)]
            if len(matches) != 1:
                print(f"[DESeq2 cont] {gof}: can't uniquely match to design matrix columns {list(cols)}. Skipping.")
                return []
            target_col = matches[0]

        contrast_vec = np.zeros(len(cols), dtype=float)
        contrast_vec[cols.get_loc(target_col)] = 1.0  # test the slope for this term

        stat_res = DeseqStats(
            dds,
            contrast=contrast_vec,    # ✅ numeric contrast vector
            inference=inference,
        )
        stat_res.summary()

        de = stat_res.results_df
        sig = de[de["padj"] < 0.05]
        return list(sig.index)

    except Exception as e:
        print(f"[DESeq2 cont] {gof}: failed with error: {e}")
        return []



def get_deseq2_signature_auto(X, Y, gof):
    """
    Automatically choose DESeq2 mode based on feature name.

    - If endswith('_MUT') or endswith('_FUSION') → binary
    - Else → continuous predictor
    """
    gof_lower = gof.lower()

    if gof_lower.endswith("_cna"):
        print(f"[DESeq2 Auto] {gof}: treated as CONT.")
        return get_deseq2_signature_cont(X, Y, gof)

    print(f"[DESeq2 Auto] {gof}: treated as BINARY")
    return get_deseq2_signature_binary(X, Y, gof)


'''
Causal (Deconfounder) Signature
'''
def get_deconfounder_signature(gof, global_results):
    coefs_tr = global_results['Deconfounder']
    if isinstance(gof, list):
        gof = gof[0]
    gene_signature = coefs_tr[[gof]].dropna()
    return list(gene_signature.index)


'''
Wrapper Functions
'''

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
    signatures['Ridge Regression']    = get_ridgereg_class_signature(Y_norm_df, x)
    return signatures


def reg_supervised_signatures(Y_norm_df, x):
    """
    Supervised signatures for continuous alterations (e.g. CNA values).
    X = normalized RNA, y = continuous alteration (CNA) value.
    """
    signatures = {}
    signatures['Random Forest (reg)']  = get_rf_reg_signature(Y_norm_df, x)
    signatures['Lasso (reg)']          = get_lasso_reg_signature(Y_norm_df, x)
    signatures['ElasticNet (reg)']     = get_elasticnet_reg_signature(Y_norm_df, x)
    signatures['SVR (reg)']            = get_svr_reg_signature(Y_norm_df, x)
    signatures['Ridge (reg)']          = get_ridge_reg_signature(Y_norm_df, x)
    return signatures



def create_supervised_signatures(
    X,
    Y,
    gof,
    global_results=None,
    Y_norm_df=None
):
    """
    Build signatures for a single alteration (gof).
    """
    # ---- align samples once ----
    common_idx = X.index.intersection(Y.index)
    X = X.loc[common_idx]
    Y = Y.loc[common_idx]

    if gof not in X.columns:
        raise KeyError(f"{gof} not found in X columns.")

    x = X[gof]

    # ---- normalize RNA for ML models (NOT for DESeq2) ----
    if Y_norm_df is None:
        Y_norm_df = normalize_counts_log_cpm(Y)

    # match x to the normalized RNA samples (in case some got dropped)
    x = x.loc[Y_norm_df.index]

    # ---- choose classification vs regression based on name ----
    if gof.endswith("_CNA"):
        # Uses your existing regression signature function
        base_sigs = reg_supervised_signatures(Y_norm_df, x) 
    else:
        # Uses your existing classification signature function
        base_sigs = class_supervised_signatures(Y_norm_df, x) 

    signatures = dict(base_sigs)

    # ---- add Deconfounder (never crash the alteration) ----
    if global_results is not None:
        try:
            signatures['Deconfounder'] = get_deconfounder_signature(gof, global_results)
        except Exception as e:
            signatures['Deconfounder_ERROR'] = repr(e)

    # ---- add DESeq2 (never crash the alteration) ----
    try:
        signatures["DESeq2"] = get_deseq2_signature_auto(X, Y, gof)
    except Exception as e:
        signatures["DESeq2_ERROR"] = repr(e)

    return signatures
