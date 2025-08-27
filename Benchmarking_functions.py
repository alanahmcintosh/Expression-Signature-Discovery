# ========================
# Libraries
# ========================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LassoCV, Lasso, RidgeCV, ElasticNetCV
from sklearn.svm import SVC, SVR
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from sklearn.model_selection import RepeatedKFold
import joblib
from Functions import ppca, predicitve_check
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, fisher_exact


# ========================
# Supervised Methods
# ========================
def deconfounder(X, Y, alpha_range=np.arange(0.01, 1, 0.01), n_splits=3, n_repeats=3, random_state=1):
    """
    Run LassoCV and train Lasso on each gene in Y with cross-validation.
    
    Parameters:
        X (pd.DataFrame): Feature matrix.
        Y (pd.DataFrame): Target gene expression matrix.
        alpha_range (np.ndarray): Grid of alpha values for LassoCV.
        n_splits (int): Number of CV folds.
        n_repeats (int): Number of CV repetitions.
        random_state (int): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: Coefficients (non-zero) for each target gene.
        list: Trained Lasso models.
        list: R² values per model.
    """
    coefs = pd.DataFrame(index=X.columns)
    models = []
    r2_scores = []

    for i in range(Y.shape[1]):
        y = Y.iloc[:, i]
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        lassocv = LassoCV(alphas=alpha_range, cv=cv, random_state=random_state)
        lassocv.fit(X, y)

        model = Lasso(alpha=lassocv.alpha_)
        model.fit(X, y)
        models.append(model)

        nz_idx = model.coef_.nonzero()[0]
        nz_features = X.columns[nz_idx]
        coefs.loc[nz_features, i] = model.coef_[nz_idx]
        r2_scores.append(model.score(X, y))

    return coefs, models, r2_scores

def choose_latent_dim_ppca(X, k_range=range(2, 99), holdout_portion=0.2, pval_cutoff=0.1, seed=123
):
    for k in k_range:
        model = ppca(factors=k)
        model.holdout(X, holdout_portion=holdout_portion, seed=seed)
        model.max_likelihood(model.x_train, standardise=False)
        pval = predicitve_check(model, k, holdout_portion=holdout_portion)
        
        if pval >= pval_cutoff:
            return k  # first valid k found
    
    # If none satisfy the cutoff, return max tested value
    return k_range[-1]

def precompute_global_results(X, Y):
    """
    Run the full Deconfounder pipeline on input matrices X and Y.
    Adds inferred latent variables, normalizes expression, runs Lasso, and returns signatures.
    """
    k = choose_latent_dim_ppca(X)
    print(f"Selected latent dimension: {k}")
    print("Precomputing Deconfounder...")

    m_ppca = ppca(k)
    m_ppca.holdout(X, seed=44)
    m_ppca.max_likelihood(m_ppca.x_train, standardise=False)
    _ = m_ppca.generate(1)
    _ = predicitve_check(m_ppca, k)

    latent_df = pd.DataFrame(m_ppca.z_mu.T, index=X.index, columns=[f"latent_{i}" for i in range(k)])
    augmented_X = pd.concat([X, latent_df], axis=1).astype(float)
    augmented_X.columns = augmented_X.columns.astype(str)

    # Normalize gene expression using library size normalization + log transform
    size_factors = 10000 / Y.sum(axis=1)
    Y_norm = np.log1p(Y.mul(size_factors, axis=0))
    Y_scaled = pd.DataFrame(StandardScaler().fit_transform(Y_norm), index=Y.index)
    Y_scaled.columns = [f'Expression_Gene_{i}' for i in range(Y.shape[1])]
    causal_signatures = {}
    coefs, models, R2 = deconfounder(augmented_X, Y_scaled)
    coefs_tr = coefs.T.set_index(Y.columns)
    causal_signatures['Deconfounder'] = coefs_tr

    print("Global precomputation completed.")
    return causal_signatures

# ========================
# Logistic Regression Methods
# ========================

def get_lasso_class_signature(X, y):
    model = LogisticRegressionCV(penalty='l1', solver='liblinear', class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0].tolist()

def get_elasticnet_class_signature(X, y):
    model = LogisticRegressionCV(penalty='elasticnet', solver='saga', class_weight='balanced',
                                  l1_ratios=[0.5], random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0].tolist()

def get_svm_class_signature(X, y): 
    model = SVC(kernel='linear', class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_).flatten() > 0.05].tolist() if hasattr(model, 'coef_') else []

def get_ridgereg_class_signature(X, y):
    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0.05].tolist()

def get_rf_class_signature(X, y):
    model = RandomForestClassifier(class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[model.feature_importances_ > 0.005].tolist()

# ========================
# Linear Regression Methods
# ========================

def get_lasso_lin_signature(X, y):
    model = LassoCV(cv=5, random_state=44, max_iter=5000)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_) > 1e-6].tolist()


def get_ridge_lin_signature(X, y):
    model = RidgeCV(alphas=np.logspace(-3, 3, 50))
    model.fit(X, y)
    return X.columns[np.abs(model.coef_) > 1e-6].tolist()


def get_rf_lin_signature(X, y):
    model = RandomForestRegressor(n_estimators=500, random_state=44, n_jobs=-1)
    model.fit(X, y)
    importances = model.feature_importances_
    # pick nonzero importance features
    return X.columns[importances > 0].tolist()

def get_svm_lin_signature(X, y):
    model = SVR(kernel="linear", C=1.0)
    model.fit(X, y)
    coefs = model.coef_.ravel()
    return X.columns[np.abs(coefs) > 1e-6].tolist()

def get_elasticnet_lin_signature(X, y):
    en = ElasticNetCV(l1_ratio=[.3, .5, .7], cv=5, random_state=44)
    en.fit(X, y)
    return X.columns[np.abs(en.coef_) > 1e-8].tolist()


def get_deconfounder_signature(gof, global_results):
    coefs_tr = global_results['Deconfounder']
    if isinstance(gof, list):
        gof = gof[0]
    gene_signature = coefs_tr[[gof]].dropna()
    return list(gene_signature.index)



def get_deseq2_signature_continuous_cna(
    X_train: pd.DataFrame,
    Y_counts: pd.DataFrame,
    gof: str,
    alpha: float = 0.05,
):
    """
    DESeq2 with a continuous CNA covariate (already centered).
    Model: counts ~ cna_ctr
    Returns (significant_gene_list, full_results_df)
    """
    # Align samples
    idx = Y_counts.index.intersection(X_train.index)
    counts_df = Y_counts.loc[idx].round().astype(int)

    # Centered CNA vector (you said it's pre-centered already)
    cna_ctr = X_train.loc[idx, gof].astype(float)
    if cna_ctr.nunique() <= 1:
        raise ValueError(f"{gof}: no variance in CNA vector (all values equal).")

    # Design (metadata)
    meta = pd.DataFrame({"cna_ctr": cna_ctr}, index=idx)

    # Fit DESeq2
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=meta,
        design_factors=["cna_ctr"],   # treat as continuous covariate
        refit_cooks=True,
    )
    dds.deseq2()

    # Pull coefficient for the continuous term
    stats = DeseqStats(dds, name="cna_ctr", inference=inference)  # <-- use name=, not contrast=
    stats.summary()
    res = (stats.results_df
           .rename(columns={
               "log2FoldChange": "log2FC_per_copy",
               "lfcSE": "SE",
               "stat": "Wald_stat",
               "pvalue": "pval",
               "padj": "qval",
               "baseMean": "baseMean",
           })
           .sort_values("pval"))

    sig = res[res["qval"] < alpha]
    return list(sig.index), res


def get_deseq2_signature_auto(
    X_train: pd.DataFrame,
    Y_counts: pd.DataFrame,
    gof: str,
    alpha: float = 0.05
):
    """
    If X[gof] is binary (0/1) -> categorical DESeq2 with contrast [gof, "1", "0"].
    Else -> continuous CNA path above.
    """
    vals = pd.unique(X_train[gof].dropna())
    is_binary = set(vals).issubset({0, 1})

    idx = Y_counts.index.intersection(X_train.index)
    counts_df = Y_counts.loc[idx].round().astype(int)
    inference = DefaultInference(n_cpus=8)

    if is_binary:
        meta = X_train.loc[idx, [gof]].copy()
        # ensure strings "0","1" with "0" as reference
        meta[gof] = meta[gof].astype(int).astype(str).astype("category")
        if {"0", "1"}.issubset(set(meta[gof].cat.categories)):
            meta[gof] = meta[gof].cat.reorder_categories(["0", "1"], ordered=True)

        dds = DeseqDataSet(
            counts=counts_df,
            metadata=meta,
            design_factors=[gof],
            refit_cooks=True,
        )
        dds.deseq2()

        stats = DeseqStats(dds, contrast=[gof, "1", "0"], inference=inference)
        stats.summary()
        res = (stats.results_df
               .rename(columns={
                   "log2FoldChange": "log2FC",
                   "lfcSE": "SE",
                   "stat": "Wald_stat",
                   "pvalue": "pval",
                   "padj": "qval",
                   "baseMean": "baseMean",
               })
               .sort_values("pval"))
        sig = res[res["qval"] < alpha]
        return list(sig.index), res

    # Continuous (already centered) CNA
    return get_deseq2_signature_continuous_cna(X_train, Y_counts, gof, alpha=alpha)


def get_deseq2_signature_auto_list(X, Y_counts, gof, alpha=0.05):
    sig_list, _ = get_deseq2_signature_auto(X, Y_counts, gof, alpha)
    return sig_list

# ========================
# Wrapper: Create Supervised Signatures
# ========================

def is_cna(col: str) -> bool:
    return col.endswith("_CNA") or col.endswith("_CNA_CNA")

def is_binary_col(col: str) -> bool:
    return col.endswith("_MUT") or col.endswith("_FUSION")

def is_binary_vector(x: pd.Series) -> bool:
    vals = pd.unique(x.dropna())
    return set(vals).issubset({0, 1})

def normalize_counts_log_cpm(Y_counts, libsize_target=1e4):
    """CPM-like scaling + log1p, then z-score per gene; returns DataFrame with same index/cols."""
    s = Y_counts.sum(axis=1).replace(0, np.nan)  # avoid div-by-zero
    p = libsize_target / s
    Y_norm = np.log1p(Y_counts.mul(p, axis=0))
    Z = pd.DataFrame(
        StandardScaler(with_mean=True, with_std=True).fit_transform(Y_norm),
        index=Y_norm.index,
        columns=Y_norm.columns,
    )
    return Z

def class_supervised_signatures(Y_norm_df, x):
    """Build mutation/fusion signatures via classifiers: X=RNA, y=alteration (0/1)."""
    signatures = {}
    signatures['Random Forest']        = get_rf_class_signature(Y_norm_df, x)
    signatures['Lasso']                = get_lasso_class_signature(Y_norm_df, x)
    signatures['ElasticNet']           = get_elasticnet_class_signature(Y_norm_df, x)
    signatures['SVM']                  = get_svm_class_signature(Y_norm_df, x)
    signatures['Ridge Regression']  = get_ridgereg_class_signature(Y_norm_df, x)
    return signatures

def reg_supervised_signatures(Y_norm_df, x):
    """Build CNA signatures via regressors: X=RNA, y=CNA (continuous/centered or raw)."""
    signatures = {}
    signatures['Random Forest']        = get_rf_lin_signature(Y_norm_df, x)
    signatures['Lasso']                = get_lasso_lin_signature(Y_norm_df, x)
    signatures['ElasticNet']           = get_elasticnet_lin_signature(Y_norm_df, x)
    signatures['SVM']                  = get_svm_lin_signature(Y_norm_df, x)
    signatures['Ridge Regression']     = get_ridge_lin_signature(Y_norm_df, x)  # <-- label fixed
    return signatures


def create_supervised_signatures(X, Y, gof, global_results = None):
    """
    Build signatures for a single alteration (gof).
    - Reverse direction models use X=RNA (normalized), y=alteration vector x.
    - DESeq2 uses raw counts Y (not normalized) under the hood.
    """
    # ---- align samples once ----
    common_idx = X.index.intersection(Y.index)
    X = X.loc[common_idx]
    Y = Y.loc[common_idx]

    # alteration vector y/x
    if gof not in X.columns:
        raise KeyError(f"{gof} not found in X columns.")
    x = X[gof]

    # normalize RNA for ML models (NOT for DESeq2)
    Y_norm_df = normalize_counts_log_cpm(Y)

    # pick path by alteration type (binary vs continuous) using the actual vector
    if is_binary_vector(x):
        signatures = class_supervised_signatures(Y_norm_df, x)
    else:
        signatures = reg_supervised_signatures(Y_norm_df, x)

    # add extra methods
    if global_results is not None:
        signatures['Deconfounder'] = get_deconfounder_signature(gof, global_results)

    signatures['DESeq2'] = get_deseq2_signature_auto_list(X, Y, gof)

    return signatures


# ========================
# Unsupervised Methods
# ========================

def create_signatures_kmeans(X, Y, method_name='K-means', verbose=True, min_cluster_size=5, pval_thresh=0.2):
    from scipy.stats import fisher_exact, chi2_contingency
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    km_signatures = {}
    Y_scaled_df = pd.DataFrame(Y, index=Y.index, columns=Y.columns)

    scores = {}
    for k in range(2, 21):
        try:
            kmeans = KMeans(n_clusters=k, random_state=44).fit(Y)
            score = silhouette_score(Y, kmeans.labels_)
            scores[k] = score
        except Exception as e:
            print(f"[{method_name}] Failed clustering for k={k}: {e}")

    if not scores:
        print(f"[{method_name}] No valid clustering configurations succeeded.")
        return {}

    best_k = max(scores, key=scores.get)
    if verbose:
        print(f"[{method_name}] Best k={best_k}, Silhouette scores: {scores}")

    cluster_labels = pd.Series(KMeans(n_clusters=best_k, random_state=44).fit_predict(Y), index=Y.index)
    cluster_sizes = cluster_labels.value_counts().to_dict()
    if verbose:
        print(f"[{method_name}] Cluster distribution: {cluster_sizes}")

    if any(count < min_cluster_size for count in cluster_sizes.values()):
        print(f"[{method_name}] Warning: One or more clusters have <{min_cluster_size} samples.")

    mutation_pvals = []
    for mutation in X.columns:
        for cluster in sorted(cluster_labels.unique()):
            try:
                contingency = pd.crosstab((cluster_labels == cluster).astype(int), X[mutation])
                if contingency.shape != (2, 2):
                    continue
                if (contingency.values < 5).any():
                    _, p = fisher_exact(contingency)
                else:
                    _, p, _, _ = chi2_contingency(contingency)
                mutation_pvals.append({'Mutation': mutation, 'Cluster': cluster, 'p_value': p})
            except Exception as e:
                print(f"[{method_name}] Warning: Stat test failed for {mutation} – {e}")

    pvals_df = pd.DataFrame(mutation_pvals)
    cluster_labels.to_csv(f"{method_name}_cluster_labels.csv")
    pvals_df.to_csv(f"{method_name}_pvals.csv")

    # Plot histogram
    plt.figure()
    pvals_df['p_value'].hist(bins=40)
    plt.axvline(pval_thresh, color='red', linestyle='--', label='Threshold')
    plt.title(f"{method_name} p-value distribution")
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{method_name}_pval_hist.png")
    plt.close()

    # Filter associations
    filtered = pvals_df[pvals_df['p_value'] < pval_thresh]
    if filtered.empty:
        print(f"[{method_name}] No associations passed p-value < {pval_thresh}")
        return {}

    best_assoc = filtered.sort_values('p_value').drop_duplicates('Mutation')

    mean_expr = Y_scaled_df.groupby(cluster_labels).mean()
    defining_genes_dict = {
        cluster: mean_expr.loc[cluster].sort_values(ascending=False).head(50).index.tolist()
        for cluster in cluster_labels.unique()
    }

    for cluster in sorted(cluster_labels.unique()):
        assoc_mut = best_assoc[best_assoc['Cluster'] == cluster]
        top_mutation = assoc_mut.sort_values('p_value').head(1)['Mutation'].values if not assoc_mut.empty else [f"Cluster_{cluster}_placeholder"]
        defining_genes = defining_genes_dict.get(cluster, [])
        km_signatures.setdefault(method_name, {})[top_mutation[0]] = defining_genes

    return km_signatures


def create_kmeans_nmf_signature(Y, X, method_name='NMF-KMeans', n_components=20, verbose=True, min_cluster_size=5, pval_thresh=0.2):
    from scipy.stats import fisher_exact, chi2_contingency
    from sklearn.decomposition import NMF
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import matplotlib.pyplot as plt

    all_signatures = {}

    try:
        nmf_model = NMF(n_components=n_components, init='random', random_state=44, max_iter=1000)
        W = nmf_model.fit_transform(Y.abs())
        H = nmf_model.components_
    except Exception as e:
        print(f"[{method_name}] NMF failed: {e}")
        return {}

    W_df = pd.DataFrame(W, index=Y.index, columns=[f'NMF_{i}' for i in range(n_components)])
    H_df = pd.DataFrame(H, columns=Y.columns, index=[f'NMF_{i}' for i in range(n_components)])

    scores = {}
    for k in range(2, 21):
        try:
            kmeans = KMeans(n_clusters=k, random_state=44).fit(W_df)
            score = silhouette_score(W_df, kmeans.labels_)
            scores[k] = score
        except Exception as e:
            print(f"[{method_name}] Failed clustering for k={k}: {e}")

    if not scores:
        print(f"[{method_name}] No valid clustering configurations succeeded.")
        return {}

    best_k = max(scores, key=scores.get)
    if verbose:
        print(f"[{method_name}] Best k={best_k}, Silhouette scores: {scores}")

    cluster_labels = pd.Series(KMeans(n_clusters=best_k, random_state=44).fit_predict(W_df), index=Y.index)
    cluster_sizes = cluster_labels.value_counts().to_dict()
    if verbose:
        print(f"[{method_name}] Cluster distribution: {cluster_sizes}")

    if any(count < min_cluster_size for count in cluster_sizes.values()):
        print(f"[{method_name}] Warning: One or more clusters have <{min_cluster_size} samples.")

    mutation_pvals = []
    for mutation in X.columns:
        for cluster in sorted(cluster_labels.unique()):
            try:
                contingency = pd.crosstab((cluster_labels == cluster).astype(int), X[mutation])
                if contingency.shape != (2, 2):
                    continue
                if (contingency.values < 5).any():
                    _, p = fisher_exact(contingency)
                else:
                    _, p, _, _ = chi2_contingency(contingency)
                mutation_pvals.append({'Mutation': mutation, 'Cluster': cluster, 'p_value': p})
            except Exception as e:
                print(f"[{method_name}] Warning: Stat test failed for {mutation} – {e}")

    pvals_df = pd.DataFrame(mutation_pvals)
    cluster_labels.to_csv(f"{method_name}_cluster_labels.csv")
    pvals_df.to_csv(f"{method_name}_pvals.csv")

    # Plot histogram
    plt.figure()
    pvals_df['p_value'].hist(bins=40)
    plt.axvline(pval_thresh, color='red', linestyle='--', label='Threshold')
    plt.title(f"{method_name} p-value distribution")
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{method_name}_pval_hist.png")
    plt.close()

    # Filter associations
    filtered = pvals_df[pvals_df['p_value'] < pval_thresh]
    if filtered.empty:
        print(f"[{method_name}] No associations passed p-value < {pval_thresh}")
        return {}

    best_assoc = filtered.sort_values('p_value').drop_duplicates('Mutation')

    mean_nmf = W_df.groupby(cluster_labels).mean()
    defining_genes_dict = {}
    for cluster in sorted(cluster_labels.unique()):
        top_component = mean_nmf.loc[cluster].idxmax()
        component_weights = H_df.loc[top_component]
        high_genes = component_weights[component_weights > 0].sort_values(ascending=False).index.tolist()
        defining_genes_dict[cluster] = high_genes

    for cluster in sorted(cluster_labels.unique()):
        assoc_mut = best_assoc[best_assoc['Cluster'] == cluster]
        top_mutation = assoc_mut.sort_values('p_value').head(1)['Mutation'].values if not assoc_mut.empty else [f"Cluster_{cluster}_placeholder"]
        defining_genes = defining_genes_dict.get(cluster, [])
        all_signatures.setdefault(method_name, {})[top_mutation[0]] = defining_genes

    return all_signatures


# ========================
# Wrapper: Create Unsupervised Signatures
# ========================

def create_unsupervised_signatures(X, Y):
    Y_norm = np.zeros(Y.shape)
    s = Y.sum(axis=1)  # Total expression per sample
    p = 10000 / s
    Y_norm = np.log(Y.mul(p, axis=0) + 1)
    ss = StandardScaler()
    Y = pd.DataFrame(ss.fit_transform(Y_norm))
    Y.columns = [f'Expression_Gene_{i}' for i in range(Y_norm.shape[1])]
    us_signatures = {}
    us_signatures.update(create_kmeans_nmf_signature(Y, X, method_name='NMF-KMeans', n_components=20))
    us_signatures.update(create_signatures_kmeans(X, Y, method_name='K-means'))
    return us_signatures


def every_signatures(d1, d2):
    d1.update(d2)
    return d1
