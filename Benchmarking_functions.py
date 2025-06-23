# ========================
# Libraries
# ========================
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LassoCV, Lasso
from sklearn.svm import SVC
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


def get_lasso_signature(X, y):
    model = LogisticRegressionCV(penalty='l1', solver='liblinear', class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0].tolist()

def get_elasticnet_signature(X, y):
    model = LogisticRegressionCV(penalty='elasticnet', solver='saga', class_weight='balanced',
                                  l1_ratios=[0.5], random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0].tolist()

def get_svm_signature(X, y):
    model = SVC(kernel='linear', class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_).flatten() > 0.05].tolist() if hasattr(model, 'coef_') else []

def get_ridgereg_signature(X, y):
    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0.05].tolist()

def get_rf_signature(X, y):
    model = RandomForestClassifier(class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[model.feature_importances_ > 0.005].tolist()

def get_deconfounder_signature(gof, global_results):
    coefs_tr = global_results['Deconfounder']
    if isinstance(gof, list):
        gof = gof[0]
    gene_signature = coefs_tr[[gof]].dropna()
    return list(gene_signature.index)

def get_deseq2_signature(X_train, Y_train, gof):
    """
    X_train: mutation/CNA/fusion matrix
    Y_train: gene expression count matrix (int)
    gof: alteration of interest
    """
    metadata = X_train.copy()
    counts_df = Y_train.astype(int)
    inference = DefaultInference(n_cpus=8)

    # Make sure it's binary categorical
    metadata[gof] = metadata[gof].astype(str).astype("category")
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

# ========================
# Wrapper: Create Supervised Signatures
# ========================

def create_supervised_signatures(X, Y, gof, global_results):
    """
    Run all supervised methods for a single GOF and return gene signature results.
    """
    signatures = {}
    
    x = X[gof] if isinstance(X[gof], pd.Series) else X[gof].values
    Y_norm = np.zeros(Y.shape)
    s = Y.sum(axis=1)  # Total expression per sample
    p = 10000 / s
    Y_norm = np.log(Y.mul(p, axis=0) + 1)
    ss = StandardScaler()
    Y_norm_df = pd.DataFrame(ss.fit_transform(Y_norm))
    Y_norm_df.columns = [f'Expression_Gene_{i}' for i in range(Y_norm.shape[1])]

    signatures['Random Forest'] = get_rf_signature(Y_norm_df, x)
    signatures['Lasso'] = get_lasso_signature(Y_norm_df, x)
    signatures['ElasticNet'] = get_elasticnet_signature(Y_norm_df, x)
    signatures['SVM'] = get_svm_signature(Y_norm_df, x)
    signatures['Logistic Regression'] = get_ridgereg_signature(Y_norm_df, x)
    signatures['Deconfounder'] = get_deconfounder_signature(gof, global_results)
    signatures['DESeq2'] = get_deseq2_signature(X, Y, gof)

    return signatures

# ========================
# Unsupervised Methods
# ========================

def create_signatures_kmeans(X, Y, method_name='K-means'):
    km_signatures = {}
    Y_scaled_df = pd.DataFrame(Y, index=Y.index, columns=Y.columns)

    scores = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=44).fit(Y)
        score = silhouette_score(Y, kmeans.labels_)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    cluster_labels = pd.Series(KMeans(n_clusters=best_k, random_state=44).fit_predict(Y), index=Y.index)

    X_clustered = X.copy()
    X_clustered['Cluster'] = cluster_labels

    mutation_pvals = []
    for mutation in X.columns:
        for cluster in sorted(cluster_labels.unique()):
            contingency = pd.crosstab((cluster_labels == cluster).astype(int), X[mutation])
            if contingency.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(contingency)
                mutation_pvals.append({'Mutation': mutation, 'Cluster': cluster, 'p_value': p})

    pvals_df = pd.DataFrame(mutation_pvals)
    best_assoc = pvals_df.sort_values('p_value').drop_duplicates('Mutation')

    mean_expr = Y_scaled_df.groupby(cluster_labels).mean()
    defining_genes_dict = {
        cluster: mean_expr.loc[cluster].sort_values(ascending=False).head(50).index.tolist()
        for cluster in cluster_labels.unique()
    }

    for cluster in sorted(cluster_labels.unique()):
        assoc_mut = best_assoc[best_assoc['Cluster'] == cluster]
        top_mutation = assoc_mut.sort_values('p_value').head(1)['Mutation'].values
        if len(top_mutation) == 0:
            continue
        defining_genes = defining_genes_dict.get(cluster, [])
        km_signatures.setdefault(method_name, {})[top_mutation[0]] = defining_genes

    return km_signatures

def create_kmeans_nmf_signature(Y, X, method_name='NMF-KMeans', n_components=20):
    all_signatures = {}
    nmf_model = NMF(n_components=n_components, init='random', random_state=44, max_iter=1000)
    W = nmf_model.fit_transform(Y.abs())
    H = nmf_model.components_

    W_df = pd.DataFrame(W, index=Y.index, columns=[f'NMF_{i}' for i in range(n_components)])
    H_df = pd.DataFrame(H, columns=Y.columns, index=[f'NMF_{i}' for i in range(n_components)])

    scores = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=44).fit(W_df)
        score = silhouette_score(W_df, kmeans.labels_)
        scores[k] = score

    best_k = max(scores, key=scores.get)
    cluster_labels = pd.Series(KMeans(n_clusters=best_k, random_state=44).fit_predict(W_df), index=Y.index)

    X_clustered = X.copy()
    X_clustered['Cluster'] = cluster_labels

    mutation_pvals = []
    for mutation in X.columns:
        for cluster in sorted(cluster_labels.unique()):
            contingency = pd.crosstab((cluster_labels == cluster).astype(int), X[mutation])
            if contingency.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(contingency)
                mutation_pvals.append({'Mutation': mutation, 'Cluster': cluster, 'p_value': p})

    pvals_df = pd.DataFrame(mutation_pvals)
    best_assoc = pvals_df.sort_values('p_value').drop_duplicates('Mutation')

    mean_nmf = W_df.groupby(cluster_labels).mean()
    defining_genes_dict = {}
    for cluster in sorted(cluster_labels.unique()):
        top_component = mean_nmf.loc[cluster].idxmax()
        component_weights = H_df.loc[top_component]
        high_genes = component_weights[component_weights > 0].sort_values(ascending=False).index.tolist()
        defining_genes_dict[cluster] = high_genes

    for cluster in sorted(cluster_labels.unique()):
        assoc_mut = best_assoc[best_assoc['Cluster'] == cluster]
        top_mutation = assoc_mut.sort_values('p_value').head(1)['Mutation'].values
        if len(top_mutation) == 0:
            continue
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
