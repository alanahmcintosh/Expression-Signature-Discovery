# ========================
# Libraries
# ========================
import numpy as np
import pandas as pd

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LassoCV, Lasso
from sklearn.svm import SVC

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

# Custom functions
from Deconfounder import ppca, predicitve_check

# Typing
from typing import Dict, Tuple, List, Optional


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
    model = LogisticRegressionCV(penalty='l1', solver='liblinear', class_weight='balanced', random_state=44, cv=10, Cs=50, max_iter=10000)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0].tolist()

def get_elasticnet_class_signature(X, y):
    model = LogisticRegressionCV(penalty='elasticnet', solver='saga', class_weight='balanced', l1_ratios=[0.5], random_state=44, max_iter=10000)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0].tolist()

def get_svm_class_signature(X, y): 
    model = SVC(kernel='linear', class_weight='balanced', random_state=44, max_iter=10000)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_).flatten() > 0.05].tolist() if hasattr(model, 'coef_') else []

def get_ridgereg_class_signature(X, y):
    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=44, max_iter=10000)
    model.fit(X, y)
    return X.columns[np.abs(model.coef_[0]) > 0.05].tolist()

def get_rf_class_signature(X, y):
    model = RandomForestClassifier(class_weight='balanced', random_state=44)
    model.fit(X, y)
    return X.columns[model.feature_importances_ > 0.005].tolist()



def get_deconfounder_signature(gof, global_results):
    coefs_tr = global_results['Deconfounder']
    if isinstance(gof, list):
        gof = gof[0]
    gene_signature = coefs_tr[[gof]].dropna()
    return list(gene_signature.index)


def get_deseq2_signature(X, Y, gof):
    """
    X_train: mutation/CNA/fusion matrix
    Y_train: gene expression count matrix (int)
    gof: alteration of interest
    """
    metadata = X.copy()
    counts_df = Y.astype(int)
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

    signatures = class_supervised_signatures(Y_norm_df, x)

    # add extra methods
    if global_results is not None:
        signatures['Deconfounder'] = get_deconfounder_signature(gof, global_results)

    signatures['DESeq2'] = get_deseq2_signature(X, Y, gof)

    return signatures


# ========================
# Unsupervised Methods
# ========================

# =========================
# Step 1 — Variable selection
# =========================
def filter_and_select_hvgs(
    Y: pd.DataFrame,
    min_mean: float = 1.0,
    top_n: int = 2000
) -> pd.DataFrame:
    """Filter low-expression genes then keep top-N HVGs by variance."""
    Yf = Y.loc[:, Y.mean(axis=0) >= min_mean]
    if Yf.shape[1] == 0:
        raise ValueError("All genes removed by min_mean filter; lower min_mean.")
    var = Yf.var(axis=0).sort_values(ascending=False)
    keep = var.head(min(top_n, len(var))).index
    return Yf[keep]


# =========================
# Step 2 — Consensus NMF & pick k by cophenetic
# =========================
def _consensus_from_labels(labels_2d: List[np.ndarray]) -> np.ndarray:
    """
    Build consensus co-assignment matrix from many label vectors.
    labels_2d: list of label arrays, each length n.
    Returns: (n x n) consensus matrix in [0,1].
    """
    n = len(labels_2d[0])
    C = np.zeros((n, n), dtype=float)
    for labs in labels_2d:
        same = (labs[:, None] == labs[None, :]).astype(float)
        C += same
    C /= len(labels_2d)
    np.fill_diagonal(C, 1.0)
    return C


def _cophenetic_from_consensus(C: np.ndarray) -> float:
    """
    Cophenetic correlation of (1 - consensus) distance matrix.
    """
    # convert to condensed distance for linkage
    D = 1.0 - C
    # ensure symmetry
    D = (D + D.T) / 2.0
    dvec = squareform(D, checks=False)
    Z = linkage(dvec, method="average")
    coph, _ = cophenet(Z, dvec)
    return float(coph)


def consensus_nmf_select_k(
    Y: pd.DataFrame,
    k_range=range(2, 8),
    n_runs: int = 50,
    sample_frac: float = 0.9,
    gene_frac: float = 0.9,
    nmf_components_cap: Optional[int] = 50,
    random_state: int = 44,
) -> Tuple[int, Dict[int, float], Dict[int, np.ndarray], Dict[int, Dict]]:
    """
    Run consensus NMF over k; pick k with highest cophenetic (TCGA-style).
    Returns:
      best_k,
      cophenetic_by_k,
      labels_by_k (labels from final run at each k),
      artifacts_by_k (W, H, final_kmeans, etc.)
    """
    rng = np.random.default_rng(random_state)
    n, g = Y.shape
    labels_by_k = {}
    artifacts_by_k = {}
    coph = {}

    for k in k_range:
        # For OV, k=4 often maxes stability; but we compute rather than assume
        label_runs = []
        for run in range(n_runs):
            # bootstrap subsampling
            rows = rng.choice(n, size=max(2, int(sample_frac * n)), replace=False)
            cols = rng.choice(g, size=max(2, int(gene_frac * g)), replace=False)
            Ysub = Y.iloc[rows, cols]

            # NMF on non-negative
            Ynn = Ysub.clip(lower=0)
            r = min(nmf_components_cap or k, Ynn.shape[1])
            r = max(r, k)  # ensure r >= k
            nmf = NMF(n_components=r, init="nndsvda", random_state=run + random_state, max_iter=600)
            W = nmf.fit_transform(Ynn)
            # cluster in W with KMeans (k groups)
            km = KMeans(n_clusters=k, n_init='auto', random_state=run + random_state)
            labs = km.fit_predict(W)

            # Map back to full index with -1 default
            labs_full = -np.ones(n, dtype=int)
            labs_full[rows] = labs
            label_runs.append(labs_full)

        # consensus from runs that labeled the same subset of samples:
        # Only count pairs co-labeled in each run (ignore -1 rows for that run)
        # A simple and robust trick: build C as average over runs of pairwise
        # equality but treat -1 as its own label and zero out rows/cols with -1
        # per run.
        C = np.zeros((n, n), dtype=float)
        counts = np.zeros((n, n), dtype=float)
        for labs in label_runs:
            mask = labs != -1
            idx = np.where(mask)[0]
            li = labs[mask]
            same = (li[:, None] == li[None, :]).astype(float)
            # add to C/counts for observed indices
            C[np.ix_(idx, idx)] += same
            counts[np.ix_(idx, idx)] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.divide(C, counts, out=np.zeros_like(C), where=counts > 0)
        np.fill_diagonal(C, 1.0)

        # cophenetic on consensus
        coph[k] = _cophenetic_from_consensus(C)

        # final fit at full data to get labels/artifacts for this k
        Ynn_full = Y.clip(lower=0)
        r_full = max(k, min(nmf_components_cap or k, Ynn_full.shape[1]))
        nmf_final = NMF(n_components=r_full, init="nndsvda", random_state=random_state, max_iter=800)
        W_full = nmf_final.fit_transform(Ynn_full)
        H_full = nmf_final.components_
        km_final = KMeans(n_clusters=k, n_init='auto', random_state=random_state).fit(W_full)
        labels_by_k[k] = km_final.labels_
        artifacts_by_k[k] = {
            "W": pd.DataFrame(W_full, index=Y.index, columns=[f"NMF_{i}" for i in range(r_full)]),
            "H": pd.DataFrame(H_full, columns=Y.columns, index=[f"NMF_{i}" for i in range(r_full)]),
            "kmeans": km_final,
            "consensus": C,
        }

    # choose k with max cophenetic
    best_k = max(coph, key=coph.get)
    return best_k, coph, labels_by_k, artifacts_by_k


# =========================
# Step 3 — Final NMF→KMeans labels
# =========================
def final_nmf_kmeans_labels(artifacts_by_k: Dict[int, Dict], k: int) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    art = artifacts_by_k[k]
    W, H, km = art["W"], art["H"], art["kmeans"]
    labels = pd.Series(km.labels_, index=W.index, name="cluster")
    return labels, W, H


# =========================
# Step 4 — KMeans (gene space) cross-check
# =========================
def kmeans_gene_space(Y_hvg: pd.DataFrame, k: int, random_state=44) -> pd.Series:
    km = KMeans(n_clusters=k, n_init='auto', random_state=random_state)
    labs = km.fit_predict(Y_hvg)
    return pd.Series(labs, index=Y_hvg.index, name="cluster_km_gene")


def overlap_metrics(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    a, b = a.loc[a.index], b.loc[a.index]
    return {
        "ARI": adjusted_rand_score(a, b),
        "NMI": normalized_mutual_info_score(a, b)
    }


# =========================
# Alteration association testing (with frequency filters)
# =========================
def prepare_alterations_binary(
    X: pd.DataFrame,
    cna_suffix: str = "_CNA",
    min_freq: float = 0.05,
    max_freq: float = 0.90
) -> pd.DataFrame:
    """
    Ensure binary; auto-binarize CNAs (any change != 0 → 1), and keep 5–90% frequency.
    """
    Xb = X.copy()
    for c in Xb.columns:
        if c.endswith(cna_suffix):
            Xb[c] = (Xb[c] != 0).astype(int)
    # keep binary only
    Xb = Xb.loc[:, Xb.nunique() == 2]
    # frequency filter
    freq = Xb.mean(axis=0)
    keep = (freq >= min_freq) & (freq <= max_freq)
    return Xb.loc[:, keep]


def cluster_alteration_associations(
    labels: pd.Series,
    Xb: pd.DataFrame
) -> pd.DataFrame:
    """
    Global χ² across clusters + per-cluster (cluster vs rest) Fisher/χ².
    Returns tidy table with raw p-values and FDR.
    """
    L = labels.astype(int)
    recs = []
    for feat in Xb.columns:
        v = Xb[feat]
        tab = pd.crosstab(L, v)
        if tab.shape[1] < 2:
            continue
        # global across all clusters
        try:
            _, p_global, _, _ = chi2_contingency(tab)
        except Exception:
            p_global = 1.0
        # per-cluster (2x2)
        for c in sorted(L.unique()):
            in_c = (L == c).astype(int)
            t22 = pd.crosstab(in_c, v)
            if t22.shape != (2, 2):
                continue
            if (t22.values < 5).any():
                _, p = fisher_exact(t22)
            else:
                _, p, _, _ = chi2_contingency(t22)
            recs.append({
                "alteration": feat,
                "cluster": int(c),
                "pval": p,
                "pval_global": p_global
            })
    out = pd.DataFrame(recs)
    if out.empty:
        return out
    out["fdr"] = multipletests(out["pval"], method="fdr_bh")[1]
    out["fdr_global"] = multipletests(out["pval_global"], method="fdr_bh")[1]
    return out.sort_values(["fdr", "pval"])


# =========================
# Cluster-vs-rest differential expression
# =========================
def de_cluster_vs_rest(
    Y: pd.DataFrame,
    labels: pd.Series,
    target_cluster: int,
    test: str = "ttest",
    fdr_thresh: Optional[float] = 0.1,
    top_n: int = 50
) -> List[str]:
    """
    Return top_n DE genes (by FDR then effect size) for cluster vs rest.
    """
    mask = labels == target_cluster
    rows = []
    for g in Y.columns:
        a, b = Y.loc[mask, g].values, Y.loc[~mask, g].values
        if test == "wilcoxon":
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = 1.0
        else:
            stat, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        eff = np.nanmean(a) - np.nanmean(b)
        rows.append((g, p, eff))
    df = pd.DataFrame(rows, columns=["gene", "pval", "effect"])
    df["fdr"] = multipletests(df["pval"], method="fdr_bh")[1]
    df = df.sort_values(["fdr", "effect"], ascending=[True, False])
    if fdr_thresh is not None:
        df = df[df["fdr"] <= fdr_thresh]
    return df.head(top_n)["gene"].tolist()


# =========================
# Wrapper to produce your signature dict
# =========================
def run_tcga_style_ov_pipeline(
    Y: pd.DataFrame,            # samples x genes (raw CPM/TPM or already log1p)
    X: pd.DataFrame,            # samples x alterations (mut/fus + CNA integers or binary)
    method_names = ("NMF-KMeans", "K-means"),
    min_mean=1.0,
    top_n_hvg=2000,
    k_range=range(2, 8),
    n_runs=50,
    freq_min=0.05,
    freq_max=0.90,
    fdr_thresh=0.1,
    de_top_n=50,
    force_k: Optional[int] = None,
    random_state=44,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns: { 'NMF-KMeans': {alteration: [genes...]}, 'K-means': {...} }
    Also saves helpful CSVs on disk (labels, assoc tables).
    """
    # --- Normalize Y like TCGA-style (library size -> log1p) if Y seems raw
    s = Y.sum(axis=1).replace(0, np.nan)
    scale = (10000 / s).fillna(0.0)
    Y_norm = np.log1p(Y.mul(scale, axis=0))
    # HVGs
    Y_hvg = filter_and_select_hvgs(Y_norm, min_mean=min_mean, top_n=top_n_hvg)

    # --- Prepare alterations (binary + frequency filtered)
    Xb = prepare_alterations_binary(X, min_freq=freq_min, max_freq=freq_max)
    # align indices
    common = Y_hvg.index.intersection(Xb.index)
    Y_hvg, Xb = Y_hvg.loc[common], Xb.loc[common]

    out = {}

    # ====== Method A: consensus NMF → KMeans ======
    if "NMF-KMeans" in method_names:
        best_k, coph, labels_by_k, artifacts_by_k = consensus_nmf_select_k(
            Y_hvg, k_range=k_range, n_runs=n_runs, random_state=random_state
        )
        k_use = force_k if force_k is not None else best_k
        labels_nmf, W, H = final_nmf_kmeans_labels(artifacts_by_k, k_use)

        # Save diagnostics
        pd.Series(labels_nmf, name="cluster").to_csv("NMF-KMeans_cluster_labels.csv")
        pd.DataFrame({"k": list(coph.keys()), "cophenetic": list(coph.values())}).to_csv(
            "NMF-KMeans_cophenetic.csv", index=False
        )

        # Associations
        assoc = cluster_alteration_associations(labels_nmf, Xb)
        assoc.to_csv("NMF-KMeans_alteration_assoc.csv", index=False)

        # Build signatures: top alteration per cluster (by FDR), then DE genes
        signatures_nmf = {}
        if assoc.empty:
            # still produce placeholders
            for c in sorted(labels_nmf.unique()):
                genes = de_cluster_vs_rest(Y_hvg, labels_nmf, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_nmf[f"Cluster_{c}_placeholder"] = genes
        else:
            sig = assoc.sort_values(["fdr", "pval"])
            for c in sorted(labels_nmf.unique()):
                sub = sig[sig["cluster"] == c]
                key = sub.iloc[0]["alteration"] if not sub.empty else f"Cluster_{c}_placeholder"
                genes = de_cluster_vs_rest(Y_hvg, labels_nmf, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_nmf[key] = genes
        out["NMF-KMeans"] = signatures_nmf

        # Cross-check with gene-space KMeans
        km_gene_labels = kmeans_gene_space(Y_hvg, k=k_use, random_state=random_state)
        pd.Series(km_gene_labels, name="cluster").to_csv("KMeansGene_cluster_labels.csv")
        ovlp = overlap_metrics(labels_nmf, km_gene_labels)
        pd.DataFrame([ovlp]).to_csv("NMF_vs_KMeansGene_overlap.csv", index=False)

    # ====== Method B: plain KMeans in gene space (HVGs) ======
    if "K-means" in method_names:
        # pick k by silhouette (gene space) but allow force_k
        scores = {}
        for k in k_range:
            try:
                km_labs = kmeans_gene_space(Y_hvg, k=k, random_state=random_state)
                if len(np.unique(km_labs)) < 2:
                    continue
                scores[k] = silhouette_score(Y_hvg, km_labs)
            except Exception:
                pass
        if len(scores) == 0:
            # degenerate case: fall back to k=2
            k_km = force_k if force_k is not None else 2
        else:
            best_km = max(scores, key=scores.get)
            k_km = force_k if force_k is not None else best_km

        labels_km = kmeans_gene_space(Y_hvg, k=k_km, random_state=random_state)
        pd.Series(labels_km, name="cluster").to_csv("K-means_cluster_labels.csv")
        pd.DataFrame({"k": list(scores.keys()), "silhouette": list(scores.values())}).to_csv(
            "K-means_silhouette.csv", index=False
        )

        assoc_km = cluster_alteration_associations(labels_km, Xb)
        assoc_km.to_csv("K-means_alteration_assoc.csv", index=False)

        signatures_km = {}
        if assoc_km.empty:
            for c in sorted(labels_km.unique()):
                genes = de_cluster_vs_rest(Y_hvg, labels_km, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_km[f"Cluster_{c}_placeholder"] = genes
        else:
            sig = assoc_km.sort_values(["fdr", "pval"])
            for c in sorted(labels_km.unique()):
                sub = sig[sig["cluster"] == c]
                key = sub.iloc[0]["alteration"] if not sub.empty else f"Cluster_{c}_placeholder"
                genes = de_cluster_vs_rest(Y_hvg, labels_km, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_km[key] = genes
        out["K-means"] = signatures_km

    return out


def every_signatures(d1, d2):
    d1.update(d2)
    return d1
