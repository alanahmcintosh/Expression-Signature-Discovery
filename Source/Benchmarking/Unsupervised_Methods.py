
'''
Unsupervised Signatures
'''
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Iterable

from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    kruskal,
    mannwhitneyu,
    ttest_ind,
)
from statsmodels.stats.multitest import multipletests


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
# Normalization guards
# =========================
def _looks_like_counts_or_linear_expr(Y: pd.DataFrame) -> bool:
    """
    Heuristic:
      - counts/TPM/CPM are non-negative and typically have large maxima
      - log1p-ish data tends to have max < ~30 (often < 15) and may include negatives if centered (but then NMF not valid)
    """
    vals = Y.to_numpy()
    if np.nanmin(vals) < 0:
        # could be centered/scaled; not safe to auto-normalize
        return False
    vmax = np.nanmax(vals)
    # If very large, it's almost certainly linear scale (counts/TPM/CPM)
    if vmax > 50:
        return True
    # If small, might already be log-scale
    return False


def normalize_library_log1p(Y: pd.DataFrame, target_sum: float = 1e4) -> pd.DataFrame:
    """Library-size normalize then log1p; expects non-negative."""
    if (Y < 0).any().any():
        raise ValueError("normalize_library_log1p got negative values; refuse to log-normalize.")
    s = Y.sum(axis=1).replace(0, np.nan)
    scale = (target_sum / s).fillna(0.0)
    return np.log1p(Y.mul(scale, axis=0))


def normalize_if_needed(Y: pd.DataFrame) -> pd.DataFrame:
    """
    If Y looks like linear expression (counts/TPM/CPM), do library-size log1p.
    If Y looks like already log-scale, return as-is.
    """
    if _looks_like_counts_or_linear_expr(Y):
        return normalize_library_log1p(Y)
    return Y.copy()


# =========================
# Step 2 — Consensus NMF & pick k by cophenetic
# =========================
def _cophenetic_from_consensus(C: np.ndarray) -> float:
    """Cophenetic correlation of (1 - consensus) distance matrix."""
    D = 1.0 - C
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
    Run consensus NMF over k; pick k with highest cophenetic (TCGA-style-ish).
    Returns:
      best_k,
      cophenetic_by_k,
      labels_by_k (labels from final full fit at each k),
      artifacts_by_k (W, H, final_kmeans, consensus matrix)
    """
    # NMF requires non-negative
    if (Y < 0).any().any():
        raise ValueError("NMF requires non-negative input. Your Y contains negatives.")

    rng = np.random.default_rng(random_state)
    n, g = Y.shape
    labels_by_k: Dict[int, np.ndarray] = {}
    artifacts_by_k: Dict[int, Dict] = {}
    coph: Dict[int, float] = {}

    for k in k_range:
        label_runs: List[np.ndarray] = []
        for run in range(n_runs):
            rows = rng.choice(n, size=max(2, int(sample_frac * n)), replace=False)
            cols = rng.choice(g, size=max(2, int(gene_frac * g)), replace=False)
            Ysub = Y.iloc[rows, cols]

            # NMF on non-negative (already checked)
            n_sub, g_sub = Ysub.shape
            cap = nmf_components_cap if nmf_components_cap is not None else k

            # r must be <= min(n_sub, g_sub) for nndsvda
            r_max = min(cap, n_sub, g_sub)
            if r_max < k:
                # not enough rank in this subsample to support k clusters/components; skip this run
                continue

            r = max(k, r_max)

            init = "nndsvda" if r <= min(n_sub, g_sub) else "random"  # should be nndsvda now, but keep safe
            nmf = NMF(n_components=r, init=init, random_state=run + random_state, max_iter=800)

            W = nmf.fit_transform(Ysub)

            km = KMeans(n_clusters=k, n_init="auto", random_state=run + random_state)
            labs = km.fit_predict(W)

            labs_full = -np.ones(n, dtype=int)
            labs_full[rows] = labs
            label_runs.append(labs_full)

        # Build consensus only over co-labeled pairs
        C = np.zeros((n, n), dtype=float)
        counts = np.zeros((n, n), dtype=float)
        for labs in label_runs:
            mask = labs != -1
            idx = np.where(mask)[0]
            li = labs[mask]
            same = (li[:, None] == li[None, :]).astype(float)
            C[np.ix_(idx, idx)] += same
            counts[np.ix_(idx, idx)] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.divide(C, counts, out=np.zeros_like(C), where=counts > 0)
        np.fill_diagonal(C, 1.0)

        coph[k] = _cophenetic_from_consensus(C)

        # Final fit on full data for this k
        n_full, g_full = Y.shape
        cap = nmf_components_cap if nmf_components_cap is not None else k

        r_max_full = min(cap, n_full, g_full)
        if r_max_full < k:
            raise ValueError(
                f"Cannot fit NMF for k={k}: need min(n_samples, n_genes) >= k, "
                f"but got min({n_full}, {g_full})={min(n_full, g_full)}."
            )

        r_full = max(k, r_max_full)
        init_full = "nndsvda" if r_full <= min(n_full, g_full) else "random"

        nmf_final = NMF(n_components=r_full, init=init_full, random_state=random_state, max_iter=1200)
        W_full = nmf_final.fit_transform(Y)

        H_full = nmf_final.components_
        km_final = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(W_full)

        labels_by_k[k] = km_final.labels_
        artifacts_by_k[k] = {
            "W": pd.DataFrame(W_full, index=Y.index, columns=[f"NMF_{i}" for i in range(r_full)]),
            "H": pd.DataFrame(H_full, columns=Y.columns, index=[f"NMF_{i}" for i in range(r_full)]),
            "kmeans": km_final,
            "consensus": C,
        }

    best_k = max(coph, key=coph.get)
    return best_k, coph, labels_by_k, artifacts_by_k


# =========================
# Step 3 — Final NMF→KMeans labels
# =========================
def final_nmf_kmeans_labels(
    artifacts_by_k: Dict[int, Dict],
    k: int
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    art = artifacts_by_k[k]
    W, H, km = art["W"], art["H"], art["kmeans"]
    labels = pd.Series(km.labels_, index=W.index, name="cluster")
    return labels, W, H


# =========================
# Step 4 — KMeans (samples using gene features)
# =========================
def kmeans_samples_by_genes(Y_hvg: pd.DataFrame, k: int, random_state=44) -> pd.Series:
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labs = km.fit_predict(Y_hvg)
    return pd.Series(labs, index=Y_hvg.index, name="cluster")


def overlap_metrics(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    a = a.loc[b.index]
    b = b.loc[a.index]
    return {
        "ARI": adjusted_rand_score(a, b),
        "NMI": normalized_mutual_info_score(a, b)
    }


# =========================
# Alteration prep (binary vs CNA continuous/int)
# =========================
def split_alterations(
    X: pd.DataFrame,
    cna_suffix: str = "_CNA",
    min_freq_bin: float = 0.05,
    max_freq_bin: float = 0.90,
    min_nonzero_cna: float = 0.01
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_bin: binary alterations (MUT/Fusion etc.), filtered by frequency
      X_cna: CNA alterations (integer/continuous), filtered by variability and nonzero fraction
    """
    X = X.copy()

    bin_cols = [c for c in X.columns if not c.endswith(cna_suffix)]
    cna_cols = [c for c in X.columns if c.endswith(cna_suffix)]

    X_bin = X[bin_cols].copy() if bin_cols else pd.DataFrame(index=X.index)
    X_cna = X[cna_cols].copy() if cna_cols else pd.DataFrame(index=X.index)

    # ---- binary cleaning ----
    if not X_bin.empty:
        # keep only true binary columns (0/1 or 2 unique values)
        # if you sometimes have boolean/float, coerce safely:
        for c in X_bin.columns:
            # if it looks boolean-ish, keep; else leave and drop later
            pass
        X_bin = X_bin.loc[:, X_bin.nunique(dropna=True) == 2].copy()

        # frequency filter on "1" rate (assumes 0/1 coding)
        # if columns aren't 0/1, this could be wrong; so enforce 0/1:
        for c in X_bin.columns:
            u = sorted(pd.unique(X_bin[c].dropna()))
            if u != [0, 1]:
                # map lowest->0, highest->1 if exactly 2 values
                if len(u) == 2:
                    X_bin[c] = (X_bin[c] == u[1]).astype(int)
                else:
                    # should not happen due to nunique==2, but keep safe
                    X_bin.drop(columns=[c], inplace=True)

        freq = X_bin.mean(axis=0)
        keep = (freq >= min_freq_bin) & (freq <= max_freq_bin)
        X_bin = X_bin.loc[:, keep].copy()

    # ---- CNA cleaning (keep continuous/int) ----
    if not X_cna.empty:
        # drop constants
        var = X_cna.var(axis=0)
        X_cna = X_cna.loc[:, var > 0].copy()

        # optional: require some nonzero fraction, avoids almost-all-zero CNAs
        nonzero_frac = (X_cna != 0).mean(axis=0)
        X_cna = X_cna.loc[:, nonzero_frac >= min_nonzero_cna].copy()

    return X_bin, X_cna


# =========================
# Association tests (correct for binary vs continuous CNA)
# =========================
def _bh_fdr(pvals: pd.Series) -> pd.Series:
    """BH-FDR for a 1D series, preserving index."""
    arr = pvals.to_numpy()
    fdr = multipletests(arr, method="fdr_bh")[1]
    return pd.Series(fdr, index=pvals.index)


def cluster_associations_binary(
    labels: pd.Series,
    X_bin: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Binary alteration enrichment.
    Returns:
      per_cluster_df: rows = (alteration, cluster) with pval + fdr + effect
      global_df: one row per alteration with p_global + fdr_global
    """
    if X_bin is None or X_bin.empty:
        return pd.DataFrame(), pd.DataFrame()

    L = labels.astype(int)

    global_rows = []
    per_rows = []

    for feat in X_bin.columns:
        v = X_bin[feat].astype(int)

        tab = pd.crosstab(L, v)
        if tab.shape[1] < 2:
            continue

        # global across clusters
        try:
            _, p_global, _, _ = chi2_contingency(tab)
        except Exception:
            p_global = 1.0

        global_rows.append({"alteration": feat, "p_global": float(p_global)})

        # per-cluster: cluster vs rest (2x2)
        for c in sorted(L.unique()):
            in_c = (L == c).astype(int)
            t22 = pd.crosstab(in_c, v)
            if t22.shape != (2, 2):
                continue

            # effect: difference in prevalence
            prev_in = v[in_c == 1].mean() if (in_c == 1).any() else np.nan
            prev_out = v[in_c == 0].mean() if (in_c == 0).any() else np.nan
            effect = float(prev_in - prev_out) if np.isfinite(prev_in) and np.isfinite(prev_out) else np.nan

            if (t22.values < 5).any():
                _, p = fisher_exact(t22)
            else:
                _, p, _, _ = chi2_contingency(t22)

            per_rows.append({
                "alteration": feat,
                "cluster": int(c),
                "pval": float(p),
                "effect": effect,
                "test": "fisher/chi2",
                "dtype": "binary"
            })

    per_df = pd.DataFrame(per_rows)
    global_df = pd.DataFrame(global_rows)

    if not per_df.empty:
        per_df["fdr"] = _bh_fdr(per_df["pval"])
        per_df = per_df.sort_values(["fdr", "pval"])

    if not global_df.empty:
        global_df["fdr_global"] = _bh_fdr(global_df["p_global"])
        global_df = global_df.sort_values(["fdr_global", "p_global"])

    return per_df, global_df


def cluster_associations_cna(
    labels: pd.Series,
    X_cna: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CNA association (integer/continuous).
    Global: Kruskal–Wallis across clusters.
    Per-cluster: Mann–Whitney (cluster vs rest).
    Returns per_cluster_df and global_df.
    """
    if X_cna is None or X_cna.empty:
        return pd.DataFrame(), pd.DataFrame()

    L = labels.astype(int)

    global_rows = []
    per_rows = []

    clusters = sorted(L.unique())

    for feat in X_cna.columns:
        v = X_cna[feat]

        # global KW (requires >=2 groups with data)
        groups = [v[L == c].dropna().to_numpy() for c in clusters]
        # if all groups empty or only one group has data, skip
        nonempty = [g for g in groups if len(g) > 0]
        if len(nonempty) < 2:
            continue

        try:
            _, p_global = kruskal(*groups, nan_policy="omit")
        except Exception:
            p_global = 1.0

        global_rows.append({"alteration": feat, "p_global": float(p_global)})

        # per cluster vs rest MWU
        for c in clusters:
            a = v[L == c].dropna().to_numpy()
            b = v[L != c].dropna().to_numpy()
            if len(a) == 0 or len(b) == 0:
                continue

            # effect: median difference (robust)
            effect = float(np.median(a) - np.median(b))

            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = 1.0

            per_rows.append({
                "alteration": feat,
                "cluster": int(c),
                "pval": float(p),
                "effect": effect,
                "test": "mannwhitney",
                "dtype": "cna"
            })

    per_df = pd.DataFrame(per_rows)
    global_df = pd.DataFrame(global_rows)

    if not per_df.empty:
        per_df["fdr"] = _bh_fdr(per_df["pval"])
        per_df = per_df.sort_values(["fdr", "pval"])

    if not global_df.empty:
        global_df["fdr_global"] = _bh_fdr(global_df["p_global"])
        global_df = global_df.sort_values(["fdr_global", "p_global"])

    return per_df, global_df


def combine_associations(
    per_bin: pd.DataFrame,
    glob_bin: pd.DataFrame,
    per_cna: pd.DataFrame,
    glob_cna: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine binary + CNA association tables.
    Global fdrs remain within-type. If you want one unified global FDR across all alterations,
    do BH on concatenated p_globals instead. For benchmarking, within-type is often clearer.
    """
    per = pd.concat([per_bin, per_cna], axis=0, ignore_index=True) if (not per_bin.empty or not per_cna.empty) else pd.DataFrame()
    glob = pd.concat([glob_bin.assign(dtype="binary"), glob_cna.assign(dtype="cna")], axis=0, ignore_index=True) if (not glob_bin.empty or not glob_cna.empty) else pd.DataFrame()
    return per, glob


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
    Operates on whatever scale Y is given (ideally log1p normalized).
    """
    mask = labels == target_cluster
    rows = []
    for g in Y.columns:
        a = Y.loc[mask, g].values
        b = Y.loc[~mask, g].values

        if test == "wilcoxon":
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = 1.0
        else:
            _, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")

        eff = float(np.nanmean(a) - np.nanmean(b))
        rows.append((g, float(p), eff))

    df = pd.DataFrame(rows, columns=["gene", "pval", "effect"])
    df["fdr"] = multipletests(df["pval"], method="fdr_bh")[1]
    df = df.sort_values(["fdr", "effect"], ascending=[True, False])

    if fdr_thresh is not None:
        df = df[df["fdr"] <= fdr_thresh]

    return df.head(top_n)["gene"].tolist()


# =========================
# Wrapper to produce your signature dict
# =========================
def run_tcga_style_unsup_pipeline(
    Y: pd.DataFrame,            # samples x genes (counts/TPM/CPM OR already-log)
    X: pd.DataFrame,            # samples x alterations (binary MUT/Fusion + integer/continuous CNA)
    method_names: Iterable[str] = ("NMF-KMeans", "K-means"),
    min_mean: float = 1.0,
    top_n_hvg: int = 2000,
    k_range=range(2, 8),
    n_runs: int = 50,
    bin_freq_min: float = 0.05,
    bin_freq_max: float = 0.90,
    cna_min_nonzero: float = 0.01,
    fdr_thresh: float = 0.1,
    de_top_n: int = 50,
    force_k: Optional[int] = None,
    random_state: int = 44,
    save_prefix: str = "",
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
      { 'NMF-KMeans': {alteration: [genes...]}, 'K-means': {...} }

    Also writes CSVs if you want (controlled by save_prefix; if empty, still writes to cwd but with plain names).
    """
    # --- Normalize Y only if needed (prevents double-log)
    Y_norm = normalize_if_needed(Y)

    # --- HVGs
    Y_hvg = filter_and_select_hvgs(Y_norm, min_mean=min_mean, top_n=top_n_hvg)

    # --- Split alterations into binary and CNA continuous/int; align indices
    X_bin, X_cna = split_alterations(
        X,
        min_freq_bin=bin_freq_min,
        max_freq_bin=bin_freq_max,
        min_nonzero_cna=cna_min_nonzero
    )

    common = Y_hvg.index.intersection(X.index)
    Y_hvg = Y_hvg.loc[common]
    X_bin = X_bin.loc[common] if not X_bin.empty else X_bin
    X_cna = X_cna.loc[common] if not X_cna.empty else X_cna

    out: Dict[str, Dict[str, List[str]]] = {}

    # ====== Method A: consensus NMF → KMeans ======
    if "NMF-KMeans" in method_names:
        best_k, coph, labels_by_k, artifacts_by_k = consensus_nmf_select_k(
            Y_hvg, k_range=k_range, n_runs=n_runs, random_state=random_state
        )
        k_use = force_k if force_k is not None else best_k
        labels_nmf, W, H = final_nmf_kmeans_labels(artifacts_by_k, k_use)

        # Save diagnostics
        pd.Series(labels_nmf, name="cluster").to_csv(f"{save_prefix}NMF-KMeans_cluster_labels.csv")
        pd.DataFrame({"k": list(coph.keys()), "cophenetic": list(coph.values())}).to_csv(
            f"{save_prefix}NMF-KMeans_cophenetic.csv", index=False
        )

        # Associations (binary + CNA)
        per_bin, glob_bin = cluster_associations_binary(labels_nmf, X_bin)
        per_cna, glob_cna = cluster_associations_cna(labels_nmf, X_cna)
        assoc_per, assoc_global = combine_associations(per_bin, glob_bin, per_cna, glob_cna)

        assoc_per.to_csv(f"{save_prefix}NMF-KMeans_alteration_assoc_per_cluster.csv", index=False)
        assoc_global.to_csv(f"{save_prefix}NMF-KMeans_alteration_assoc_global.csv", index=False)

        # Build signatures: per cluster, pick best-associated alteration (lowest FDR then pval)
        signatures_nmf: Dict[str, List[str]] = {}
        if assoc_per.empty:
            for c in sorted(labels_nmf.unique()):
                genes = de_cluster_vs_rest(Y_hvg, labels_nmf, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_nmf[f"Cluster_{c}_placeholder"] = genes
        else:
            sig = assoc_per.sort_values(["fdr", "pval"])
            for c in sorted(labels_nmf.unique()):
                sub = sig[sig["cluster"] == c]
                top_alt = sub.iloc[0]["alteration"] if not sub.empty else "placeholder"
                key = f"{top_alt}"
                genes = de_cluster_vs_rest(Y_hvg, labels_nmf, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_nmf[key] = genes



        out["NMF-KMeans"] = signatures_nmf

        # Cross-check with direct KMeans on samples
        km_labels = kmeans_samples_by_genes(Y_hvg, k=k_use, random_state=random_state)
        pd.Series(km_labels, name="cluster").to_csv(f"{save_prefix}KMeans_cluster_labels.csv")
        ovlp = overlap_metrics(labels_nmf, km_labels)
        pd.DataFrame([ovlp]).to_csv(f"{save_prefix}NMF_vs_KMeans_overlap.csv", index=False)

    # ====== Method B: plain KMeans (samples using genes) ======
    if "K-means" in method_names:
        # pick k by silhouette unless force_k
        scores = {}
        for k in k_range:
            try:
                km_labs = kmeans_samples_by_genes(Y_hvg, k=k, random_state=random_state)
                if len(np.unique(km_labs)) < 2:
                    continue
                scores[k] = silhouette_score(Y_hvg, km_labs)
            except Exception:
                pass

        if force_k is not None:
            k_km = force_k
        else:
            k_km = max(scores, key=scores.get) if len(scores) else 2

        labels_km = kmeans_samples_by_genes(Y_hvg, k=k_km, random_state=random_state)
        pd.Series(labels_km, name="cluster").to_csv(f"{save_prefix}K-means_cluster_labels.csv")
        pd.DataFrame({"k": list(scores.keys()), "silhouette": list(scores.values())}).to_csv(
            f"{save_prefix}K-means_silhouette.csv", index=False
        )

        per_bin, glob_bin = cluster_associations_binary(labels_km, X_bin)
        per_cna, glob_cna = cluster_associations_cna(labels_km, X_cna)
        assoc_per, assoc_global = combine_associations(per_bin, glob_bin, per_cna, glob_cna)

        assoc_per.to_csv(f"{save_prefix}K-means_alteration_assoc_per_cluster.csv", index=False)
        assoc_global.to_csv(f"{save_prefix}K-means_alteration_assoc_global.csv", index=False)

        signatures_km: Dict[str, List[str]] = {}
        if assoc_per.empty:
            for c in sorted(labels_km.unique()):
                genes = de_cluster_vs_rest(Y_hvg, labels_km, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_km[f"Cluster_{c}_placeholder"] = genes
        else:
            sig = assoc_per.sort_values(["fdr", "pval"])
            for c in sorted(labels_km.unique()):
                sub = sig[sig["cluster"] == c]
                top_alt = sub.iloc[0]["alteration"] if not sub.empty else "placeholder"
                key = f"{top_alt}"
                genes = de_cluster_vs_rest(Y_hvg, labels_km, c, fdr_thresh=fdr_thresh, top_n=de_top_n)
                signatures_km[key] = genes


        out["K-means"] = signatures_km

    return out

