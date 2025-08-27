import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from scipy.stats import nbinom, binom
from pydeseq2.default_inference import DefaultInference
import random 
import numpy.random as npr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

### ======================== ###
### Real Data preprocessing ####
### ======================== ###


def to_patient_id(index, study=None):
    index = index.astype(str).str.strip()
    if study == 'TCGA':
        return index.str.extract(r'(TCGA-\w{2}-\w{4})')[0]
    elif study == 'TARGET':
        return index.str.extract(r'(TARGET-\d{2}-[A-Z0-9]{6})')[0]
    else:
        return index


### Mutation Loader ###
def load_mutations_raw(path, rename=True):
    df = pd.read_csv(path, sep=None, engine='python', index_col=0).astype(int)
    if rename:
        df.columns = [f"{gene}_MUT" for gene in df.columns]
    return df


### CNA Loader ###
def load_cna(path, top_n=500, cna_process=True, rename=True):
    df = pd.read_csv(path, sep=None, engine='python', index_col=0)

    if cna_process:
        # Drop any columns like Entrez_ID before transposing
        df = df.drop(columns=[c for c in df.columns if 'Entrez' in c], errors='ignore')

        # Remove any genes with ambiguous names like "GENE1|GENE2"
        df = df.loc[:, ~df.columns.str.contains('\|', regex=True)]
        
        # Transpose: samples become rows, genes become columns
        df = df.T

    if rename:
        df.columns = [f"{gene}_CNA" for gene in df.columns]
    
    # Keep top N most variable CNA features
    top_vars = df.var().sort_values(ascending=False).head(top_n).index
    return df[top_vars]


### Fusion Loader ###
def load_fusions_raw(path):
    import pandas as pd

    sv_df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    sv_df.columns = sv_df.columns.str.strip()

    # Determine Patient/Sample ID
    if "Sample_Id" in sv_df.columns:
        sv_df["Patient_Id"] = sv_df["Sample_Id"].astype(str)
    elif "Unnamed: 0" in sv_df.columns:
        sv_df["Patient_Id"] = sv_df["Unnamed: 0"].astype(str)
    else:
        raise ValueError("No sample ID column found in fusion file.")

    # Extract 5' and 3' gene names in original order
    if "Site1_Hugo_Symbol" not in sv_df.columns or "Site2_Hugo_Symbol" not in sv_df.columns:
        raise ValueError("Missing required columns: 'Site1_Hugo_Symbol' and/or 'Site2_Hugo_Symbol'")
    
    gene1 = sv_df["Site1_Hugo_Symbol"].astype(str).str.strip()
    gene2 = sv_df["Site2_Hugo_Symbol"].astype(str).str.strip()
    sv_df["Fusion_Gene"] = gene1 + "--" + gene2  # Preserve order (5' → 3')

    # Drop any rows with missing patient ID
    sv_df = sv_df.dropna(subset=["Patient_Id"])

    # One-hot encode and aggregate
    dummies = pd.get_dummies(sv_df["Fusion_Gene"])
    dummies["Patient_Id"] = sv_df["Patient_Id"]
    fusion_df = dummies.groupby("Patient_Id").sum()

    fusion_df.columns = [f"{col}_FUSION" for col in fusion_df.columns]

    return fusion_df


### Subtype Handler ###
def process_subtypes(sample_info, min_samples=5):
    """
    Process clinical DataFrame to extract a single 'Subtype' column with sample IDs as index.
    Filters subtypes with fewer than `min_samples` samples.
    """
    if sample_info is None or sample_info.empty:
        raise ValueError("[process_subtypes] Input DataFrame is empty or None.")

    # Standardize column names
    sample_info.columns = sample_info.columns.str.strip().str.upper().str.replace(" ", "_")

    subtype_candidates = [
        "CANCER_SUBTYPE_CURATED", "ONCOTREE_CODE", "ONCOTREE", "SUBTYPE",
        "DISEASE_SUBTYPE", "DISEASE", "CANCER_TYPE", "CANCER_TYPE_DETAILED"
    ]

    for col in subtype_candidates:
        if col in sample_info.columns:
            sample_info = sample_info[[col]].copy()
            sample_info.columns = ["Subtype"]
            break
    else:
        raise ValueError(f"[process_subtypes] No valid subtype column found in: {sample_info.columns.tolist()}")

    sample_info = sample_info.dropna(subset=["Subtype"])
    subtype_counts = sample_info["Subtype"].value_counts()
    valid_subtypes = subtype_counts[subtype_counts >= min_samples].index
    sample_info = sample_info[sample_info["Subtype"].isin(valid_subtypes)]

    return sample_info


def read_clinical_file(path):
    import pandas as pd

    for sep_try in [None, '\t', ',', r'\s+']:
        try:
            df = pd.read_csv(path, sep=sep_try, engine='python', header=0, index_col=0)
            df.index = df.index.astype(str).str.strip().str.upper()
            df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
            return df
        except Exception as e:
            continue

    raise ValueError(f"[read_clinical_file] Failed to load clinical file: {path}")


### Safe ID Mapping ###
def safe_map_index(df, study, name):
    mapped = to_patient_id(df.index.to_series(), study=study)
    df = df[mapped.notna()].copy()
    df.index = mapped[mapped.notna()].str.strip()
    print(f"[{name}] Mapped {len(df)} samples")
    return df


### Main Integration Function ###
def integrate_data(
    mut_path,
    cna_path,
    fusion_info_path,
    subtype_clinical_path,
    study=None,
    cna_process=True,
    cna_top_n=500,
    min_subtype_n=5,
    mut_freq_thresh=0.02,
    fusion_freq_thresh=0.02,
    is_tcga=True,
    rename=True,
):

    mut_raw = load_mutations_raw(mut_path)
    cna_raw = load_cna(cna_path, cna_top_n, cna_process,rename=rename)
    fusion_raw = load_fusions_raw(fusion_info_path) if fusion_info_path else None
    clinical = read_clinical_file(subtype_clinical_path)
    clinical = process_subtypes(clinical)

    # Mapping IDs
    mut_raw = safe_map_index(mut_raw, study, "Mutations")
    cna_raw = safe_map_index(cna_raw, study, "CNA")
    clinical = safe_map_index(clinical, study, "Clinical")
    if fusion_raw is not None:
        fusion_raw = safe_map_index(fusion_raw, study, "Fusions")

    print("Mut ∩ CNA:", len(mut_raw.index.intersection(cna_raw.index)))
    print("Mut ∩ Clinical:", len(mut_raw.index.intersection(clinical.index)))
    print("CNA ∩ Clinical:", len(cna_raw.index.intersection(clinical.index)))
    print("[DEBUG] Mutation mapped samples:", sorted(mut_raw.index.tolist())[:10])
    print("[DEBUG] CNA mapped samples:", sorted(cna_raw.index.tolist())[:10])
    print("[DEBUG] Fusion mapped samples:", sorted(fusion_raw.index.tolist())[:10] if fusion_raw is not None else "None")
    print("[DEBUG] Clinical mapped samples:", sorted(clinical.index.tolist())[:10])

    # Shared samples
    modalities = [mut_raw, cna_raw, clinical]
    include_fusions = fusion_raw is not None and isinstance(fusion_raw, pd.DataFrame) \
    and fusion_raw.shape[0] > 0 and fusion_raw.shape[1] > 0
    if include_fusions:
        modalities.append(fusion_raw)
    else:
        fusion_raw = pd.DataFrame(index=clinical.index)  # Dummy frame to avoid downstream errors
        print("[INFO] No valid fusions found. Proceeding without fusion data.")

    common = modalities[0].index
    for df in modalities[1:]:
        common = common.intersection(df.index)

    if len(common) == 0:
        raise ValueError("No shared samples across modalities. Check ID mapping or thresholds.")

    print(f"[integrate_data] Final shared sample count: {len(common)}")

    # Subset + log transform
    mut_df = mut_raw.loc[common]
    cna_df = np.power(2, cna_raw.loc[common]) if is_tcga else cna_raw.loc[common]
    clinical = clinical.loc[common]
    fusion_df = fusion_raw.loc[common] if fusion_raw is not None else pd.DataFrame(index=common)

    # Frequency filter
    if mut_df.shape[0] < 100:
        mut_df = mut_df.loc[:, (mut_df > 0).sum() >= 5]
    else:
        mut_df = mut_df.loc[:, (mut_df > 0).mean() >= mut_freq_thresh]
        top_300 = (mut_df > 0).mean().sort_values(ascending=False).head(300).index
        mut_df = mut_df[top_300]

    if include_fusions:
        if fusion_df.shape[0] < 100:
            fusion_df = fusion_df.loc[:, (fusion_df > 0).sum() >= 5]
        else:
            fusion_freq = (fusion_df > 0).sum(axis=0) / len(fusion_df)
            fusion_keep = fusion_freq[fusion_freq >= fusion_freq_thresh].index
            fusion_df = fusion_df.loc[:, fusion_keep]

        if fusion_df.shape[1] == 0:
            print("[Warning] No fusions passed frequency threshold. Dropping fusion modality.")
            fusion_df = pd.DataFrame(index=common)
    else:
        fusion_df = pd.DataFrame(index=common)

    # Rename to Sample_1, Sample_2, ...
    sample_names = [f"Sample_{i+1}" for i in range(len(common))]
    for df in (mut_df, cna_df, fusion_df, clinical):
        df.index = sample_names

    return mut_df, cna_df, fusion_df, clinical


#### Synthetic Data Creation ####

def sim_mut(mutation_probabilities, n_samples, seed=44):
    if seed is not None:
        np.random.seed(seed)
    sim_mutations = np.zeros((n_samples, len(mutation_probabilities)))
    for i in range(len(mutation_probabilities)):
        sim_mutations[:, i] = np.random.binomial(
            n=1,
            p=mutation_probabilities.iloc[i] if isinstance(mutation_probabilities, pd.Series) else mutation_probabilities[i],
            size=n_samples
        )
    return sim_mutations


def sim_cna(means, variances, n_samples, seed=44):
    if seed is not None:
        np.random.seed(seed)
    n_genes = len(means)
    sim_cna = np.zeros((n_samples, n_genes))
    for i in range(n_genes):
        mu = means.iloc[i]
        var = variances.iloc[i]
        if not np.isfinite(mu) or not np.isfinite(var) or mu <= 0:
            continue
        if var <= mu:
            p = 1 - (var / mu)
            n = mu / p
            sim_cna[:, i] = np.random.binomial(n=int(n), p=p, size=n_samples)
        else:
            r = mu**2 / (var - mu)
            p = r / (r + mu)
            sim_cna[:, i] = nbinom.rvs(n=r, p=p, size=n_samples)
    return sim_cna


def sim_by_cluster(mut, subtype, n_samples, fusions=None, seed=44):
    M = pd.concat([mut, subtype], axis=1)
    if fusions is not None:
        F = pd.concat([fusions, subtype], axis=1)
    

    subtype_counts = subtype['Subtype'].value_counts()
    subtypes = subtype_counts.index.tolist()
    proportions = subtype_counts / subtype_counts.sum()
    rounded_sizes = (proportions * n_samples).round().astype(int)
    rounded_sizes.iloc[-1] += n_samples - rounded_sizes.sum()

    new_mut = []
    new_fusions = [] if fusions is not None else None
    

    for i, (s, n) in enumerate(zip(subtypes, rounded_sizes)):
        seed_for_subtype = seed + i if seed is not None else None
        M_s = M[M['Subtype'] == s].drop(columns='Subtype')
        M_s = M_s.apply(pd.to_numeric, errors='coerce')
        n_subtype_samples = M_s.shape[0]
        mut_probs = M_s.mean(axis=0).fillna(0).clip(0, 1)
        mut_cluster = sim_mut(mut_probs, n_samples=n, seed=seed_for_subtype)
        new_mut.append(mut_cluster)

        if fusions is not None:
            F_s = F[F['Subtype'] == s].drop(columns='Subtype')
            F_s = F_s.apply(pd.to_numeric, errors='coerce')
            fus_probs = F_s.mean(axis=0).fillna(0).clip(0, 1)
            fus_cluster = sim_mut(fus_probs, n_samples=n, seed=seed_for_subtype + 100 if seed is not None else None)
            new_fusions.append(fus_cluster)


    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    output = {
        'mut': pd.DataFrame(np.vstack(new_mut), columns=mut.columns, index=sample_names)
    }
    if fusions is not None:
        output['fusion'] = pd.DataFrame(np.vstack(new_fusions), columns=fusions.columns, index=sample_names)

    return output


def normalize_weights(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def preprocess_alteration_matrix_weighted(mut_df, fus_df=None, cna_df=None):
    components_scaled = []
    components_unscaled = {}

    if mut_df is not None:
        mut_df = mut_df.astype(int)
        components_scaled.append(mut_df)
        components_unscaled['mut'] = mut_df

    if fus_df is not None:
        fus_df = fus_df.astype(int)
        components_scaled.append(fus_df)
        components_unscaled['fusion'] = fus_df

    if cna_df is not None:
        scaler = StandardScaler()
        cna_scaled = pd.DataFrame(
            scaler.fit_transform(cna_df),
            index=cna_df.index,
            columns=cna_df.columns
        )
        components_scaled.append(cna_scaled)
        components_unscaled['cna'] = cna_df

    combined_scaled = pd.concat(components_scaled, axis=1)
    return combined_scaled, components_unscaled

def adjust_prob_with_minimum(values, min_freq=0.02):
    prob = np.mean(values)
    return max(prob, min_freq)


def apply_feature_weights(scaled_df, mut_cols, fusion_cols, cna_cols, weights):
    df_weighted = scaled_df.copy()
    for col in mut_cols:
        if col in df_weighted:
            df_weighted[col] *= weights.get('mut', 1.0)
    for col in fusion_cols:
        if col in df_weighted:
            df_weighted[col] *= weights.get('fusion', 1.0)
    for col in cna_cols:
        if col in df_weighted:
            df_weighted[col] *= weights.get('cna', 1.0)
    return df_weighted


def sample_from_neighbors_weighted(scaled_df, unscaled_dfs, n_samples, k_neighbors=5, seed=44):
    np.random.seed(seed)
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
    nn_model.fit(scaled_df.values)

    sampled_rows = []

    for _ in range(n_samples):
        anchor_idx = np.random.choice(scaled_df.shape[0])
        anchor = scaled_df.iloc[anchor_idx]
        neighbors_idx = nn_model.kneighbors([anchor.values], return_distance=False)[0]
        neighborhood_idx = scaled_df.iloc[neighbors_idx].index

        synthetic_row = {}
        for col in scaled_df.columns:
            if col in unscaled_dfs.get('mut', {}) and col in unscaled_dfs['mut'].columns:
                values = unscaled_dfs['mut'].loc[neighborhood_idx, col]
                prob = adjust_prob_with_minimum(values, min_freq=0.02)
                synthetic_row[col] = np.random.binomial(1, prob)
            elif  col in unscaled_dfs.get('fusion', {}) and col in unscaled_dfs['fusion'].columns:
                values = unscaled_dfs['fusion'].loc[neighborhood_idx, col].dropna()
                if len(values) == 0:
                    prob = 0.0  # fallback for empty neighborhood
                else:
                    prob = np.clip(values.mean(), 0, 1)
                synthetic_row[col] = np.random.binomial(1, prob)
            elif col in unscaled_dfs.get('cna', {}) and col in unscaled_dfs['cna'].columns:
                values = unscaled_dfs['cna'].loc[neighborhood_idx, col].values
                mu, var = np.mean(values), np.var(values)
                if not np.isfinite(mu) or mu <= 0:
                    synthetic_row[col] = mu
                elif var <= mu:
                    p = 1 - (var / mu)
                    n = mu / p
                    synthetic_row[col] = np.random.binomial(n=int(max(n, 1)), p=p)
                else:
                    r = mu**2 / (var - mu)
                    p = r / (r + mu)
                    synthetic_row[col] = nbinom.rvs(n=r, p=p)
        sampled_rows.append(synthetic_row)

    return pd.DataFrame(sampled_rows)


def sim_by_cluster_neighbors_weighted(mut, subtype, n_samples, cna=None, fusions=None, weights=None, seed=44, k_neighbors=5):
    weights = weights or {'mut': 1.0, 'fusion': 2.0, 'cna': 1.0}
    scaled_combined, unscaled_dfs = preprocess_alteration_matrix_weighted(mut, fusions, cna)
    scaled_combined = scaled_combined.loc[subtype.index]
    combined = pd.concat([scaled_combined, subtype], axis=1)

    subtype_counts = subtype['Subtype'].value_counts()
    proportions = subtype_counts / subtype_counts.sum()
    rounded_sizes = (proportions * n_samples).round().astype(int)
    rounded_sizes.iloc[-1] += n_samples - rounded_sizes.sum()

    new_data = []
    for i, (s, n) in enumerate(zip(rounded_sizes.index, rounded_sizes)):
        subset_scaled = combined[combined['Subtype'] == s].drop(columns='Subtype')
        subset_indices = subset_scaled.index
        subset_unscaled = {k: df.loc[subset_indices] for k, df in unscaled_dfs.items()}

        k_actual = min(k_neighbors, subset_scaled.shape[0] - 1)
        if k_actual < 1:
            raise ValueError(f"Subtype '{s}' has too few samples to simulate.")

        weighted_scaled = apply_feature_weights(
            subset_scaled,
            mut.columns if mut is not None else [],
            fusions.columns if fusions is not None else [],
            cna.columns if cna is not None else [],
            weights
        )

        synthetic = sample_from_neighbors_weighted(weighted_scaled, subset_unscaled, n_samples=n, k_neighbors=k_actual, seed=seed + i)
        new_data.append(synthetic)

    synthetic_full = pd.concat(new_data, axis=0)
    synthetic_full.index = [f"Sample_{i+1}" for i in range(n_samples)]

    output = {}
    if mut is not None:
        output['mut'] = synthetic_full[mut.columns].astype(int)
    if fusions is not None:
        output['fusion'] = synthetic_full[fusions.columns].astype(int)
    if cna is not None:
        output['cna'] = synthetic_full[cna.columns]

    return output


def center_cna(CN: pd.DataFrame, diploid: int = 2) -> pd.DataFrame:
    """
    Center integer copy numbers so diploid=0 (CN - diploid).
    """
    return CN.astype(float) - diploid


"""
EXPRESSION  SIMULATION FUNCTIONS
"""

def preprocess_rna_for_simulation(rna_df, strategy="auto", user_scale=None, verbose=True):
    """
    Preprocess RNA-seq expression data to convert from normalized form (e.g., CPM, RPKM, RSEM)
    into pseudo-counts for negative binomial simulation.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Normalized RNA expression (samples × genes).
    strategy : str
        'auto' to choose scale based on data, or 'manual' to specify your own.
    user_scale : float or None
        Custom scale factor if strategy is 'manual'.
    verbose : bool
        Whether to print summary info.

    Returns
    -------
    rna_scaled : pd.DataFrame
        Pseudo-count integer RNA expression matrix.
    scale_factor : float
        Scale factor that was applied.
    """

    if strategy == "manual":
        if user_scale is None:
            raise ValueError("You must provide user_scale if using manual strategy.")
        scale_factor = user_scale
    elif strategy == "auto":
        q75 = rna_df.quantile(0.75).median()
        if q75 < 1:
            scale_factor = 100
        elif q75 < 10:
            scale_factor = 50
        elif q75 < 100:
            scale_factor = 10
        else:
            scale_factor = 1
    else:
        raise ValueError("strategy must be 'auto' or 'manual'")

    # Apply pseudocounting and scaling
    rna_scaled = (rna_df * scale_factor).round().astype(int)
    rna_scaled[rna_scaled < 0] = 0

    if verbose:
        print(f"[preprocess_rna] Applied scale factor: {scale_factor}")
        print(rna_scaled.describe().T.head(5))

    return rna_scaled, scale_factor

def select_genes_with_expr_filter(
    rna_df: pd.DataFrame,
    alterations_df: pd.DataFrame,
    target_total: int = 10_000,
    min_cpm: float = 1.0,
    min_prop_samples: float = 0.20,   # keep gene if expressed in ≥20% samples
    use_mad: bool = False,            # alternative variability metric
    verbose: bool = True
):
    """
    rna_df: samples x genes (counts or log-like)
    alterations_df: samples x alteration features (cols like 'TP53_mut', 'ERBB2_CNA', etc.)
    target_total: desired total number of genes (including all altered genes present in RNA)
    min_cpm: CPM threshold to call a gene 'expressed' (if rna_df looks like counts)
    min_prop_samples: fraction of samples that must be expressed to keep a gene
    use_mad: if True, rank by MAD of log1p-CPM instead of variance
    verbose: print a short summary
    """
    # ---------- 0) Clean inputs ----------
    # Ensure gene columns are strings
    rna_df = rna_df.copy()
    rna_df.columns = rna_df.columns.astype(str)
    alterations_df = alterations_df.copy()
    alterations_df.columns = alterations_df.columns.astype(str)

    # ---------- 1) Identify altered genes present in RNA ----------
    altered_genes = list({c.split('_')[0] for c in alterations_df.columns})
    altered_in_expr = [g for g in altered_genes if g in rna_df.columns]

    # ---------- 2) Decide if rna_df looks like raw counts ----------
    # Heuristic: if most values are non-negative integers and max >= 50, treat as counts
    values = rna_df.to_numpy()
    nonneg = (values >= 0).mean() > 0.999
    int_like = (np.isclose(values, np.round(values)).mean() > 0.98)
    looks_like_counts = bool(nonneg and int_like and np.nanmax(values) >= 50)

    # ---------- 3) Library-size normalization & log1p-CPM ----------
    if looks_like_counts:
        lib_sizes = rna_df.sum(axis=1).replace(0, np.nan)  # avoid div by 0
        cpm = (rna_df.div(lib_sizes, axis=0)) * 1e6
        log_cpm = np.log1p(cpm)
        # Expression filter: CPM >= min_cpm in ≥ min_prop_samples of samples
        expressed_mask = (cpm >= min_cpm).mean(axis=0) >= min_prop_samples
    else:
        # Already TPM / FPKM / log2(TPM+1) style. Use a small floor for expression.
        # Treat values > 0.1 as "expressed" (tune if needed).
        log_cpm = np.log1p(rna_df)  # still stabilize before variance/MAD
        expressed_mask = (rna_df > 0.1).mean(axis=0) >= min_prop_samples

    # ---------- 4) Keep all altered genes regardless of expression ----------
    # But we’ll still report which altered genes were lowly expressed.
    low_expr_altered = [g for g in altered_in_expr if not expressed_mask.get(g, False)]

    # Genes eligible for variability ranking = non-altered & pass expression filter
    non_altered = [g for g in rna_df.columns if g not in altered_in_expr]
    non_alt_keep = [g for g in non_altered if expressed_mask.get(g, False)]

    # ---------- 5) Rank by variability among eligible non-altered genes ----------
    if len(non_alt_keep) > 0:
        X = log_cpm[non_alt_keep]  # samples x kept genes
        if use_mad:
            # Median absolute deviation across samples, per gene
            med = X.median(axis=0)
            variability = (X.sub(med, axis=1)).abs().median(axis=0)
        else:
            # Sample variance across samples, per gene
            variability = X.var(axis=0, ddof=1)
        variability = variability.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked_non_altered = variability.sort_values(ascending=False).index.tolist()
    else:
        ranked_non_altered = []

    # ---------- 6) Decide how many non-altered to take ----------
    # We *must* keep all altered genes that exist in RNA.
    n_altered = len(altered_in_expr)
    n_remaining = max(0, target_total - n_altered)

    # If altered genes already exceed target_total, we keep them all and add none
    top_non_altered = ranked_non_altered[:n_remaining] if n_remaining > 0 else []

    # ---------- 7) Compose final set ----------
    genes_to_keep = sorted(set(altered_in_expr + top_non_altered))

    # ---------- 8) Book-keeping / reporting ----------
    low_expr_non_altered = [g for g in non_altered if not expressed_mask.get(g, False)]
    genes_dropped_low_expr = sorted(low_expr_non_altered)  # altered are not dropped, just flagged

    if verbose:
        total_genes = rna_df.shape[1]
        kept_non_altered = len(top_non_altered)
        dropped_low_expr = len(genes_dropped_low_expr)
        print(f"[Gene selection]")
        print(f"  Total RNA genes: {total_genes}")
        print(f"  Altered genes present in RNA: {n_altered}")
        print(f"  Target total: {target_total} -> selecting {kept_non_altered} non-altered to fill")
        print(f"  Low-expression (non-altered) genes dropped: {dropped_low_expr}")
        if low_expr_altered:
            print(f"  NOTE: {len(low_expr_altered)} altered gene(s) are lowly expressed but KEPT: {', '.join(low_expr_altered[:10])}{' ...' if len(low_expr_altered) > 10 else ''}")

    return {
        "genes_to_keep": genes_to_keep,
        "altered_genes_kept": sorted(altered_in_expr),
        "low_expr_altered_flagged": sorted(low_expr_altered),
        "genes_dropped_low_expr": genes_dropped_low_expr
    }

def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=None, condition='A'):
    """
    Estimate DESeq2-style parameters from real RNA-seq data.
    Returns gene means, variances, dispersions, and size factors.
    """
    if seed is not None:
        np.random.seed(seed)

    rna_df = rna_df.round().astype(int)
    # Alternate conditions to ensure at least two levels
    fake_conditions = ['A'] * (rna_df.shape[0] // 2) + ['B'] * (rna_df.shape[0] - rna_df.shape[0] // 2)
    metadata = pd.DataFrame({'condition': fake_conditions}, index=rna_df.index)

    dds = DeseqDataSet(
        counts=rna_df,
        metadata=metadata,
        design_factors="condition",
        ref_level=condition
    )
    dds.deseq2()

    gene_means = rna_df.mean()
    gene_vars = rna_df.var()
    dispersions = pd.Series(dds.varm["dispersions"], index=rna_df.columns)
    dispersions = dispersions.fillna(dispersions.median())
    dispersions = dispersions.clip(lower=1e-6, upper=1e6)

    size_factors = dds.obsm['size_factors']

    return gene_means, gene_vars, dispersions, size_factors


def simulate_rna_background(gene_means, dispersions, size_factors, n_samples):
    """
    Simulate RNA expression background using NB with DESeq2-style parameters.
    """
    mu_matrix = np.outer(size_factors, gene_means)
    return pd.DataFrame(mu_matrix, columns=gene_means.index)


def generate_signatures(genes, alteration_features, min_size=1, max_size=150, seed=44):
    """
    Generate random expression signatures for mutations or fusions.
    For fusions, ensure both partner genes are targets.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    signatures = {}
    for alt in alteration_features:
        if "_FUSION" in alt:
            # Fusion case: extract partner genes
            partners = alt.replace("_FUSION", "").split("-")
            fusion_targets = [g for g in partners if g in genes]

            # Fill out target list
            n_targets = random.randint(min_size, max_size)
            potential_targets = [g for g in genes if g not in fusion_targets]
            n_remaining = n_targets - len(fusion_targets)
            if len(potential_targets) < n_remaining:
                continue
            targets = fusion_targets + random.sample(potential_targets, k=n_remaining)

        else:
            base_gene = alt.split('_')[0]
            if base_gene not in genes:
                continue
            n_targets = random.randint(min_size, max_size)
            potential_targets = [g for g in genes if g != base_gene]
            if len(potential_targets) < n_targets - 1:
                continue
            targets = [base_gene] + random.sample(potential_targets, k=n_targets - 1)

        # Assign random effects (fusion or mutation)
        effects = {t: np.round(np.random.uniform(-2.5, 2.5), 2) for t in targets}
        signatures[alt] = {'targets': targets, 'effects': effects}
    return signatures

def inject_expression_effects(expr_df, alteration_df, signatures, cna_df=None):
    """
    Inject additive alteration effects into expression matrix.
    Optionally includes CNA multiplicative modulation.
    """
    modified = expr_df.copy()
    for sample in expr_df.index:
        active_alts = alteration_df.columns[alteration_df.loc[sample] == 1]
        for gene in expr_df.columns:
            alpha_i = expr_df.loc[sample, gene]  # baseline
            beta_sum = 0
            for alt in active_alts:
                if alt in signatures and gene in signatures[alt]['targets']:
                    beta_sum += signatures[alt]['effects'][gene]

            # Safely extract CNA multiplier
            copy_factor = 1.0
            if cna_df is not None and gene in cna_df.columns:
                val = cna_df.loc[sample, gene]
                if isinstance(val, pd.Series):
                    copy_factor = float(val.iloc[0])
                else:
                    copy_factor = float(val)

            expr_value = (beta_sum + alpha_i) * copy_factor / 2
            modified.loc[sample, gene] = max(0, expr_value)

    return modified


def sample_nb(mu, dispersions):
    """
    Sample RNA-seq counts using a Negative Binomial distribution.
    Handles edge cases and clips invalid values.
    """
    mu = np.asarray(mu)
    if isinstance(dispersions, pd.Series):
        dispersions = dispersions.values

    # Check for NaNs or infs in inputs
    if np.any(~np.isfinite(mu)):
        raise ValueError("mu contains non-finite values")
    if np.any(~np.isfinite(dispersions)):
        raise ValueError("dispersions contain non-finite values")

    mu = np.clip(mu, 1e-8, 1e6)
    dispersions = np.clip(dispersions, 1e-8, 1e6)

    r = 1.0 / dispersions
    r = np.clip(r, 1e-6, 1e6)

    if mu.shape[1] != r.shape[0]:
        raise ValueError(f"Shape mismatch: mu {mu.shape}, r {r.shape}")

    r = r[np.newaxis, :]
    p = r / (r + mu)

    if np.any(np.isnan(p)) or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError(
            f"Invalid p in NB sampling: min={np.nanmin(p)}, max={np.nanmax(p)}, NaNs={np.isnan(p).sum()}"
        )

    return np.random.negative_binomial(n=r, p=p)


def simulate_rna_with_signatures(
    rna_df,
    alteration_df,
    cna_df=None,
    n_samples=600,
    min_sig_size=1,
    max_sig_size=150,
    n_genes_to_sim=10000,
    seed=44
):
    """
    Simulate RNA-seq counts with expression effects from mutations, fusions, and optional CNAs.
    """
    if seed is not None:
        np.random.seed(seed)
   
    # Step 1: Estimate RNA-seq parameters
    gene_means, gene_vars, dispersions, size_factors = estimate_deseq2_parameters(rna_df, seed=seed)

    # Step 2: Identify genes affected by alterations (including fusion partners)
    altered_genes = set()
    for alt in alteration_df.columns:
        if "_FUSION" in alt:
            partners = alt.replace("_FUSION", "").split("-")
            altered_genes.update(partners)
        else:
            base_gene = alt.split("_")[0]
            altered_genes.add(base_gene)

    # Step 3: Select expression genes to simulate
    top_genes = gene_means.sort_values(ascending=False).head(n_genes_to_sim).index
    genes_to_sim = sorted(set(top_genes).union(altered_genes).intersection(rna_df.columns))

    # Step 4: Subset all expression-related objects
    rna_df = rna_df[genes_to_sim]
    gene_means = gene_means[genes_to_sim]
    gene_vars = gene_vars[genes_to_sim]
    dispersions = dispersions[genes_to_sim]

    # Step 5: Subset CNA (if provided)
    if cna_df is not None:
        if all(col.endswith('_CNA') for col in cna_df.columns):
            cna_df = cna_df.copy()
            cna_df.columns = cna_df.columns.str.replace('_CNA', '', regex=False)
        cna_df = cna_df.loc[:, cna_df.columns.isin(genes_to_sim)]

    # Step 6: Simulate background expression
    expr_bg = simulate_rna_background(gene_means, dispersions, size_factors[:n_samples], n_samples)
    expr_bg.index = [f"Sample_{i+1}" for i in range(n_samples)]

    # Step 7: Align alteration and CNA data
    alteration_df = alteration_df.copy()
    alteration_df.index = expr_bg.index
    if cna_df is not None:
        cna_df = cna_df.copy()
        cna_df.index = expr_bg.index

    # Step 8: Generate signatures (fusion-aware version assumed)
    all_alts = alteration_df.columns.tolist()
    true_signatures = generate_signatures(expr_bg.columns.tolist(), all_alts, min_sig_size, max_sig_size, seed=seed)

    # Step 9: Inject alteration-driven effects
    expr_effected = inject_expression_effects(expr_bg, alteration_df, true_signatures, cna_df)

    # Step 10: Sample final RNA-seq counts
    dispersions = dispersions[expr_effected.columns]
    print("expr_effected shape:", expr_effected.shape)
    print("dispersions shape:", dispersions.shape)
    print("NaNs in dispersions:", dispersions.isna().sum())
    print("NaNs in expr_effected:", np.isnan(expr_effected.values).sum())

    expr_counts = sample_nb(expr_effected.values, dispersions)
    expr_sim = pd.DataFrame(expr_counts, columns=expr_bg.columns, index=expr_bg.index)

    return expr_sim, true_signatures

import pandas as pd
import json
import os

def save_simulation_outputs(
    rna_real,
    expr_sim,
    alterations_df,
    cna_df,
    true_signatures,
    output_dir,
    suffix,
    compress=False
):
    """
    Save real and simulated RNA data, alteration data, and true signatures with dataset-specific suffix.

    Parameters
    ----------
    rna_real : pd.DataFrame
        The real expression matrix (scaled or original).
    expr_sim : pd.DataFrame
        The simulated RNA-seq expression matrix.
    alterations_df : pd.DataFrame
        Binary mutation/fusion/CNA alteration matrix.
    cna_df : pd.DataFrame or None
        CNA matrix (log2 or count-based).
    true_signatures : dict
        Dictionary of true signatures used in the simulation.
    output_dir : str
        Directory path where files will be saved.
    suffix : str
        Identifier for the dataset (e.g., "IBC", "COAD").
    compress : bool
        Whether to save CSVs as compressed `.csv.gz`.
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = ".csv.gz" if compress else ".csv"

    # Save DataFrames
    rna_real.to_csv(os.path.join(output_dir, f"rna_real_{suffix}{ext}"))
    expr_sim.to_csv(os.path.join(output_dir, f"rna_simulated_{suffix}{ext}"))
    alterations_df.to_csv(os.path.join(output_dir, f"alterations_{suffix}{ext}"))

    if cna_df is not None:
        cna_df.to_csv(os.path.join(output_dir, f"cna_{suffix}{ext}"))

    # Save true signatures dict as JSON
    with open(os.path.join(output_dir, f"true_signatures_{suffix}.json"), "w") as f:
        json.dump(true_signatures, f, indent=2)

    print(f"[✓] All files saved for {suffix} in: {output_dir}")
