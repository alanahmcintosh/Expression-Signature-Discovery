
"""
==============================================================
SIMULATION PIPELINE
==============================================================

This module simulates:
  1. Multi-omic alteration matrices (mutations, fusions, CNAs, clinical)
     using subtype-stratified KNN + ratio-based CNA modeling.
  2. RNA-seq expression data based on those alterations, using
     DESeq2-style negative binomial parameterization and
     alteration-driven expression effects.

Developed for benchmarking gene-expression signature discovery methods.
==============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==============================================================
# PART 1 — PREPROCESSING + ALTERATION SIMULATION
# ==============================================================

def preprocess_X_weighted(mut=None, fusion=None, cna=None, clinical=None, weights=None):
    """
    Standardizes and weights alteration data blocks before KNN sampling.

    Parameters
    ----------
    mut, fusion, cna, clinical : pd.DataFrame or None
        Input alteration matrices (samples x features).
    weights : dict
        Relative weight for each omic type during scaling.

    Returns
    -------
    combined_scaled : pd.DataFrame
        Weighted + scaled concatenation of all available blocks.
    unscaled_blocks : dict
        Original unscaled blocks (used for downstream resampling).
    """
    weights = weights or {'mut': 1.0, 'fusion': 1.5, 'cna': 1.0, 'clinical': 0.5}
    scaler = StandardScaler()
    scaled_blocks, unscaled_blocks = [], {}

    # Mutations
    if mut is not None:
        mut = mut.astype(float)
        scaled_blocks.append(mut * weights['mut'])
        unscaled_blocks['mut'] = mut

    # Fusions
    if fusion is not None:
        fusion = fusion.astype(float)
        scaled_blocks.append(fusion * weights['fusion'])
        unscaled_blocks['fusion'] = fusion

    # CNA: z-score scale across genes (continuous)
    if cna is not None:
        cna_scaled = pd.DataFrame(
            scaler.fit_transform(cna),
            index=cna.index,
            columns=cna.columns
        )
        scaled_blocks.append(cna_scaled * weights['cna'])
        unscaled_blocks['cna'] = cna

    # Clinical covariates (numerical)
    if clinical is not None:
        clin_scaled = pd.DataFrame(
            scaler.fit_transform(clinical),
            index=clinical.index,
            columns=clinical.columns
        )
        scaled_blocks.append(clin_scaled * weights['clinical'])
        unscaled_blocks['clinical'] = clinical

    # Combine scaled omic blocks
    combined_scaled = pd.concat(scaled_blocks, axis=1)
    return combined_scaled, unscaled_blocks


def sample_from_neighbors_ratioCNA(
    scaled_df, unscaled_dfs, n_samples, k_neighbors=5, seed=44
):
    """
    Samples synthetic alteration profiles by drawing from the local
    neighborhood of real samples (KNN-based multivariate structure preservation).

    Parameters
    ----------
    scaled_df : pd.DataFrame
        Scaled concatenated matrix for neighbor search.
    unscaled_dfs : dict
        Dict of original unscaled blocks for resampling.
    n_samples : int
        Number of synthetic samples to generate.
    k_neighbors : int
        Size of local neighborhood to sample from.
    """
    np.random.seed(seed)
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(scaled_df)), metric="euclidean")
    nn.fit(scaled_df.values)

    samples = []

    for _ in range(n_samples):
        # Randomly pick an anchor real sample
        anchor = scaled_df.sample(1, random_state=np.random.randint(0, 1e6))
        neighbors_idx = nn.kneighbors(anchor.values, return_distance=False)[0]
        neighborhood_idx = scaled_df.iloc[neighbors_idx].index
        synthetic = {}

        # --- MUTATIONS ---
        for col in scaled_df.columns:
            if col in unscaled_dfs.get('mut', {}) and col in unscaled_dfs['mut'].columns:
                values = unscaled_dfs['mut'].loc[neighborhood_idx, col]
                prob = np.clip(values.mean(), 0.02, 0.98)
                synthetic[col] = np.random.binomial(1, prob)

            # --- FUSIONS ---
            elif col in unscaled_dfs.get('fusion', {}) and col in unscaled_dfs['fusion'].columns:
                values = unscaled_dfs['fusion'].loc[neighborhood_idx, col].dropna()
                prob = np.clip(values.mean() if len(values) > 0 else 0.0, 0, 1)
                synthetic[col] = np.random.binomial(1, prob)

            # --- CNAs ---
            elif col in unscaled_dfs.get('cna', {}):
                vals = unscaled_dfs['cna'].loc[neighborhood_idx, col].dropna().values
                if len(vals) == 0:
                    continue

                # Log-normal sampling approximates multiplicative copy-number variation
                mu = np.mean(np.log(np.clip(vals, 1e-3, None)))
                sigma = np.std(np.log(np.clip(vals, 1e-3, None))) + 1e-6
                val = np.random.lognormal(mean=mu, sigma=sigma)

                # Clamp to realistic CNA range
                synthetic[col] = np.clip(val, 0.2, 3.5)

            # --- CLINICAL ---
            elif col in unscaled_dfs.get('clinical', {}):
                vals = unscaled_dfs['clinical'].loc[neighborhood_idx, col].dropna().values
                if len(vals) == 0:
                    continue
                synthetic[col] = np.random.choice(vals)

        samples.append(synthetic)

    return pd.DataFrame(samples)


def simulate_X_hybrid_ratioCNA(
    mut, fusion, cna, clinical, subtype, n_samples,
    weights=None, k_neighbors=5, seed=44
):
    """
    Subtype-aware hybrid simulator for mutations, fusions, CNAs, and clinical data.
    Each subtype is simulated independently to preserve within-subtype structure.
    """
    np.random.seed(seed)
    weights = weights or {'mut': 1.0, 'fusion': 1.5, 'cna': 1.0, 'clinical': 0.5}

    # Determine how many samples to simulate per subtype (proportional to real subtype distribution)
    subtype_counts = subtype['Subtype'].value_counts()
    proportions = subtype_counts / subtype_counts.sum()
    sizes = (proportions * n_samples).round().astype(int)
    sizes.iloc[-1] += n_samples - sizes.sum()  # fix rounding mismatch

    blocks = []
    for i, (s, n) in enumerate(zip(sizes.index, sizes)):
        idx = subtype[subtype['Subtype'] == s].index

        scaled, unscaled = preprocess_X_weighted(
            mut=mut.loc[idx] if mut is not None else None,
            fusion=fusion.loc[idx] if fusion is not None else None,
            cna=cna.loc[idx] if cna is not None else None,
            weights=weights
        )

        sim_df = sample_from_neighbors_ratioCNA(
            scaled_df=scaled,
            unscaled_dfs=unscaled,
            n_samples=n,
            k_neighbors=k_neighbors,
            seed=seed + i
        )
        sim_df['Subtype'] = s
        blocks.append(sim_df)

    X_sim = pd.concat(blocks, axis=0)
    X_sim.index = [f"Sample_{i+1}" for i in range(len(X_sim))]
    return X_sim


def split_simulated_blocks_v2(X_sim):
    """
    Splits a combined simulated matrix into separate dataframes for each omic block.
    """
    X_sim.columns = X_sim.columns.astype(str)

    mut_cols = [c for c in X_sim.columns if c.endswith(("_MUT", "_GOF", "_LOF"))]
    fusion_cols = [c for c in X_sim.columns if "_FUSION" in c or "_FUS" in c]
    cna_cols = [c for c in X_sim.columns if c.endswith("_CNA")]
    subtype_col = "SUBTYPE" if "SUBTYPE" in X_sim.columns else None

    exclude = set(mut_cols + fusion_cols + cna_cols + ([subtype_col] if subtype_col else []))
    clinical_cols = [c for c in X_sim.columns if c not in exclude]

    mut_sim = X_sim[mut_cols] if mut_cols else pd.DataFrame(index=X_sim.index)
    fusion_sim = X_sim[fusion_cols] if fusion_cols else pd.DataFrame(index=X_sim.index)
    cna_sim = X_sim[cna_cols] if cna_cols else pd.DataFrame(index=X_sim.index)
    clin_sim = X_sim[clinical_cols] if clinical_cols else pd.DataFrame(index=X_sim.index, dtype='object')
    subtype_sim = X_sim[subtype_col] if subtype_col else pd.Series(index=X_sim.index, dtype="object")

    print(f" Mutations: {mut_sim.shape[1]} features")
    print(f" Fusions:   {fusion_sim.shape[1]} features")
    print(f" CNAs:      {cna_sim.shape[1]} features")
    print(f" Clinical:  {clin_sim.shape[1]} features")
    if subtype_col:
        print(f" Subtypes:  {subtype_sim.nunique()} unique")

    return mut_sim, fusion_sim, cna_sim, clin_sim, subtype_sim

