

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==============================================================
# PART 1 — PREPROCESSING + ALTERATION SIMULATION
# ==============================================================
def estimate_cna_event_params(cna_real):
    """
    Estimate per-gene CNA event probabilities from GISTIC-like integer CNAs in {-2,-1,0,1,2}.

    Robust to duplicated columns: if duplicates exist for a gene, we collapse
    them by mean and round back to int in [-2,2].
    
    Used for simulation of CNA events
    """
    if cna_real is None or cna_real.empty:
        return pd.DataFrame(columns=["gene", "p_amp", "p_del", "p_neu", "q_amp2", "q_del2"]).set_index("gene")

    df = cna_real.copy()
    df.columns = df.columns.astype(str)

    # Coerce numeric and clip to expected range
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = np.rint(df).astype(int).clip(-2, 2)

    # Collapse duplicate gene columns safely (common in TCGA merges)
    if df.columns.duplicated().any():
        df = df.groupby(axis=1, level=0).mean()
        df = np.rint(df).astype(int).clip(-2, 2)

    rows = []
    for g in df.columns:
        x = df[g]  # guaranteed Series now

        p_amp = float((x > 0).mean())
        p_del = float((x < 0).mean())
        p_neu = float((x == 0).mean())  # safer than 1 - p_amp - p_del

        x_amp = x[x > 0]
        x_del = x[x < 0]

        q_amp2 = float((x_amp == 2).mean()) if len(x_amp) > 0 else 0.0
        q_del2 = float((x_del == -2).mean()) if len(x_del) > 0 else 0.0

        rows.append((g, p_amp, p_del, p_neu, q_amp2, q_del2))

    out = pd.DataFrame(rows, columns=["gene", "p_amp", "p_del", "p_neu", "q_amp2", "q_del2"]).set_index("gene")
    return out


def robust_scale_with_floor(df, cna_std_floor):
    '''
    Column wise z scoring with a min standard deviation floor. (CNA deatures have low/near constant variance, having an min std floor prevents explosive values). 
    Rare/ Constant altertaions remain bounf instead of dominating distance matrix
    '''
    df = df.astype(float)
    mu = df.mean(axis=0)
    
    # Col standard deviation
    # Replace wih floor 0 so no NaNs occur from division by 0
    sd = df.std(axis=0).fillna(0.0)
    
    # Minimimum allowed denominator
    # Prevents dividion by small/constant variance
    denom = np.maximum(sd.to_numpy(), float(cna_std_floor))
    
    #Standardize
    out = (df - mu) / denom
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def preprocess_X_weighted(
    mut = None,
    fusion = None,
    cna = None,
    clinical = None,
    weights = {"mut": 1.0, "fusion": 1.5, "cna": 2.0, "clinical": 0.5},
    cna_std_floor = 0.25,
    cna_clip: tuple[int, int] = (-2, 2),
):
    """
    Standardizes and weights alteration data blocks before KNN sampling.

    Key points:
      - MUT/FUS are treated as numeric binary blocks and weighted directly.
      - CNA is assumed to be GISTIC-like integers in [-2,-1,0,1,2].
        We:
          (i) coerce to numeric, round, clip,
          (ii) collapse duplicated CNA columns (critical for downstream sampling),
          (iii) create two non-negative "views": AMP_LVL (0/1/2) and DEL_LVL (0/1/2),
          (iv) robust z-score each with a std floor to avoid divide-by-zero,
          (v) weight and concatenate for neighbor search.
      - Clinical: numeric-only scaling for KNN; original is preserved for sampling.

    Returns
    -------
    combined_scaled : pd.DataFrame
        Weighted + scaled concatenation of all available blocks for neighbor search.
    unscaled_blocks : dict
        Original unscaled blocks (used for downstream resampling).
    """
    weights = weights
    scaler = StandardScaler()

    scaled_blocks: list[pd.DataFrame] = []
    unscaled_blocks: dict[str, pd.DataFrame] = {}

    # -------------------------
    # Mutations
    # -------------------------
    if mut is not None and not mut.empty:
        mut = mut.copy()
        mut.columns = mut.columns.astype(str)
        mut = mut.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        scaled_blocks.append(mut * float(weights.get("mut", 1.0)))
        unscaled_blocks["mut"] = mut

    # -------------------------
    # Fusions
    # -------------------------
    if fusion is not None and not fusion.empty:
        fusion = fusion.copy()
        fusion.columns = fusion.columns.astype(str)
        fusion = fusion.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        scaled_blocks.append(fusion * float(weights.get("fusion", 1.5)))
        unscaled_blocks["fusion"] = fusion

    # -------------------------
    # CNA (GISTIC-like integers)
    # -------------------------
    if cna is not None and not cna.empty:
        cna_proc = cna.copy()
        cna_proc.columns = cna_proc.columns.astype(str)

        # Coerce -> int -> clip into expected range
        cna_num = cna_proc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        lo, hi = cna_clip
        g = np.rint(cna_num).astype(int).clip(lo, hi)

        # collapse duplicated CNA columns so block[col] is always a Series
        if g.columns.duplicated().any():
            g = g.groupby(axis=1, level=0).mean()
            g = np.rint(g).astype(int).clip(lo, hi)

        # Keep unscaled discrete CNA for later sampling/output
        unscaled_blocks["cna"] = g

        # Two non-negative views (dosage levels)
        amp_lvl = g.where(g > 0, 0).astype(float)            # 0,1,2
        del_lvl = (-g.where(g < 0, 0)).astype(float)         # 0,1,2

        amp_scaled = robust_scale_with_floor(amp_lvl, cna_std_floor=cna_std_floor)
        del_scaled = robust_scale_with_floor(del_lvl, cna_std_floor=cna_std_floor)

        # Rename so both views can coexist
        amp_scaled.columns = [f"{c}__AMP_LVL" for c in amp_scaled.columns]
        del_scaled.columns = [f"{c}__DEL_LVL" for c in del_scaled.columns]

        cna_scaled = pd.concat([amp_scaled, del_scaled], axis=1)
        scaled_blocks.append(cna_scaled * float(weights.get("cna", 2.0)))

    # -------------------------
    # Clinical (numeric only for scaling)
    # -------------------------
    if clinical is not None and not clinical.empty:
        clinical = clinical.copy()
        clinical.columns = clinical.columns.astype(str)

        # Keep original for sampling later
        unscaled_blocks["clinical"] = clinical

        # Numeric-only for neighbor search
        clin_num = clinical.apply(pd.to_numeric, errors="coerce")
        clin_num = clin_num.loc[:, clin_num.notna().any(axis=0)]  # drop fully non-numeric cols

        if clin_num.shape[1] > 0:
            clin_num = clin_num.fillna(0.0)

            clin_scaled = pd.DataFrame(
                scaler.fit_transform(clin_num),
                index=clin_num.index,
                columns=clin_num.columns,
            )
            scaled_blocks.append(clin_scaled * float(weights.get("clinical", 0.5)))

    # -------------------------
    # Combine scaled blocks
    # -------------------------
    if len(scaled_blocks) == 0:
        # Return empty but well-formed outputs
        combined_scaled = pd.DataFrame(index=pd.Index([]))
        return combined_scaled, unscaled_blocks

    combined_scaled = pd.concat(scaled_blocks, axis=1)
    combined_scaled.columns = combined_scaled.columns.astype(str)

    return combined_scaled, unscaled_blocks



def sample_from_neighbors_ratioCNA(
    scaled_df,
    unscaled_dfs,
    n_samples,
    k_neighbors=3,
    seed=44,
):
    """
    Samples synthetic alteration profiles by drawing from the local
    neighborhood of real samples (KNN-based multivariate structure preservation).
      - scaled_df contains CNA distance features (e.g. GENE_CNA__AMP_LVL / __DEL_LVL)
      - unscaled_dfs['cna'] contains the true CNA matrix (e.g. GENE_CNA in {-2..2})
    We must use scaled_df ONLY for neighbor search, and sample values ONLY from unscaled_dfs.
    """
    rng = np.random.default_rng(seed)

    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(scaled_df)), metric="euclidean")
    nn.fit(scaled_df.values)

    # blocks (may be missing)
    mut_df = unscaled_dfs.get("mut", None)
    fus_df = unscaled_dfs.get("fusion", None)
    cna_df = unscaled_dfs.get("cna", None)
    clin_df = unscaled_dfs.get("clinical", None)
    
    # Precomoute gene specfic CNA probabilties
    cna_params= estimate_cna_event_params(cna_df)
    
    samples = []

    for _ in range(n_samples):
        
        # Choose a real anchor sample
        # Determines biological context we simulate from
        anchor_pos = int(rng.integers(0, len(scaled_df))) # chosen sample (index)
        anchor_vec = scaled_df.iloc[[anchor_pos]].values # Values within that sample

        # Find the samples local neighbourhood in the altertaion space
        neighbors_idx = nn.kneighbors(anchor_vec, return_distance=False)[0]
        neighborhood_idx = scaled_df.iloc[neighbors_idx].index

        synthetic = {}

        # -------------------------
        # MUTATIONS: iterate over mut columns (Binary) (not scaled_df.columns)
        # -------------------------
        # Sample using neighbour frequency - preserves co-mutation structure 
        if mut_df is not None and not mut_df.empty:
            for col in mut_df.columns:
                values = mut_df.loc[neighborhood_idx, col]
                
                # Relative Frequency of mutation in neighbourhood 
                prob = float(np.clip(values.mean(), 0.02, 0.98))
                
                #sample binary mutation from binomial 
                synthetic[col] = rng.binomial(1, prob)

        # -------------------------
        # FUSIONS: Binary, Same process as above 
        # -------------------------
        if fus_df is not None and not fus_df.empty:
            for col in fus_df.columns:
                values = fus_df.loc[neighborhood_idx, col]
                prob = float(np.clip(values.mean() if len(values) > 0 else 0.0, 0.0, 1.0))
                synthetic[col] = rng.binomial(1, prob)

        # -------------------------
        # CNA (GISTIC discrete states)
        # 2 stage 
        #   1: Event Type (AMP/DEL/NEU) from local neighbourhood
        #   2: Severity level of AMP or DEL from gene specfic distribution
        # Helps keep local correlation structure and correct marginal frequencies
        # -------------------------
        if cna_df is not None and not cna_df.empty:
            for col in cna_df.columns:
                vals = cna_df.loc[neighborhood_idx, col].dropna().astype(int)
                
                
                # if neighbours contain no information - neutral
                if len(vals) == 0:
                    synthetic[col] = 0
                    continue

                # Stage 1: event type from neighbors (preserves co-occurrence)
                p_amp_nb = float((vals > 0).mean())
                p_del_nb = float((vals < 0).mean())
                p_neu_nb = float((vals == 0).mean()) 

                probs = np.array([p_neu_nb, p_amp_nb, p_del_nb], dtype=float) #categorical distribution
                probs = probs / probs.sum() if probs.sum() > 0 else np.array([1.0, 0.0, 0.0]) 

                event = rng.choice([0, 1, 2], p=probs) # Sample from categorical distribution, 0=NEU,1=AMP,2=DEL

                # Stage 2: severity from real gene-wise rates (anchors marginals)
                if cna_params is None or col not in cna_params.index:
                    # fallback: just sample directly from neighbor states
                    states, counts = np.unique(vals.values, return_counts=True)
                    synthetic[col] = int(rng.choice(states, p=counts / counts.sum()))
                    continue
                 
                # Using CNA probs calculated earlier to get gene-wise probability of 'severe' AMP/DEL
                if event == 1:
                    q2 = float(cna_params.at[col, "q_amp2"]) # Check severity probabilities for that gene
                    synthetic[col] = 2 if rng.random() < q2 else 1
                elif event == 2:
                    q2 = float(cna_params.at[col, "q_del2"])
                    synthetic[col] = -2 if rng.random() < q2 else -1
                else:
                    synthetic[col] = 0


        # -------------------------
        # CLINICAL METADATA
        # Sample categorical variables from neighbours
        # -------------------------
        if clin_df is not None and not clin_df.empty:
            for col in clin_df.columns:
                vals = clin_df.loc[neighborhood_idx, col].dropna().values
                if vals.size == 0:
                    continue
                synthetic[col] = rng.choice(vals)

        samples.append(synthetic)

        
    #--------------------------
    # ASSEMBLE DATA
    #--------------------------
    
    out = pd.DataFrame(samples)

    # Optional: ensure consistent column order (mut + fusion + cna + clinical)
    cols = []
    if mut_df is not None and not mut_df.empty:
        cols += list(mut_df.columns)
    if fus_df is not None and not fus_df.empty:
        cols += list(fus_df.columns)
    if cna_df is not None and not cna_df.empty:
        cols += list(cna_df.columns)
    if clin_df is not None and not clin_df.empty:
        cols += list(clin_df.columns)

    # Keep any missing cols (if a block was empty) but align if possible
    cols = [c for c in cols if c in out.columns]
    out = out.reindex(columns=cols)

    return out



def simulate_X_hybrid_ratioCNA(
    mut, fusion, cna, clinical, subtype, n_samples,
    weights={'mut': 1.0, 'fusion': 1.5, 'cna': 2.0, 'clinical': 0.5}, k_neighbors=3, seed=44
):
    """
    Subtype-aware hybrid simulator for mutations, fusions, CNAs (GISTIC), and clinical data.
    Each subtype is simulated independently to preserve within-subtype structure.
    """
    rng = np.random.default_rng(seed)
    weights = weights

    # Determine how many samples to simulate per subtype (proportional to real subtype distribution)
    subtype_counts = subtype['Subtype'].value_counts()
    proportions = subtype_counts / subtype_counts.sum()
    sizes = (proportions * n_samples).round().astype(int)
    sizes.iloc[-1] += n_samples - sizes.sum()  # fix rounding mismatch

    blocks = []
    for block_i, (s, n) in enumerate(zip(sizes.index, sizes)):
        if n <= 0:
            continue

        idx = subtype[subtype['Subtype'] == s].index

        # defensive intersection
        if mut is not None:
            idx = idx.intersection(mut.index)
        if fusion is not None:
            idx = idx.intersection(fusion.index)
        if cna is not None:
            idx = idx.intersection(cna.index)
        if clinical is not None:
            idx = idx.intersection(clinical.index)

        if len(idx) < 2:
            continue


        scaled, unscaled = preprocess_X_weighted(
            mut=mut.loc[idx] if mut is not None else None,
            fusion=fusion.loc[idx] if fusion is not None else None,
            cna=cna.loc[idx] if cna is not None else None,
            clinical=clinical.loc[idx] if clinical is not None else None,
            weights=weights,
        )

        # vary the seed per subtype block to avoid identical RNG streams
        block_seed = int(rng.integers(0, 2**31 - 1))

        sim_df = sample_from_neighbors_ratioCNA(
            scaled_df=scaled,
            unscaled_dfs=unscaled,
            n_samples=int(n),
            k_neighbors=k_neighbors,
            seed=block_seed,
        )
        
        sim_df['Subtype'] = s
        blocks.append(sim_df)

    X_sim = pd.concat(blocks, axis=0)
    X_sim.index = [f"Sample_{i+1}" for i in range(len(X_sim))]
    return X_sim


def split_simulated_blocks_v2(X_sim):
    """
    Splits a combined simulated matrix into separate dataframes for each omic block.

    Assumes:
      - Mutations end with _MUT/_GOF/_LOF
      - Fusions contain _FUSION or _FUS
      - CNA GISTIC calls end with _CNA
      - Optional derived CNA views might end with _AMP/_DEL/_AMP_LVL/_DEL_LVL
      - Subtype column might be "Subtype" or "SUBTYPE"
    """
    X_sim = X_sim.copy()
    X_sim.columns = X_sim.columns.astype(str)

    mut_cols = [c for c in X_sim.columns if c.endswith(("_MUT", "_GOF", "_LOF"))]
    fusion_cols = [c for c in X_sim.columns if ("_FUSION" in c or "_FUS" in c)]

    # Keep CNA as GISTIC calls by default, but allow derived views if present
    cna_cols = [
        c for c in X_sim.columns
        if c.endswith(("_CNA", "_AMP", "_DEL", "_AMP_LVL", "_DEL_LVL"))
    ]

    # Subtype col detection (handles your current simulator output)
    subtype_col = None
    for cand in ("Subtype", "SUBTYPE", "subtype"):
        if cand in X_sim.columns:
            subtype_col = cand
            break

    exclude = set(mut_cols + fusion_cols + cna_cols + ([subtype_col] if subtype_col else []))
    clinical_cols = [c for c in X_sim.columns if c not in exclude]

    mut_sim = X_sim[mut_cols] if mut_cols else pd.DataFrame(index=X_sim.index)
    fusion_sim = X_sim[fusion_cols] if fusion_cols else pd.DataFrame(index=X_sim.index)
    cna_sim = X_sim[cna_cols] if cna_cols else pd.DataFrame(index=X_sim.index)
    clin_sim = X_sim[clinical_cols] if clinical_cols else pd.DataFrame(index=X_sim.index, dtype="object")

    if subtype_col:
        subtype_sim = X_sim[subtype_col]
    else:
        subtype_sim = pd.Series(index=X_sim.index, dtype="object")

    print(f" Mutations: {mut_sim.shape[1]} features")
    print(f" Fusions:   {fusion_sim.shape[1]} features")
    print(f" CNAs:      {cna_sim.shape[1]} features")
    print(f" Clinical:  {clin_sim.shape[1]} features")
    if subtype_col:
        print(f" Subtypes:  {subtype_sim.nunique()} unique")

    return mut_sim, fusion_sim, cna_sim, clin_sim, subtype_sim
