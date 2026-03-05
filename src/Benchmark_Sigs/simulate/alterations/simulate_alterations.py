"""
Alteration simulation (subtype-aware) using KNN neighborhood sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark_sigs.simulate.alterations.scaling import preprocess_X_weighted
from benchmark_sigs.simulate.alterations.knn_sampler import sample_from_neighbors_ratioCNA


def simulate_X_hybrid_ratioCNA(
    mut,
    fusion,
    cna,
    clinical,
    subtype,
    n_samples,
    weights={"mut": 1.0, "fusion": 1.5, "cna": 2.0, "clinical": 0.5},
    k_neighbors=3,
    seed=44,
):
    """
    Subtype-aware hybrid simulator for mutations, fusions, CNAs (GISTIC), and clinical data.
    Each subtype is simulated independently to preserve within-subtype structure.

    Parameters
    ----------
    mut, fusion, cna, clinical : pd.DataFrame or None
        Real data blocks (index = samples).
    subtype : pd.DataFrame
        Must include a column named 'Subtype' (index = samples).
    n_samples : int
        Total number of synthetic samples to simulate.
    weights : dict
        Modality weights passed to preprocess_X_weighted for neighbor search.
    k_neighbors : int
        Number of neighbors used by KNN sampler.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Combined simulated alteration matrix with an added 'Subtype' column.
    """
    rng = np.random.default_rng(seed)

    if subtype is None or subtype.empty or "Subtype" not in subtype.columns:
        raise ValueError("[simulate_X_hybrid_ratioCNA] subtype must be a DataFrame with a 'Subtype' column.")
    if n_samples <= 0:
        return pd.DataFrame()

    # Determine how many samples to simulate per subtype (proportional to real distribution)
    subtype_counts = subtype["Subtype"].value_counts()
    proportions = subtype_counts / subtype_counts.sum()
    sizes = (proportions * n_samples).round().astype(int)
    if len(sizes) > 0:
        sizes.iloc[-1] += n_samples - sizes.sum()  # fix rounding mismatch

    blocks = []

    for s, n in zip(sizes.index, sizes):
        n = int(n)
        if n <= 0:
            continue

        idx = subtype[subtype["Subtype"] == s].index

        # Defensive intersection across provided modalities
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

        # Vary the seed per subtype block to avoid identical RNG streams
        block_seed = int(rng.integers(0, 2**31 - 1))

        sim_df = sample_from_neighbors_ratioCNA(
            scaled_df=scaled,
            unscaled_dfs=unscaled,
            n_samples=n,
            k_neighbors=k_neighbors,
            seed=block_seed,
        )

        sim_df["Subtype"] = s
        blocks.append(sim_df)

    if len(blocks) == 0:
        return pd.DataFrame()

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
    if X_sim is None or not isinstance(X_sim, pd.DataFrame) or X_sim.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(dtype="object"),
        )

    X_sim = X_sim.copy()
    X_sim.columns = X_sim.columns.astype(str)

    mut_cols = [c for c in X_sim.columns if c.endswith(("_MUT", "_GOF", "_LOF"))]
    fusion_cols = [c for c in X_sim.columns if ("_FUSION" in c or "_FUS" in c)]

    cna_cols = [
        c
        for c in X_sim.columns
        if c.endswith(("_CNA", "_AMP", "_DEL", "_AMP_LVL", "_DEL_LVL"))
    ]

    # Subtype col detection
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
    clin_sim = (
        X_sim[clinical_cols] if clinical_cols else pd.DataFrame(index=X_sim.index, dtype="object")
    )

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
