"""
KNN-based alteration sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from benchmark_sigs.simulate.alterations.cna_params import estimate_cna_event_params


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

    Parameters
    ----------
    scaled_df : pd.DataFrame
        Scaled/weighted matrix used ONLY for neighbor search (samples x features).
        Typically includes CNA distance features (e.g. GENE__AMP_LVL / __DEL_LVL views)
        and optionally other weighted blocks.
    unscaled_dfs : dict
        Dictionary of original (unscaled) blocks. Expected keys:
          - "mut" (binary)
          - "fusion" (binary)
          - "cna" (GISTIC-like ints in {-2,-1,0,1,2})
          - "clinical" (mixed)
    n_samples : int
        Number of synthetic samples to generate.
    k_neighbors : int
        Neighborhood size for KNN.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Synthetic alteration matrix (n_samples x features).
    """
    if scaled_df is None or not isinstance(scaled_df, pd.DataFrame) or scaled_df.empty:
        raise ValueError("[sample_from_neighbors_ratioCNA] scaled_df is empty or invalid.")
    if n_samples <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)

    nn = NearestNeighbors(
        n_neighbors=min(int(k_neighbors), len(scaled_df)),
        metric="euclidean",
    )
    nn.fit(scaled_df.values)

    # blocks (may be missing)
    mut_df = unscaled_dfs.get("mut", None)
    fus_df = unscaled_dfs.get("fusion", None)
    cna_df = unscaled_dfs.get("cna", None)
    clin_df = unscaled_dfs.get("clinical", None)

    # Precompute gene-specific CNA probabilities (for severity sampling)
    cna_params = estimate_cna_event_params(cna_df) if cna_df is not None else None

    samples = []

    for _ in range(int(n_samples)):
        # Choose a real anchor sample (biological context)
        anchor_pos = int(rng.integers(0, len(scaled_df)))
        anchor_vec = scaled_df.iloc[[anchor_pos]].values

        # Find local neighborhood in alteration space
        neighbors_idx = nn.kneighbors(anchor_vec, return_distance=False)[0]
        neighborhood_idx = scaled_df.iloc[neighbors_idx].index

        synthetic = {}

        # -------------------------
        # MUTATIONS: binary, sample from neighborhood frequency
        # -------------------------
        if mut_df is not None and not mut_df.empty:
            for col in mut_df.columns:
                values = mut_df.loc[neighborhood_idx, col]
                prob = float(np.clip(values.mean(), 0.02, 0.98))
                synthetic[col] = rng.binomial(1, prob)

        # -------------------------
        # FUSIONS: binary, sample from neighborhood frequency
        # -------------------------
        if fus_df is not None and not fus_df.empty:
            for col in fus_df.columns:
                values = fus_df.loc[neighborhood_idx, col]
                prob = float(np.clip(values.mean() if len(values) > 0 else 0.0, 0.0, 1.0))
                synthetic[col] = rng.binomial(1, prob)

        # -------------------------
        # CNA (GISTIC discrete states)
        # Two-stage:
        #   1) event type (NEU/AMP/DEL) from neighborhood (preserves correlation)
        #   2) severity level from gene-specific rates (anchors marginals)
        # -------------------------
        if cna_df is not None and not cna_df.empty:
            for col in cna_df.columns:
                vals = cna_df.loc[neighborhood_idx, col].dropna().astype(int)

                # If neighbors contain no information, default to neutral
                if len(vals) == 0:
                    synthetic[col] = 0
                    continue

                # Stage 1: event type from neighbors
                p_amp_nb = float((vals > 0).mean())
                p_del_nb = float((vals < 0).mean())
                p_neu_nb = float((vals == 0).mean())

                probs = np.array([p_neu_nb, p_amp_nb, p_del_nb], dtype=float)
                probs = probs / probs.sum() if probs.sum() > 0 else np.array([1.0, 0.0, 0.0])

                # 0=NEU, 1=AMP, 2=DEL
                event = rng.choice([0, 1, 2], p=probs)

                # Stage 2: severity from gene-wise rates (fallback to neighbor sampling)
                if cna_params is None or col not in cna_params.index:
                    states, counts = np.unique(vals.values, return_counts=True)
                    synthetic[col] = int(rng.choice(states, p=counts / counts.sum()))
                    continue

                if event == 1:
                    q2 = float(cna_params.at[col, "q_amp2"])
                    synthetic[col] = 2 if rng.random() < q2 else 1
                elif event == 2:
                    q2 = float(cna_params.at[col, "q_del2"])
                    synthetic[col] = -2 if rng.random() < q2 else -1
                else:
                    synthetic[col] = 0

        # -------------------------
        # CLINICAL METADATA: sample from neighbors
        # -------------------------
        if clin_df is not None and not clin_df.empty:
            for col in clin_df.columns:
                vals = clin_df.loc[neighborhood_idx, col].dropna().values
                if vals.size == 0:
                    continue
                synthetic[col] = rng.choice(vals)

        samples.append(synthetic)

    # --------------------------
    # Assemble output
    # --------------------------
    out = pd.DataFrame(samples)

    # Ensure consistent column order (mut + fusion + cna + clinical)
    cols = []
    if mut_df is not None and not mut_df.empty:
        cols += list(mut_df.columns)
    if fus_df is not None and not fus_df.empty:
        cols += list(fus_df.columns)
    if cna_df is not None and not cna_df.empty:
        cols += list(cna_df.columns)
    if clin_df is not None and not clin_df.empty:
        cols += list(clin_df.columns)

    cols = [c for c in cols if c in out.columns]
    out = out.reindex(columns=cols)

    return out
