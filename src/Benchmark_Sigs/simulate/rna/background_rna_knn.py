"""
Background RNA simulation via KNN in alteration space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def simulate_background_from_alterations_knn(
    real_rna,
    real_alts,
    sim_alts,
    k=5,
    mix_conc=1.0,
    residual_scale=0.0,
    seed=44,
    metric="euclidean",
):
    """
    Simulate background RNA expression means for synthetic samples using KNN in alteration space.

    Parameters
    ----------
    real_rna : pd.DataFrame
        Real RNA matrix (samples x genes). Values should be non-negative.
    real_alts : pd.DataFrame
        Real alteration matrix (samples x features).
    sim_alts : pd.DataFrame
        Simulated alteration matrix (samples x features).
    k : int
        Number of neighbors in alteration space (min 2 enforced).
    mix_conc : float
        Dirichlet concentration parameter for mixing weights.
        higher -> more uniform mixtures.
    residual_scale : float
        Optional Gaussian noise added in z-space to prevent overfitting.
    seed : int
        RNG seed.
    metric : str
        Distance metric for NearestNeighbors.

    Returns
    -------
    pd.DataFrame
        Simulated background expression matrix (samples x genes), indexed by sim_alts.index,
        columns matching real_rna.columns.
    """
    if real_rna is None or real_rna.empty:
        raise ValueError("[simulate_background_from_alterations_knn] real_rna is empty.")
    if real_alts is None or real_alts.empty:
        raise ValueError("[simulate_background_from_alterations_knn] real_alts is empty.")
    if sim_alts is None or sim_alts.empty:
        raise ValueError("[simulate_background_from_alterations_knn] sim_alts is empty.")

    rng = np.random.default_rng(seed)

    # --- 1. Align alteration features ---
    common = real_alts.columns.intersection(sim_alts.columns)
    if len(common) == 0:
        raise ValueError(
            "[simulate_background_from_alterations_knn] "
            "No overlapping alteration features between real_alts and sim_alts."
        )

    # Use the same columns, same order
    Ar = real_alts.loc[:, common].copy()
    As = sim_alts.loc[:, common].copy()
    As = As[Ar.columns]  # enforce identical order

    # --- 2. Scale in alteration space ---
    scaler = StandardScaler(with_mean=True, with_std=True)
    Ar_scaled = scaler.fit_transform(Ar.to_numpy())
    As_scaled = scaler.transform(As.to_numpy())

    # --- 3. KNN in alteration space (fit on REAL alterations) ---
    k_ = max(2, min(int(k), Ar_scaled.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_, metric=metric)
    nn.fit(Ar_scaled)

    # --- 4. Prepare RNA (log1p + z-score) ---
    # Align real RNA rows to real alterations index
    R_counts = real_rna.loc[Ar.index].clip(lower=0)
    L = np.log1p(R_counts)

    mu = L.mean(axis=0)
    sd = L.std(axis=0, ddof=0).replace(0, 1.0)
    Z = (L - mu) / sd

    # --- 5. Generate simulated expression for each synthetic sample ---
    mixes = []
    for i in range(As_scaled.shape[0]):
        # Find genetically similar real samples
        _, idxs = nn.kneighbors(As_scaled[i : i + 1])
        nbr_idx = Ar.index[idxs[0]]
        nbrZ = Z.loc[nbr_idx]

        # Random convex mixture of neighbours (Dirichlet weights sum to 1)
        w = rng.dirichlet(np.full(nbrZ.shape[0], float(mix_conc)))
        z_mix = np.average(nbrZ.values, axis=0, weights=w)

        # Optional noise to avoid overfitting
        if residual_scale and residual_scale > 0:
            z_mix += rng.normal(0, float(residual_scale), size=z_mix.shape[0])

        # Transform back to expression scale
        L_mix = z_mix * sd.values + mu.values
        mu_expr = np.expm1(L_mix)

        # No negative values
        mu_expr[mu_expr < 0] = 0
        mixes.append(mu_expr)

    sim_mu = pd.DataFrame(mixes, columns=real_rna.columns, index=sim_alts.index)
    return sim_mu
