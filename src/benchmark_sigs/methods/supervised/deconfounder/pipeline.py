from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .ppca import ppca, predicitve_check, choose_latent_dim_ppca
from .outcome_model import deconfounder


def compute_deconfounder(X, Y):
    """
    Full Deconfounder pipeline:
    - fit PPCA on X to infer latent factors
    - augment X with latent factors
    - normalize + log + scale Y (keeping gene names!)
    - run LassoCV/Lasso per gene to get coefficients
    """
    k = choose_latent_dim_ppca(X)
    print(f"Selected latent dimension: {k}")
    print("Precomputing Deconfounder...")

    m_ppca = ppca(k)
    m_ppca.holdout(X, seed=44)
    m_ppca.max_likelihood(m_ppca.x_train, standardise=False)
    _ = m_ppca.generate(1)
    _ = predicitve_check(m_ppca, k)

    latent_df = pd.DataFrame(
        m_ppca.z_mu.T,
        index=X.index,
        columns=[f"latent_{i}" for i in range(k)],
    )
    augmented_X = pd.concat([X, latent_df], axis=1).astype(float)
    augmented_X.columns = augmented_X.columns.astype(str)

    # Normalize gene expression using library size normalization + log transform
    size_factors = 10000 / Y.sum(axis=1)
    Y_norm = np.log1p(Y.mul(size_factors, axis=0))

    # KEEP gene names; critical for alignment
    Y_scaled = pd.DataFrame(
        StandardScaler().fit_transform(Y_norm),
        index=Y.index,
        columns=Y.columns,
    )

    coefs, models, R2 = deconfounder(augmented_X, Y_scaled)  # option1; swap if needed

    # coefs: rows = features, cols = genes
    # transpose to: rows = genes, cols = features
    coefs_tr = coefs.T  # index == gene names (deconfounder() pre-allocates columns=Y.columns)

    causal_signatures = {"Deconfounder": coefs_tr}

    print("Global precomputation completed.")
    return causal_signatures
