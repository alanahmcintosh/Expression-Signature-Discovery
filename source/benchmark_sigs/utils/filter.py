from __future__ import annotations

import pandas as pd
import numpy as np


def sanitize_binary_design(
    X,
    min_group_n=5,
    priority_keywords=("GOF", "LOF", "FUSION"),
    min_residual_df=5,
    return_drop_info=False,
    verbose=False,
):
    """
    Clean a binary design matrix and prune it to a DESeq2-safe subset.
    """
    X = X.copy()
    n0, p0 = X.shape

    drop_info = {
        "constant": [],
        "imbalanced": [],
        "duplicate": [],
        "dependent": [],
        "capacity": [],
        "kept": [],
    }

    # 0) force numeric binary
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = (X > 0).astype(int)

    # 1) remove constant columns
    nunique = X.nunique(axis=0)
    constant_cols = nunique[nunique <= 1].index.tolist()
    drop_info["constant"] = constant_cols
    X = X.loc[:, nunique > 1]

    # 2) require enough 1s and enough 0s
    ones = X.sum(axis=0)
    zeros = X.shape[0] - ones
    keep_balanced = (ones >= min_group_n) & (zeros >= min_group_n)
    drop_info["imbalanced"] = X.columns[~keep_balanced].tolist()
    X = X.loc[:, keep_balanced]

    # 3) remove exact duplicate columns
    before_cols = X.columns.tolist()
    X = X.T.drop_duplicates().T
    drop_info["duplicate"] = [c for c in before_cols if c not in X.columns]

    # 4) priority ordering: GOF/LOF/FUSION first, then prevalence
    prevalence = X.sum(axis=0)

    priority_cols = [
        c for c in X.columns
        if any(k in str(c).upper() for k in priority_keywords)
    ]
    nonpriority_cols = [c for c in X.columns if c not in priority_cols]

    priority_cols = sorted(priority_cols, key=lambda c: prevalence[c], reverse=True)
    nonpriority_cols = sorted(nonpriority_cols, key=lambda c: prevalence[c], reverse=True)

    ordered_cols = priority_cols + nonpriority_cols
    X = X.loc[:, ordered_cols]

    # 5) compute max allowable columns for DESeq2
    max_cols_allowed = max(0, X.shape[0] - 1 - min_residual_df)

    # 6) iteratively keep columns if they preserve full rank and capacity
    keep_cols = []

    for col in X.columns:
        if len(keep_cols) >= max_cols_allowed:
            drop_info["capacity"].append(col)
            continue

        trial_cols = keep_cols + [col]
        trial = X.loc[:, trial_cols]
        dm = np.column_stack([np.ones(trial.shape[0]), trial.values])
        rank = np.linalg.matrix_rank(dm)

        if rank == dm.shape[1]:
            keep_cols.append(col)
        else:
            drop_info["dependent"].append(col)

    X_clean = X.loc[:, keep_cols].copy()
    drop_info["kept"] = X_clean.columns.tolist()

    if verbose:
        print(f"Design shape: {n0}x{p0} -> {X_clean.shape}")
        print(f"Removed {len(drop_info['constant'])} constant columns")
        print(f"Removed {len(drop_info['imbalanced'])} sparse/imbalanced columns")
        print(f"Removed {len(drop_info['duplicate'])} duplicate columns")
        print(f"Removed {len(drop_info['dependent'])} linearly dependent columns")
        print(f"Removed {len(drop_info['capacity'])} columns due to DESeq2 df limit")
        print(f"Max columns allowed: {max_cols_allowed}")

        dm_final = np.column_stack([np.ones(X_clean.shape[0]), X_clean.values])
        print(f"Final design matrix shape: {dm_final.shape}")
        print(f"Final rank: {np.linalg.matrix_rank(dm_final)} / {dm_final.shape[1]}")
        print(f"Residual df: {X_clean.shape[0] - dm_final.shape[1]}")

    if return_drop_info:
        return X_clean, drop_info

    return X_clean