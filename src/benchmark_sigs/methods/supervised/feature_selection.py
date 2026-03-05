# supervised/feature_selection.py

from __future__ import annotations

import numpy as np


def select_features_elbow(feature_names, weights, tol=1e-8):
    """
    Elbow-based feature selector.

    - Takes absolute weights/importances.
    - Finds the 'knee' of the sorted curve (rank vs |weight|).
    - Returns all features up to the knee index.

    Parameters
    ----------
    feature_names : sequence of str
        Feature names aligned to `weights`.
    weights : array-like
        Weights/importances aligned to `feature_names`.
    tol : float
        Threshold below which weights are treated as zero.

    Returns
    -------
    list[str]
        Selected feature names (up to the knee), sorted by descending |weight|.
    """
    weights = np.asarray(weights).ravel()
    abs_w = np.abs(weights)

    feature_names = np.asarray(feature_names)
    if feature_names.shape[0] != abs_w.shape[0]:
        raise ValueError(
            f"feature_names and weights must have the same length "
            f"({feature_names.shape[0]} vs {abs_w.shape[0]})."
        )

    # Only consider clearly non-zero weights
    mask = abs_w > tol
    if mask.sum() == 0:
        return []

    abs_w_nz = abs_w[mask]
    names_nz = feature_names[mask]

    n = len(abs_w_nz)
    if n <= 1:
        return names_nz.tolist()

    # Sort descending by |weight|
    order = np.argsort(abs_w_nz)[::-1]
    w_sorted = abs_w_nz[order]
    names_sorted = names_nz[order]

    # Coordinates (rank, value)
    x = np.arange(n, dtype=float)
    y = w_sorted.astype(float)

    # Line from first to last point
    x0, y0 = 0.0, y[0]
    x1, y1 = float(n - 1), y[-1]
    dx = x1 - x0
    dy = y1 - y0
    denom = np.sqrt(dx**2 + dy**2)

    if denom == 0:
        # Flat line — all weights equal: keep all non-zero
        return names_sorted.tolist()

    # Per-point distance to the straight line
    # |dy*x_i - dx*y_i + x1*y0 - y1*x0| / sqrt(dx^2+dy^2)
    dist = np.abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom
    knee_idx = int(np.argmax(dist))

    # Keep all features up to and including the knee
    return names_sorted[: knee_idx + 1].tolist()


def signature_from_weights_for_alt(
    W,
    alt,
    mode = "nonzero",   # "nonzero" or "elbow"
    coef_tol = 1e-6,
    elbow_tol = 1e-8,
):
    """
    W: predictors x genes (weights or importances)
    alt: predictor name (e.g., TP53_LOF, MYC_AMP)
    mode:
      - "nonzero": return genes with |weight| > coef_tol (good for Lasso/ElasticNet)
      - "elbow": use select_features_elbow on |weights| (good for Ridge/RF/SVR)
    """
    if alt not in W.index:
        return []

    row = W.loc[alt].copy()
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if mode == "nonzero":
        return row.index[np.abs(row.values) > coef_tol].tolist()

    if mode == "elbow":
        return select_features_elbow(row.index, row.values, tol=elbow_tol)

    raise ValueError("mode must be 'nonzero' or 'elbow'")
