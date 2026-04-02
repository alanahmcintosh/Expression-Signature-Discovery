# supervised/feature_selection.py

from __future__ import annotations

import numpy as np


def select_features_by_stability(feature_names, stability_scores, threshold=0.6, tol=1e-12):
    """
    Select features whose stability frequency is >= threshold.

    Parameters
    ----------
    feature_names : sequence of str
        Feature names aligned to `stability_scores`.
    stability_scores : array-like
        Selection frequencies in [0, 1], aligned to `feature_names`.
    threshold : float
        Minimum stability frequency required to keep a feature.
    tol : float
        Small numerical tolerance.

    Returns
    -------
    list[str]
        Selected feature names, sorted by descending stability score.
    """
    scores = np.asarray(stability_scores, dtype=float).ravel()
    feature_names = np.asarray(feature_names)

    if feature_names.shape[0] != scores.shape[0]:
        raise ValueError(
            f"feature_names and stability_scores must have the same length "
            f"({feature_names.shape[0]} vs {scores.shape[0]})."
        )

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    keep = scores >= (threshold - tol)
    if keep.sum() == 0:
        return []

    kept_names = feature_names[keep]
    kept_scores = scores[keep]

    order = np.argsort(kept_scores)[::-1]
    return kept_names[order].tolist()



def score_threshold_mask(scores, rule="mean", z=1.0, tol=1e-12):
    """
    Convert a dense score vector into a boolean selection mask using
    a threshold rule.
    
    Parameters
    ----------
    scores : array-like
        Non-negative scores for one gene across predictors.
    rule : str
        Thresholding rule:
        - 'mean'      : keep scores > mean(scores)
        - 'median'    : keep scores > median(scores)
        - 'mean+sd'   : keep scores > mean(scores) + z * sd(scores)
    z : float
        Multiplier used only for rule='mean+sd'.
    tol : float
        Numerical tolerance.

    Returns
    -------
    np.ndarray
        Boolean mask of selected predictors.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if scores.size == 0:
        return np.zeros(0, dtype=bool)

    if rule == "mean":
        thr = float(np.mean(scores))
    elif rule == "median":
        thr = float(np.median(scores))
    elif rule == "mean+sd":
        thr = float(np.mean(scores) + z * np.std(scores))
    else:
        raise ValueError("rule must be one of: 'mean', 'median', 'mean+sd'")

    return scores > (thr + tol)



def signature_from_weights_for_alt(
    W,
    alt,
    mode="nonzero",   # "nonzero" or "stability"
    coef_tol=1e-6,
    stability_threshold=0.6,
):
    """
    Extract selected genes for one alteration from a predictors x genes matrix.

    Parameters
    ----------
    W : pd.DataFrame
        predictors x genes matrix.
        - For mode='nonzero': coefficient / importance matrix
        - For mode='stability': stability-frequency matrix in [0, 1]
    alt : str
        Predictor / alteration name.
    mode : str
        Selection mode: 'nonzero' or 'stability'
    coef_tol : float
        Absolute coefficient threshold for 'nonzero' mode.
    stability_threshold : float
        Inclusion threshold for 'stability' mode.

    Returns
    -------
    list[str]
        Selected genes.
    """
    if alt not in W.index:
        return []

    row = W.loc[alt].copy()
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if mode == "nonzero":
        vals = np.asarray(row.values, dtype=float)
        return row.index[np.abs(vals) > coef_tol].tolist()

    if mode == "stability":
        return select_features_by_stability(
            feature_names=row.index,
            stability_scores=row.values,
            threshold=stability_threshold,
        )

    raise ValueError("mode must be 'nonzero' or 'stability'")