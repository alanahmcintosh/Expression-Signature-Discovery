from __future__ import annotations

import pandas as pd

from .models import (
    fit_alt_to_expr_weights_lasso,
    fit_alt_to_expr_weights_elasticnet,
    fit_alt_to_expr_weights_ridge,
    fit_alt_to_expr_weights_svm,
    fit_alt_to_expr_importances_rf,
)
from .deseq2 import get_deseq2_signature_binary
from .signature_extraction import signature_from_weights_for_alt
from .deconfounder import get_deconfounder_signature

def class_supervised_signatures(W_dict, gof):
    """
    Extract signatures for one alteration from precomputed weights/importances.
    """
    signatures = {}
    signatures["Lasso"] = signature_from_weights_for_alt(W_dict["Lasso"], gof, mode="nonzero")
    signatures["ElasticNet"] = signature_from_weights_for_alt(W_dict["ElasticNet"], gof, mode="nonzero")
    signatures["Ridge"] = signature_from_weights_for_alt(W_dict["Ridge"], gof, mode="elbow")
    signatures["SVM"] = signature_from_weights_for_alt(W_dict["SVM"], gof, mode="elbow")
    signatures["Random Forest"] = signature_from_weights_for_alt(W_dict["Random Forest"], gof, mode="elbow")
    return signatures


def precompute_supervised_weights(X_alt_df, Y_norm_df):
    """
    Fit effect models ONCE: expr ~ alterations.
    Returns dict of method -> weights matrix (predictors x genes).
    """
    W = {}
    W["Lasso"] = fit_alt_to_expr_weights_lasso(X_alt_df, Y_norm_df)
    W["ElasticNet"] = fit_alt_to_expr_weights_elasticnet(X_alt_df, Y_norm_df)
    W["Ridge"] = fit_alt_to_expr_weights_ridge(X_alt_df, Y_norm_df)
    W["SVM"] = fit_alt_to_expr_weights_svm(X_alt_df, Y_norm_df)
    W["Random Forest"] = fit_alt_to_expr_importances_rf(X_alt_df, Y_norm_df)
    return W


def create_supervised_signatures(
    X,
    Y,
    gof,
    global_results=None,
    W_dict=None,
    min_unique_x=2,
    min_std_x=1e-8,
    min_group_n=1,
):
    # ---- align samples once ----
    common = X.index.intersection(Y.index)
    X = X.loc[common]
    Y = Y.loc[common]

    if gof not in X.columns:
        raise KeyError(f"{gof} not found in X columns.")

    x = pd.to_numeric(X[gof], errors="coerce")
    valid_idx = x.dropna().index
    x = x.loc[valid_idx].astype(int)

    Y_sub = Y.loc[valid_idx]

    nunq = x.nunique(dropna=True)
    std = float(x.std())
    if nunq < min_unique_x or not (std > min_std_x):
        return {"SKIPPED": f"{gof}: predictor constant/low-var (n_unique={nunq}, std={std})"}

    vc = x.value_counts()
    if (0 not in vc) or (1 not in vc) or (vc[0] < min_group_n) or (vc[1] < min_group_n):
        return {"SKIPPED": f"{gof}: insufficient group sizes {vc.to_dict()} (min_group_n={min_group_n})"}

    signatures = {}

    # ---- effect-style ML models: extract from precomputed weights ----
    if W_dict is not None:
        signatures.update(class_supervised_signatures(W_dict, gof))

    # ---- Deconfounder ----
    if global_results is not None:
        signatures["Deconfounder"] = get_deconfounder_signature(gof, global_results)

    # ---- DESeq2 ----
    try:
        if (Y_sub < 0).any().any():
            raise ValueError(f"{gof}: Y contains negative values; DESeq2 requires non-negative counts.")

        X_design = pd.DataFrame({gof: x}, index=x.index)
        signatures["DESeq2"] = get_deseq2_signature_binary(X_design, Y_sub, gof, min_group_n=min_group_n)

    except Exception as e:
        signatures["DESeq2_ERROR"] = repr(e)

    return signatures
