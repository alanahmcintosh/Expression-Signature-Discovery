from __future__ import annotations

import pandas as pd

from .models import (
    fit_alt_to_expr_weights_lasso,
    fit_alt_to_expr_weights_elasticnet,
    fit_alt_to_expr_weights_ridge,
    fit_alt_to_expr_weights_svm,
    fit_alt_to_expr_weights_rf,
)
from .deseq2 import get_deseq2_signature_binary, precompute_deseq2_results
from .feature_selection import signature_from_weights_for_alt
from .deconfounder import get_deconfounder_signature


def class_supervised_signatures(W_dict, gof, stability_threshold=0.6, method='all'):
    """
    Extract signatures for one alteration from precomputed weights/importances.
    """
    signatures = {}
    run_all = method == "all"

    if run_all or method == "Lasso":
        signatures["Lasso"] = signature_from_weights_for_alt(
        W_dict["Lasso"], gof, mode="nonzero"
        )
    if run_all or method == "ElasticNet":
        signatures["ElasticNet"] = signature_from_weights_for_alt(
        W_dict["ElasticNet"], gof, mode="nonzero"
        )
    if run_all or method == "Ridge":
        signatures["Ridge"] = signature_from_weights_for_alt(
        W_dict["Ridge"], gof, mode="stability", stability_threshold=stability_threshold
        )
    if run_all or method == "SVM":
        signatures["SVM"] = signature_from_weights_for_alt(
        W_dict["SVM"], gof, mode="stability", stability_threshold=stability_threshold
        )
    if run_all or method == "RandomForest":
        signatures["RandomForest"] = signature_from_weights_for_alt(
            W_dict["RandomForest"], gof, mode="stability", stability_threshold=stability_threshold
        )
    return signatures



def precompute_supervised_weights(
    X_alt_df,
    Y_norm_df,
    ridge_n_bootstraps=20,
    ridge_sample_fraction=0.5,
    ridge_threshold_rule="mean",
    ridge_threshold_z=1.0,
    svm_n_bootstraps=20,
    svm_sample_fraction=0.5,
    svm_threshold_rule="mean+sd",
    svm_threshold_z=1.0,
    rf_n_bootstraps=20,
    rf_sample_fraction=0.5,
    rf_threshold_rule="mean",
    rf_threshold_z=1.0,
    method='all'
):
    """
    Fit effect models ONCE: expr ~ alterations.

    Returns
    -------
    dict
        Method -> predictors x genes matrix

        - Lasso / ElasticNet / SVM: coefficient matrices
        - Ridge / Random Forest: stability-frequency matrices
    """
    W = {}

    run_all = method == "all"

    if run_all or method == "Lasso":
        W["Lasso"] = fit_alt_to_expr_weights_lasso(X_alt_df, Y_norm_df)
    if run_all or method == "ElasticNet":
        W["ElasticNet"] = fit_alt_to_expr_weights_elasticnet(X_alt_df, Y_norm_df)
    if run_all or method == "Ridge":
        W["Ridge"] = fit_alt_to_expr_weights_ridge(
        X_alt_df,
        Y_norm_df,
        n_bootstraps=ridge_n_bootstraps,
        sample_fraction=ridge_sample_fraction,
        threshold_rule=ridge_threshold_rule,
        threshold_z=ridge_threshold_z,
    )
    if run_all or method == "SVM":
            W["SVM"] = fit_alt_to_expr_weights_svm(
            X_alt_df, Y_norm_df,
            n_bootstraps=svm_n_bootstraps,
            sample_fraction=svm_sample_fraction,
            threshold_rule=svm_threshold_rule,
            threshold_z=svm_threshold_z,
        )

    if run_all or method == "RandomForest":
        W["RandomForest"] = fit_alt_to_expr_weights_rf(
        X_alt_df,
        Y_norm_df,
        n_bootstraps=rf_n_bootstraps,
        sample_fraction=rf_sample_fraction,
        threshold_rule=rf_threshold_rule,
        threshold_z=rf_threshold_z,
    )
    return W


def create_supervised_signatures(
    X,
    Y,
    gof,
    global_results=None,
    W_dict=None,
    deseq2_sigs=None,
    method="all",
    min_unique_x=1,
    min_std_x=1e-8,
    min_group_n=2,
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

    run_all = method == "all"

    # ---- ML models from weights ----
        # ---- ML models from weights ----
    if W_dict is not None:

        if run_all or method == "Lasso":
            signatures["Lasso"] = signature_from_weights_for_alt(
                W_dict["Lasso"], gof, mode="nonzero"
            )
            print("Lasso Success")

        if run_all or method == "ElasticNet":
            signatures["ElasticNet"] = signature_from_weights_for_alt(
                W_dict["ElasticNet"], gof, mode="nonzero"
            )
            print("EN Success")

        if run_all or method == "Ridge":
            signatures["Ridge"] = signature_from_weights_for_alt(
                W_dict["Ridge"], gof, mode="stability", stability_threshold=0.6
            )
            print("Ridge Success")

        if run_all or method == "SVM":
            signatures["SVM"] = signature_from_weights_for_alt(
                W_dict["SVM"], gof, mode="stability", stability_threshold=0.6
            )
            print("SVM Success")

        if run_all or method == "RandomForest":
            signatures["RandomForest"] = signature_from_weights_for_alt(
                W_dict["RandomForest"], gof, mode="stability", stability_threshold=0.6
            )
            print("RF Success")

    # ---- Deconfounder ----
    if global_results is not None and (run_all or method == "Deconfounder"):
        signatures["Deconfounder"] = get_deconfounder_signature(gof, global_results)
        print('DECF Success')
    # ---- DESeq2 ----
    if run_all or method == "DESeq2":
        if deseq2_sigs is not None:
            signatures["DESeq2"] = deseq2_sigs.get(gof, [])
            print("DESeq2 Success")

    return signatures
