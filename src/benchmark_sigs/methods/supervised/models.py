

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
    RidgeCV,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from benchmark_sigs.utils import align_XY  

from benchmark_sigs.methods.supervised.feature_selection import score_threshold_mask


def fit_alt_to_expr_weights_lasso(
    X_alt,
    Y_expr,
    alpha_range=np.logspace(-3, 0, 25),
    n_splits=3,
    n_repeats=1,
    random_state=44,
    n_jobs=8,
    max_iter=5000,
    tol=1e-3,
):
    """
    Per-gene LassoCV on: expr_gene ~ X_alt
    Returns weights: predictors x genes (NaN where coef==0).
    """
    X, Y = align_XY(X_alt, Y_expr)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values

        lcv = LassoCV(
            alphas=alpha_range,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=max_iter,
            tol=tol,
        )
        lcv.fit(X, y)

        model = Lasso(
            alpha=lcv.alpha_,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        model.fit(X, y)

        coef = model.coef_
        nz = np.flatnonzero(coef)
        if nz.size:
            W.iloc[nz, W.columns.get_loc(gene)] = coef[nz]

    return W


def fit_alt_to_expr_weights_elasticnet(
    X_alt,
    Y_expr,
    alpha_range=np.logspace(-3, 0, 25),
    l1_ratios=(0.5,),
    n_splits=3,
    n_repeats=1,
    random_state=44,
    n_jobs=8,
    max_iter=5000,
    tol=1e-3,
):
    """
    Per-gene ElasticNetCV on: expr_gene ~ X_alt
    Returns weights: predictors x genes (NaN where coef==0).
    """
    X, Y = align_XY(X_alt, Y_expr)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values

        encv = ElasticNetCV(
            alphas=alpha_range,
            l1_ratio=list(l1_ratios),
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=max_iter,
            tol=tol,
        )
        encv.fit(X, y)

        model = ElasticNet(
            alpha=encv.alpha_,
            l1_ratio=encv.l1_ratio_,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        model.fit(X, y)

        coef = model.coef_
        nz = np.flatnonzero(coef)
        if nz.size:
            W.iloc[nz, W.columns.get_loc(gene)] = coef[nz]

    return W


def fit_alt_to_expr_weights_ridge(
    X_alt,
    Y_expr,
    alphas=np.logspace(-3, 3, 13),
    n_bootstraps=20,
    sample_fraction=0.5,
    random_state=44,
    threshold_rule="mean",
    threshold_z=1.0,
):
    """
    Stability selection for Ridge.

    For each resample:
      1. fit RidgeCV on all genes jointly
      2. for each gene, compute abs(coef)
      3. select predictors whose abs(coef) passes the threshold rule
      4. accumulate selection counts

    Returns
    -------
    pd.DataFrame
        predictors x genes matrix of selection frequencies in [0, 1].
    """
    X, Y = align_XY(X_alt, Y_expr)

    rng = np.random.default_rng(random_state)
    n, p = X.shape
    g = Y.shape[1]

    sub_n = max(2, int(np.floor(sample_fraction * n)))
    counts = np.zeros((p, g), dtype=float)

    Xv = X.values
    Yv = Y.values

    for b in range(n_bootstraps):
        idx = rng.choice(n, size=sub_n, replace=False)
        Xb = Xv[idx]
        Yb = Yv[idx]

        model = RidgeCV(alphas=alphas)
        model.fit(Xb, Yb)

        coef_abs = np.abs(model.coef_)   # shape: (n_genes, n_features)

        for j in range(g):
            mask = score_threshold_mask(
                coef_abs[j, :],
                rule=threshold_rule,
                z=threshold_z,
            )
            counts[mask, j] += 1.0

    freqs = counts / float(n_bootstraps)
    return pd.DataFrame(freqs, index=X.columns, columns=Y.columns, dtype=float)


def fit_alt_to_expr_weights_svm(
    X_alt,
    Y_expr,
    C=1.0,
    epsilon=0.1,
    max_iter=5000,
    tol=1e-3,
    random_state=44,
):
    """
    LinearSVR per gene: expr_gene ~ X_alt
    Returns weights: predictors x genes (NaN where coef==0).
    """
    X, Y = align_XY(X_alt, Y_expr)
    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values
        svr = LinearSVR(C=C, epsilon=epsilon, max_iter=max_iter, tol=tol, random_state=random_state)
        svr.fit(X.values, y)
        coef = svr.coef_
        nz = np.flatnonzero(coef)
        if nz.size:
            W.iloc[nz, W.columns.get_loc(gene)] = coef[nz]

    return W


def fit_alt_to_expr_weights_rf(
    X_alt,
    Y_expr,
    n_estimators=50,
    max_depth=8,
    random_state=44,
    n_jobs=1,
    n_bootstraps=20,
    sample_fraction=0.5,
    threshold_rule="mean",
    threshold_z=1.0,
):
    """
    Stability selection for Random Forest.

    For each resample and each gene:
      1. fit RF regressor
      2. get feature_importances_
      3. select predictors whose importance passes the threshold rule
      4. accumulate selection counts

    Returns
    -------
    pd.DataFrame
        predictors x genes matrix of selection frequencies in [0, 1].
    """
    X, Y = align_XY(X_alt, Y_expr)

    rng = np.random.default_rng(random_state)
    n, p = X.shape
    g = Y.shape[1]

    sub_n = max(2, int(np.floor(sample_fraction * n)))
    counts = np.zeros((p, g), dtype=float)

    for b in range(n_bootstraps):
        idx = rng.choice(n, size=sub_n, replace=False)
        Xb = X.iloc[idx]
        Yb = Y.iloc[idx]

        for j, gene in enumerate(Y.columns):
            yb = Yb[gene].values

            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state + b,
                n_jobs=n_jobs,
            )
            rf.fit(Xb.values, yb)

            scores = rf.feature_importances_

            mask = score_threshold_mask(
                scores,
                rule=threshold_rule,
                z=threshold_z,
            )
            counts[mask, j] += 1.0

    freqs = counts / float(n_bootstraps)
    return pd.DataFrame(freqs, index=X.columns, columns=Y.columns, dtype=float)