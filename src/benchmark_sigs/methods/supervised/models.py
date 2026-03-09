

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


def fit_alt_to_expr_weights_ridge(X_alt, Y_expr, alphas=np.logspace(-3, 3, 13)):
    """
    RidgeCV multi-output regression: Y_expr ~ X_alt (fits ALL genes at once, fast).
    Returns dense weights: predictors x genes.
    """
    X, Y = align_XY(X_alt, Y_expr)
    rcv = RidgeCV(alphas=alphas)
    rcv.fit(X.values, Y.values)
    return pd.DataFrame(rcv.coef_.T, index=X.columns, columns=Y.columns, dtype=float)


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


def fit_alt_to_expr_importances_rf(
    X_alt,
    Y_expr,
    n_estimators=200,
    max_depth=12,
    random_state=44,
    n_jobs=8,
):
    """
    RF regressor per gene: expr_gene ~ X_alt
    Returns importances: predictors x genes (dense).
    """
    X, Y = align_XY(X_alt, Y_expr)
    W = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    for gene in Y.columns:
        y = Y[gene].values
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        rf.fit(X.values, y)
        W[gene] = rf.feature_importances_

    return W
