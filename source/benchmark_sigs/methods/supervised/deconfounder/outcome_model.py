from __future__ import annotations

import os

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import RepeatedKFold


def deconfounder(
    X,
    Y,
    alpha_range=np.logspace(-3, 0, 25),
    n_splits=3,
    n_repeats=1,
    random_state=44,
    n_jobs=-1,
    max_iter=5000,
    tol=1e-3,
):
    """
    Run LassoCV + Lasso per gene in Y.
    """
    if n_jobs == -1:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))

    # Pre-allocate ALL gene columns using real gene names
    coefs = pd.DataFrame(index=X.columns, columns=Y.columns, dtype=float)

    models = []
    r2_scores = []

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    for gene in Y.columns:
        y = Y[gene]

        lassocv = LassoCV(
            alphas=alpha_range,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=max_iter,
            tol=tol,
        )
        lassocv.fit(X, y)

        model = Lasso(alpha=lassocv.alpha_, max_iter=max_iter, tol=tol)
        model.fit(X, y)

        models.append(model)

        nz_idx = np.flatnonzero(model.coef_)
        if nz_idx.size:
            nz_features = X.columns[nz_idx]
            coefs.loc[nz_features, gene] = model.coef_[nz_idx]

        r2_scores.append(model.score(X, y))

    coefs = coefs.copy()
    return coefs, models, r2_scores
