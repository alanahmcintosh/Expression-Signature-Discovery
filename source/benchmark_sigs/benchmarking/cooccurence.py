'''
Calculate alteration co-occurence
'''
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import itertools


def co_occurence(X):
    """
    Calculate co-occurrence rates between pairs of alterations.

    Returns
    -------
    co_occ : pd.DataFrame
        Observed co-occurrence counts.
    exp : pd.DataFrame
        Expected co-occurrence counts under independence.
    co_occ_diff : pd.DataFrame
        Observed minus expected counts.
    fischers_exact_df : pd.DataFrame
        Pairwise Fisher exact summary table.
    """
    X = X.copy().astype(int)
    co_occ = X.T @ X

    freq = X.sum(axis=0)
    n = X.shape[0]

    exp = np.outer(freq, freq) / n
    exp = pd.DataFrame(exp, index=X.columns, columns=X.columns)
    co_occ_diff = co_occ - exp

    def safe_odds_ratio(a, b, c, d, alpha=0.5):
        return ((a + alpha) * (d + alpha)) / ((b + alpha) * (c + alpha))

    results = []
    for g1, g2 in itertools.combinations(X.columns, 2):
        a = int(((X[g1] == 1) & (X[g2] == 1)).sum())
        b = int(((X[g1] == 1) & (X[g2] == 0)).sum())
        c = int(((X[g1] == 0) & (X[g2] == 1)).sum())
        d = int(((X[g1] == 0) & (X[g2] == 0)).sum())

        odds_fisher, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        odds_smooth = safe_odds_ratio(a, b, c, d, alpha=0.5)

        results.append((str(g1), str(g2), a, (a / n) * 100.0, odds_fisher, odds_smooth, p))

        fischers_exact_df = pd.DataFrame(
            results,
            columns=["alt1", "alt2", "co_count", "co_percent", "odds_ratio", "odds_smooth", "p"],
        )

    return co_occ, exp, co_occ_diff, fischers_exact_df

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return np.nan
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else np.nan
