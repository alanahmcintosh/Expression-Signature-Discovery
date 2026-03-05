"""
Build per-alteration signature parameter dictionaries from DE summary tables.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_lognormal_from_mean_median(mean_abs, median_abs):
    """
    Fit lognormal parameters (mu, sigma) from mean and median on the ORIGINAL scale.

    For lognormal X ~ LogNormal(mu, sigma):
      median = exp(mu)
      mean   = exp(mu + sigma^2/2)

    => mu = log(median)
       sigma^2 = 2*(log(mean) - log(median))

    Returns (mu, sigma) or (None, None) if invalid.
    """
    try:
        m = float(mean_abs)
        med = float(median_abs)
        if not (np.isfinite(m) and np.isfinite(med) and m > 0 and med > 0):
            return None, None

        mu = float(np.log(med))
        sigma2 = 2.0 * (np.log(m) - np.log(med))
        sigma2 = max(float(sigma2), 1e-6)
        sigma = float(np.sqrt(sigma2))
        return mu, sigma
    except Exception:
        return None, None


def cap_abs_log2fc(mean_abs, abs_max, cap_k=8.0, global_cap=12.0):
    """
    Choose a realistic cap for absolute log2FC magnitudes.
    """
    cap = None
    try:
        m = float(mean_abs)
        mx = float(abs_max)
        if np.isfinite(m) and m > 0 and np.isfinite(mx) and mx > 0:
            cap = max(mx, cap_k * m)
    except Exception:
        cap = None

    if cap is None or not np.isfinite(cap) or cap <= 0:
        cap = float(global_cap)

    return float(min(cap, float(global_cap)))


def build_alt_params_from_deseq2_summary(
    deseq2_summary,
    alteration_col="alteration",
    n_sig_col="n_sig",
    mean_abs_col="mean_abs_log2FC_sig",
    median_abs_col="median_abs_log2FC_sig",
    max_abs_col="abs_max_log2FC_sig",
    cap_k=8.0,
    global_cap=12.0,
):
    """
    Given a dataframe containing summary statistics from differential expression,
    build parameters used to define simulated expression signatures.

    Each alteration gets:
      - size_mean : expected signature size
      - abs_mu, abs_sigma : lognormal model for absolute |log2FC|
      - abs_cap : upper bound on |log2FC|

    Returns
    -------
    dict[str, dict]
        alt -> {"size_mean", "abs_mu", "abs_sigma", "abs_cap"}
    """
    df = deseq2_summary.copy()
    if alteration_col in df.columns:
        df = df.set_index(alteration_col)

    params = {}

    for alt, row in df.iterrows():
        alt = str(alt)

        n_sig = row.get(n_sig_col, np.nan)
        mean_abs = row.get(mean_abs_col, np.nan)
        median_abs = row.get(median_abs_col, np.nan)
        abs_max = row.get(max_abs_col, np.nan)

        mu, sigma = fit_lognormal_from_mean_median(mean_abs, median_abs)
        cap = cap_abs_log2fc(mean_abs, abs_max, cap_k=cap_k, global_cap=global_cap)

        size_mean = None
        try:
            if np.isfinite(n_sig) and float(n_sig) > 0:
                size_mean = int(round(float(n_sig)))
        except Exception:
            size_mean = None

        params[alt] = {
            "size_mean": size_mean,
            "abs_mu": mu,
            "abs_sigma": sigma,
            "abs_cap": cap,
        }

    return params
