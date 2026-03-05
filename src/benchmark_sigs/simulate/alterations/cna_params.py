"""
CNA parameter estimation for simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_cna_event_params(cna_real):
    """
    Estimate per-gene CNA event probabilities from GISTIC-like integer CNAs in {-2,-1,0,1,2}.

    Robust to duplicated columns: if duplicates exist for a gene, we collapse
    them by mean and round back to int in [-2,2].

    Parameters
    ----------
    cna_real : pd.DataFrame
        Samples x genes CNA matrix with values typically in {-2,-1,0,1,2}.

    Returns
    -------
    pd.DataFrame
        Indexed by gene with columns:
            p_amp  : probability CNA > 0
            p_del  : probability CNA < 0
            p_neu  : probability CNA == 0
            q_amp2 : conditional probability of +2 given amplification
            q_del2 : conditional probability of -2 given deletion
    """
    if cna_real is None or cna_real.empty:
        return (
            pd.DataFrame(
                columns=["gene", "p_amp", "p_del", "p_neu", "q_amp2", "q_del2"]
            )
            .set_index("gene")
        )

    df = cna_real.copy()
    df.columns = df.columns.astype(str)

    # Coerce numeric and clip to expected range
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = np.rint(df).astype(int).clip(-2, 2)

    # Collapse duplicate gene columns safely (common in TCGA merges)
    if df.columns.duplicated().any():
        df = df.groupby(axis=1, level=0).mean()
        df = np.rint(df).astype(int).clip(-2, 2)

    rows = []

    for g in df.columns:
        x = df[g]

        p_amp = float((x > 0).mean())
        p_del = float((x < 0).mean())
        p_neu = float((x == 0).mean())  # safer than 1 - p_amp - p_del

        x_amp = x[x > 0]
        x_del = x[x < 0]

        q_amp2 = float((x_amp == 2).mean()) if len(x_amp) > 0 else 0.0
        q_del2 = float((x_del == -2).mean()) if len(x_del) > 0 else 0.0

        rows.append((g, p_amp, p_del, p_neu, q_amp2, q_del2))

    out = pd.DataFrame(
        rows,
        columns=["gene", "p_amp", "p_del", "p_neu", "q_amp2", "q_del2"],
    ).set_index("gene")

    return out
