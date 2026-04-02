"""
Scaling + weighting utilities for alteration blocks before KNN sampling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def robust_scale_with_floor(df, cna_std_floor):
    """
    Column-wise z-scoring with a minimum standard deviation floor.

    CNA features can have very low/near-constant variance; a std floor prevents
    explosive standardized values so rare/constant alterations do not dominate
    distance computations.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric matrix (samples x features).
    cna_std_floor : float
        Minimum standard deviation used in denominator.

    Returns
    -------
    pd.DataFrame
        Standardized matrix with same index/columns.
    """
    df = df.astype(float)
    mu = df.mean(axis=0)

    # Column standard deviation; fill NaNs to avoid division by zero artifacts
    sd = df.std(axis=0).fillna(0.0)

    # Minimum allowed denominator to avoid dividing by tiny/zero variance
    denom = np.maximum(sd.to_numpy(), float(cna_std_floor))

    out = (df - mu) / denom
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def preprocess_X_weighted(
    mut=None,
    fusion=None,
    cna=None,
    clinical=None,
    weights={"mut": 1.0, "fusion": 1.5, "cna": 2.0, "clinical": 0.5},
    cna_std_floor=0.25,
    cna_clip=(-2, 2),
):
    """
    Standardizes and weights alteration data blocks before KNN sampling.

    Returns
    -------
    combined_scaled : pd.DataFrame
        Weighted + scaled concatenation of all available blocks for neighbor search.
    unscaled_blocks : dict
        Original unscaled blocks (used for downstream resampling).
    """
    scaler = StandardScaler()

    scaled_blocks = []
    unscaled_blocks = {}

    # -------------------------
    # Mutations
    # -------------------------
    if mut is not None and not mut.empty:
        mut = mut.copy()
        mut.columns = mut.columns.astype(str)
        mut = mut.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        scaled_blocks.append(mut * float(weights.get("mut", 1.0)))
        unscaled_blocks["mut"] = mut

    # -------------------------
    # Fusions
    # -------------------------
    if fusion is not None and not fusion.empty:
        fusion = fusion.copy()
        fusion.columns = fusion.columns.astype(str)
        fusion = fusion.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        scaled_blocks.append(fusion * float(weights.get("fusion", 1.5)))
        unscaled_blocks["fusion"] = fusion

    # -------------------------
    # CNA (GISTIC-like integers)
    # -------------------------
    if cna is not None and not cna.empty:
        cna_proc = cna.copy()
        cna_proc.columns = cna_proc.columns.astype(str)

        # Coerce -> int -> clip into expected range
        cna_num = cna_proc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        lo, hi = cna_clip
        g = np.rint(cna_num).astype(int).clip(lo, hi)

        # Collapse duplicated CNA columns so g[col] is always a Series, not a DataFrame
        if g.columns.duplicated().any():
            g = g.groupby(axis=1, level=0).mean()
            g = np.rint(g).astype(int).clip(lo, hi)

        # Keep unscaled discrete CNA for later sampling/output
        unscaled_blocks["cna"] = g

        # Two non-negative dosage views (0,1,2)
        amp_lvl = g.where(g > 0, 0).astype(float)
        del_lvl = (-g.where(g < 0, 0)).astype(float)

        amp_scaled = robust_scale_with_floor(amp_lvl, cna_std_floor=cna_std_floor)
        del_scaled = robust_scale_with_floor(del_lvl, cna_std_floor=cna_std_floor)

        # Rename so both views can coexist
        amp_scaled.columns = [f"{c}__AMP_LVL" for c in amp_scaled.columns]
        del_scaled.columns = [f"{c}__DEL_LVL" for c in del_scaled.columns]

        cna_scaled = pd.concat([amp_scaled, del_scaled], axis=1)
        scaled_blocks.append(cna_scaled * float(weights.get("cna", 2.0)))

    # -------------------------
    # Clinical (numeric only for scaling)
    # -------------------------
    if clinical is not None and not clinical.empty:
        clinical = clinical.copy()
        clinical.columns = clinical.columns.astype(str)

        # Keep original for sampling later
        unscaled_blocks["clinical"] = clinical

        # Numeric-only for neighbor search
        clin_num = clinical.apply(pd.to_numeric, errors="coerce")
        clin_num = clin_num.loc[:, clin_num.notna().any(axis=0)]  # drop fully non-numeric cols

        if clin_num.shape[1] > 0:
            clin_num = clin_num.fillna(0.0)

            clin_scaled = pd.DataFrame(
                scaler.fit_transform(clin_num),
                index=clin_num.index,
                columns=clin_num.columns,
            )
            scaled_blocks.append(clin_scaled * float(weights.get("clinical", 0.5)))

    # -------------------------
    # Combine scaled blocks
    # -------------------------
    if len(scaled_blocks) == 0:
        combined_scaled = pd.DataFrame(index=pd.Index([]))
        return combined_scaled, unscaled_blocks

    combined_scaled = pd.concat(scaled_blocks, axis=1)
    combined_scaled.columns = combined_scaled.columns.astype(str)

    return combined_scaled, unscaled_blocks
