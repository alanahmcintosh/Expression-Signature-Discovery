from __future__ import annotations

import pandas as pd
from typing import Optional

def to_patient_index(df, study):
    """
    Standardize sample index formatting.
    For TCGA, trims to first 12 characters (TCGA-XX-YYYY).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    idx = df.index.astype(str).str.strip().str.upper()
    if study and study.upper() == "TCGA":
        idx = idx.str.slice(0, 12)

    out = df.copy()
    out.index = idx
    return out


def to_patient_id(index, study):
    """
    Map raw sample IDs to patient IDs for specific studies.
    """
    s = index.astype(str).str.strip()
    if study and study.upper() == "TCGA":
        return s.str.extract(r"(TCGA-\w{2}-\w{4})")[0]
    if study and study.upper() == "TARGET":
        return s.str.extract(r"(TARGET-\d{2}-[A-Z0-9]{6})")[0]
    return s


def safe_map_index(df, study, name):
    """
    Map df.index to patient IDs, drop unmapped, standardize formatting, and log.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[INFO] {name}: empty/unusable dataframe")
        return df

    mapped = to_patient_id(df.index.to_series(), study=study)
    keep = mapped.notna()

    out = df.loc[keep].copy()
    out.index = mapped.loc[keep].str.strip().str.upper()
    out = to_patient_index(out, study)

    print(f"[INFO] {name}: mapped {len(out)} samples")
    return out
