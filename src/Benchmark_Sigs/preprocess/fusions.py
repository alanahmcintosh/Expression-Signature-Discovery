"""
Fusion preprocessing utilities.

"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


# =============================================================================
# FUSIONS
# =============================================================================

def read_fusions_raw(path):
    """
    Load raw fusion file and return one-hot encoded fusion matrix.

    Returns
    -------
    pd.DataFrame
        Samples x fusion_features (binary)
        Columns formatted as: GENE1--GENE2_FUSION
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[read_fusions_file] File not found: {path}")

    sv_df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    sv_df.columns = sv_df.columns.str.strip()

    # Standardize sample column
    if "Sample_Id" not in sv_df.columns and "sample_id" in sv_df.columns:
        sv_df["Sample_Id"] = sv_df["sample_id"].astype(str)

    # Derive gene symbols from fusion_name
    if (
        "fusion_name" in sv_df.columns
        and not {"Site1_Hugo_Symbol", "Site2_Hugo_Symbol"}.issubset(sv_df.columns)
    ):
        g1g2 = sv_df["fusion_name"].astype(str).str.split("--", n=1, expand=True)
        sv_df["Site1_Hugo_Symbol"] = g1g2[0].str.strip()
        if g1g2.shape[1] > 1:
            sv_df["Site2_Hugo_Symbol"] = g1g2[1].str.strip()

    # Validate required columns
    if "Site1_Hugo_Symbol" not in sv_df.columns or "Site2_Hugo_Symbol" not in sv_df.columns:
        raise ValueError("[read_fusions_file] Missing Site1_Hugo_Symbol/Site2_Hugo_Symbol")

    # Standardize patient/sample ID
    if "Sample_Id" in sv_df.columns:
        sv_df["Patient_Id"] = sv_df["Sample_Id"].astype(str)
    elif "Unnamed: 0" in sv_df.columns:
        sv_df["Patient_Id"] = sv_df["Unnamed: 0"].astype(str)
    else:
        raise ValueError("[read_fusions_file] No valid sample ID column found")

    sv_df = sv_df.dropna(subset=["Patient_Id"])

    # Build fusion feature names
    fusion_names = (
        sv_df["Site1_Hugo_Symbol"].astype(str).str.strip()
        + "--"
        + sv_df["Site2_Hugo_Symbol"].astype(str).str.strip()
    )

    # One-hot encode and aggregate per patient
    dummies = pd.get_dummies(fusion_names)
    dummies["Patient_Id"] = sv_df["Patient_Id"]

    fusion_df = dummies.groupby("Patient_Id").sum()
    fusion_df.columns = [f"{c}_FUSION" for c in fusion_df.columns]

    return fusion_df
