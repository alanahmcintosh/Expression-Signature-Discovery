"""
Readers for Core Data Types
"""




from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]



# =============================================================================
# CNA
# =============================================================================

def read_cna_file(
    path,
    cna_process,
    rename,
):
    """
    Load and preprocess a TCGA-style CNA file.

    Parameters
    ----------
    path : str or Path
        Path to CNA file.
    cna_process : bool
        If True:
            - Drop Entrez columns
            - Drop columns containing '|'
            - Transpose to samples x genes
    rename : bool
        If True:
            Rename columns to <GENE>_CNA

    Returns
    -------
    pd.DataFrame
        Samples x genes CNA matrix.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[read_cna_file] File not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python", index_col=0)

    if cna_process:
        # Remove Entrez columns
        df = df.drop(columns=[c for c in df.columns if "Entrez" in c], errors="ignore")

        # Remove gene columns with "|"
        df = df.loc[:, ~df.columns.str.contains(r"\|", regex=True)]

        # Transpose to samples x genes
        df = df.T

    if rename:
        df.columns = [f"{g}_CNA" for g in df.columns]

    return df


# =============================================================================
# FUSIONS
# =============================================================================

def read_fusions_file(path):
    """
    Load raw fusion file and return one-hot encoded fusion matrix.

    Expected columns:
    - Sample_Id or sample_id
    - fusion_name  (GENE1--GENE2) OR
    - Site1_Hugo_Symbol + Site2_Hugo_Symbol

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
        raise ValueError(
            "[read_fusions_file] Missing Site1_Hugo_Symbol/Site2_Hugo_Symbol"
        )

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

    dummies = pd.get_dummies(fusion_names)
    dummies["Patient_Id"] = sv_df["Patient_Id"]

    fusion_df = dummies.groupby("Patient_Id").sum()
    fusion_df.columns = [f"{c}_FUSION" for c in fusion_df.columns]

    return fusion_df


# =============================================================================
# CLINICAL
# =============================================================================

def read_clinical_file(path):
    """
    Load clinical file.

    Standardizes:
    - Index to uppercase strings
    - Column names to uppercase with underscores

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[read_clinical_file] File not found: {path}")

    for sep_try in [None, "\t", ",", r"\s+"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep_try,
                engine="python",
                header=0,
                index_col=0,
            )

            df.index = df.index.astype(str).str.strip().str.upper()
            df.columns = (
                df.columns.str.strip()
                .str.upper()
                .str.replace(" ", "_", regex=False)
            )

            return df

        except Exception:
            continue

    raise ValueError(f"[read_clinical_file] Failed to load clinical file: {path}")


# =============================================================================
# RNA
# =============================================================================

def read_rna_file(path):
    """
    Load RNA expression matrix.

    Assumes:
    - First column is gene or sample index
    - Returns raw DataFrame without transformation

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[read_rna_file] File not found: {path}")

    for sep_try in [None, "\t", ",", r"\s+"]:
        try:
            df = pd.read_csv(
                path,
                sep=sep_try,
                engine="python",
                header=0,
                index_col=0,
            )
            return df

        except Exception:
            continue

    raise ValueError(f"[read_rna_file] Failed to load RNA file: {path}")
