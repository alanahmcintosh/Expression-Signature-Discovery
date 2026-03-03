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
