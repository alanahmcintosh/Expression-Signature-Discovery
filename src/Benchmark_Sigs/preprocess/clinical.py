"""
CLinical Data Processing 
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from benchmark_sigs.config.defaults import CLIN_FEATURES

# =============================================================================
# Subtypes
# =============================================================================

def process_subtypes(sample_info, min_samples = 5):
    """
    Extract and clean a subtype column from a clinical DataFrame.

    Parameters
    ----------
    sample_info : pd.DataFrame
        Clinical/sample metadata (index = samples).
    min_samples : int
        Minimum number of samples required for a subtype to be retained.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column 'Subtype', filtered to subtypes with
        at least `min_samples`.
    """
    if sample_info is None or sample_info.empty:
        raise ValueError("[process_subtypes] Empty clinical DataFrame.")

    df = sample_info.copy()
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_", regex=False)

    candidates = [
        "CANCER_SUBTYPE_CURATED",
        "ONCOTREE_CODE",
        "ONCOTREE",
        "SUBTYPE",
        "DISEASE_SUBTYPE",
        "DISEASE",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
    ]

    subtype_col: Optional[str] = None
    for col in candidates:
        if col in df.columns:
            subtype_col = col
            break

    if subtype_col is None:
        raise ValueError(
            f"[process_subtypes] No subtype column found in: {df.columns.tolist()}"
        )

    out = df[[subtype_col]].copy()
    out.columns = ["Subtype"]
    out = out.dropna(subset=["Subtype"])

    # Keep only subtypes with sufficient sample counts
    keep = out["Subtype"].value_counts()
    keep = keep[keep >= min_samples].index

    return out[out["Subtype"].isin(keep)]


# =============================================================================
# Baseline clinical variable selection
# =============================================================================

def select_known_clinicals(clin_df, cancer):
    """
    Keep only 'known at diagnosis' baseline clinical variables relevant to `cancer`.

    Parameters
    ----------
    clin_df : pd.DataFrame
        Clinical dataframe (rows = samples).
    cancer : str
        One of {'AML','ALL','IBC','OV','COAD'} (case-insensitive).

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only matched baseline clinical columns.
        Returns empty DataFrame (with same index) if nothing matches.
    """
    cancer_u = cancer.upper()
    if cancer_u not in CLIN_FEATURES:
        raise ValueError(
            f"Unknown cancer type '{cancer_u}'. Must be one of {list(CLIN_FEATURES.keys())}"
        )

    keep_patterns = [re.compile(pat, re.I) for pat in CLIN_FEATURES[cancer_u]]

    matched_cols: list[str] = []
    for c in clin_df.columns:
        for pat in keep_patterns:
            if pat.search(c):
                matched_cols.append(c)
                break

    matched_cols = sorted(set(matched_cols))
    if not matched_cols:
        print(
            f"[Warning] No baseline clinicals matched for {cancer_u}. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame(index=clin_df.index)

    return clin_df.loc[:, matched_cols]


# =============================================================================
# Encoding + alignment
# =============================================================================

def encode_alterations_clinical(
    mutation_df,
    cna_df,
    fusion_df,
    clinical_df,
    rna_df,
    disease,
    missingness_threshold = 0.25,
):
    """
    Align alterations + baseline clinicals to RNA samples, one-hot encode categorical
    clinicals, and scale all features.

    Parameters
    ----------
    mutation_df, cna_df, fusion_df : pd.DataFrame
        Alteration matrices with index as samples.
    clinical_df : pd.DataFrame
        Clinical metadata with index as samples.
    rna_df : pd.DataFrame
        Expression matrix with index as samples.
    disease : str
        Cancer/disease key for CLIN_FEATURES selection.
    missingness_threshold : float
        Drop clinical columns with missingness > threshold (fraction).

    Returns
    -------
    pd.DataFrame
        Scaled feature matrix Xs (samples x features), NaNs filled with 0.0.
    """
    #  Select and clean baseline clinical variables 
    clin_all_filtered = select_known_clinicals(clinical_df, disease)
    print("Kept columns:", list(clin_all_filtered.columns))

    # Apply missingness threshold on clinical columns only
    clin_filtered = clin_all_filtered.loc[
        :, clin_all_filtered.isna().mean() <= missingness_threshold
    ].copy()

    # Fill remaining NaNs in categorical columns (numeric are left as-is for now)
    for col in clin_filtered.columns:
        if clin_filtered[col].dtype == object:
            clin_filtered[col] = clin_filtered[col].fillna("Unknown")

    print(f"Kept {clin_filtered.shape[1]} columns out of {clin_all_filtered.shape[1]}")
    print(f"Remaining NaNs (clinical): {clin_filtered.isna().sum().sum()}")

    # Align all data types by sample
    bin_alt_real = pd.concat(
        [mutation_df, fusion_df, cna_df, clin_filtered],
        axis=1,
        join="inner",
    )

    # Align to RNA expression samples
    common_samples = bin_alt_real.index.intersection(rna_df.index)
    X_aligned = bin_alt_real.loc[common_samples].sort_index()
    Y_aligned = rna_df.loc[common_samples].sort_index()  # kept for logging parity

    print(f"Aligned shapes → X: {X_aligned.shape}, Y: {Y_aligned.shape}")

    # Coerce common numeric clinical columns if present
    possible_numeric = [
        "DIAGNOSIS_AGE",
        "TMB",
        "MSISENSOR_SCORE",
        "MSI_MANTIS_SCORE",
        "AGE",
        "ANEUPLOIDY_SCORE",
        "TUMOR_BREAK_LOAD",
        "TMB_(NONSYNONYMOUS)",
    ]
    for col in possible_numeric:
        if col in X_aligned.columns:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce")

    #  One-hot encode categoricals and scale everything 
    non_numeric_cols = X_aligned.select_dtypes(include=["object", "category"]).columns
    print(f"Encoding {len(non_numeric_cols)} non-numeric columns")

    X_encoded = pd.get_dummies(
        X_aligned,
        columns=non_numeric_cols,
        drop_first=False,
        dtype=float,
    )
    print(f"Remaining NaNs (post-encoding): {X_encoded.isna().sum().sum()}")

    scaler = StandardScaler()
    Xs_arr = scaler.fit_transform(X_encoded)

    Xs = pd.DataFrame(Xs_arr, index=X_encoded.index, columns=X_encoded.columns)
    Xs = Xs.fillna(0.0)

    print(f"Remaining NaNs (final): {Xs.isna().sum().sum()}")
    print("Final Xs shape:", Xs.shape)

    return Xs
    print("Final Xs shape:", Xs.shape)
    
    return Xs
