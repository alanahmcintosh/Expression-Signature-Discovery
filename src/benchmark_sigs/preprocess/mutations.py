"""
Mutation preprocessing utilities.
"""

from __future__ import annotations

from typing import Optional, Set

import pandas as pd

from benchmark_sigs.config.defaults import (
    DEFAULT_ONCOGENES,
    DEFAULT_TUMOR_SUPPRESSORS,
    DISEASE_ONCOGENES,
    DISEASE_TUMOR_SUPPRESSORS,
    DEFAULT_DROP_CLASSES,
    TRUNCATING,
    SPLICE,
    MISSENSE,
    INFRAME,
)
# =============================================================================
# Variant classification
# =============================================================================

def classify_variant(gene, disease, variant_class):
    """
    Classify a single variant into GOF/LOF/Unclear/Drop.

    Parameters
    ----------
    gene : str
        Hugo symbol.
    disease : str
        Disease label (e.g. AML, ALL).
    variant_class : str
        Variant classification string (must match your config sets).

    Returns
    -------
    str
        One of {"GOF", "LOF", "Unclear", "Drop"}.
    """
    gene_u = str(gene).upper()
    disease_u = str(disease).upper()
    variant_u = str(variant_class)

    onc = DEFAULT_ONCOGENES.union(DISEASE_ONCOGENES.get(disease_u, set()))
    tsg = DEFAULT_TUMOR_SUPPRESSORS.union(DISEASE_TUMOR_SUPPRESSORS.get(disease_u, set()))

    if variant_u in DEFAULT_DROP_CLASSES:
        return "Drop"

    # --- special cases ---
    if gene_u == "NOTCH1" and disease_u in {"ALL", "T-ALL"}:
        return "GOF"
    if gene_u == "NPM1" and disease_u == "AML" and variant_u in TRUNCATING:
        return "GOF"
    if gene_u in {"SF3B1", "SRSF2", "U2AF1"} and variant_u in MISSENSE:
        return "GOF"

    # --- general logic ---
    if gene_u in onc:
        if variant_u in (MISSENSE | INFRAME):
            return "GOF"
        if variant_u in TRUNCATING:
            return "Unclear"

    if gene_u in tsg:
        if variant_u in (TRUNCATING | SPLICE):
            return "LOF"
        if variant_u in MISSENSE:
            return "Unclear"

    return "Unclear"


# =============================================================================
# MAF to one-hot encoded Dataframe
# =============================================================================

def maf_to_onehot(
    maf_annot,
    sample_col = "Tumor_Sample_Barcode",
    func_col  = "Functional_Label",
    include_uncertain = True,
    uncertain_top_k = 100,          # max 100 uncertain (_MUT) allowed
    min_uncertain_freq = 0.02,    # _MUT must have min 2% freq if using thresholding
    uncertain_labels = None,
):
    """
    One-hot encode a MAF-like dataframe into *_GOF, *_LOF, and *_MUT columns.

    Parameters
    ----------
    maf_annot : pd.DataFrame
        Annotated mutation table.
    sample_col : str
        Sample identifier column.
    func_col : str
        Functional label column (e.g. GOF/LOF/Unclear).
    include_uncertain : bool
        If True, include passenger-like '_MUT' features for uncertain mutations.
    uncertain_top_k : int
        If >0, keep only top-k uncertain genes by frequency. If 0/None, use
        min_uncertain_freq thresholding instead.
    min_uncertain_freq : float
        Minimum mean frequency across samples required to keep uncertain gene
        if not using top-k.
    uncertain_labels : set[str] or None
        Labels considered "uncertain". Defaults to {"Unclear","Uncertain","Unknown","MUT"}.

    Returns
    -------
    pd.DataFrame
        One-hot matrix (samples x alteration features), ints {0,1}.
    """
    if maf_annot is None or maf_annot.empty:
        return pd.DataFrame()

    df = maf_annot.copy()

    if sample_col not in df.columns:
        raise KeyError(f"[maf_to_onehot] sample_col '{sample_col}' not found in columns.")
    if "Hugo_Symbol" not in df.columns:
        raise KeyError("[maf_to_onehot] Required column 'Hugo_Symbol' not found.")
    if func_col not in df.columns:
        raise KeyError(f"[maf_to_onehot] func_col '{func_col}' not found in columns.")

    df[sample_col] = df[sample_col].astype(str).str.strip()
    df["Hugo_Symbol"] = df["Hugo_Symbol"].astype(str).str.strip()

    uncertain_labels = uncertain_labels or {"Unclear", "Uncertain", "Unknown", "MUT"}

    # --- GOF and LOF (binary presence per sample-gene) ---
    gof = (
        (df[func_col] == "GOF")
        .groupby([df[sample_col], df["Hugo_Symbol"]])
        .any()
        .unstack(fill_value=False)
    )
    lof = (
        (df[func_col] == "LOF")
        .groupby([df[sample_col], df["Hugo_Symbol"]])
        .any()
        .unstack(fill_value=False)
    )

    if not gof.empty:
        gof.columns = [f"{g}_GOF" for g in gof.columns]
    if not lof.empty:
        lof.columns = [f"{g}_LOF" for g in lof.columns]

    out = pd.DataFrame(index=sorted(set(df[sample_col])))

    for mat in (gof, lof):
        if mat is not None and not mat.empty:
            out = out.join(mat.astype(int), how="left")

    # --- uncertain (_MUT) ---
    if include_uncertain:
        unc = df[df[func_col].isin(uncertain_labels)]
        if not unc.empty:
            uncpairs = (
                unc.groupby([unc[sample_col], unc["Hugo_Symbol"]])
                .size()
                .unstack(fill_value=0)
            )

            freqs = uncpairs.mean(axis=0)

            if uncertain_top_k:
                keep = freqs.sort_values(ascending=False).head(uncertain_top_k).index
            else:
                keep = freqs[freqs >= min_uncertain_freq].index

            uncpairs = uncpairs.reindex(columns=keep, fill_value=0)

            # Drop genes that already appear as functional GOF/LOF features
            functional_genes = {c.split("_")[0] for c in out.columns if out[c].sum() > 0}
            uncpairs = uncpairs.drop(
                columns=[g for g in uncpairs.columns if g in functional_genes],
                errors="ignore",
            )

            if not uncpairs.empty:
                uncpairs = (uncpairs > 0).astype(int)
                uncpairs.columns = [f"{g}_MUT" for g in uncpairs.columns]
                out = out.join(uncpairs, how="left")

            # Keep your original logging behaviour
            print(
                f"[INFO] Added {len(uncpairs.columns)} passenger (_MUT) features "
                f"from {len(unc)} uncertain mutations."
            )

    # --- cleanup ---
    out = out.fillna(0).astype(int)
    out = out.loc[:, out.sum(axis=0) > 0]

    print(f"[INFO] One-hot matrix: {out.shape[0]} samples × {out.shape[1]} features")
    print(f"  • {sum('_GOF' in c for c in out.columns)} GOF")
    print(f"  • {sum('_LOF' in c for c in out.columns)} LOF")
    print(f"  • {sum('_MUT' in c for c in out.columns)} MUT")

    return out
