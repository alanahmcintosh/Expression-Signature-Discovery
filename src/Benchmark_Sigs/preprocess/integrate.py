"""
Integration of multi-omics modalities into aligned matrices.

This module orchestrates:
- clinical loading + subtype extraction
- mutation parsing / one-hot creation
- CNA loading + processing
- fusion loading (optional)
- RNA loading + orientation handling
- sample ID mapping + intersection
- frequency-based feature filtering

Returns aligned matrices ready for downstream modelling/simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark_sigs.io.readers import read_clinical_file, read_rna_file, read_cna_file
from benchmark_sigs.preprocess.clinical import process_subtypes
from benchmark_sigs.preprocess.mutations import classify_variant, maf_to_onehot
from benchmark_sigs.preprocess.fusions import read_fusions_file as load_fusions_raw 
from benchmark_sigs.utils.sample_ids import to_patient_id, to_patient_index, safe_map_index
from benchmark_sigs.utils.checks import nonempty


def integrate_data(
    mut_path,
    cna_path,
    fusion_info_path,
    patient_path,
    sample_path,
    rna_path,
    study=None,
    disease=None,
    cna_process=True,
    is_tcga=False,
    cna_top_n=200,
    min_subtype_n=3,
    mut_freq_thresh=0.02,
    fusion_freq_thresh=0.02,
    rename=True,
    input_format="maf",  # "maf" | "custom_tsv" | "onehot"
    uncertain_top_k=100,
):
    """Integrate mutation, CNA, fusion, and clinical data into aligned matrices."""
    # --- 1. Clinical + subtype
    clinical_start = read_clinical_file(sample_path)
    clinical_start = clinical_start.iloc[4:,]
    clinical = process_subtypes(clinical_start, min_samples=min_subtype_n)
    clinical = safe_map_index(clinical, study, "Clinical")

    sample = read_clinical_file(patient_path)
    sample = sample.iloc[4:,]

    # --- 2. Mutations
    if input_format == "onehot":
        mut_raw = pd.read_csv(mut_path, sep=None, engine="python", index_col=0).astype(int)
        mut_raw = safe_map_index(mut_raw, study, "Mutations")
    else:
        maf = pd.read_csv(mut_path, sep="\t", comment="#")
        maf["Tumor_Sample_Barcode"] = maf["Tumor_Sample_Barcode"].astype(str).str.strip().str.upper()

        maf["Functional_Label"] = maf.apply(
            lambda r: classify_variant(r["Hugo_Symbol"], disease, r["Variant_Classification"]),
            axis=1,
        )

        mut_raw = maf_to_onehot(maf, uncertain_top_k=uncertain_top_k)
        mut_raw.index = to_patient_id(mut_raw.index.to_series(), study=study).str.strip().str.upper()
        mut_raw = to_patient_index(mut_raw[mut_raw.index.notna()], study)

    # --- 3. CNA & Fusions & RNA
    cna_raw = read_cna_file(cna_path, cna_process=cna_process, rename=rename)
    cna_raw = safe_map_index(cna_raw, study, "CNA")

    fusion_raw = load_fusions_raw(fusion_info_path) if fusion_info_path else None
    if fusion_raw is not None:
        fusion_raw = safe_map_index(fusion_raw, study, "Fusions")
        if not nonempty(fusion_raw):
            print("[INFO] No valid fusions after mapping; dropping fusion modality.")
            fusion_raw = None
    else:
        print("[INFO] No fusion file provided; skipping fusions.")

    rna = read_rna_file(rna_path)
    rna.index = to_patient_id(rna.index, study=study)
    if rna.shape[0] > rna.shape[1]:
        rna = rna.T

    # --- 4. Intersections
    modalities = {"mut": mut_raw, "cna": cna_raw, "clin": clinical, "RNA": rna}
    if fusion_raw is not None:
        modalities["fus"] = fusion_raw

    common = None
    for name, df in modalities.items():
        if not nonempty(df) and name != "clin":
            print(f"[WARN] {name} modality has no data; ignored for intersection.")
            continue
        idx = df.index if isinstance(df, pd.DataFrame) else pd.Index([])
        common = idx if common is None else common.intersection(idx)

    if common is None or len(common) == 0:
        print("[DEBUG] Zero overlap after ID coercion. Pairwise overlaps:")
        keys = list(modalities.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ai = modalities[keys[i]].index if nonempty(modalities[keys[i]]) else pd.Index([])
                bj = modalities[keys[j]].index if nonempty(modalities[keys[j]]) else pd.Index([])
                print(f"  {keys[i]} ∩ {keys[j]}: {len(ai.intersection(bj))}")
        raise ValueError("No shared samples across modalities after ID coercion.")

    print(f"[INFO] Shared sample count: {len(common)}")

    # --- 5. Subset & normalize
    mut_df = mut_raw.loc[common]

    cna_sub = cna_raw.loc[common]
    cna_df = cna_sub.apply(pd.to_numeric, errors="coerce").fillna(0)
    cna_df = np.rint(cna_df).astype(int).clip(-2, 2)
    cna_neutral = 0

    clinical = clinical.loc[common]
    fusion_df = fusion_raw.loc[common] if fusion_raw is not None else pd.DataFrame(index=common)
    rna_df = rna.loc[common]

    clinical_df = pd.concat([clinical_start, sample], axis=1, join="inner")

    # --- 6. Frequency filters
    if cna_top_n is not None and cna_top_n > 0:
        alt_freq = (cna_df != cna_neutral).mean(axis=0)
        top_genes = alt_freq.sort_values(ascending=False).head(cna_top_n).index
        cna_df = cna_df[top_genes]

    if mut_df.shape[0] < 100:
        mut_df = mut_df.loc[:, (mut_df > 0).sum() >= 3]
    else:
        mut_df = mut_df.loc[:, (mut_df > 0).mean() >= mut_freq_thresh]
        top_300 = (mut_df > 0).mean().sort_values(ascending=False).head(300).index
        mut_df = mut_df[top_300]

    if nonempty(fusion_df):
        if fusion_df.shape[0] < 100:
            fusion_df = fusion_df.loc[:, (fusion_df > 0).sum() >= 3]
        else:
            freq = (fusion_df > 0).mean()
            fusion_df = fusion_df.loc[:, freq[freq >= fusion_freq_thresh].index]
        if fusion_df.shape[1] == 0:
            print("[WARN] No fusions passed frequency filter.")
            fusion_df = pd.DataFrame(index=common)

    return mut_df, cna_df, fusion_df, clinical, clinical_df, rna_df
