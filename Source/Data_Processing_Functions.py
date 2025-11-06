"""
===============================================================
Multi-omic Integration and Preprocessing Pipeline
---------------------------------------------------------------
This script handles:
  • Gene-based GOF/LOF/Unclear mutation classification
  • One-hot encoding of MAF-style variant data
  • Loading and filtering of CNA, fusion, and clinical data
  • Subtype extraction and sample harmonization
  • Integration of modalities into aligned multi-omic matrices
===============================================================
"""

import pandas as pd
import numpy as np
from typing import Optional

# ============================================================
# 1. Default Gene Sets
# ============================================================

DEFAULT_ONCOGENES = {
    "KRAS", "NRAS", "HRAS", "BRAF", "PIK3CA", "IDH1", "IDH2", "EGFR",
    "ERBB2", "ALK", "FGFR2", "FGFR3", "KIT", "PDGFRA", "JAK2", "MYD88",
    "CTNNB1", "GNAQ", "GNAS"
}

DEFAULT_TUMOR_SUPPRESSORS = {
    "TP53", "PTEN", "RB1", "NF1", "TET2", "DNMT3A", "ARID1A", "ARID1B",
    "KEAP1", "CDKN2A", "SMAD4", "STK11", "KMT2D", "APC", "VHL",
    "ATRX", "BRCA1", "BRCA2"
}

TRUNCATING = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Stop_Codon_Del", "Stop_Codon_Ins"
}
SPLICE = {"Splice_Site", "Splice_Acceptor", "Splice_Donor"}
INFRAME = {"In_Frame_Del", "In_Frame_Ins"}
MISSENSE = {"Missense_Mutation"}
DEFAULT_DROP_CLASSES = {
    "Silent", "Intron", "UTR_3", "UTR_5", "IGR", "RNA",
    "Non-coding_Transcript_Exon", "In_Frame_Shift"
}


# ============================================================
# 2. Disease-Specific Gene Sets
# ============================================================

DISEASE_ONCOGENES = {
    "ALL": {"NOTCH1", "JAK1", "JAK2", "JAK3", "IL7R", "FLT3", "PTPN11",
            "CRLF2", "NCOA2", "ETV6"},
    "T-ALL": {"NOTCH1", "JAK1", "JAK3", "IL7R"},
    "B-ALL": {"JAK2", "IL7R", "FLT3", "PTPN11", "CRLF2", "NCOA2"},
    "AML": {"FLT3", "KIT", "NRAS", "KRAS", "IDH1", "IDH2", "JAK2", "NPM1"},
    "BRCA": {"ERBB2", "PIK3CA", "AKT1", "ESR1", "FGFR1"},
    "COAD": {"KRAS", "NRAS", "BRAF", "PIK3CA", "ERBB2", "FGFR2"},
    "OV": {"KRAS", "NRAS", "BRAF", "PIK3CA", "ERBB2"},
}

DISEASE_TUMOR_SUPPRESSORS = {
    "ALL": {"IKZF1", "PAX5", "ETV6", "CDKN2A", "CDKN2B", "FBXW7", "PHF6",
            "PTEN", "CREBBP", "RB1"},
    "T-ALL": {"FBXW7", "PHF6", "PTEN", "CDKN2A", "CDKN2B"},
    "B-ALL": {"IKZF1", "PAX5", "ETV6", "CDKN2A", "CDKN2B", "CREBBP", "RB1"},
    "AML": {"TP53", "DNMT3A", "TET2", "ASXL1", "RUNX1", "NPM1", "CEBPA", "WT1"},
    "BRCA": {"TP53", "BRCA1", "BRCA2", "PTEN", "RB1", "CDH1", "NF1"},
    "COAD": {"APC", "TP53", "SMAD4", "FBXW7", "ARID1A"},
    "OV": {"TP53", "BRCA1", "BRCA2", "NF1", "RB1"},
}


# ============================================================
# 3. Variant Classification
# ============================================================

def classify_variant(gene: str, disease: str, variant_class: str) -> str:
    """Return GOF, LOF, Unclear, or Drop classification for a single variant."""
    gene, disease = gene.upper(), disease.upper()

    onc = DEFAULT_ONCOGENES.union(DISEASE_ONCOGENES.get(disease, set()))
    tsg = DEFAULT_TUMOR_SUPPRESSORS.union(DISEASE_TUMOR_SUPPRESSORS.get(disease, set()))

    if variant_class in DEFAULT_DROP_CLASSES:
        return "Drop"

    # --- special cases ---
    if gene == "NOTCH1" and disease in {"ALL", "T-ALL"}:
        return "GOF"
    if gene == "NPM1" and disease == "AML" and variant_class in TRUNCATING:
        return "GOF"
    if gene in {"SF3B1", "SRSF2", "U2AF1"} and variant_class in MISSENSE:
        return "GOF"

    # --- general logic ---
    if gene in onc:
        if variant_class in MISSENSE | INFRAME:
            return "GOF"
        elif variant_class in TRUNCATING:
            return "Unclear"
    elif gene in tsg:
        if variant_class in TRUNCATING | SPLICE:
            return "LOF"
        elif variant_class in MISSENSE:
            return "Unclear"

    return "Unclear"


# ============================================================
# 4. MAF → One-hot Conversion
# ============================================================

def maf_to_onehot(
    maf_annot: pd.DataFrame,
    sample_col: str = "Tumor_Sample_Barcode",
    func_col: str = "Functional_Label",
    include_uncertain: bool = True,
    uncertain_top_k: Optional[int] = 100,
    min_uncertain_freq: float = 0.02,
    uncertain_labels: Optional[set] = None,
) -> pd.DataFrame:
    """
    One-hot encode a MAF-like dataframe into *_GOF, *_LOF, and *_MUT columns.
    """
    df = maf_annot.copy()
    df[sample_col] = df[sample_col].astype(str).str.strip()
    df["Hugo_Symbol"] = df["Hugo_Symbol"].astype(str).str.strip()
    uncertain_labels = uncertain_labels or {"Unclear", "Uncertain", "Unknown", "MUT"}

    # --- GOF and LOF ---
    gof = (df[func_col] == "GOF").groupby([df[sample_col], df["Hugo_Symbol"]]).any().unstack(fill_value=False)
    lof = (df[func_col] == "LOF").groupby([df[sample_col], df["Hugo_Symbol"]]).any().unstack(fill_value=False)
    gof.columns = [f"{g}_GOF" for g in gof.columns]
    lof.columns = [f"{g}_LOF" for g in lof.columns]

    out = pd.DataFrame(index=sorted(set(df[sample_col])))
    for mat in [gof, lof]:
        if not mat.empty:
            out = out.join(mat.astype(int), how="left")

    # --- uncertain (_MUT) ---
    if include_uncertain:
        unc = df[df[func_col].isin(uncertain_labels)]
        if not unc.empty:
            uncpairs = unc.groupby([df[sample_col], df["Hugo_Symbol"]]).size().unstack(fill_value=0)
            freqs = uncpairs.mean(axis=0)

            keep = (freqs.sort_values(ascending=False).head(uncertain_top_k).index
                    if uncertain_top_k else freqs[freqs >= min_uncertain_freq].index)
            uncpairs = uncpairs.reindex(columns=keep, fill_value=0)

            functional_genes = {c.split("_")[0] for c in out.columns if out[c].sum() > 0}
            uncpairs = uncpairs.drop(columns=[g for g in uncpairs.columns if g in functional_genes],
                                     errors="ignore")

            if not uncpairs.empty:
                uncpairs = (uncpairs > 0).astype(int)
                uncpairs.columns = [f"{g}_MUT" for g in uncpairs.columns]
                out = out.join(uncpairs, how="left").fillna(0)

            print(f"[INFO] Added {len(uncpairs.columns)} passenger (_MUT) features "
                  f"from {len(unc)} uncertain mutations.")

    # --- cleanup ---
    out = out.fillna(0).astype(int)
    out = out.loc[:, out.sum(0) > 0]
    print(f"[INFO] One-hot matrix: {out.shape[0]} samples × {out.shape[1]} features")
    print(f"  • {sum('_GOF' in c for c in out.columns)} GOF")
    print(f"  • {sum('_LOF' in c for c in out.columns)} LOF")
    print(f"  • {sum('_MUT' in c for c in out.columns)} MUT")
    return out


# ============================================================
# 5. CNA / Fusion / Clinical Loaders
# ============================================================

def load_cna(path: str, top_n: int = 500, cna_process: bool = True, rename: bool = True) -> pd.DataFrame:
    """Load and preprocess a CNA file (TCGA-style)."""
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    if cna_process:
        df = df.drop(columns=[c for c in df.columns if "Entrez" in c], errors="ignore")
        df = df.loc[:, ~df.columns.str.contains(r"\|", regex=True)]
        df = df.T
    if rename:
        df.columns = [f"{g}_CNA" for g in df.columns]

    top_vars = df.var().sort_values(ascending=False).head(top_n).index
    return df[top_vars]


def load_fusions_raw(path: str) -> pd.DataFrame:
    """Load raw fusion data and one-hot encode fusions."""
    sv_df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    sv_df.columns = sv_df.columns.str.strip()

    if "Sample_Id" not in sv_df.columns and "sample_id" in sv_df.columns:
        sv_df["Sample_Id"] = sv_df["sample_id"].astype(str)

    if {"fusion_name"}.issubset(sv_df.columns) and not {"Site1_Hugo_Symbol", "Site2_Hugo_Symbol"}.issubset(sv_df.columns):
        g1g2 = sv_df["fusion_name"].astype(str).str.split("--", n=1, expand=True)
        sv_df["Site1_Hugo_Symbol"] = g1g2[0].str.strip()
        sv_df["Site2_Hugo_Symbol"] = g1g2[1].str.strip()

    sv_df["Patient_Id"] = sv_df.get("Sample_Id", sv_df.get("Unnamed: 0")).astype(str)
    if "Site1_Hugo_Symbol" not in sv_df or "Site2_Hugo_Symbol" not in sv_df:
        raise ValueError("Missing Site1_Hugo_Symbol/Site2_Hugo_Symbol")

    sv_df = sv_df.dropna(subset=["Patient_Id"])
    dummies = pd.get_dummies(
        sv_df["Site1_Hugo_Symbol"].astype(str).str.strip() + "--" +
        sv_df["Site2_Hugo_Symbol"].astype(str).str.strip()
    )
    dummies["Patient_Id"] = sv_df["Patient_Id"]
    fusion_df = dummies.groupby("Patient_Id").sum()
    fusion_df.columns = [f"{c}_FUSION" for c in fusion_df.columns]
    return fusion_df


def read_clinical_file(path: str) -> pd.DataFrame:
    """Try reading a clinical file with flexible delimiters."""
    for sep_try in [None, "\t", ",", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep_try, engine="python", header=0, index_col=0)
            df.index = df.index.astype(str).str.strip().str.upper()
            df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
            return df
        except Exception:
            continue
    raise ValueError(f"[read_clinical_file] Failed to load clinical file: {path}")


def process_subtypes(sample_info: pd.DataFrame, min_samples: int = 5) -> pd.DataFrame:
    """Extract and clean subtype column, filtering for sufficient sample counts."""
    if sample_info is None or sample_info.empty:
        raise ValueError("[process_subtypes] Empty clinical DataFrame.")

    sample_info.columns = sample_info.columns.str.strip().str.upper().str.replace(" ", "_")
    candidates = [
        "CANCER_SUBTYPE_CURATED", "ONCOTREE_CODE", "ONCOTREE", "SUBTYPE",
        "DISEASE_SUBTYPE", "DISEASE", "CANCER_TYPE", "CANCER_TYPE_DETAILED"
    ]
    for col in candidates:
        if col in sample_info.columns:
            sample_info = sample_info[[col]].copy()
            sample_info.columns = ["Subtype"]
            break
    else:
        raise ValueError(f"[process_subtypes] No subtype column found in: {sample_info.columns.tolist()}")

    sample_info = sample_info.dropna(subset=["Subtype"])
    keep = sample_info["Subtype"].value_counts()[lambda x: x >= min_samples].index
    return sample_info[sample_info["Subtype"].isin(keep)]


# ============================================================
# 6. Helper Functions
# ============================================================

def _to_patient_index(df: Optional[pd.DataFrame], study: Optional[str]) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    idx = df.index.astype(str).str.strip().str.upper()
    if study and study.upper() == "TCGA":
        idx = idx.str.slice(0, 12)
    out = df.copy()
    out.index = idx
    return out


def _nonempty(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0


def to_patient_id(index: pd.Series, study: Optional[str] = None) -> pd.Series:
    index = index.astype(str).str.strip()
    if study == "TCGA":
        return index.str.extract(r"(TCGA-\w{2}-\w{4})")[0]
    elif study == "TARGET":
        return index.str.extract(r"(TARGET-\d{2}-[A-Z0-9]{6})")[0]
    return index


def safe_map_index(df: pd.DataFrame, study: Optional[str], name: str) -> pd.DataFrame:
    mapped = to_patient_id(df.index.to_series(), study=study)
    df = df[mapped.notna()].copy()
    df.index = mapped[mapped.notna()].str.strip().str.upper()
    df = _to_patient_index(df, study)
    print(f"[INFO] {name}: mapped {len(df)} samples")
    return df


# ============================================================
# 7. Main Integration Function
# ============================================================

def integrate_data(
    mut_path: str,
    cna_path: str,
    fusion_info_path: Optional[str],
    subtype_clinical_path: str,
    study: Optional[str] = None,
    disease: Optional[str] = None,
    cna_process: bool = True,
    cna_top_n: int = 500,
    min_subtype_n: int = 5,
    mut_freq_thresh: float = 0.02,
    fusion_freq_thresh: float = 0.02,
    is_tcga: bool = True,
    rename: bool = True,
    input_format: str = "maf",  # "maf" | "custom_tsv" | "onehot"
    uncertain_top_k: Optional[int] = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Integrate mutation, CNA, fusion, and clinical data into aligned matrices."""
    # --- 1. Clinical + subtype
    clinical_start = read_clinical_file(subtype_clinical_path)
    clinical = process_subtypes(clinical_start, min_samples=min_subtype_n)
    clinical = safe_map_index(clinical, study, "Clinical")

    # --- 2. Mutations
    if input_format == "onehot":
        mut_raw = pd.read_csv(mut_path, sep=None, engine="python", index_col=0).astype(int)
        mut_raw = safe_map_index(mut_raw, study, "Mutations")
    else:
        maf = pd.read_csv(mut_path, sep="\t", comment="#")
        maf["Tumor_Sample_Barcode"] = maf["Tumor_Sample_Barcode"].astype(str).str.strip().str.upper()

        maf["Functional_Label"] = maf.apply(
            lambda r: classify_variant(r["Hugo_Symbol"], disease, r["Variant_Classification"]), axis=1
        )

        mut_raw = maf_to_onehot(maf, uncertain_top_k=uncertain_top_k)
        mut_raw.index = to_patient_id(mut_raw.index.to_series(), study=study).str.strip().str.upper()
        mut_raw = _to_patient_index(mut_raw[mut_raw.index.notna()], study)

    # --- 3. CNA & Fusions
    cna_raw = load_cna(cna_path, cna_top_n, cna_process, rename)
    cna_raw = safe_map_index(cna_raw, study, "CNA")

    fusion_raw = load_fusions_raw(fusion_info_path) if fusion_info_path else None
    if fusion_raw is not None:
        fusion_raw = safe_map_index(fusion_raw, study, "Fusions")
        if not _nonempty(fusion_raw):
            print("[INFO] No valid fusions after mapping; dropping fusion modality.")
            fusion_raw = None
    else:
        print("[INFO] No fusion file provided; skipping fusions.")

    # --- 4. Intersections
    modalities = {"mut": mut_raw, "cna": cna_raw, "clin": clinical}
    if fusion_raw is not None:
        modalities["fus"] = fusion_raw

    common = None
    for name, df in modalities.items():
        if not _nonempty(df) and name != "clin":
            print(f"[WARN] {name} modality has no data; ignored for intersection.")
            continue
        idx = df.index if isinstance(df, pd.DataFrame) else pd.Index([])
        common = idx if common is None else common.intersection(idx)

    if common is None or len(common) == 0:
        print("[DEBUG] Zero overlap after ID coercion. Pairwise overlaps:")
        keys = list(modalities.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ai = modalities[keys[i]].index if _nonempty(modalities[keys[i]]) else pd.Index([])
                bj = modalities[keys[j]].index if _nonempty(modalities[keys[j]]) else pd.Index([])
                print(f"  {keys[i]} ∩ {keys[j]}: {len(ai.intersection(bj))}")
        raise ValueError("No shared samples across modalities after ID coercion.")

    print(f"[INFO] Shared sample count: {len(common)}")

    # --- 5. Subset & normalize
    mut_df = mut_raw.loc[common]
    cna_df = np.power(2, cna_raw.loc[common]) if is_tcga else cna_raw.loc[common]
    clinical = clinical.loc[common]
    fusion_df = fusion_raw.loc[common] if fusion_raw is not None else pd.DataFrame(index=common)

    # --- 6. Frequency filters
    if mut_df.shape[0] < 100:
        mut_df = mut_df.loc[:, (mut_df > 0).sum() >= 5]
    else:
        mut_df = mut_df.loc[:, (mut_df > 0).mean() >= mut_freq_thresh]
        top_300 = (mut_df > 0).mean().sort_values(ascending=False).head(300).index
        mut_df = mut_df[top_300]

    if _nonempty(fusion_df):
        if fusion_df.shape[0] < 100:
            fusion_df = fusion_df.loc[:, (fusion_df > 0).sum() >= 5]
        else:
            freq = (fusion_df > 0).mean()
            fusion_df = fusion_df.loc[:, freq[freq >= fusion_freq_thresh].index]
        if fusion_df.shape[1] == 0:
            print("[WARN] No fusions passed frequency filter.")
            fusion_df = pd.DataFrame(index=common)

    return mut_df, cna_df, fusion_df, clinical

