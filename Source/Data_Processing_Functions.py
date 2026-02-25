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
from sklearn.preprocessing import StandardScaler
import re

# ============================================================
# 1. Default Gene Sets
# ============================================================

# Default genes - Common Oncogenes and TSGs which occur across multiple cancer types.
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


#DIfferent Mutation Types
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

def classify_variant(gene, disease, variant_class):
    """Return GOF, LOF, Unclear, or Drop classification for a single variant."""
    # Logic for classifying variants, using the default and disease specfic genes as well as variant types.
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
    maf_annot,
    sample_col = "Tumor_Sample_Barcode",
    func_col = "Functional_Label",
    include_uncertain = True,
    uncertain_top_k = 100, # max 100 uncertain (_MUT) allowed
    min_uncertain_freq = 0.02, #_MUT must have a min 2% freq to pass
    uncertain_labels = None,
):
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

def load_cna(path, cna_process=True, rename= False):
    """Load and preprocess a CNA file (TCGA-style)."""
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    if cna_process:
        df = df.drop(columns=[c for c in df.columns if "Entrez" in c], errors="ignore")
        df = df.loc[:, ~df.columns.str.contains(r"\|", regex=True)]
        df = df.T
    if rename:
        df.columns = [f"{g}_CNA" for g in df.columns]

    return df


def load_fusions_raw(path):
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


def read_clinical_file(path):
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


def read_rna_file(path):
    """Try reading a clinical file with flexible delimiters."""
    for sep_try in [None, "\t", ",", r"\s+"]:
        try:
            df = pd.read_csv(path, sep=sep_try, engine="python", header=0, index_col=0)
            return df
        except Exception:
            continue
    raise ValueError(f"[read_clinical_file] Failed to load clinical file: {path}")


def process_subtypes(sample_info, min_samples = 5) ->:
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

def to_patient_index(df, study):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    idx = df.index.astype(str).str.strip().str.upper()
    if study and study.upper() == "TCGA":
        idx = idx.str.slice(0, 12)
    out = df.copy()
    out.index = idx
    return out


def nonempty(df):
    return isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0


def to_patient_id(index, study):
    index = index.astype(str).str.strip()
    if study == "TCGA":
        return index.str.extract(r"(TCGA-\w{2}-\w{4})")[0]
    elif study == "TARGET":
        return index.str.extract(r"(TARGET-\d{2}-[A-Z0-9]{6})")[0]
    return index


def safe_map_index(df, study, name):
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
    mut_path,
    cna_path,
    fusion_info_path:,
    patient_path:,
    sample_path:,
    rna_path:, 
    study = None,
    disease = None,
    cna_process = True,
    is_tcga=False,
    cna_top_n = 200,
    min_subtype_n = 3,
    mut_freq_thresh = 0.02,
    fusion_freq_thresh = 0.02,
    rename = True,
    input_format = "maf",  # "maf" | "custom_tsv" | "onehot"
    uncertain_top_k = 100,
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
            lambda r: classify_variant(r["Hugo_Symbol"], disease, r["Variant_Classification"]), axis=1
        )

        mut_raw = maf_to_onehot(maf, uncertain_top_k=uncertain_top_k)
        mut_raw.index = to_patient_id(mut_raw.index.to_series(), study=study).str.strip().str.upper()
        mut_raw = to_patient_index(mut_raw[mut_raw.index.notna()], study)

    # --- 3. CNA & Fusions & RNA
    cna_raw = load_cna(cna_path, cna_process, rename)
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
    modalities = {"mut": mut_raw, "cna": cna_raw, "clin": clinical, 'RNA':rna}
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
    clinical_df= pd.concat([clinical_start, sample], axis=1, join='inner')

    # --- 6. Frequency filters
    if cna_top_n is not None and cna_top_n > 0:
        # integer CN, assume diploid = 2
        alt_freq = (cna_df != cna_neutral).mean(axis=0)  # fraction of samples altered (up or down)
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




# ============================================================
# 8. Clinical Feature Selection
# ============================================================


CLIN_FEATURES = {
    "AML": [
        "SUBTYPE", "TCGA_PANCANATLAS_CANCER_TYPE_ACRONYM",
        "DIAGNOSIS_AGE", "SEX", "RACE_CATEGORY", "ETHNICITY_CATEGORY",
        "TMB_(NONSYNONYMOUS)", "ANEUPLOIDY_SCORE", "TUMOR_BREAK_LOAD",
        "NEOPLASM_HISTOLOGIC_GRADE", "SAMPLE_TYPE"
    ],
    "ALL": [
        "Cancer Subtype Curated",
        "Cancer Type Detailed",
        "Oncotree Code",
        "Site of Sample",
        "Sex",
        "Reported Ethnicity",
        "Age"
    ],

    "IBC": [
        "SUBTYPE", "TCGA_PANCANATLAS_CANCER_TYPE_ACRONYM",
        "DIAGNOSIS_AGE", "SEX", "RACE_CATEGORY",
        "NEOPLASM_HISTOLOGIC_GRADE",
        "NEOPLASM_DISEASE_STAGE_AMERICAN_JOINT_COMMITTEE_ON_CANCER_CODE",
        "AMERICAN_JOINT_COMMITTEE_ON_CANCER_TUMOR_STAGE_CODE",
        "NEOPLASM_DISEASE_LYMPH_NODE_STAGE_AMERICAN_JOINT_COMMITTEE_ON_CANCER_CODE",
        "TMB", "ANEUPLOIDY_SCORE",
        "MSI_MANTIS_SCORE", "MSISENSOR_SCORE",
        "SAMPLE_TYPE"
    ],
    "OV": [
        "SUBTYPE", "TCGA_PANCANATLAS_CANCER_TYPE_ACRONYM",
        "DIAGNOSIS_AGE", "SEX", "RACE_CATEGORY",
        "NEOPLASM_HISTOLOGIC_GRADE",
        "NEOPLASM_DISEASE_STAGE_AMERICAN_JOINT_COMMITTEE_ON_CANCER_CODE",
        "ANEUPLOIDY_SCORE", "TMB", "TUMOR_BREAK_LOAD",
        "MSI_MANTIS_SCORE", "MSISENSOR_SCORE",
        "SAMPLE_TYPE"
    ],
    "COAD": [
        "SUBTYPE", "DIAGNOSIS_AGE", "SEX", "RACE_CATEGORY",
        "TUMOR_DISEASE_ANATOMIC_SITE", "NEOPLASM_HISTOLOGIC_GRADE",
        "NEOPLASM_DISEASE_STAGE_AMERICAN_JOINT_COMMITTEE_ON_CANCER_CODE",
        "MSI", "MSISENSOR", "MANTIS", "TMB", "ANEUPLOIDY_SCORE",
        "SAMPLE_TYPE"
    ]
}

def select_known_clinicals(clin_df, cancer):
    """
    Subset the clinical dataframe to keep only 'known at diagnosis' variables
    relevant for the given cancer type.

    Parameters
    ----------
    clin_df : pd.DataFrame
        Clinical dataframe (rows = samples)
    cancer : str
        One of {'AML','ALL','IBC','OV','COAD'} (case-insensitive)

    Returns
    -------
    pd.DataFrame : filtered dataframe with only known baseline clinicals
    """
    cancer = cancer.upper()
    if cancer not in CLIN_FEATURES:
        raise ValueError(f"Unknown cancer type '{cancer}'. Must be one of {list(CLIN_FEATURES.keys())}")

    keep_patterns = [re.compile(k, re.I) for k in CLIN_FEATURES[cancer]]
    matched_cols = []
    for c in clin_df.columns:
        for pat in keep_patterns:
            if pat.search(c):
                matched_cols.append(c)
                break

    matched_cols = sorted(set(matched_cols))
    if not matched_cols:
        print(f"[Warning] No baseline clinicals matched for {cancer}. Returning empty DataFrame.")
        return pd.DataFrame(index=clin_df.index)

    return clin_df.loc[:, matched_cols]


# ============================================================
# 9. Altertaion and Feature Encoding
# ============================================================


def encode_altertaions_clinical(mutation_df, cna_df, fusion_df, clinical_df, rna_df, disease):

    # --- Select and clean clinical variables ---
    clin_all_filtered = select_known_clinicals(clinical_df, disease)
    print("Kept columns:", list(clin_all_filtered.columns))

    # Apply 25% missingness threshold
    threshold = 0.25
    clin_filtered = clin_all_filtered.loc[:, clin_all_filtered.isna().mean() <= threshold].copy()

    # Fill remaining NaNs (categorical = 'Unknown', numeric = 0)
    for col in clin_filtered.columns:
        if clin_filtered[col].dtype == object:
            clin_filtered[col] = clin_filtered[col].fillna("Unknown")

    print(f"Kept {clin_filtered.shape[1]} columns out of {clin_all_filtered.shape[1]}")
    print(f"Remaining NaNs: {clin_filtered.isna().sum().sum()}")

    # ---Align all data types by sample ---
    bin_alt_real = pd.concat(
        [mutation_df, fusion_df, cna_df, clin_filtered],
        axis=1, join='inner'
    )

    # Align to RNA expression samples
    common_samples = bin_alt_real.index.intersection(rna_df.index)
    X_aligned = bin_alt_real.loc[common_samples].sort_index()
    Y_aligned = rna_df.loc[common_samples].sort_index()

    print(f"Aligned shapes → X: {X_aligned.shape}, Y: {Y_aligned.shape}")

    possible_numeric = ["DIAGNOSIS_AGE", "TMB", "MSISENSOR_SCORE", "MSI_MANTIS_SCORE", 'Age','ANEUPLOIDY_SCORE', 'TUMOR_BREAK_LOAD', 'TMB_(NONSYNONYMOUS)']

    for col in possible_numeric:
        if col in X_aligned.columns:
            X_aligned[col] = pd.to_numeric(X_aligned[col], errors="coerce")


    # ---Encode categorical variables and scale everything ---
    non_numeric_cols = X_aligned.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X_aligned.select_dtypes(exclude=["object", "category"]).columns

    print(f"Encoding {len(non_numeric_cols)} non-numeric columns")

    # One-hot encode only categorical columns
    X_encoded = pd.get_dummies(X_aligned, columns=non_numeric_cols, drop_first=False, dtype=float)
    print(f"Remaining NaNs: {X_encoded.isna().sum().sum()}")

    # Scale all features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_encoded)

    # Rebuild DataFrame
    Xs = pd.DataFrame(Xs, index=X_encoded.index, columns=X_encoded.columns)
    Xs = Xs.fillna(0.0)
    print(f"Remaining NaNs: {Xs.isna().sum().sum()}")
    print("Final Xs shape:", Xs.shape)
    
    return Xs


# ============================================================
# 10. PRE-RNA SIMULATION PROCESSING
# ============================================================


def preprocess_rna_for_simulation(rna_df, strategy="auto", user_scale=None, verbose=True):
    """
    Preprocess RNA-seq expression data to convert from normalized form (e.g., CPM, RPKM, RSEM)
    into pseudo-counts for negative binomial simulation.
    """

    if strategy == "manual":
        if user_scale is None:
            raise ValueError("You must provide user_scale if using manual strategy.")
        scale_factor = user_scale
    elif strategy == "auto":
        q75 = rna_df.quantile(0.75).median()
        if q75 < 1:
            scale_factor = 100
        elif q75 < 10:
            scale_factor = 50
        elif q75 < 100:
            scale_factor = 10
        else:
            scale_factor = 1
    else:
        raise ValueError("strategy must be 'auto' or 'manual'")

    # Apply pseudocounting and scaling
    # Multiply expression and round to int
    rna_scaled = (rna_df * scale_factor).round().astype(int)
    
    # NB counts cannot be negative
    rna_scaled[rna_scaled < 0] = 0

    if verbose:
        print(f"[preprocess_rna] Applied scale factor: {scale_factor}")

    return rna_scaled, scale_factor


def select_genes_with_expr_filter(
    rna_df,
    alterations_df,
    target_total= 10_000,
    min_cpm= 1.0,
    min_prop_samples = 0.20,   # keep gene if expressed in ≥20% samples
    use_mad = False,            # alternative variability metric, if True, rank by MAD of log1p-CPM instead of variance
    verbose = False
):

    # ---------- 0) Clean inputs ----------
    # Ensure gene columns are strings
    rna_df = rna_df.copy()
    rna_df.columns = rna_df.columns.astype(str)
    alterations_df = alterations_df.copy()
    alterations_df.columns = alterations_df.columns.astype(str)

    # ---------- 1) Identify altered genes present in RNA ----------
    altered_genes = list({c.split('_')[0] for c in alterations_df.columns})
    altered_in_expr = [g for g in altered_genes if g in rna_df.columns]

    # ---------- 2) Decide if rna_df looks like raw counts ----------
    # Heuristic: if most values are non-negative integers and max >= 50, treat as counts
    values = rna_df.to_numpy()
    nonneg = (values >= 0).mean() > 0.999
    int_like = (np.isclose(values, np.round(values)).mean() > 0.98)
    looks_like_counts = bool(nonneg and int_like and np.nanmax(values) >= 50)

    # ---------- 3) Library-size normalization & log1p-CPM ----------
    if looks_like_counts:
        lib_sizes = rna_df.sum(axis=1).replace(0, np.nan)  # avoid div by 0
        cpm = (rna_df.div(lib_sizes, axis=0)) * 1e6
        log_cpm = np.log1p(cpm)
        # Expression filter: CPM >= min_cpm in ≥ min_prop_samples of samples
        expressed_mask = (cpm >= min_cpm).mean(axis=0) >= min_prop_samples
    else:
        # Already TPM / FPKM / log2(TPM+1) style. Use a small floor for expression.
        # Treat values > 0.1 as "expressed" (tune if needed).
        log_cpm = np.log1p(rna_df)  # still stabilize before variance/MAD
        expressed_mask = (rna_df > 0.1).mean(axis=0) >= min_prop_samples

    # ---------- 4) Keep all altered genes regardless of expression ----------
    # But we’ll still report which altered genes were lowly expressed.
    low_expr_altered = [g for g in altered_in_expr if not expressed_mask.get(g, False)]

    # Genes eligible for variability ranking = non-altered & pass expression filter
    non_altered = [g for g in rna_df.columns if g not in altered_in_expr]
    non_alt_keep = [g for g in non_altered if expressed_mask.get(g, False)]

    # ---------- 5) Rank by variability among eligible non-altered genes ----------
    if len(non_alt_keep) > 0:
        X = log_cpm[non_alt_keep]  # samples x kept genes
        if use_mad:
            # Median absolute deviation across samples, per gene
            med = X.median(axis=0)
            variability = (X.sub(med, axis=1)).abs().median(axis=0)
        else:
            # Sample variance across samples, per gene
            variability = X.var(axis=0, ddof=1)
        variability = variability.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ranked_non_altered = variability.sort_values(ascending=False).index.tolist()
    else:
        ranked_non_altered = []

    # ---------- 6) Decide how many non-altered to take ----------
    # We *must* keep all altered genes that exist in RNA.
    n_altered = len(altered_in_expr)
    n_remaining = max(0, target_total - n_altered)

    # If altered genes already exceed target_total, we keep them all and add none
    top_non_altered = ranked_non_altered[:n_remaining] if n_remaining > 0 else []

    # ---------- 7) Compose final set ----------
    genes_to_keep = sorted(set(altered_in_expr + top_non_altered))

    # ---------- 8) Book-keeping / reporting ----------
    low_expr_non_altered = [g for g in non_altered if not expressed_mask.get(g, False)]
    genes_dropped_low_expr = sorted(low_expr_non_altered)  # altered are not dropped, just flagged

    if verbose:
        total_genes = rna_df.shape[1]
        kept_non_altered = len(top_non_altered)
        dropped_low_expr = len(genes_dropped_low_expr)
        print(f"[Gene selection]")
        print(f"  Total RNA genes: {total_genes}")
        print(f"  Altered genes present in RNA: {n_altered}")
        print(f"  Target total: {target_total} -> selecting {kept_non_altered} non-altered to fill")
        print(f"  Low-expression (non-altered) genes dropped: {dropped_low_expr}")
        if low_expr_altered:
            print(f"  NOTE: {len(low_expr_altered)} altered gene(s) are lowly expressed but KEPT: {', '.join(low_expr_altered[:10])}{' ...' if len(low_expr_altered) > 10 else ''}")

    return {
        "genes_to_keep": genes_to_keep,
        "altered_genes_kept": sorted(altered_in_expr),
        "low_expr_altered_flagged": sorted(low_expr_altered),
        "genes_dropped_low_expr": genes_dropped_low_expr
    }
