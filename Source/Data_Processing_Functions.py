import pandas as pd
import numpy as np


### ======================== ###
### Real Data preprocessing ####
### ======================== ###


def to_patient_id(index, study=None):
    index = index.astype(str).str.strip()
    if study == 'TCGA':
        return index.str.extract(r'(TCGA-\w{2}-\w{4})')[0]
    elif study == 'TARGET':
        return index.str.extract(r'(TARGET-\d{2}-[A-Z0-9]{6})')[0]
    else:
        return index


### Mutation Loader ###
def load_mutations_raw(path, rename=True):
    df = pd.read_csv(path, sep=None, engine='python', index_col=0).astype(int)
    if rename:
        df.columns = [f"{gene}_MUT" for gene in df.columns]
    return df


### CNA Loader ###
def load_cna(path, top_n=500, cna_process=True, rename=True):
    df = pd.read_csv(path, sep=None, engine='python', index_col=0)

    if cna_process:
        # Drop any columns like Entrez_ID before transposing
        df = df.drop(columns=[c for c in df.columns if 'Entrez' in c], errors='ignore')

        # Remove any genes with ambiguous names like "GENE1|GENE2"
        df = df.loc[:, ~df.columns.str.contains('\|', regex=True)]
        
        # Transpose: samples become rows, genes become columns
        df = df.T

    if rename:
        df.columns = [f"{gene}_CNA" for gene in df.columns]
    
    # Keep top N most variable CNA features
    top_vars = df.var().sort_values(ascending=False).head(top_n).index
    return df[top_vars]


### Fusion Loader ###
def load_fusions_raw(path):
    import pandas as pd

    sv_df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
    sv_df.columns = sv_df.columns.str.strip()

    # Determine Patient/Sample ID
    if "Sample_Id" in sv_df.columns:
        sv_df["Patient_Id"] = sv_df["Sample_Id"].astype(str)
    elif "Unnamed: 0" in sv_df.columns:
        sv_df["Patient_Id"] = sv_df["Unnamed: 0"].astype(str)
    else:
        raise ValueError("No sample ID column found in fusion file.")

    # Extract 5' and 3' gene names in original order
    if "Site1_Hugo_Symbol" not in sv_df.columns or "Site2_Hugo_Symbol" not in sv_df.columns:
        raise ValueError("Missing required columns: 'Site1_Hugo_Symbol' and/or 'Site2_Hugo_Symbol'")
    
    gene1 = sv_df["Site1_Hugo_Symbol"].astype(str).str.strip()
    gene2 = sv_df["Site2_Hugo_Symbol"].astype(str).str.strip()
    sv_df["Fusion_Gene"] = gene1 + "--" + gene2  # Preserve order (5' → 3')

    # Drop any rows with missing patient ID
    sv_df = sv_df.dropna(subset=["Patient_Id"])

    # One-hot encode and aggregate
    dummies = pd.get_dummies(sv_df["Fusion_Gene"])
    dummies["Patient_Id"] = sv_df["Patient_Id"]
    fusion_df = dummies.groupby("Patient_Id").sum()

    fusion_df.columns = [f"{col}_FUSION" for col in fusion_df.columns]

    return fusion_df


### Subtype Handler ###
def process_subtypes(sample_info, min_samples=5):
    """
    Process clinical DataFrame to extract a single 'Subtype' column with sample IDs as index.
    Filters subtypes with fewer than `min_samples` samples.
    """
    if sample_info is None or sample_info.empty:
        raise ValueError("[process_subtypes] Input DataFrame is empty or None.")

    # Standardize column names
    sample_info.columns = sample_info.columns.str.strip().str.upper().str.replace(" ", "_")

    subtype_candidates = [
        "CANCER_SUBTYPE_CURATED", "ONCOTREE_CODE", "ONCOTREE", "SUBTYPE",
        "DISEASE_SUBTYPE", "DISEASE", "CANCER_TYPE", "CANCER_TYPE_DETAILED"
    ]

    for col in subtype_candidates:
        if col in sample_info.columns:
            sample_info = sample_info[[col]].copy()
            sample_info.columns = ["Subtype"]
            break
    else:
        raise ValueError(f"[process_subtypes] No valid subtype column found in: {sample_info.columns.tolist()}")

    sample_info = sample_info.dropna(subset=["Subtype"])
    subtype_counts = sample_info["Subtype"].value_counts()
    valid_subtypes = subtype_counts[subtype_counts >= min_samples].index
    sample_info = sample_info[sample_info["Subtype"].isin(valid_subtypes)]

    return sample_info


def read_clinical_file(path):
    import pandas as pd

    for sep_try in [None, '\t', ',', r'\s+']:
        try:
            df = pd.read_csv(path, sep=sep_try, engine='python', header=0, index_col=0)
            df.index = df.index.astype(str).str.strip().str.upper()
            df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")
            return df
        except Exception as e:
            continue

    raise ValueError(f"[read_clinical_file] Failed to load clinical file: {path}")


### Safe ID Mapping ###
def safe_map_index(df, study, name):
    mapped = to_patient_id(df.index.to_series(), study=study)
    df = df[mapped.notna()].copy()
    df.index = mapped[mapped.notna()].str.strip()
    print(f"[{name}] Mapped {len(df)} samples")
    return df


### Main Integration Function ###
def integrate_data(
    mut_path,
    cna_path,
    fusion_info_path,
    subtype_clinical_path,
    study=None,
    cna_process=True,
    cna_top_n=500,
    min_subtype_n=5,
    mut_freq_thresh=0.02,
    fusion_freq_thresh=0.02,
    is_tcga=True,
    rename=True,
):

    mut_raw = load_mutations_raw(mut_path)
    cna_raw = load_cna(cna_path, cna_top_n, cna_process,rename=rename)
    fusion_raw = load_fusions_raw(fusion_info_path) if fusion_info_path else None
    clinical = read_clinical_file(subtype_clinical_path)
    clinical = process_subtypes(clinical)

    # Mapping IDs
    mut_raw = safe_map_index(mut_raw, study, "Mutations")
    cna_raw = safe_map_index(cna_raw, study, "CNA")
    clinical = safe_map_index(clinical, study, "Clinical")
    if fusion_raw is not None:
        fusion_raw = safe_map_index(fusion_raw, study, "Fusions")

    print("Mut ∩ CNA:", len(mut_raw.index.intersection(cna_raw.index)))
    print("Mut ∩ Clinical:", len(mut_raw.index.intersection(clinical.index)))
    print("CNA ∩ Clinical:", len(cna_raw.index.intersection(clinical.index)))
    print("[DEBUG] Mutation mapped samples:", sorted(mut_raw.index.tolist())[:10])
    print("[DEBUG] CNA mapped samples:", sorted(cna_raw.index.tolist())[:10])
    print("[DEBUG] Fusion mapped samples:", sorted(fusion_raw.index.tolist())[:10] if fusion_raw is not None else "None")
    print("[DEBUG] Clinical mapped samples:", sorted(clinical.index.tolist())[:10])

    # Shared samples
    modalities = [mut_raw, cna_raw, clinical]
    include_fusions = fusion_raw is not None and isinstance(fusion_raw, pd.DataFrame) \
    and fusion_raw.shape[0] > 0 and fusion_raw.shape[1] > 0
    if include_fusions:
        modalities.append(fusion_raw)
    else:
        fusion_raw = pd.DataFrame(index=clinical.index)  # Dummy frame to avoid downstream errors
        print("[INFO] No valid fusions found. Proceeding without fusion data.")

    common = modalities[0].index
    for df in modalities[1:]:
        common = common.intersection(df.index)

    if len(common) == 0:
        raise ValueError("No shared samples across modalities. Check ID mapping or thresholds.")

    print(f"[integrate_data] Final shared sample count: {len(common)}")

    # Subset + log transform
    mut_df = mut_raw.loc[common]
    cna_df = np.power(2, cna_raw.loc[common]) if is_tcga else cna_raw.loc[common]
    clinical = clinical.loc[common]
    fusion_df = fusion_raw.loc[common] if fusion_raw is not None else pd.DataFrame(index=common)

    # Frequency filter
    if mut_df.shape[0] < 100:
        mut_df = mut_df.loc[:, (mut_df > 0).sum() >= 5]
    else:
        mut_df = mut_df.loc[:, (mut_df > 0).mean() >= mut_freq_thresh]
        top_300 = (mut_df > 0).mean().sort_values(ascending=False).head(300).index
        mut_df = mut_df[top_300]

    if include_fusions:
        if fusion_df.shape[0] < 100:
            fusion_df = fusion_df.loc[:, (fusion_df > 0).sum() >= 5]
        else:
            fusion_freq = (fusion_df > 0).sum(axis=0) / len(fusion_df)
            fusion_keep = fusion_freq[fusion_freq >= fusion_freq_thresh].index
            fusion_df = fusion_df.loc[:, fusion_keep]

        if fusion_df.shape[1] == 0:
            print("[Warning] No fusions passed frequency threshold. Dropping fusion modality.")
            fusion_df = pd.DataFrame(index=common)
    else:
        fusion_df = pd.DataFrame(index=common)

    # Rename to Sample_1, Sample_2, ...
    sample_names = [f"Sample_{i+1}" for i in range(len(common))]
    for df in (mut_df, cna_df, fusion_df, clinical):
        df.index = sample_names

    return mut_df, cna_df, fusion_df, clinical
