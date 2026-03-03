
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
