import pandas as pd
import numpy as np
import pytest
import Data_Processing_Functions as dpf


# ----------------------------
# classify_variant
# ----------------------------
@pytest.mark.parametrize(
    "gene,disease,variant_class,expected",
    [
        # Drop classes
        ("TP53", "AML", "Silent", "Drop"),
        # Special cases
        ("NOTCH1", "ALL", "Missense_Mutation", "GOF"),
        ("NOTCH1", "T-ALL", "Nonsense_Mutation", "GOF"),
        ("NPM1", "AML", "Frame_Shift_Ins", "GOF"),
        ("SF3B1", "AML", "Missense_Mutation", "GOF"),
        # General oncogene logic
        ("KRAS", "COAD", "Missense_Mutation", "GOF"),
        ("KRAS", "COAD", "Frame_Shift_Del", "Unclear"),
        # General TSG logic
        ("TP53", "OV", "Nonsense_Mutation", "LOF"),
        ("TP53", "OV", "Splice_Site", "LOF"),
        ("TP53", "OV", "Missense_Mutation", "Unclear"),
        # Unknown gene default
        ("SOMEGENE", "AML", "Missense_Mutation", "Unclear"),
    ],
)
def test_classify_variant(gene, disease, variant_class, expected):
    assert dpf.classify_variant(gene, disease, variant_class) == expected


# ----------------------------
# maf_to_onehot
# ----------------------------
def test_maf_to_onehot_gof_lof_and_mut_filters():
    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S1", "S2", "S3", "S3", "S4"],
            "Hugo_Symbol": ["KRAS", "TP53", "KRAS", "PIK3CA", "PIK3CA", "KRAS"],
            "Functional_Label": ["GOF", "LOF", "GOF", "Unclear", "Unclear", "Unclear"],
        }
    )

    out = dpf.maf_to_onehot(
        maf,
        sample_col="Tumor_Sample_Barcode",
        func_col="Functional_Label",
        include_uncertain=True,
        uncertain_top_k=100,
        min_uncertain_freq=0.0,
        uncertain_labels={"Unclear"},
    )

    # GOF/LOF columns exist
    assert "KRAS_GOF" in out.columns
    assert "TP53_LOF" in out.columns

    # Passenger _MUT for KRAS should be dropped because KRAS already functional (GOF)
    assert "KRAS_MUT" not in out.columns

    # PIK3CA is only uncertain -> should become _MUT
    assert "PIK3CA_MUT" in out.columns

    # Values are 0/1 ints
    assert set(np.unique(out.to_numpy())).issubset({0, 1})


def test_maf_to_onehot_uncertain_top_k_limits_features():
    # 3 uncertain genes; top_k=1 should keep only one MUT feature
    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S2", "S3", "S1", "S2", "S3"],
            "Hugo_Symbol": ["G1", "G1", "G1", "G2", "G2", "G3"],
            "Functional_Label": ["Unclear"] * 6,
        }
    )
    out = dpf.maf_to_onehot(
        maf,
        func_col="Functional_Label",
        include_uncertain=True,
        uncertain_top_k=1,
        min_uncertain_freq=0.0,
        uncertain_labels={"Unclear"},
    )

    mut_cols = [c for c in out.columns if c.endswith("_MUT")]
    assert len(mut_cols) == 1


# ----------------------------
# load_cna
# ----------------------------
def test_load_cna_transposes_and_renames(tmp_path):
    # Simulate TCGA-style CNA file: genes as rows, samples as columns
    cna = pd.DataFrame(
        {
            "S1": [2, 1],
            "S2": [0, -2],
            "Entrez_Gene_Id": [111, 222],
        },
        index=["GENE1", "GENE2"],
    )
    p = tmp_path / "cna.tsv"
    cna.to_csv(p, sep="\t")

    df = dpf.load_cna(str(p), cna_process=True, rename=True)

    # After processing: samples as rows, genes as columns with _CNA suffix
    assert list(df.index) == ["S1", "S2"]
    assert "GENE1_CNA" in df.columns
    assert "GENE2_CNA" in df.columns
    assert "Entrez_Gene_Id" not in df.columns


# ----------------------------
# load_fusions_raw
# ----------------------------
def test_load_fusions_raw_parses_fusion_name_and_suffixes(tmp_path):
    sv = pd.DataFrame(
        {
            "sample_id": ["P1", "P1", "P2"],
            "fusion_name": ["ETV6--RUNX1", "ETV6--RUNX1", "BCR--ABL1"],
        }
    )
    p = tmp_path / "fusions.csv"
    sv.to_csv(p, index=False)

    fus = dpf.load_fusions_raw(str(p))

    assert "ETV6--RUNX1_FUSION" in fus.columns
    assert "BCR--ABL1_FUSION" in fus.columns
    assert fus.loc["P1", "ETV6--RUNX1_FUSION"] == 2  # counted twice then grouped


def test_load_fusions_raw_raises_if_missing_symbols(tmp_path):
    sv = pd.DataFrame({"Sample_Id": ["P1"], "foo": ["bar"]})
    p = tmp_path / "bad.csv"
    sv.to_csv(p, index=False)

    with pytest.raises(ValueError):
        dpf.load_fusions_raw(str(p))


# ----------------------------
# to_patient_id / to_patient_index / nonempty
# ----------------------------
def test_to_patient_id_tcga_extraction():
    idx = pd.Index(["TCGA-AB-1234-01A", "TCGA-ZZ-9999", "NOT_TCGA"])
    out = dpf.to_patient_id(idx, study="TCGA")
    assert out.iloc[0] == "TCGA-AB-1234"
    assert out.iloc[1] == "TCGA-ZZ-9999"
    assert pd.isna(out.iloc[2])


def test_to_patient_index_tcga_slices_to_12():
    df = pd.DataFrame({"A": [1, 2]}, index=["TCGA-AB-1234-01A", "TCGA-AB-1234-02A"])
    out = dpf.to_patient_index(df, study="TCGA")
    assert list(out.index) == ["TCGA-AB-1234", "TCGA-AB-1234"]


def test_nonempty():
    assert dpf.nonempty(pd.DataFrame({"A": [1]}))
    assert not dpf.nonempty(pd.DataFrame())
    assert not dpf.nonempty(None)


# ----------------------------
# select_known_clinicals
# ----------------------------
def test_select_known_clinicals_matches_patterns():
    clin = pd.DataFrame(
        {
            "diagnosis_age": [10, 12],
            "Sex": ["F", "M"],
            "some_other": [1, 2],
        },
        index=["S1", "S2"],
    )
    out = dpf.select_known_clinicals(clin, "AML")
    # AML patterns include DIAGNOSIS_AGE and SEX (case-insensitive match)
    assert any("diagnosis_age" == c for c in out.columns)
    assert any("Sex" == c for c in out.columns)
    assert "some_other" not in out.columns


def test_select_known_clinicals_unknown_cancer_raises():
    clin = pd.DataFrame({"A": [1]}, index=["S1"])
    with pytest.raises(ValueError):
        dpf.select_known_clinicals(clin, "NOTACANCER")


# ----------------------------
# preprocess_rna_for_simulation
# ----------------------------
def test_preprocess_rna_auto_scale_and_nonnegative():
    rna = pd.DataFrame(
        {"G1": [0.1, 0.2, -0.5], "G2": [0.0, 0.0, 0.0]},
        index=["S1", "S2", "S3"],
    )
    scaled, sf = dpf.preprocess_rna_for_simulation(rna, strategy="auto", verbose=False)
    assert isinstance(sf, (int, float))
    assert (scaled.to_numpy() >= 0).all()
    assert scaled.dtypes.nunique() == 1
    assert str(scaled.dtypes[0]).startswith("int")


def test_preprocess_rna_manual_requires_scale():
    rna = pd.DataFrame({"G1": [1.0]}, index=["S1"])
    with pytest.raises(ValueError):
        dpf.preprocess_rna_for_simulation(rna, strategy="manual", user_scale=None, verbose=False)


# ----------------------------
# select_genes_with_expr_filter
# ----------------------------
def test_select_genes_with_expr_filter_keeps_altered_genes_even_if_low_expr():
    # RNA counts-like matrix
    rna = pd.DataFrame(
        {
            "ALTERED": [0, 0, 0, 0, 0],  # low expression
            "G2": [100, 120, 90, 110, 80],
            "G3": [60, 55, 70, 65, 75],
            "G4": [10, 11, 9, 12, 8],
        },
        index=[f"S{i}" for i in range(5)],
    )

    alts = pd.DataFrame(
        {"ALTERED_GOF": [1, 0, 0, 0, 1]},
        index=rna.index,
    )

    res = dpf.select_genes_with_expr_filter(
        rna_df=rna,
        alterations_df=alts,
        target_total=2,  # target smaller than altered genes count -> should still keep altered
        min_cpm=1.0,
        min_prop_samples=0.8,
        verbose=False,
    )

    assert "ALTERED" in res["genes_to_keep"]
    assert "ALTERED" in res["altered_genes_kept"]
    assert "ALTERED" in res["low_expr_altered_flagged"]


