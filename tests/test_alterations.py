import numpy as np
import pandas as pd
import pytest

import alterations as alt


# ----------------------------
# estimate_cna_event_params
# ----------------------------
def test_estimate_cna_event_params_empty_returns_expected_columns():
    out = alt.estimate_cna_event_params(pd.DataFrame())
    assert list(out.columns) == ["p_amp", "p_del", "p_neu", "q_amp2", "q_del2"]
    assert out.index.name == "gene"
    assert out.empty


def test_estimate_cna_event_params_clips_rounds_and_computes_probs():
    # values include floats and out-of-range; should be rint + clip to [-2,2]
    cna = pd.DataFrame(
        {
            "G1": [0, 1.2, 2.7, -1.4, -3.0],  # -> [0,1,2,-1,-2]
            "G2": [0, 0, 0, 0, 0],
        },
        index=[f"S{i}" for i in range(5)],
    )

    out = alt.estimate_cna_event_params(cna)

    # For G1: amp fraction = 2/5, del fraction = 2/5, neu = 1/5
    assert out.loc["G1", "p_amp"] == pytest.approx(2 / 5)
    assert out.loc["G1", "p_del"] == pytest.approx(2 / 5)
    assert out.loc["G1", "p_neu"] == pytest.approx(1 / 5)

    # q_amp2 among amp entries: values >0 are [1,2] => q_amp2=1/2
    assert out.loc["G1", "q_amp2"] == pytest.approx(0.5)
    # q_del2 among del entries: values <0 are [-1,-2] => q_del2=1/2
    assert out.loc["G1", "q_del2"] == pytest.approx(0.5)

    # G2 is all neutral
    assert out.loc["G2", "p_neu"] == pytest.approx(1.0)
    assert out.loc["G2", "p_amp"] == pytest.approx(0.0)
    assert out.loc["G2", "p_del"] == pytest.approx(0.0)
    assert out.loc["G2", "q_amp2"] == pytest.approx(0.0)
    assert out.loc["G2", "q_del2"] == pytest.approx(0.0)


def test_estimate_cna_event_params_collapses_duplicate_gene_columns():
    # duplicate "G1" appears twice; function should collapse by mean then rint/clip
    cna = pd.DataFrame(
        np.array(
            [
                [2, 2, 0],
                [1, 2, 0],
                [-2, -1, 0],
                [0, 0, 0],
            ]
        ),
        columns=["G1", "G1", "G2"],
        index=["S1", "S2", "S3", "S4"],
    )

    out = alt.estimate_cna_event_params(cna)

    assert "G1" in out.index and "G2" in out.index
    # sanity: probabilities exist and sum ~1 for each gene
    for g in ["G1", "G2"]:
        s = out.loc[g, ["p_amp", "p_del", "p_neu"]].sum()
        assert s == pytest.approx(1.0)


# ----------------------------
# robust_scale_with_floor
# ----------------------------
def test_robust_scale_with_floor_constant_column_is_bounded():
    df = pd.DataFrame({"A": [5, 5, 5], "B": [0, 1, 2]}, index=["S1", "S2", "S3"])
    out = alt.robust_scale_with_floor(df, cna_std_floor=0.25)

    assert out.shape == df.shape
    assert list(out.columns) == ["A", "B"]

    # Constant column A should become all zeros (x - mean == 0)
    assert np.allclose(out["A"].values, 0.0)

    # Column B should be standardized (not necessarily unit variance due to floor, but finite)
    assert np.isfinite(out["B"].values).all()


def test_robust_scale_with_floor_applies_floor_not_zero_division():
    df = pd.DataFrame({"A": [0, 0, 0], "B": [1, 1, 2]}, index=["S1", "S2", "S3"])
    out = alt.robust_scale_with_floor(df, cna_std_floor=0.5)
    assert np.isfinite(out.to_numpy()).all()


# ----------------------------
# preprocess_X_weighted
# ----------------------------
def test_preprocess_X_weighted_returns_empty_if_no_blocks():
    combined, unscaled = alt.preprocess_X_weighted(mut=None, fusion=None, cna=None, clinical=None)
    assert isinstance(combined, pd.DataFrame)
    assert combined.empty
    assert isinstance(unscaled, dict)
    assert unscaled == {}


def test_preprocess_X_weighted_mut_fusion_weighting():
    mut = pd.DataFrame({"A_GOF": [1, 0], "B_LOF": [0, 1]}, index=["S1", "S2"])
    fus = pd.DataFrame({"F1_FUSION": [0, 1]}, index=["S1", "S2"])

    combined, unscaled = alt.preprocess_X_weighted(
        mut=mut,
        fusion=fus,
        cna=None,
        clinical=None,
        weights={"mut": 2.0, "fusion": 3.0, "cna": 2.0, "clinical": 0.5},
    )

    # Combined should contain weighted blocks with same columns (no scaling for mut/fusion)
    assert set(["A_GOF", "B_LOF", "F1_FUSION"]).issubset(combined.columns)
    assert combined.loc["S1", "A_GOF"] == pytest.approx(2.0)
    assert combined.loc["S2", "F1_FUSION"] == pytest.approx(3.0)

    # unscaled stored as float
    assert "mut" in unscaled and "fusion" in unscaled
    assert unscaled["mut"].dtypes["A_GOF"] == float


def test_preprocess_X_weighted_cna_creates_amp_del_views_and_collapses_dupes():
    # duplicate CNA col "G1_CNA" appears twice
    cna = pd.DataFrame(
        {
            "G1_CNA": [2, 0, -2],
            "G1_CNA": [2, 1, -1],  # duplicate key intentionally (last one wins in dict),
        },
        index=["S1", "S2", "S3"],
    )
    # The above dict would overwrite; build duplicates properly:
    cna = pd.DataFrame(
        np.array([[2, 2], [0, 1], [-2, -1]]),
        columns=["G1_CNA", "G1_CNA"],
        index=["S1", "S2", "S3"],
    )

    combined, unscaled = alt.preprocess_X_weighted(
        mut=None,
        fusion=None,
        cna=cna,
        clinical=None,
        weights={"mut": 1.0, "fusion": 1.5, "cna": 2.0, "clinical": 0.5},
        cna_std_floor=0.25,
    )

    # unscaled CNA should be discrete ints in [-2,2] with duplicates collapsed to one column
    assert "cna" in unscaled
    assert list(unscaled["cna"].columns) == ["G1_CNA"]
    assert set(np.unique(unscaled["cna"].to_numpy())).issubset({-2, -1, 0, 1, 2})

    # combined has AMP/DEL view columns
    assert "G1_CNA__AMP_LVL" in combined.columns
    assert "G1_CNA__DEL_LVL" in combined.columns

    # Weighted (cna weight=2.0): values should be finite
    assert np.isfinite(combined.to_numpy()).all()


def test_preprocess_X_weighted_clinical_numeric_only_scaled_but_unscaled_kept():
    clinical = pd.DataFrame(
        {
            "Age": [10, 12, np.nan],
            "Sex": ["F", "M", "F"],  # non-numeric
            "Batch": [1, 2, 3],
        },
        index=["S1", "S2", "S3"],
    )

    combined, unscaled = alt.preprocess_X_weighted(
        clinical=clinical,
        mut=None,
        fusion=None,
        cna=None,
        weights={"mut": 1.0, "fusion": 1.5, "cna": 2.0, "clinical": 0.5},
    )

    assert "clinical" in unscaled
    assert "Sex" in unscaled["clinical"].columns  # preserved unscaled

    # combined should NOT include non-numeric Sex
    assert "Sex" not in combined.columns
    # Age/Batch included and scaled (roughly z-scored then weighted)
    assert "Age" in combined.columns
    assert "Batch" in combined.columns
    assert np.isfinite(combined[["Age", "Batch"]].to_numpy()).all()


# ----------------------------
# sample_from_neighbors_ratioCNA
# ----------------------------
def _toy_blocks_for_sampling():
    idx = pd.Index([f"S{i}" for i in range(6)])
    mut = pd.DataFrame({"A_GOF": [1, 0, 1, 0, 0, 1]}, index=idx)
    fus = pd.DataFrame({"F1_FUSION": [0, 1, 0, 0, 1, 0]}, index=idx)
    cna = pd.DataFrame({"G1_CNA": [2, 1, 0, -1, -2, 0]}, index=idx)
    clin = pd.DataFrame({"SubtypeLike": ["X", "X", "Y", "Y", "Y", "X"]}, index=idx)
    combined, unscaled = alt.preprocess_X_weighted(mut=mut, fusion=fus, cna=cna, clinical=clin)
    return combined, unscaled


def test_sample_from_neighbors_ratioCNA_output_shape_and_columns():
    scaled, unscaled = _toy_blocks_for_sampling()
    out = alt.sample_from_neighbors_ratioCNA(
        scaled_df=scaled,
        unscaled_dfs=unscaled,
        n_samples=10,
        k_neighbors=3,
        seed=123,
    )

    # Expected column ordering: mut + fusion + cna + clinical (only those present)
    assert list(out.columns) == ["A_GOF", "F1_FUSION", "G1_CNA", "SubtypeLike"]
    assert out.shape[0] == 10

    # Binary columns are 0/1
    assert set(np.unique(out["A_GOF"].to_numpy())).issubset({0, 1})
    assert set(np.unique(out["F1_FUSION"].to_numpy())).issubset({0, 1})

    # CNA column in discrete states
    assert set(np.unique(out["G1_CNA"].to_numpy())).issubset({-2, -1, 0, 1, 2})


def test_sample_from_neighbors_ratioCNA_deterministic_given_seed():
    scaled, unscaled = _toy_blocks_for_sampling()

    out1 = alt.sample_from_neighbors_ratioCNA(scaled, unscaled, n_samples=5, k_neighbors=3, seed=7)
    out2 = alt.sample_from_neighbors_ratioCNA(scaled, unscaled, n_samples=5, k_neighbors=3, seed=7)

    pd.testing.assert_frame_equal(out1, out2)


def test_sample_from_neighbors_ratioCNA_handles_missing_blocks():
    # scaled_df still needed for KNN, but unscaled missing mut/fusion/clinical etc.
    idx = pd.Index(["S1", "S2", "S3"])
    scaled = pd.DataFrame({"X": [0.0, 1.0, 2.0]}, index=idx)
    unscaled = {"cna": pd.DataFrame({"G1_CNA": [0, 1, -1]}, index=idx)}

    out = alt.sample_from_neighbors_ratioCNA(scaled, unscaled, n_samples=4, k_neighbors=2, seed=1)
    assert list(out.columns) == ["G1_CNA"]
    assert out.shape == (4, 1)


# ----------------------------
# simulate_X_hybrid_ratioCNA
# ----------------------------
def test_simulate_X_hybrid_ratioCNA_returns_n_samples_and_subtype_column():
    idx = pd.Index([f"S{i}" for i in range(10)])
    mut = pd.DataFrame({"A_GOF": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]}, index=idx)
    fus = pd.DataFrame({"F1_FUSION": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]}, index=idx)
    cna = pd.DataFrame({"G1_CNA": [2, 1, 0, -1, -2, 0, 1, 0, -1, 2]}, index=idx)
    clin = pd.DataFrame({"Age": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}, index=idx)

    subtype = pd.DataFrame(
        {"Subtype": ["A"] * 6 + ["B"] * 4},
        index=idx,
    )

    X_sim = alt.simulate_X_hybrid_ratioCNA(
        mut=mut,
        fusion=fus,
        cna=cna,
        clinical=clin,
        subtype=subtype,
        n_samples=25,
        k_neighbors=3,
        seed=99,
    )

    assert X_sim.shape[0] == 25
    assert "Subtype" in X_sim.columns
    assert X_sim["Subtype"].isin(["A", "B"]).all()
    assert list(X_sim.index) == [f"Sample_{i+1}" for i in range(25)]


def test_simulate_X_hybrid_ratioCNA_deterministic_at_top_level_seed():
    # Note: internal per-subtype seeds are derived from RNG with top-level seed.
    idx = pd.Index([f"S{i}" for i in range(8)])
    mut = pd.DataFrame({"A_GOF": [1, 0, 1, 0, 0, 1, 0, 0]}, index=idx)
    subtype = pd.DataFrame({"Subtype": ["A"] * 4 + ["B"] * 4}, index=idx)

    X1 = alt.simulate_X_hybrid_ratioCNA(
        mut=mut, fusion=None, cna=None, clinical=None, subtype=subtype, n_samples=12, seed=123
    )
    X2 = alt.simulate_X_hybrid_ratioCNA(
        mut=mut, fusion=None, cna=None, clinical=None, subtype=subtype, n_samples=12, seed=123
    )
    pd.testing.assert_frame_equal(X1, X2)


# ----------------------------
# split_simulated_blocks_v2
# ----------------------------
def test_split_simulated_blocks_v2_splits_by_suffix_and_subtype():
    X = pd.DataFrame(
        {
            "KRAS_GOF": [1, 0],
            "TP53_LOF": [0, 1],
            "ETV6--RUNX1_FUSION": [1, 0],
            "G1_CNA": [2, -1],
            "G1__AMP_LVL": [2, 0],
            "Age": [10, 12],
            "Subtype": ["A", "B"],
        },
        index=["Sample_1", "Sample_2"],
    )

    mut_sim, fusion_sim, cna_sim, clin_sim, subtype_sim = alt.split_simulated_blocks_v2(X)

    assert list(mut_sim.columns) == ["KRAS_GOF", "TP53_LOF"]
    assert list(fusion_sim.columns) == ["ETV6--RUNX1_FUSION"]

    # CNA includes both _CNA and derived view suffixes
    assert set(cna_sim.columns) == {"G1_CNA", "G1__AMP_LVL"}
    assert list(clin_sim.columns) == ["Age"]
    assert subtype_sim.name == "Subtype"
    assert subtype_sim.tolist() == ["A", "B"]


def test_split_simulated_blocks_v2_handles_missing_subtype_column():
    X = pd.DataFrame(
        {
            "A_MUT": [1, 0],
            "B_FUSION": [0, 1],
            "C_CNA": [0, 2],
            "SomeClin": ["x", "y"],
        },
        index=["S1", "S2"],
    )
    mut_sim, fusion_sim, cna_sim, clin_sim, subtype_sim = alt.split_simulated_blocks_v2(X)

    assert mut_sim.shape[1] == 1
    assert fusion_sim.shape[1] == 1
    assert cna_sim.shape[1] == 1
    assert clin_sim.shape[1] == 1
    assert isinstance(subtype_sim, pd.Series)
    assert subtype_sim.empty or subtype_sim.isna().all()


def test_split_simulated_blocks_v2_detects_subtype_case_insensitive():
    X = pd.DataFrame(
        {
            "A_GOF": [1, 0],
            "SUBTYPE": ["X", "Y"],
            "Age": [5, 6],
        },
        index=["S1", "S2"],
    )
    _, _, _, clin_sim, subtype_sim = alt.split_simulated_blocks_v2(X)

    assert "Age" in clin_sim.columns
    assert subtype_sim.name == "SUBTYPE"
    assert subtype_sim.tolist() == ["X", "Y"]
