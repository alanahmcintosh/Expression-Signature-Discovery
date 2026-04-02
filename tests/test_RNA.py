import numpy as np
import pandas as pd
import pytest

import RNA as rna


# ----------------------------
# 0) Helper utilities for tests
# ----------------------------
def _toy_rng(seed=0):
    return np.random.default_rng(seed)


# ----------------------------
# sample_size
# ----------------------------
def test_sample_size_fallback_when_invalid_mean():
    # n_mean invalid -> uses fallback range
    v = rna.sample_size(None, seed=1, fallback_size_range=(10, 20))
    assert 10 <= v <= 20

    v2 = rna.sample_size(-5, seed=1, fallback_size_range=(10, 20))
    assert 10 <= v2 <= 20

    v3 = rna.sample_size(np.nan, seed=1, fallback_size_range=(10, 20))
    assert 10 <= v3 <= 20


def test_sample_size_rounds_and_min_one():
    assert rna.sample_size(3.4, seed=1, fallback_size_range=(1, 2)) == 3
    assert rna.sample_size(0.4, seed=1, fallback_size_range=(1, 2)) == 1


# ----------------------------
# sample_abs
# ----------------------------
def test_sample_abs_fallback_uniform_when_params_missing():
    val = rna.sample_abs(None, None, cap=None, seed=2, fallback_abs_range=(1.0, 2.0))
    assert 1.0 <= val <= 2.0


def test_sample_abs_applies_cap():
    # force a large value via lognormal by using large mu; cap should bind
    val = rna.sample_abs(mu=5.0, sigma=0.1, cap=3.0, seed=2, fallback_abs_range=(1.0, 2.0))
    assert val <= 3.0


# ----------------------------
# build_targets_default
# ----------------------------
def test_build_targets_default_forces_base_gene_and_no_duplicates():
    genes = [f"G{i}" for i in range(10)]
    gene_set = set(genes)

    t = rna.build_targets_default("G3", L=5, seed=1, genes=genes, gene_set=gene_set)
    assert t[0] == "G3"
    assert len(t) == 5
    assert len(set(t)) == 5


def test_build_targets_default_no_base_gene_samples_L():
    genes = [f"G{i}" for i in range(10)]
    gene_set = set(genes)

    t = rna.build_targets_default("NOTHERE", L=4, seed=1, genes=genes, gene_set=gene_set)
    assert len(t) == 4
    assert len(set(t)) == 4


# ----------------------------
# build_targets_shared
# ----------------------------
def test_build_targets_shared_includes_base_gene_and_shares_fraction():
    genes = [f"G{i}" for i in range(20)]
    gene_set = set(genes)

    base = "G0"
    ref_targets = ["G0", "G1", "G2", "G3", "G4", "G5"]
    L = 10
    share_frac = 0.5

    t = rna.build_targets_shared(
        base_gene=base,
        L=L,
        ref_targets=ref_targets,
        share_frac=share_frac,
        seed=3,
        gene_set=gene_set,
        genes=genes,
    )

    assert t[0] == base
    assert len(t) == L
    assert len(set(t)) == L

    # shared should be drawn from ref_targets excluding base
    shared_possible = set(ref_targets) - {base}
    # expected shared count ≈ round(share_frac*(L-1))
    expected = int(round(share_frac * (L - 1)))
    # actual shared count = overlap with shared_possible (excluding base)
    actual = len((set(t) - {base}) & shared_possible)
    assert actual == expected


def test_build_targets_shared_handles_small_specific_pool_by_relaxing():
    genes = ["A", "B", "C", "D"]
    gene_set = set(genes)
    base = "A"
    ref_targets = ["A", "B", "C", "D"]
    # L is large vs available; function should still return without error by relaxing
    t = rna.build_targets_shared(
        base_gene=base,
        L=4,
        ref_targets=ref_targets,
        share_frac=0.75,
        seed=1,
        gene_set=gene_set,
        genes=genes,
    )
    assert len(t) == 4
    assert len(set(t)) == 4


# ----------------------------
# gistic_to_amp_del_binary
# ----------------------------
def test_gistic_to_amp_del_binary_outputs_expected_columns_and_values():
    cna = pd.DataFrame(
        {"G1": [2, 1, 0, -1, -2, np.nan], "G2": [0, 0, 2, -2, 1, -1]},
        index=[f"S{i}" for i in range(6)],
    )
    out = rna.gistic_to_amp_del_binary(cna)

    assert set(out.columns) == {"G1_AMP", "G1_DEL", "G2_AMP", "G2_DEL"}
    # AMP is 1 when >0, DEL is 1 when <0
    assert out.loc["S0", "G1_AMP"] == 1 and out.loc["S0", "G1_DEL"] == 0
    assert out.loc["S4", "G1_DEL"] == 1 and out.loc["S4", "G1_AMP"] == 0
    assert set(np.unique(out.to_numpy())).issubset({0, 1})


# ----------------------------
# apply_ampdel_severity_weights
# ----------------------------
def test_apply_ampdel_severity_weights_replaces_amp_del_with_severity():
    # alteration_df has AMP/DEL binaries but we replace with 0/1/2 severity from gistic
    A = pd.DataFrame(
        {"G1_AMP": [1, 0, 1], "G1_DEL": [0, 1, 0], "KRAS_GOF": [1, 0, 1]},
        index=["S1", "S2", "S3"],
    )
    cna = pd.DataFrame({"G1": [2, -2, 1]}, index=["S1", "S2", "S3"])

    out = rna.apply_ampdel_severity_weights(A, cna)

    # AMP becomes clipped positive copy (0/1/2), DEL becomes -cna clipped
    assert out.loc["S1", "G1_AMP"] == 2
    assert out.loc["S2", "G1_DEL"] == 2
    assert out.loc["S3", "G1_AMP"] == 1

    # Non-AMP/DEL columns unchanged
    assert out["KRAS_GOF"].tolist() == [1, 0, 1]


# ----------------------------
# parse_alt
# ----------------------------
@pytest.mark.parametrize(
    "alt_str,expected_kind,expected_base",
    [
        ("TP53_LOF", "LOF", "TP53"),
        ("KRAS_GOF", "GOF", "KRAS"),
        ("MYC_AMP", "AMP", "MYC"),
        ("CDKN2A_DEL", "DEL", "CDKN2A"),
        ("ETV6--RUNX1_FUSION", "FUSION", "ETV6"),
        ("SOMETHING", "OTHER", None),
    ],
)
def test_parse_alt(alt_str, expected_kind, expected_base):
    kind, base, partners = rna.parse_alt(alt_str)
    assert kind == expected_kind
    assert base == expected_base
    if kind == "FUSION":
        assert isinstance(partners, list) and len(partners) >= 2


# ----------------------------
# cap_abs_log2fc
# ----------------------------
def test_cap_abs_log2fc_rule():
    # cap = min(abs_max, mean+cap_k, global_cap)
    cap = rna.cap_abs_log2fc(mean_abs=2.0, abs_max=20.0, cap_k=8.0, global_cap=12.0)
    assert cap == pytest.approx(10.0)

    cap2 = rna.cap_abs_log2fc(mean_abs=2.0, abs_max=9.0, cap_k=8.0, global_cap=12.0)
    assert cap2 == pytest.approx(9.0)

    cap3 = rna.cap_abs_log2fc(mean_abs=100.0, abs_max=200.0, cap_k=8.0, global_cap=12.0)
    assert cap3 == pytest.approx(12.0)


def test_cap_abs_log2fc_invalid_falls_back_to_global_cap():
    cap = rna.cap_abs_log2fc(mean_abs=np.nan, abs_max=5.0, cap_k=8.0, global_cap=12.0)
    assert cap == pytest.approx(12.0)


# ----------------------------
# fit_lognormal_from_mean_median
# ----------------------------
def test_fit_lognormal_from_mean_median_valid():
    mu, sigma = rna.fit_lognormal_from_mean_median(mean_val=2.0, median_val=1.0)
    assert mu == pytest.approx(np.log(1.0))
    assert sigma >= 0


def test_fit_lognormal_from_mean_median_invalid_returns_none():
    mu, sigma = rna.fit_lognormal_from_mean_median(mean_val=-1.0, median_val=1.0)
    assert mu is None and sigma is None


# ----------------------------
# build_alt_params_from_deseq2_summary
# ----------------------------
def test_build_alt_params_from_deseq2_summary_builds_expected_dict():
    df = pd.DataFrame(
        {
            "alteration": ["KRAS_GOF", "TP53_LOF"],
            "n_sig": [10, 20],
            "mean_abs_log2FC_sig": [1.5, 2.0],
            "median_abs_log2FC_sig": [1.0, 1.8],
            "abs_max_log2FC_sig": [8.0, 9.0],
        }
    )
    params = rna.build_alt_params_from_deseq2_summary(df)
    assert set(params.keys()) == {"KRAS_GOF", "TP53_LOF"}
    assert params["KRAS_GOF"]["size_mean"] == 10
    assert params["TP53_LOF"]["abs_cap"] <= 12.0


# ----------------------------
# draw_size_factors_from_deseq
# ----------------------------
def test_draw_size_factors_from_deseq_no_subtype_resamples():
    sf = pd.Series([0.9, 1.0, 1.1], index=["S1", "S2", "S3"])
    out = rna.draw_size_factors_from_deseq(sf, n_samples=5, subtype=None, rng=1)
    assert len(out) == 5
    assert set(np.unique(out)).issubset(set(sf.values))


def test_draw_size_factors_from_deseq_with_subtype_preserves_proportions_length():
    sf = pd.Series([0.9, 1.0, 1.1, 1.2], index=["S1", "S2", "S3", "S4"])
    subtype = pd.Series(["A", "A", "B", "B"], index=sf.index)
    out = rna.draw_size_factors_from_deseq(sf, n_samples=10, subtype=subtype, rng=2)
    assert len(out) == 10


# ----------------------------
# sample_nb
# ----------------------------
def test_sample_nb_shapes_and_nonnegativity():
    rng = _toy_rng(0)
    mu = np.full((4, 3), 10.0)
    disp = pd.Series([0.1, 0.2, 0.3], index=["G1", "G2", "G3"])
    out = rna.sample_nb(mu, disp, rng=rng)
    assert out.shape == (4, 3)
    assert (out >= 0).all()


def test_sample_nb_raises_on_nonfinite_inputs():
    mu = np.array([[np.nan, 1.0]])
    disp = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        rna.sample_nb(mu, disp, rng=_toy_rng(0))


def test_sample_nb_raises_on_shape_mismatch():
    mu = np.ones((3, 2))
    disp = np.ones(3)  # expects length 2
    with pytest.raises(ValueError):
        rna.sample_nb(mu, disp, rng=_toy_rng(0))


# ----------------------------
# sample_nb_for_signature_genes
# ----------------------------
def test_sample_nb_for_signature_genes_only_samples_sig_genes_nb(monkeypatch):
    mu_df = pd.DataFrame(
        {"G1": [10.0, 10.0], "G2": [5.0, 5.0], "G3": [1.2, 1.2]},
        index=["S1", "S2"],
    )
    disp = pd.Series({"G1": 0.1, "G2": 0.2, "G3": 0.3})

    # make NB deterministic for test
    def fake_sample_nb(mu, dispersions, rng=None):
        return np.full_like(mu, 7, dtype=int)

    monkeypatch.setattr(rna, "sample_nb", fake_sample_nb)

    out = rna.sample_nb_for_signature_genes(mu_df, disp, sig_genes=["G1", "G3"], seed=1)

    # signature genes set to 7, others rounded
    assert out.loc["S1", "G1"] == 7
    assert out.loc["S1", "G3"] == 7
    assert out.loc["S1", "G2"] == int(round(5.0))
    assert (out.to_numpy() >= 0).all()


def test_sample_nb_for_signature_genes_no_sig_genes_rounds():
    mu_df = pd.DataFrame({"G1": [0.2, 0.9]}, index=["S1", "S2"])
    disp = pd.Series({"G1": 0.1})
    out = rna.sample_nb_for_signature_genes(mu_df, disp, sig_genes=["NOTHERE"], seed=1)
    assert out.dtypes["G1"] in (np.int64, np.int32, int)
    assert out.loc["S1", "G1"] == 0


# ----------------------------
# simulate_background_from_alterations_knn
# ----------------------------
def test_simulate_background_from_alterations_knn_requires_overlap_features():
    real_rna = pd.DataFrame({"G1": [10, 11]}, index=["S1", "S2"])
    real_alts = pd.DataFrame({"A": [1, 0]}, index=["S1", "S2"])
    sim_alts = pd.DataFrame({"B": [1, 0]}, index=["X1", "X2"])  # no overlap
    with pytest.raises(ValueError):
        rna.simulate_background_from_alterations_knn(real_rna, real_alts, sim_alts)


def test_simulate_background_from_alterations_knn_outputs_mu_matrix_shape_and_nonnegative():
    real_rna = pd.DataFrame(
        {"G1": [10, 20, 15], "G2": [5, 6, 7]},
        index=["S1", "S2", "S3"],
    )
    real_alts = pd.DataFrame(
        {"A": [1, 0, 1], "B": [0, 1, 1]},
        index=real_rna.index,
    )
    sim_alts = pd.DataFrame(
        {"A": [1, 0], "B": [1, 0]},
        index=["X1", "X2"],
    )

    mu = rna.simulate_background_from_alterations_knn(
        real_rna=real_rna,
        real_alts=real_alts,
        sim_alts=sim_alts,
        k=2,
        mix_conc=1.0,
        residual_scale=0.0,
        seed=0,
    )

    assert mu.shape == (2, 2)
    assert list(mu.columns) == ["G1", "G2"]
    assert (mu.to_numpy() >= 0).all()


# ----------------------------
# generate_signatures_from_deseq2_params
# ----------------------------
def test_generate_signatures_from_deseq2_params_includes_base_gene_and_sign_bias():
    genes = ["KRAS", "TP53", "G1", "G2", "G3", "G4"]
    alteration_features = ["KRAS_GOF", "TP53_LOF"]
    alt_params = {
        "KRAS_GOF": {"size_mean": 3, "abs_mu": np.log(1.0), "abs_sigma": 0.0, "abs_cap": 2.0},
        "TP53_LOF": {"size_mean": 3, "abs_mu": np.log(1.0), "abs_sigma": 0.0, "abs_cap": 2.0},
    }

    sigs = rna.generate_signatures_from_deseq2_params(
        genes=genes,
        alteration_features=alteration_features,
        alt_params=alt_params,
        seed=1,
        gof_p_pos=1.0,  # force positive
        lof_p_pos=0.0,  # force negative
    )

    assert "KRAS_GOF" in sigs and "TP53_LOF" in sigs
    assert "KRAS" in sigs["KRAS_GOF"]["targets"]
    assert "TP53" in sigs["TP53_LOF"]["targets"]

    # With p_pos forced, all GOF effects positive, all LOF effects negative
    assert all(v > 0 for v in sigs["KRAS_GOF"]["effects"].values())
    assert all(v < 0 for v in sigs["TP53_LOF"]["effects"].values())


def test_generate_signatures_shares_targets_for_amp_with_gof_when_enabled():
    genes = [f"G{i}" for i in range(30)] + ["MYC"]
    alteration_features = ["MYC_GOF", "MYC_AMP"]
    alt_params = {
        "MYC_GOF": {"size_mean": 10, "abs_mu": np.log(1.0), "abs_sigma": 0.0, "abs_cap": 2.0},
        "MYC_AMP": {"size_mean": 10, "abs_mu": np.log(1.0), "abs_sigma": 0.0, "abs_cap": 2.0},
    }

    sigs = rna.generate_signatures_from_deseq2_params(
        genes=genes,
        alteration_features=alteration_features,
        alt_params=alt_params,
        seed=4,
        share_amp_with_gof=True,
        share_frac=0.7,
        gof_p_pos=1.0,
    )

    gof_targets = sigs["MYC_GOF"]["targets"]
    amp_targets = sigs["MYC_AMP"]["targets"]
    # exclude base gene
    gof_set = set(gof_targets) - {"MYC"}
    amp_set = set(amp_targets) - {"MYC"}
    overlap = len(gof_set & amp_set)
    expected = int(round(0.7 * (len(amp_targets) - 1)))
    assert overlap == expected


# ----------------------------
# induce_expression_effects
# ----------------------------
def test_induce_expression_effects_applies_log2fc_multiplicatively():
    mu = pd.DataFrame({"G1": [10.0, 10.0], "G2": [8.0, 8.0]}, index=["S1", "S2"])
    A = pd.DataFrame({"ALT1_GOF": [1, 0]}, index=["S1", "S2"])
    signatures = {
        "ALT1_GOF": {"effects": {"G1": 1.0, "G2": -1.0}, "targets": ["G1", "G2"], "effect_mode": "log2fc"}
    }

    out = rna.induce_expression_effects(mu, A, signatures, cna_gistic_df=None)

    # S1 affected: G1 *2^(+1)=*2, G2 *2^(-1)=/2
    assert out.loc["S1", "G1"] == pytest.approx(20.0)
    assert out.loc["S1", "G2"] == pytest.approx(4.0)
    # S2 unaffected
    assert out.loc["S2", "G1"] == pytest.approx(10.0)


def test_induce_expression_effects_returns_original_if_no_signature_alts():
    mu = pd.DataFrame({"G1": [1.0]}, index=["S1"])
    A = pd.DataFrame({"NOPE": [1]}, index=["S1"])
    out = rna.induce_expression_effects(mu, A, signatures={}, cna_gistic_df=None)
    pd.testing.assert_frame_equal(out, mu)


def test_induce_expression_effects_amp_del_severity_from_gistic():
    mu = pd.DataFrame({"MYC": [10.0, 10.0]}, index=["S1", "S2"])
    A = pd.DataFrame({"MYC_AMP": [1, 1]}, index=["S1", "S2"])
    cna = pd.DataFrame({"MYC": [2, 1]}, index=["S1", "S2"])
    signatures = {"MYC_AMP": {"effects": {"MYC": 1.0}, "targets": ["MYC"], "effect_mode": "log2fc"}}

    out = rna.induce_expression_effects(mu, A, signatures, cna_gistic_df=cna)

    # severity weights: S1 weight=2 => multiply by 2^(2*1)=4; S2 weight=1 => multiply by 2
    assert out.loc["S1", "MYC"] == pytest.approx(40.0)
    assert out.loc["S2", "MYC"] == pytest.approx(20.0)


# ----------------------------
# simulate_rna_with_signatures (wrapper)
#   We monkeypatch DESeq2 + background + NB sampling so it becomes fast & deterministic.
# ----------------------------
def test_simulate_rna_with_signatures_wrapper_happy_path(monkeypatch):
    # Small counts matrix (real)
    real_rna = pd.DataFrame(
        {"KRAS": [10, 12, 11], "TP53": [8, 7, 9], "G1": [2, 3, 2]},
        index=["R1", "R2", "R3"],
    )

    # simulated alterations on simulated samples
    sim_alts = pd.DataFrame(
        {"KRAS_GOF": [1, 0], "TP53_LOF": [0, 1]},
        index=["S1", "S2"],
    )

    # "real" alterations aligned to real_rna index
    real_alts = pd.DataFrame(
        {"KRAS_GOF": [1, 0, 1], "TP53_LOF": [0, 1, 0]},
        index=real_rna.index,
    )

    # Patch DESeq2 estimation to avoid running pydeseq2
    def fake_estimate_deseq2_parameters(rna_df, seed=44, **kwargs):
        means = rna_df.mean()
        vars_ = rna_df.var()
        disp = pd.Series(0.1, index=rna_df.columns)
        sf = pd.Series(1.0, index=rna_df.index, name="size_factor")
        return means, vars_, disp, sf

    monkeypatch.setattr(rna, "estimate_deseq2_parameters", fake_estimate_deseq2_parameters)

    # Patch background μ builder to something simple/deterministic
    def fake_background(real_rna, real_alts, sim_alts, **kwargs):
        # return per-sample mu equal to the global mean of real_rna
        mu = real_rna.mean(axis=0).to_frame().T
        mu = pd.concat([mu] * sim_alts.shape[0], axis=0)
        mu.index = sim_alts.index
        return mu

    monkeypatch.setattr(rna, "simulate_background_from_alterations_knn", fake_background)

    # Patch NB sampler so we can assert exact output values
    def fake_sample_nb_for_signature_genes(mu_df, dispersions, sig_genes, seed=44):
        # just round everything to int for test
        out = mu_df.round().astype(int)
        out[out < 0] = 0
        return out

    monkeypatch.setattr(rna, "sample_nb_for_signature_genes", fake_sample_nb_for_signature_genes)

    # Provide deseq2 summary so signatures get generated
    deseq2_summary = pd.DataFrame(
        {
            "alteration": ["KRAS_GOF", "TP53_LOF"],
            "n_sig": [2, 2],
            "mean_abs_log2FC_sig": [1.0, 1.0],
            "median_abs_log2FC_sig": [1.0, 1.0],
            "abs_max_log2FC_sig": [2.0, 2.0],
        }
    )

    expr_sim, true_sigs, sim_bg = rna.simulate_rna_with_signatures(
        rna_df=real_rna,
        alteration_df=sim_alts,
        altertaions_real=real_alts,
        cna_df=None,
        n_samples=2,
        n_genes_to_sim=3,
        seed=1,
        deseq2_summary_df=deseq2_summary,
        true_signatures=None,
    )

    # shapes & keys
    assert expr_sim.shape == (2, 3)
    assert sim_bg.shape == (2, 3)
    assert set(true_sigs.keys()) == {"KRAS_GOF", "TP53_LOF"}
    assert list(expr_sim.index) == ["S1", "S2"]


def test_simulate_rna_with_signatures_uses_provided_true_signatures(monkeypatch):
    real_rna = pd.DataFrame({"G1": [10, 11]}, index=["R1", "R2"])
    sim_alts = pd.DataFrame({"ALT1_GOF": [1]}, index=["S1"])
    real_alts = pd.DataFrame({"ALT1_GOF": [1, 0]}, index=real_rna.index)

    # Patch DESeq2 + background + NB as above
    monkeypatch.setattr(
        rna,
        "estimate_deseq2_parameters",
        lambda rna_df, seed=44, **kwargs: (
            rna_df.mean(),
            rna_df.var(),
            pd.Series(0.1, index=rna_df.columns),
            pd.Series(1.0, index=rna_df.index),
        ),
    )
    monkeypatch.setattr(
        rna,
        "simulate_background_from_alterations_knn",
        lambda real_rna, real_alts, sim_alts, **kwargs: pd.DataFrame(
            {"G1": [10.0]}, index=sim_alts.index
        ),
    )
    monkeypatch.setattr(
        rna,
        "sample_nb_for_signature_genes",
        lambda mu_df, dispersions, sig_genes, seed=44: mu_df.round().astype(int),
    )

    provided = {
        "ALT1_GOF": {"targets": ["G1"], "effects": {"G1": 1.0}, "effect_mode": "log2fc"}
    }

    expr_sim, true_sigs, _ = rna.simulate_rna_with_signatures(
        rna_df=real_rna,
        alteration_df=sim_alts,
        altertaions_real=real_alts,
        n_samples=1,
        n_genes_to_sim=1,
        seed=0,
        true_signatures=provided,
        deseq2_summary_df=None,
        deseq2_summary_path=None,
    )

    assert true_sigs is provided
    # Since ALT1_GOF=1 and effect is +1, mu 10 -> 20 -> rounds to 20
    assert expr_sim.loc["S1", "G1"] == 20
