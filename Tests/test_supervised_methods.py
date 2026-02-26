import numpy as np
import pandas as pd
import pytest

# Import the module under test
import supervised_methods as sm


# -----------------------------
# Helper fixtures
# -----------------------------
@pytest.fixture
def toy_counts_and_X():
    """
    Small, deterministic toy dataset.

    X: binary alteration column ALT1 with balanced groups
    Y: non-negative "counts" (DataFrame)
    """
    idx = [f"S{i}" for i in range(8)]
    X = pd.DataFrame(
        {
            "ALT1": [0, 0, 0, 0, 1, 1, 1, 1],
            "ALT2": [1, 1, 1, 1, 1, 1, 1, 1],  # constant
        },
        index=idx,
    )

    # Non-negative integer counts
    Y = pd.DataFrame(
        {
            "G1": [10, 12, 11, 9, 40, 39, 41, 38],
            "G2": [5, 4, 6, 5, 20, 22, 21, 19],
            "G3": [0, 1, 0, 2, 1, 0, 1, 0],
        },
        index=idx,
    )
    return X, Y


@pytest.fixture
def toy_W_dict():
    """
    Minimal W_dict shaped like: predictors x genes.
    """
    predictors = ["ALT1", "ALT2"]
    genes = ["G1", "G2", "G3"]

    # For nonzero selection: ALT1 has strong weights on G1,G2
    W_lasso = pd.DataFrame(
        [[1.2, -0.9, 0.0],
         [0.0,  0.0, 0.0]],
        index=predictors, columns=genes
    )

    W_enet = pd.DataFrame(
        [[0.5, 0.0, 0.2],
         [0.0, 0.0, 0.0]],
        index=predictors, columns=genes
    )

    # Ridge/SVM/RF: elbow-based selection (we just need *some* values)
    W_ridge = pd.DataFrame(
        [[0.9, 0.2, 0.1],
         [0.1, 0.1, 0.1]],
        index=predictors, columns=genes
    )

    W_svm = pd.DataFrame(
        [[0.8, 0.05, 0.01],
         [0.0, 0.0, 0.0]],
        index=predictors, columns=genes
    )

    W_rf = pd.DataFrame(
        [[0.7, 0.2, 0.1],
         [0.0, 0.0, 0.0]],
        index=predictors, columns=genes
    )

    return {
        "Lasso": W_lasso,
        "ElasticNet": W_enet,
        "Ridge": W_ridge,
        "SVM": W_svm,
        "Random Forest": W_rf,
    }


# -----------------------------
# Tests: select_features_elbow
# -----------------------------
def test_select_features_elbow_all_zero_returns_empty():
    names = ["a", "b", "c"]
    w = [0.0, 0.0, 0.0]
    assert sm.select_features_elbow(names, w) == []


def test_select_features_elbow_single_nonzero_returns_that_feature():
    names = ["a", "b", "c"]
    w = [0.0, 2.0, 0.0]
    out = sm.select_features_elbow(names, w)
    assert out == ["b"]


def test_select_features_elbow_flat_nonzero_returns_top1():
    # With equal weights, dist-to-line is 0 for all points, argmax -> 0,
    # so the function returns only the top-ranked feature.
    names = ["a", "b", "c"]
    w = [1.0, 1.0, 1.0]
    out = sm.select_features_elbow(names, w)
    assert len(out) == 1
    assert out[0] in names



# -----------------------------
# Tests: normalize_counts_log_cpm
# -----------------------------
def test_normalize_counts_log_cpm_drops_all_zero_samples_and_preserves_shape_otherwise():
    idx = ["S1", "S2", "S3"]
    Y = pd.DataFrame(
        {"G1": [0, 10, 5], "G2": [0, 0, 5]},
        index=idx
    )
    Z = sm.normalize_counts_log_cpm(Y)

    # S1 is all-zero => dropped
    assert "S1" not in Z.index
    assert set(Z.index) == {"S2", "S3"}
    assert list(Z.columns) == ["G1", "G2"]


def test_normalize_counts_log_cpm_zscores_per_gene_nonproportional():
    idx = ["S1", "S2", "S3"]
    Y = pd.DataFrame(
        {"G1": [10, 20, 30], "G2": [30, 5, 2]},
        index=idx
    )
    Z = sm.normalize_counts_log_cpm(Y)

    means = Z.mean(axis=0).values
    stds = Z.std(axis=0, ddof=0).values

    assert np.allclose(means, 0.0, atol=1e-7)
    assert np.allclose(stds, 1.0, atol=1e-7)
# -----------------------------
# Tests: align_XY
# -----------------------------
def test_align_XY_intersection_order():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=["S1", "S2", "S3"])
    Y = pd.DataFrame({"g": [10, 20]}, index=["S2", "S1"])  # different order
    X2, Y2 = sm.align_XY(X, Y)

    common = X.index.intersection(Y.index)
    assert list(X2.index) == list(common)
    assert list(Y2.index) == list(common)


# -----------------------------
# Tests: signature_from_weights_for_alt
# -----------------------------
def test_signature_from_weights_for_alt_missing_alt_returns_empty(toy_W_dict):
    W = toy_W_dict["Lasso"]
    assert sm.signature_from_weights_for_alt(W, "NOPE", mode="nonzero") == []


def test_signature_from_weights_for_alt_nonzero_mode(toy_W_dict):
    W = toy_W_dict["Lasso"]
    genes = sm.signature_from_weights_for_alt(W, "ALT1", mode="nonzero", coef_tol=1e-6)
    assert set(genes) == {"G1", "G2"}  # G3 is 0


def test_signature_from_weights_for_alt_elbow_mode_returns_nonempty(toy_W_dict):
    W = toy_W_dict["Ridge"]
    genes = sm.signature_from_weights_for_alt(W, "ALT1", mode="elbow")
    assert isinstance(genes, list)
    assert len(genes) >= 1
    assert all(g in W.columns for g in genes)


def test_signature_from_weights_for_alt_invalid_mode_raises(toy_W_dict):
    W = toy_W_dict["Ridge"]
    with pytest.raises(ValueError):
        sm.signature_from_weights_for_alt(W, "ALT1", mode="bogus")


# -----------------------------
# Tests: get_deconfounder_signature
# -----------------------------
def test_get_deconfounder_signature_handles_list_gof():
    coefs = pd.DataFrame({"ALT1": [np.nan, 1.0, 0.5]}, index=["G1", "G2", "G3"])
    global_results = {"Deconfounder": coefs}
    out = sm.get_deconfounder_signature(["ALT1"], global_results)
    assert out == ["G2", "G3"]


# -----------------------------
# Tests: create_supervised_signatures
# -----------------------------
def test_create_supervised_signatures_missing_gof_raises(toy_counts_and_X):
    X, Y = toy_counts_and_X
    with pytest.raises(KeyError):
        sm.create_supervised_signatures(X, Y, gof="NOT_A_COL")


def test_create_supervised_signatures_constant_predictor_skips(toy_counts_and_X):
    X, Y = toy_counts_and_X
    out = sm.create_supervised_signatures(X, Y, gof="ALT2", W_dict=None, global_results=None)
    assert "SKIPPED" in out
    assert "constant/low-var" in out["SKIPPED"]


def test_create_supervised_signatures_insufficient_groups_skips(toy_counts_and_X):
    X, Y = toy_counts_and_X
    X = X.copy()
    X["ALT1"] = [0] * 7 + [1]  # only one "1"
    out = sm.create_supervised_signatures(X, Y, gof="ALT1", min_group_n=2, W_dict=None, global_results=None)
    assert "SKIPPED" in out
    assert "insufficient group sizes" in out["SKIPPED"]


def test_create_supervised_signatures_includes_W_dict_and_deconfounder_and_mocks_deseq2(
    monkeypatch, toy_counts_and_X, toy_W_dict
):
    X, Y = toy_counts_and_X

    # Mock DESeq2 signature function so tests do not need pydeseq2
    def fake_deseq2_signature_binary(X_design, Y_sub, gof, **kwargs):
        # Return deterministic "significant genes"
        return ["G2", "G3"]

    monkeypatch.setattr(sm, "get_deseq2_signature_binary", fake_deseq2_signature_binary)

    # Deconfounder global_results shape expectation:
    # global_results["Deconfounder"] is a DataFrame with column gof, index genes
    global_results = {
        "Deconfounder": pd.DataFrame({"ALT1": [0.2, np.nan, 1.0]}, index=["G1", "G2", "G3"])
    }

    out = sm.create_supervised_signatures(
        X=X,
        Y=Y,
        gof="ALT1",
        global_results=global_results,
        W_dict=toy_W_dict,
        min_group_n=1,
    )

    # W_dict-derived
    assert "Lasso" in out and isinstance(out["Lasso"], list)
    assert "ElasticNet" in out and isinstance(out["ElasticNet"], list)
    assert "Ridge" in out and isinstance(out["Ridge"], list)
    assert "SVM" in out and isinstance(out["SVM"], list)
    assert "Random Forest" in out and isinstance(out["Random Forest"], list)

    # Deconfounder-derived
    assert out["Deconfounder"] == ["G1", "G3"]

    # Mocked DESeq2-derived
    assert out["DESeq2"] == ["G2", "G3"]
    assert "DESeq2_ERROR" not in out


def test_create_supervised_signatures_deseq2_negative_values_errors(monkeypatch, toy_counts_and_X):
    X, Y = toy_counts_and_X
    Y_bad = Y.copy()
    Y_bad.iloc[0, 0] = -1  # negative count should trigger error path before DESeq2

    # If code changes and tries to call DESeq2 anyway, ensure we notice
    def fail_if_called(*args, **kwargs):
        raise RuntimeError("DESeq2 should not be called when Y contains negatives")

    monkeypatch.setattr(sm, "get_deseq2_signature_binary", fail_if_called)

    out = sm.create_supervised_signatures(X, Y_bad, gof="ALT1", W_dict=None, global_results=None)
    assert "DESeq2_ERROR" in out
    assert "requires non-negative counts" in out["DESeq2_ERROR"]