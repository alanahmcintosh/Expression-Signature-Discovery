import numpy as np
import pandas as pd
import pytest

import Deconfounder as dc


# ----------------------------
# Helpers
# ----------------------------
def _toy_XY(n=12, p=5, q=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, p)),
        index=[f"S{i}" for i in range(n)],
        columns=[f"feat_{j}" for j in range(p)],
    )
    # positive counts-like Y
    Y = pd.DataFrame(
        rng.poisson(lam=20, size=(n, q)),
        index=X.index,
        columns=[f"gene_{g}" for g in range(q)],
    )
    return X, Y


# ============================================================
# ppca class tests
# ============================================================

def test_ppca_max_likelihood_sets_attributes_and_shapes():
    X, _ = _toy_XY(n=10, p=6, q=3, seed=1)
    m = dc.ppca(factors=2)

    m.max_likelihood(X.values, standardise=True)

    assert m.ran is True
    assert m.n == 10
    assert m.D == 6
    assert m.W.shape == (6, 2)
    assert m.z_mu.shape == (2, 10)  # (factors, n)
    assert m.z_cov.shape == (2, 2)
    assert m.means.shape == (6,)


def test_ppca_max_likelihood_mask_shape_mismatch_raises():
    X, _ = _toy_XY(n=8, p=4, q=2, seed=2)
    m = dc.ppca(factors=2)

    bad_mask = np.ones((X.shape[0], X.shape[1] + 1), dtype=bool)
    with pytest.raises(ValueError):
        m.max_likelihood(X.values, standardise=False, mask=bad_mask)


def test_ppca_generate_requires_ran_then_returns_shapes():
    X, _ = _toy_XY(n=6, p=5, q=2, seed=3)
    m = dc.ppca(factors=2)
    m.max_likelihood(X.values, standardise=False)

    gen_x, z = m.generate(1)

    assert gen_x.shape == (m.n, m.D)          # n*n_rep, D ; here n_rep=1
    assert z.shape == (m.n, m.factors)        # n*n_rep, k
    assert np.isfinite(gen_x).all()
    assert np.isfinite(z).all()


def test_ppca_get_W_cov_returns_k_by_k():
    X, _ = _toy_XY(n=10, p=6, q=3, seed=4)
    m = dc.ppca(factors=3)
    m.max_likelihood(X.values, standardise=False)

    W_cov = m.get_W_cov()
    assert W_cov.shape == (3, 3)
    assert np.all(np.diag(W_cov) >= 0)


def test_ppca_holdout_creates_masks_and_arrays():
    X, _ = _toy_XY(n=10, p=6, q=3, seed=5)
    m = dc.ppca(factors=2)

    x_np = X.values
    m.holdout(x_np, holdout_portion=0.2, n_rep=10, seed=123)

    assert m.holdout_mask.shape == x_np.shape
    assert m.x_val.shape == x_np.shape
    assert m.holdout_gen.shape == (10, *x_np.shape)

    # ✅ now x_train should be a 2D masked matrix
    assert m.x_train.shape == x_np.shape
    assert m.x_train.ndim == 2

    # ✅ optional 1D vector of observed entries (if you kept it)
    assert hasattr(m, "x_train_vec")
    assert m.x_train_vec.ndim == 1


# ============================================================
# predicitve_check tests
# ============================================================

def test_predicitve_check_returns_probability_in_0_1():
    X, _ = _toy_XY(n=12, p=6, q=3, seed=6)
    m = dc.ppca(factors=2)

    m.holdout(X.values, holdout_portion=0.2, n_rep=5, seed=1)

    # ✅ Fit PPCA on the 2D training matrix
    m.max_likelihood(m.x_train, standardise=False)

    # generate once so the model has latent variables etc
    _ = m.generate(1)

    pval = dc.predicitve_check(m, factors=2, holdout_portion=0.2, n_rep=5)
    assert 0.0 <= pval <= 1.0

# ============================================================
# deconfounder (Option 1) tests
# ============================================================

def test_deconfounder_preallocates_all_gene_columns_even_if_all_zero(monkeypatch):
    """
    Key behavior described in the docstring: coefs must have columns for ALL Y genes,
    even if a gene ends up with all-zero coefficients.
    """
    X, Y = _toy_XY(n=15, p=5, q=4, seed=7)

    # Force LassoCV and Lasso to choose all-zero coefs deterministically.
    class FakeLassoCV:
        def __init__(self, alphas, cv, random_state, n_jobs, max_iter, tol):
            self.alpha_ = float(alphas[0])
        def fit(self, X, y):
            return self

    class FakeLasso:
        def __init__(self, alpha, max_iter, tol):
            self.alpha = alpha
            self.coef_ = None
        def fit(self, X, y):
            self.coef_ = np.zeros(X.shape[1], dtype=float)
            return self
        def score(self, X, y):
            return 0.0

    monkeypatch.setattr(dc, "LassoCV", FakeLassoCV)
    monkeypatch.setattr(dc, "Lasso", FakeLasso)

    coefs, models, r2 = dc.deconfounder(X, Y, n_splits=2, n_repeats=1, random_state=0)

    # coefs rows are X features, columns are Y genes
    assert list(coefs.index) == list(X.columns)
    assert list(coefs.columns) == list(Y.columns)
    # all NaN because no non-zeros were written, but columns must exist
    assert coefs.shape == (X.shape[1], Y.shape[1])
    assert len(models) == Y.shape[1]
    assert len(r2) == Y.shape[1]


def test_deconfounder_writes_nonzero_coeffs_into_correct_gene_column(monkeypatch):
    X, Y = _toy_XY(n=20, p=3, q=2, seed=8)

    # Make gene_0 have non-zero coefficient on feat_1 only
    class FakeLassoCV:
        def __init__(self, alphas, cv, random_state, n_jobs, max_iter, tol):
            self.alpha_ = float(alphas[0])
        def fit(self, X, y):
            return self

    class FakeLasso:
        def __init__(self, alpha, max_iter, tol):
            self.alpha = alpha
            self.coef_ = None
        def fit(self, X, y):
            if np.allclose(y, Y["gene_0"].values):
                self.coef_ = np.array([0.0, 1.5, 0.0])
            else:
                self.coef_ = np.array([0.0, 0.0, 0.0])
            return self
        def score(self, X, y):
            return 0.1

    monkeypatch.setattr(dc, "LassoCV", FakeLassoCV)
    monkeypatch.setattr(dc, "Lasso", FakeLasso)

    coefs, models, r2 = dc.deconfounder(X, Y, n_splits=2, n_repeats=1, random_state=0)

    assert coefs.loc["feat_1", "gene_0"] == pytest.approx(1.5)
    assert pd.isna(coefs.loc["feat_0", "gene_0"])
    assert len(models) == 2
    assert len(r2) == 2


# ============================================================
# lasso (Option 2 wrapper around cross_validation from Lasso.py)
# ============================================================

def test_lasso_returns_models_and_scores_and_coef_table(monkeypatch):
    X, Y = _toy_XY(n=10, p=4, q=3, seed=9)

    # Patch cross_validation (imported into dc namespace via "from Lasso import *")
    def fake_cross_validation(X_np, Y_np, a, e, Kfold, K, tol, max_ite, seed):
        p = X_np.shape[1]
        q = Y_np.shape[1]
        # B_best: list of q vectors length p
        B_best = [np.array([0.0, 1.0, 0.0, 0.0])] * q
        MSE = np.zeros((Kfold, q))
        B_full = None
        return B_best, MSE, B_full

    monkeypatch.setattr(dc, "cross_validation", fake_cross_validation)

    coefs, models, r2 = dc.lasso(X, Y, Kfold=2, K=5, seed=1)

    assert coefs.shape == (X.shape[1], Y.shape[1])
    assert list(coefs.index) == list(X.columns)
    assert list(coefs.columns) == list(range(Y.shape[1]))
    assert len(models) == Y.shape[1]
    assert len(r2) == Y.shape[1]

    # model API
    pred = models[0].predict(X.values)
    assert pred.shape[0] == X.shape[0]
    assert np.isfinite(pred).all()


# ============================================================
# choose_latent_dim_ppca
# ============================================================

def test_choose_latent_dim_ppca_returns_first_k_meeting_cutoff(monkeypatch):
    X, _ = _toy_XY(n=12, p=6, q=3, seed=10)
    X_np = X.values

    # Make predicitve_check return <cutoff for k=2, then >=cutoff for k=3
    def fake_predictive_check(model, k, holdout_portion=0.2, n_rep=100):
        return 0.05 if k == 2 else 0.5

    monkeypatch.setattr(dc, "predicitve_check", fake_predictive_check)

    k = dc.choose_latent_dim_ppca(X_np, k_range=range(2, 5), pval_cutoff=0.1, seed=1)
    assert k == 3


def test_choose_latent_dim_ppca_returns_max_if_none_meet_cutoff(monkeypatch):
    X, _ = _toy_XY(n=12, p=6, q=3, seed=11)
    X_np = X.values

    monkeypatch.setattr(dc, "predicitve_check", lambda *args, **kwargs: 0.01)
    k = dc.choose_latent_dim_ppca(X_np, k_range=range(2, 6), pval_cutoff=0.9, seed=1)
    assert k == 5  # max of range(2,6)


# ============================================================
# compute_deconfounder (full pipeline wrapper)
# ============================================================

def test_compute_deconfounder_returns_causal_signatures_with_gene_index(monkeypatch):
    X, Y = _toy_XY(n=10, p=4, q=3, seed=12)

    # 1) Avoid looping over many k
    monkeypatch.setattr(dc, "choose_latent_dim_ppca", lambda X_in: 2)

    # 2) Avoid heavy predictive check
    monkeypatch.setattr(dc, "predicitve_check", lambda *args, **kwargs: 0.5)

    # 3) Make ppca deterministic and fast: patch methods on the class
    def fake_holdout(self, x, holdout_portion=0.2, n_rep=100, seed=None):
        self.holdout_mask = np.zeros_like(x)
        self.x_train = x  # use full X as "train"
        self.x_val = np.zeros_like(x)
        self.holdout_row = np.array([0])
        self.holdout_col = np.array([0])
        self.holdout_gen = np.zeros((n_rep, *x.shape))

    def fake_max_likelihood(self, x, factors=None, standardise=True, mask=None):
        x = np.asarray(x, dtype=float)
        self.n, self.D = x.shape
        self.factors = 2
        self.ran = True
        self.W = np.zeros((self.D, self.factors))
        self.sigma2 = 1.0
        self.M_inv = np.eye(self.factors)
        self.z_mu = np.zeros((self.factors, self.n))
        self.z_cov = np.eye(self.factors)
        self.means = np.zeros((self.D,))

    def fake_generate(self, n, standardise=True):
        gen_x = np.zeros((n * self.n, self.D))
        z = np.zeros((n * self.n, self.factors))
        return gen_x, z

    monkeypatch.setattr(dc.ppca, "holdout", fake_holdout)
    monkeypatch.setattr(dc.ppca, "max_likelihood", fake_max_likelihood)
    monkeypatch.setattr(dc.ppca, "generate", fake_generate)

    # 4) Avoid sklearn LassoCV loop: patch dc.deconfounder to return a known coef table
    def fake_deconf(augX, Y_scaled, **kwargs):
        # coefs expected: rows=features cols=genes
        coefs = pd.DataFrame(0.0, index=augX.columns, columns=Y_scaled.columns)
        models = [object() for _ in Y_scaled.columns]
        r2 = [0.0] * len(Y_scaled.columns)
        return coefs, models, r2

    monkeypatch.setattr(dc, "deconfounder", fake_deconf)

    out = dc.compute_deconfounder(X, Y)

    assert isinstance(out, dict)
    assert "Deconfounder" in out
    coef_tr = out["Deconfounder"]

    # coefs transposed: rows = genes, cols = augmented_X features
    assert list(coef_tr.index) == list(Y.columns)
    assert all(isinstance(c, str) for c in coef_tr.columns)
    assert coef_tr.shape[0] == Y.shape[1]
    # should include latent factors columns
    assert any(c.startswith("latent_") for c in coef_tr.columns)
