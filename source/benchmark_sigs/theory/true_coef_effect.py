from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from benchmark_sigs.simulate.rna.nb_sampling import sample_nb


def theoretical_c1(
    mu_x1, # true mean of expression when x=1 (mutant)
    mu_x0, # true mean of expression when x=0 (wildtype)
    n1, # mutant group n
    n0, # wildtype group n
    p, # Rate at which z co-occurs when x =1
    q, # Rate at which z co-occurs when x = 0
):
    """
    Expected single-variable coefficient for z when true expression is driven by x.
    Calculates coefficient equation is meant to predict 
    p = P(z=1 | x=1)
    q = P(z=1 | x=0)
    n1 = number of x=1 samples
    n0 = number of x=0 samples
    """

    mean_z1 = ((p * n1 * mu_x1) + (q * n0 * mu_x0)) / ((p * n1) + (q * n0))

    mean_z0 = (
        ((1 - p) * n1 * mu_x1) + ((1 - q) * n0 * mu_x0)
    ) / (((1 - p) * n1) + ((1 - q) * n0))

    return np.log(mean_z1) - np.log(mean_z0)


def simulate_x_z(
    n0 = 500, # wildtype sample size
    n1= 500, # mutant sample size
    p = 0.8, # rate at which z co occurs when = 1
    q = 0.2, # rate at which z co occurs when x = 0
    seed = 1,
):
    """
    Simulate two binary alteration variables:
    x = true causal alteration
    z = co-occurring alteration
    """

    rng = np.random.default_rng(seed)

    x = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int),
    ])

    z = np.zeros(n0 + n1, dtype=int)

    z[x == 1] = rng.binomial(1, p, size=n1) # for samples where x = 1, z is sampled with probability p
    z[x == 0] = rng.binomial(1, q, size=n0) # for samples where x = 0, z is sampled with probability q

    return pd.DataFrame({"x": x, "z": z})


def simulate_nb_expression(
    design,
    mu_x0 = 100.0,
    log_fc_x = 1.0,
    dispersion = 0.1,
    seed= 1,
):
    """
    Simulate one gene where expression depends only on x.

    log(mu_i) = beta0 + beta1*x_i
    """

    mu_x1 = mu_x0 * np.exp(log_fc_x) # calculate mutant mean, expression hugher in mutant samples

    mu = np.where(design["x"].values == 1, mu_x1, mu_x0) # creates sample level expected means, ie if x =1 then mean exp = mu_x1

    counts = sample_nb(
        mu=mu.reshape(-1, 1),
        dispersions=np.array([dispersion]),
        rng=np.random.default_rng(seed),
    ).ravel() # turn expected counts into RNA seq counts, NB adds overdispersion (like real RNA)

    return pd.Series(counts, index=design.index, name="gene")


def fit_single_variable_nb(
    y,
    variable,
    dispersion,
):
    """
    Fit simple NB GLM:

        y ~ variable

    Returns the coefficient for variable.
    If run with design["x"] returns estimated effect of x 
    If run with design["z"] returns estimated effect of z 
    """

    X = sm.add_constant(variable.astype(float)) # adds intercept

    # Fit an NB generalized linear model
    model = sm.GLM(
        y.values,
        X.values,
        family=sm.families.NegativeBinomial(alpha=dispersion),
    )

    result = model.fit()

    return result.params[1] # return co-efficient for the variable


def run_one_check(
    n0 = 500,
    n1 = 500,
    p = 0.8,
    q = 0.2,
    mu_x0 = 100.0,
    log_fc_x = 1.0,
    dispersion= 0.1,
    seed = 1,
):
    """
    Run one simulated experiment and compare:

    1. true x coefficient
    2. fitted coefficient from y ~ x
    3. theoretical expected coefficient from y ~ z
    4. fitted coefficient from y ~ z
    """

    design = simulate_x_z(n0=n0, n1=n1, p=p, q=q, seed=seed)

    y = simulate_nb_expression(
        design=design,
        mu_x0=mu_x0,
        log_fc_x=log_fc_x,
        dispersion=dispersion,
        seed=seed,
    )

    mu_x1 = mu_x0 * np.exp(log_fc_x)

    c1_expected = theoretical_c1(
        mu_x1=mu_x1,
        mu_x0=mu_x0,
        n1=n1,
        n0=n0,
        p=p,
        q=q,
    )

    beta_x_hat = fit_single_variable_nb(y, design["x"], dispersion=dispersion)
    c1_z_hat = fit_single_variable_nb(y, design["z"], dispersion=dispersion)

    return {
        "n0": n0,
        "n1": n1,
        "p": p,
        "q": q,
        "mu_x0": mu_x0,
        "mu_x1": mu_x1,
        "true_log_fc_x": log_fc_x,
        "fitted_coef_y_on_x": beta_x_hat,
        "theoretical_coef_y_on_z": c1_expected,
        "fitted_coef_y_on_z": c1_z_hat,
        "abs_error_z": abs(c1_z_hat - c1_expected),
    }


def run_replicates(
    n_reps = 100,
    **kwargs,
):
    rows = []

    for seed in range(n_reps):
        rows.append(run_one_check(seed=seed, **kwargs))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run_replicates(
        n_reps=100,
        n0=1000,
        n1=1000,
        p=0.8,
        q=0.2,
        mu_x0=100.0,
        log_fc_x=1.0,
        dispersion=0.1,
    )

    print("\nMean results across replicates:")
    print(
        results[
            [
                "true_log_fc_x",
                "fitted_coef_y_on_x",
                "theoretical_coef_y_on_z",
                "fitted_coef_y_on_z",
                "abs_error_z",
            ]
        ].mean()
    )

    results.to_csv("nb_cooccurrence_theory_check.csv", index=False)
    print("\nSaved: nb_cooccurrence_theory_check.csv")