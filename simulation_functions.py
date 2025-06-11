import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from scipy.stats import nbinom
from pydeseq2.default_inference import DefaultInference
import random 
import numpy.random as npr

def sim_mut(mutation_probabilities, n_samples):
    """
    Simulate binary mutation matrix from per-gene mutation probabilities.

    Parameters:
    -----------
    mutation_probabilities : list or np.ndarray
        Length-G array, where each value is the probability of mutation for one gene.
        (G = number of genes/alterations)
    n_samples : int
        Number of synthetic samples to generate.

    Returns:
    --------
    sim_mutations : np.ndarray
        Binary mutation matrix of shape (n_samples × G),
        where 1 = mutated, 0 = wild-type.
    """
    sim_mutations = np.zeros((n_samples, len(mutation_probabilities)))

    # For each gene, simulate mutation presence across samples
    for i in range(len(mutation_probabilities)):
        sim_mutations[:, i] = np.random.binomial(
            n=1,
            p = mutation_probabilities.iloc[i] if isinstance(mutation_probabilities, pd.Series) else mutation_probabilities[i],
            size=n_samples
        )

    return sim_mutations



def sim_cna(means, variances, n_samples):
    """
    Simulate CNA values using a Negative Binomial distribution.

    Parameters
    ----------
    means : array-like or pd.Series, shape (n_genes,)
        Mean CNA values per gene.
    variances : array-like or pd.Series, shape (n_genes,)
        Variance of CNA values per gene.
    n_samples : int
        Number of samples to simulate.

    Returns
    -------
    sim_cna : np.ndarray, shape (n_samples, n_genes)
        Simulated CNA values.
    """
    n_genes = len(means)
    sim_cna = np.zeros((n_samples, n_genes))

    for i in range(n_genes):
        mu = means.iloc[i] if isinstance(means, pd.Series) else means[i]
        var = variances.iloc[i] if isinstance(variances, pd.Series) else variances[i]

        # Ensure valid variance
        if var <= mu or not np.isfinite(var):
            var = mu + max(1e-3, mu * 0.05)

        r = mu**2 / (var - mu)
        r = max(r, 1e-3)  # must be > 0
        p = r / (r + mu)
        p = min(max(p, 1e-6), 1 - 1e-6)  # keep p in (0, 1)

        try:
            sim_cna[:, i] = nbinom.rvs(n=r, p=p, size=n_samples)
        except ValueError as e:
            print(f"[sim_cna] Skipping gene {i}: Invalid NB params r={r}, p={p} (mu={mu}, var={var})")
            sim_cna[:, i] = 0

    return sim_cna


def sim_by_cluster(mut, cna, subtype, n_samples):
    """
    Simulate mutational and copy number data by subtype cluster.

    Parameters
    ----------
    mut : pd.DataFrame
        Binary mutation matrix (samples × genes).
    cna : pd.DataFrame
        Log2 CNA matrix (samples × genes).
    subtype : pd.DataFrame
        DataFrame with a 'Subtype' column indexed by sample.
    n_samples : int
        Total number of samples to simulate.

    Returns
    -------
    synthetic_mut : pd.DataFrame
        Simulated mutation matrix.
    synthetic_cna : pd.DataFrame
        Simulated CNA matrix.
    """
    M = pd.concat([mut, subtype], axis=1)
    C = pd.concat([cna, subtype], axis=1)
    subtype_counts = subtype['Subtype'].value_counts()
    subtypes = subtype_counts.index.tolist()

    proportions = subtype_counts / subtype_counts.sum()

    # Step 2: Get rounded sample sizes
    rounded_sizes = (proportions * n_samples).round().astype(int)

    # Step 3: Adjust the last subtype to make total exactly 600
    difference = n_samples - rounded_sizes.sum()
    rounded_sizes.iloc[-1] += difference  # may be negative or positive


    new_mut = []
    new_cna = []

    for s, n in zip(subtypes, rounded_sizes):
        # Subtype-specific samples
        M_s = M[M['Subtype'] == s].drop(columns='Subtype')
        C_s = C[C['Subtype'] == s].drop(columns='Subtype')

        # Estimate mutation probabilities
        mut_probs = M_s.mean(axis=0)

        # Shift CNA to be non-negative for Poisson
        means = C_s.mean()
        var = C_s.var()

        # Simulate mutations and CNAs
        mut_cluster = sim_mut(mut_probs, n_samples=n)
        cna_cluster = sim_cna(means, var, n_samples=n)

        new_mut.append(mut_cluster)
        new_cna.append(cna_cluster)

    # Recombine cluster-wise results
    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]

    synthetic_mut = pd.DataFrame(np.vstack(new_mut), columns=mut.columns, index=sample_names)
    synthetic_cna = pd.DataFrame(np.vstack(new_cna), columns=cna.columns, index=sample_names)

    return synthetic_mut, synthetic_cna

"""
EXPRESSION  SIMULATION FUNCTIONS
"""

"""
EXPRESSION SIMULATION FUNCTIONS
"""

import numpy as np
import pandas as pd
import random
from scipy.stats import nbinom
from pydeseq2.dds import DeseqDataSet


def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=None, condition='A'):
    """
    Estimate DESeq2-style parameters from real RNA-seq data.
    Returns gene means, variances, dispersions, and size factors.
    """
    if seed is not None:
        np.random.seed(seed)

    rna_df = rna_df.round().astype(int)
    metadata = pd.DataFrame({'condition': [condition] * rna_df.shape[0]}, index=rna_df.index)

    dds = DeseqDataSet(
        counts=rna_df,
        metadata=metadata,
        design_factors="condition",
        ref_level=condition
    )
    dds.deseq2()

    gene_means = rna_df.mean()
    gene_vars = rna_df.var()
    dispersions = pd.Series(dds.varm["dispersions"], index=rna_df.columns)
    size_factors = dds.obsm['size_factors']

    return gene_means, gene_vars, dispersions, size_factors


def simulate_rna_background(gene_means, dispersions, size_factors, n_samples):
    """
    Simulate RNA expression background using NB with DESeq2-style parameters.
    """
    mu_matrix = np.outer(size_factors, gene_means)
    return pd.DataFrame(mu_matrix, columns=gene_means.index)


def generate_signatures(genes, alteration_features, min_size=1, max_size=150):
    """
    Create random expression signatures for each alteration.
    Ensures each altered gene is a target of its own signature.
    """
    signatures = {}
    for alt in alteration_features:
        base_gene = alt.split('_')[0]
        if base_gene not in genes:
            continue
        n_targets = random.randint(min_size, max_size)
        potential_targets = [g for g in genes if g != base_gene]
        if len(potential_targets) < n_targets - 1:
            continue
        targets = [base_gene] + random.sample(potential_targets, k=n_targets - 1)
        effects = {t: np.round(np.random.uniform(-2.5, 2.5), 2) for t in targets}
        signatures[alt] = {'targets': targets, 'effects': effects}
    return signatures


def inject_expression_effects(expr_df, alteration_df, signatures, cna_df=None):
    """
    Inject additive alteration effects into expression matrix.
    Optionally includes CNA multiplicative modulation.
    """
    modified = expr_df.copy()
    for sample in expr_df.index:
        active_alts = alteration_df.columns[alteration_df.loc[sample] == 1]
        for gene in expr_df.columns:
            alpha_i = expr_df.loc[sample, gene]  # baseline
            beta_sum = 0
            for alt in active_alts:
                if alt in signatures and gene in signatures[alt]['targets']:
                    beta_sum += signatures[alt]['effects'][gene]
            copy_factor = cna_df.loc[sample, gene] if (cna_df is not None and gene in cna_df.columns) else 1.0
            expr_value = (beta_sum + alpha_i) * copy_factor
            modified.loc[sample, gene] = max(0, expr_value)
    return modified


def sample_nb(mu, dispersions):
    """
    Sample from Negative Binomial for each gene/sample using DESeq2-style dispersion.
    """
    r = 1 / dispersions.values
    r = np.clip(r, 1e-6, 1e6)
    p = r / (r + mu)
    counts = np.random.negative_binomial(n=r, p=p)
    return counts


def simulate_rna_with_signatures(
    rna_df,
    alteration_df,
    cna_df,
    n_samples=600,
    min_sig_size=1,
    max_sig_size=150,
    n_genes_to_sim=10000,
    seed=None
):
    """
    Main simulation function: combines background expression, mutation/CNA effects, and sampling.
    """
    if seed is not None:
        np.random.seed(seed)

    gene_means, gene_vars, dispersions, size_factors = estimate_deseq2_parameters(rna_df, seed=seed)
   
    altered_genes = set([a.split('_')[0] for a in alteration_df.columns])
    top_genes = gene_means.sort_values(ascending=False).head(n_genes_to_sim).index
    genes_to_sim = sorted(set(top_genes).union(altered_genes).intersection(rna_df.columns))

    rna_df = rna_df[genes_to_sim]
    gene_means = gene_means[genes_to_sim]
    gene_vars = gene_vars[genes_to_sim]
    dispersions = dispersions[genes_to_sim]

    if all(col.endswith('_CNA') for col in cna_df.columns):
        cna_df.columns = cna_df.columns.str.replace('_CNA', '', regex=False)
    cna_df = cna_df.loc[:, cna_df.columns.isin(genes_to_sim)]

    expr_bg = simulate_rna_background(gene_means, dispersions, size_factors[:n_samples], n_samples)
    expr_bg.index = [f"Sample_{i+1}" for i in range(n_samples)]
    alteration_df = alteration_df.copy()
    alteration_df.index = expr_bg.index
    if cna_df is not None:
        cna_df = cna_df.copy()
        cna_df.index = expr_bg.index


    all_alts = alteration_df.columns.tolist()
    true_signatures = generate_signatures(expr_bg.columns.tolist(), all_alts, min_sig_size, max_sig_size)

    expr_effected = inject_expression_effects(expr_bg, alteration_df, true_signatures, cna_df)
    expr_counts = sample_nb(expr_effected.values, dispersions)

    expr_sim = pd.DataFrame(expr_counts, columns=expr_bg.columns, index=expr_bg.index)
    return expr_sim, true_signatures
