import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from scipy.stats import nbinom, binom
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
    Simulate CNA values using Binomial or Negative Binomial distribution 
    based on the relationship between mean and variance.

    Parameters
    ----------
    means : pd.Series, shape (n_genes,)
        Mean CNA values per gene.
    variances : pd.Series, shape (n_genes,)
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
        mu = means.iloc[i]
        var = variances.iloc[i]

        if not np.isfinite(mu) or not np.isfinite(var) or mu <= 0:
            continue

        if var <= mu:
            # Binomial Distribution
            p = 1 - (var/mu)
            n = mu/p
            sim_cna[:, i] = np.random.binomial(n=n, p=p, size=n_samples)
        else:
            # Use Negative Binomial: variance > mean
            r = mu**2 / (var - mu)
            p = r / (r + mu)
            sim_cna[:, i] = nbinom.rvs(n=r, p=p, size=n_samples)

    return sim_cna



def sim_by_cluster(mut, subtype, n_samples, cna=None, fusions=None):
    """
    Simulate mutation, CNA, and/or fusion data stratified by subtype.

    Parameters
    ----------
    mut : pd.DataFrame
        Binary mutation matrix (samples × genes).
    subtype : pd.DataFrame
        DataFrame with 'Subtype' column, indexed by sample.
    n_samples : int
        Total number of samples to simulate.
    cna : pd.DataFrame, optional
        CNA matrix (samples × genes), typically continuous.
    fusions : pd.DataFrame, optional
        Binary fusion matrix (samples × fusions).

    Returns
    -------
    dict
        Dictionary with keys corresponding to each simulated data type.
        e.g. {'mut': synthetic_mut, 'fusion': synthetic_fusion, 'cna': synthetic_cna}
    """
    # Combine mutation and subtype
    M = pd.concat([mut, subtype], axis=1)

    if fusions is not None:
        F = pd.concat([fusions, subtype], axis=1)
    if cna is not None:
        C = pd.concat([cna, subtype], axis=1)

    # Count subtype distributions
    
    subtype_counts = subtype['Subtype'].value_counts()
    subtypes = subtype_counts.index.tolist()
    proportions = subtype_counts / subtype_counts.sum()

    # Compute sample sizes per cluster
    rounded_sizes = (proportions * n_samples).round().astype(int)
    rounded_sizes.iloc[-1] += n_samples - rounded_sizes.sum()  # Adjust total

    new_mut = []
    new_fusions = [] if fusions is not None else None
    new_cna = [] if cna is not None else None

    for s, n in zip(subtypes, rounded_sizes):
        # Subtype-specific splits
        M_s = M[M['Subtype'] == s].drop(columns='Subtype')
        M_s = M_s.apply(pd.to_numeric, errors='coerce')
        mut_probs = M_s.mean(axis=0).fillna(0).clip(0, 1)
        mut_cluster = sim_mut(mut_probs, n_samples=n)
        new_mut.append(mut_cluster)

        if fusions is not None:
            F_s = F[F['Subtype'] == s].drop(columns='Subtype')
            F_s = F_s.apply(pd.to_numeric, errors='coerce')
            fus_probs = F_s.mean(axis=0).fillna(0).clip(0, 1)
            fus_cluster = sim_mut(fus_probs, n_samples=n)
            new_fusions.append(fus_cluster)

        if cna is not None:
            C_s = C[C['Subtype'] == s].drop(columns='Subtype')
            means = C_s.mean()
            var = C_s.var()
            cna_cluster = sim_cna(means, var, n_samples=n)
            new_cna.append(cna_cluster)

    # Assemble final outputs
    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    output = {
        'mut': pd.DataFrame(np.vstack(new_mut), columns=mut.columns, index=sample_names)
    }

    if fusions is not None:
        output['fusion'] = pd.DataFrame(np.vstack(new_fusions), columns=fusions.columns, index=sample_names)

    if cna is not None:
        output['cna'] = pd.DataFrame(np.vstack(new_cna), columns=cna.columns, index=sample_names)

    return output

"""
EXPRESSION  SIMULATION FUNCTIONS
"""



def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=None, condition='A'):
    """
    Estimate DESeq2-style parameters from real RNA-seq data.
    Returns gene means, variances, dispersions, and size factors.
    """
    if seed is not None:
        np.random.seed(seed)

    rna_df = rna_df.round().astype(int)
    # Alternate conditions to ensure at least two levels
    fake_conditions = ['A'] * (rna_df.shape[0] // 2) + ['B'] * (rna_df.shape[0] - rna_df.shape[0] // 2)
    metadata = pd.DataFrame({'condition': fake_conditions}, index=rna_df.index)

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
    Generate random expression signatures for mutations or fusions.
    For fusions, ensure both partner genes are targets.
    """
    signatures = {}
    for alt in alteration_features:
        if "_FUSION" in alt:
            # Fusion case: extract partner genes
            partners = alt.replace("_FUSION", "").split("-")
            fusion_targets = [g for g in partners if g in genes]

            # Fill out target list
            n_targets = random.randint(min_size, max_size)
            potential_targets = [g for g in genes if g not in fusion_targets]
            n_remaining = n_targets - len(fusion_targets)
            if len(potential_targets) < n_remaining:
                continue
            targets = fusion_targets + random.sample(potential_targets, k=n_remaining)

        else:
            base_gene = alt.split('_')[0]
            if base_gene not in genes:
                continue
            n_targets = random.randint(min_size, max_size)
            potential_targets = [g for g in genes if g != base_gene]
            if len(potential_targets) < n_targets - 1:
                continue
            targets = [base_gene] + random.sample(potential_targets, k=n_targets - 1)

        # Assign random effects (fusion or mutation)
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
            expr_value = (beta_sum + alpha_i) * copy_factor /2
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
    cna_df=None,
    n_samples=600,
    min_sig_size=1,
    max_sig_size=150,
    n_genes_to_sim=10000,
    seed=None
):
    """
    Simulate RNA-seq counts with expression effects from mutations, fusions, and optional CNAs.
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Estimate RNA-seq parameters
    gene_means, gene_vars, dispersions, size_factors = estimate_deseq2_parameters(rna_df, seed=seed)

    # Step 2: Identify genes affected by alterations (including fusion partners)
    altered_genes = set()
    for alt in alteration_df.columns:
        if "_FUSION" in alt:
            partners = alt.replace("_FUSION", "").split("-")
            altered_genes.update(partners)
        else:
            base_gene = alt.split("_")[0]
            altered_genes.add(base_gene)

    # Step 3: Select expression genes to simulate
    top_genes = gene_means.sort_values(ascending=False).head(n_genes_to_sim).index
    genes_to_sim = sorted(set(top_genes).union(altered_genes).intersection(rna_df.columns))

    # Step 4: Subset all expression-related objects
    rna_df = rna_df[genes_to_sim]
    gene_means = gene_means[genes_to_sim]
    gene_vars = gene_vars[genes_to_sim]
    dispersions = dispersions[genes_to_sim]

    # Step 5: Subset CNA (if provided)
    if cna_df is not None:
        if all(col.endswith('_CNA') for col in cna_df.columns):
            cna_df = cna_df.copy()
            cna_df.columns = cna_df.columns.str.replace('_CNA', '', regex=False)
        cna_df = cna_df.loc[:, cna_df.columns.isin(genes_to_sim)]

    # Step 6: Simulate background expression
    expr_bg = simulate_rna_background(gene_means, dispersions, size_factors[:n_samples], n_samples)
    expr_bg.index = [f"Sample_{i+1}" for i in range(n_samples)]

    # Step 7: Align alteration and CNA data
    alteration_df = alteration_df.copy()
    alteration_df.index = expr_bg.index
    if cna_df is not None:
        cna_df = cna_df.copy()
        cna_df.index = expr_bg.index

    # Step 8: Generate signatures (fusion-aware version assumed)
    all_alts = alteration_df.columns.tolist()
    true_signatures = generate_signatures(expr_bg.columns.tolist(), all_alts, min_sig_size, max_sig_size)

    # Step 9: Inject alteration-driven effects
    expr_effected = inject_expression_effects(expr_bg, alteration_df, true_signatures, cna_df)

    # Step 10: Sample final RNA-seq counts
    expr_counts = sample_nb(expr_effected.values, dispersions)
    expr_sim = pd.DataFrame(expr_counts, columns=expr_bg.columns, index=expr_bg.index)

    return expr_sim, true_signatures

