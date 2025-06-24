import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from scipy.stats import nbinom, binom
from pydeseq2.default_inference import DefaultInference
import random 
import numpy.random as npr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def preprocess_alteration_matrix(mut_df, fus_df=None, cna_df=None):
    components = []

    if mut_df is not None:
        mut_df = mut_df.astype(int)
        components.append(mut_df)

    if fus_df is not None:
        fus_df = fus_df.astype(int)
        components.append(fus_df)

    if cna_df is not None:
        cna_scaled = pd.DataFrame(
            StandardScaler().fit_transform(cna_df),
            index=cna_df.index,
            columns=cna_df.columns
        )
        components.append(cna_scaled)

    combined = pd.concat(components, axis=1)
    return combined


def sample_from_neighbors(real_df, n_samples, k_neighbors=5, seed=44):
    np.random.seed(seed)
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
    nn_model.fit(real_df.values)

    sampled_rows = []

    for _ in range(n_samples):
        anchor_idx = np.random.choice(real_df.shape[0])
        anchor = real_df.iloc[anchor_idx]
        neighbors_idx = nn_model.kneighbors([anchor.values], return_distance=False)[0]
        neighborhood = real_df.iloc[neighbors_idx]

        synthetic_row = {}
        for col in real_df.columns:
            values = neighborhood[col].values
            if set(values) <= {0, 1}:
                prob = np.mean(values)
                synthetic_row[col] = np.random.binomial(1, prob)
            else:
                mu, sigma = np.mean(values), np.std(values)
                synthetic_row[col] = np.random.normal(mu, sigma)

        sampled_rows.append(synthetic_row)

    return pd.DataFrame(sampled_rows)


def sim_by_cluster_neighbors(mut, subtype, n_samples, cna=None, fusions=None, seed=44, k_neighbors=5):
    combined = preprocess_alteration_matrix(mut, fusions, cna)
    combined = combined.loc[subtype.index]
    combined = pd.concat([combined, subtype], axis=1)

    subtype_counts = subtype['Subtype'].value_counts()
    proportions = subtype_counts / subtype_counts.sum()
    rounded_sizes = (proportions * n_samples).round().astype(int)
    rounded_sizes.iloc[-1] += n_samples - rounded_sizes.sum()

    new_data = []

    for i, (s, n) in enumerate(zip(rounded_sizes.index, rounded_sizes)):
        subset = combined[combined['Subtype'] == s].drop(columns='Subtype')

        # Clip neighbors to avoid exceeding available samples
        k_actual = min(k_neighbors, subset.shape[0] - 1)
        if k_actual < 1:
            raise ValueError(f"Subtype '{s}' has too few samples ({subset.shape[0]}) to support neighbor sampling.")

        synthetic = sample_from_neighbors(subset, n_samples=n, k_neighbors=k_actual, seed=seed + i)
        new_data.append(synthetic)

    synthetic_full = pd.concat(new_data, axis=0)
    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    synthetic_full.index = sample_names

    output = {}
    if mut is not None:
        output['mut'] = synthetic_full[mut.columns].astype(int)
    if fusions is not None:
        output['fusion'] = synthetic_full[fusions.columns].astype(int)
    if cna is not None:
        output['cna'] = synthetic_full[cna.columns]

    return output


"""
EXPRESSION  SIMULATION FUNCTIONS
"""

def preprocess_rna_for_simulation(rna_df, strategy="auto", user_scale=None, verbose=True):
    """
    Preprocess RNA-seq expression data to convert from normalized form (e.g., CPM, RPKM, RSEM)
    into pseudo-counts for negative binomial simulation.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Normalized RNA expression (samples × genes).
    strategy : str
        'auto' to choose scale based on data, or 'manual' to specify your own.
    user_scale : float or None
        Custom scale factor if strategy is 'manual'.
    verbose : bool
        Whether to print summary info.

    Returns
    -------
    rna_scaled : pd.DataFrame
        Pseudo-count integer RNA expression matrix.
    scale_factor : float
        Scale factor that was applied.
    """

    if strategy == "manual":
        if user_scale is None:
            raise ValueError("You must provide user_scale if using manual strategy.")
        scale_factor = user_scale
    elif strategy == "auto":
        q75 = rna_df.quantile(0.75).median()
        if q75 < 1:
            scale_factor = 100
        elif q75 < 10:
            scale_factor = 50
        elif q75 < 100:
            scale_factor = 10
        else:
            scale_factor = 1
    else:
        raise ValueError("strategy must be 'auto' or 'manual'")

    # Apply pseudocounting and scaling
    rna_scaled = (rna_df * scale_factor).round().astype(int)
    rna_scaled[rna_scaled < 0] = 0

    if verbose:
        print(f"[preprocess_rna] Applied scale factor: {scale_factor}")
        print(rna_scaled.describe().T.head(5))

    return rna_scaled, scale_factor



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
    dispersions = dispersions.fillna(dispersions.median())
    dispersions = dispersions.clip(lower=1e-6, upper=1e6)

    size_factors = dds.obsm['size_factors']

    return gene_means, gene_vars, dispersions, size_factors


def simulate_rna_background(gene_means, dispersions, size_factors, n_samples):
    """
    Simulate RNA expression background using NB with DESeq2-style parameters.
    """
    mu_matrix = np.outer(size_factors, gene_means)
    return pd.DataFrame(mu_matrix, columns=gene_means.index)


def generate_signatures(genes, alteration_features, min_size=1, max_size=150, seed=44):
    """
    Generate random expression signatures for mutations or fusions.
    For fusions, ensure both partner genes are targets.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

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
    Sample RNA-seq counts using a Negative Binomial distribution.
    Handles edge cases and clips invalid values.
    """
    mu = np.asarray(mu)
    if isinstance(dispersions, pd.Series):
        dispersions = dispersions.values

    # Check for NaNs or infs in inputs
    if np.any(~np.isfinite(mu)):
        raise ValueError("mu contains non-finite values")
    if np.any(~np.isfinite(dispersions)):
        raise ValueError("dispersions contain non-finite values")

    mu = np.clip(mu, 1e-8, 1e6)
    dispersions = np.clip(dispersions, 1e-8, 1e6)

    r = 1.0 / dispersions
    r = np.clip(r, 1e-6, 1e6)

    if mu.shape[1] != r.shape[0]:
        raise ValueError(f"Shape mismatch: mu {mu.shape}, r {r.shape}")

    r = r[np.newaxis, :]
    p = r / (r + mu)

    if np.any(np.isnan(p)) or np.any(p <= 0) or np.any(p >= 1):
        raise ValueError(
            f"Invalid p in NB sampling: min={np.nanmin(p)}, max={np.nanmax(p)}, NaNs={np.isnan(p).sum()}"
        )

    return np.random.negative_binomial(n=r, p=p)





def simulate_rna_with_signatures(
    rna_df,
    alteration_df,
    cna_df=None,
    n_samples=600,
    min_sig_size=1,
    max_sig_size=150,
    n_genes_to_sim=10000,
    seed=44
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
    true_signatures = generate_signatures(expr_bg.columns.tolist(), all_alts, min_sig_size, max_sig_size, seed=seed)

    # Step 9: Inject alteration-driven effects
    expr_effected = inject_expression_effects(expr_bg, alteration_df, true_signatures, cna_df)

    # Step 10: Sample final RNA-seq counts
    dispersions = dispersions[expr_effected.columns]
    print("expr_effected shape:", expr_effected.shape)
    print("dispersions shape:", dispersions.shape)
    print("NaNs in dispersions:", dispersions.isna().sum())
    print("NaNs in expr_effected:", np.isnan(expr_effected.values).sum())

    expr_counts = sample_nb(expr_effected.values, dispersions)
    expr_sim = pd.DataFrame(expr_counts, columns=expr_bg.columns, index=expr_bg.index)

    return expr_sim, true_signatures

