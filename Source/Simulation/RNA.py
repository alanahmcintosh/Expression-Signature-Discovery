import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from scipy.stats import norm, nbinom
from pydeseq2.dds import DeseqDataSet  # or your wrapper import

# ==============================================================
# 1️⃣  DESeq2-LIKE PARAMETER ESTIMATION
# ==============================================================

def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=44, condition='A'):
    """
    Estimate DESeq2-style parameters (mean, dispersion, and size factors)
    from a real RNA count matrix.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Real RNA count matrix (samples x genes).
    size_factor_sd : float
        Std deviation for log-normal size factor variation.
    seed : int
        Random seed for reproducibility.
    condition : str
        Reference level for DESeq2.

    Returns
    -------
    gene_means : pd.Series
        Mean counts per gene.
    gene_vars : pd.Series
        Variance per gene.
    dispersions : pd.Series
        Gene-level dispersion estimates.
    size_factors : pd.Series
        Per-sample normalization factors.
    """
    if seed is not None:
        np.random.seed(seed)

    # Round to integers — DESeq2 expects counts
    rna_df = rna_df.round().astype(int)

    # Fake two-group design to trigger DESeq2 estimation
    fake_conditions = (
        ['A'] * (rna_df.shape[0] // 2)
        + ['B'] * (rna_df.shape[0] - rna_df.shape[0] // 2)
    )
    metadata = pd.DataFrame({'condition': fake_conditions}, index=rna_df.index)

    # Run DESeq2
    dds = DeseqDataSet(
        counts=rna_df,
        metadata=metadata,
        design_factors="condition",
        ref_level=condition
    )
    dds.deseq2()

    # Retrieve means, variances, dispersions, and size factors
    gene_means = rna_df.mean()
    gene_vars = rna_df.var()
    dispersions = pd.Series(dds.varm["dispersions"], index=rna_df.columns).fillna(0.1)
    dispersions = dispersions.clip(lower=1e-6, upper=1e6)
    size_factors = pd.Series(dds.obsm['size_factors'], index=rna_df.index, name="size_factor")

    return gene_means, gene_vars, dispersions, size_factors


# ==============================================================
# 2️⃣  SIZE FACTOR RESAMPLING (SUBTYPE-AWARE)
# ==============================================================

def draw_size_factors_from_deseq(size_factors, n_samples, subtype=None, seed=44):
    """
    Resample DESeq2 size factors (sample-level normalization coefficients)
    to assign realistic per-sample scaling in the simulated data.

    Optionally stratified by subtype proportions.
    """
    rng = np.random.default_rng(seed)
    sf = size_factors.dropna()

    # No subtypes: resample directly
    if subtype is None:
        return rng.choice(sf.values, size=n_samples, replace=True)

    # Subtype-aware sampling: preserve subtype proportions
    subtype = subtype.loc[sf.index]
    counts = subtype.value_counts()
    props = counts / counts.sum()
    alloc = (props * n_samples).round().astype(int)
    alloc.iloc[-1] += n_samples - alloc.sum()  # ensure sum == n_samples

    out = []
    for s, n in alloc.items():
        pool = sf.loc[subtype[subtype == s].index]
        if len(pool) == 0:
            pool = sf  # fallback
        out.append(rng.choice(pool.values, size=int(n), replace=True))
    return np.concatenate(out)[:n_samples]


# ==============================================================
# 3️⃣  PCA + KNN MIXING: RNA BACKGROUND GENERATOR
# ==============================================================

def knn_mix_block_pca(real_block, n_samples, n_pcs=30, k=30,
                       mix_conc=1.0, residual_scale=0.5,
                       seed=44, metric="euclidean"):
    """
    Generate synthetic RNA mean expression profiles (μ) by mixing
    nearest neighbours in PCA space.

    This preserves the overall correlation structure of real data
    while allowing unlimited new samples.

    Procedure:
      1. log1p-transform and z-score the real data
      2. run PCA to capture major gene–gene structure
      3. for each synthetic sample:
         - select a random "anchor" sample
         - find its k nearest neighbors in PCA space
         - mix neighbors using Dirichlet weights (controls diversity)
         - optionally add scaled residual noise (fine-grained variation)
         - transform back to original expression scale
    """

    # Random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # -----------------------------------------
    # 1. Log-transform input
    # -----------------------------------------
    R = real_block.clip(lower=0)      # ensure no negatives
    L = np.log1p(R)                   # log1p stabilizes variance

    # -----------------------------------------
    # 2. Z-score normalization (gene-wise)
    # -----------------------------------------
    mu = L.mean(axis=0)
    sd = L.std(axis=0, ddof=0).replace(0, 1.0)  # avoid division by zero
    Z = (L - mu) / sd                           # standardized matrix

    # -----------------------------------------
    # 3. PCA: capture major structure (dimension reduction)
    # -----------------------------------------
    # Limit PCs to something meaningful (can't exceed n_samples or n_genes)
    n_comp = max(2, min(n_pcs, Z.shape[0] - 1, Z.shape[1]))
    pca = PCA(n_components=n_comp, random_state=0).fit(Z)

    S = pca.transform(Z)              # samples in PC space
    Z_recon = pca.inverse_transform(S)
    Z_resid = Z - Z_recon             # residuals = fine-scale variation

    # -----------------------------------------
    # 4. KNN model in PCA space
    # -----------------------------------------
    k_ = max(2, min(k, len(S)))       # ensure k is valid
    nn = NearestNeighbors(n_neighbors=k_, metric=metric).fit(S)

    mixes = []

    # -----------------------------------------
    # 5. Generate synthetic samples
    # -----------------------------------------
    for i in range(n_samples):

        # Choose anchor sample
        a = int(rng.integers(0, len(S)))

        # Get its k nearest neighbors in PC space
        _, idxs = nn.kneighbors(S[a:a+1])
        nbrS = S[idxs[0]]

        # -----------------------------------------
        # Mix neighbors using Dirichlet weights
        # mix_conc controls how "spread out" the mixture is:
        #   - low conc => more diverse, sharp mixtures
        #   - high conc => neighbors weighted more evenly
        # -----------------------------------------
        w = rng.dirichlet(np.full(nbrS.shape[0], mix_conc))
        s_mix = np.average(nbrS, axis=0, weights=w)

        # Back-project from PCA space to standardized gene space
        z_mix = pca.inverse_transform(s_mix)

        # -----------------------------------------
        # Add local fine-grained variation
        # (residuals represent structure not captured by PCA)
        # -----------------------------------------
        if residual_scale and residual_scale > 0:
            j = int(rng.choice(idxs[0]))     # pick one of the neighbors
            z_mix = z_mix + residual_scale * Z_resid.iloc[j].values

        # -----------------------------------------
        # 6. Transform back to original expression scale
        # -----------------------------------------
        mu_expr = np.expm1(z_mix * sd.values + mu.values)
        mu_expr[mu_expr < 0] = 0.0           # numerical protection
        mixes.append(mu_expr)

    # Final synthetic μ matrix
    sim_mu = pd.DataFrame(mixes, columns=real_block.columns)
    return sim_mu



def simulate_background_knn(real_rna, n_samples, n_pcs=30, k=30,
                            mix_conc=1.0, residual_scale=0.5, seed=44,
                            subtype=None, size_factors_to_apply=None,
                            metric="euclidean"):
    """
    Generate background RNA mean expression (μ) using KNN–PCA mixing.

    If subtype labels are provided, the simulation preserves subtype 
    proportions and generates expression separately within each subtype.

    Optionally apply DESeq2-style size factors afterward.
    """

    # ------------------------------------------------------
    # Case 1: No subtype structure — simulate from entire dataset
    # ------------------------------------------------------
    if subtype is None:
        sim_mu = knn_mix_block_pca(
            real_block=real_rna,
            n_samples=n_samples,
            n_pcs=n_pcs,
            k=k,
            mix_conc=mix_conc,
            residual_scale=residual_scale,
            seed=seed,
            metric=metric
        )

    # ------------------------------------------------------
    # Case 2: Subtype-aware simulation
    # - simulate each subtype separately
    # - keep subtype proportions in the synthetic dataset
    # ------------------------------------------------------
    else:
        subtype = subtype.loc[real_rna.index]  # align labels
        counts = subtype.value_counts()
        props = counts / counts.sum()

        # Allocate number of synthetic samples per subtype
        alloc = (props * n_samples).round().astype(int)
        alloc.iloc[-1] += n_samples - alloc.sum()  # fix rounding

        parts = []

        # Simulate each subtype independently
        for i, (s, n) in enumerate(alloc.items()):
            block = real_rna.loc[subtype[subtype == s].index]

            # Use different seeds for each subtype block
            # so they don’t produce identical samples
            block_seed = seed + i if seed is not None else None

            part = knn_mix_block_pca(
                real_block=block,
                n_samples=int(n),
                n_pcs=n_pcs,
                k=k,
                mix_conc=mix_conc,
                residual_scale=residual_scale,
                seed=block_seed,
                metric=metric
            )
            parts.append(part)

        # Combine subtype-specific simulations into one dataframe
        sim_mu = pd.concat(parts, axis=0)

    # ------------------------------------------------------
    # Final formatting + apply size factors
    # ------------------------------------------------------

    # Give synthetic samples clean names
    sim_mu.index = [f"Sample_{i+1}" for i in range(n_samples)]

    # Apply provided DESeq2 size factors (optional)
    if size_factors_to_apply is not None:
        if len(size_factors_to_apply) != n_samples:
            raise ValueError("size_factors_to_apply length mismatch.")

        # Multiply rows by their corresponding size factor
        sim_mu = sim_mu.mul(size_factors_to_apply, axis=0)

    return sim_mu



# ==============================================================
# 4️⃣  SIGNATURE + EFFECT INJECTION
# ==============================================================

def generate_signatures_simplified(
    genes,
    alteration_features,
    min_size=10,
    max_size=150,
    seed=44,
    gof_range=(-0.8, +2.5),
    lof_range=(-2.5, +0.8),
    fusion_range=(-2.0, +2.0)
):
    """
    Generate synthetic expression signatures for each alteration based on
    simplified biological rules.

    Rules:
      - GOF mutations & CNA gains → upregulated signatures
      - LOF mutations & CNA losses → downregulated signatures
      - MUT / uncertain → no signature
      - Fusions → unique mixed-direction signatures
    """

    # Set global random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    signatures = {}
    gene_set = set(genes)

    for alt in alteration_features:

        # Parse alteration name → type, gene, partners (for fusions)
        kind, base_gene, partners = _parse_alt(alt)

        # ------------------------------------------------------
        # 1. Skip uncertain/passenger mutations entirely
        # ------------------------------------------------------
        if kind == "UNCERTAIN":
            continue

        # ------------------------------------------------------
        # 2. Fusions have their own signature logic
        # ------------------------------------------------------
        if kind == "FUSION":
            # Keep only fusion partner genes that exist in the gene list
            fusion_targets = [g for g in (partners or []) if g in gene_set]

            # If none of the partners are in the expression matrix, skip
            if not fusion_targets:
                continue

            # Choose signature size
            n_targets = random.randint(min_size, max_size)

            # Add remaining targets from other available genes
            remaining = [g for g in genes if g not in fusion_targets]
            extra = random.sample(remaining, k=max(0, n_targets - len(fusion_targets)))

            targets = fusion_targets + extra

            # Fusions get mixed-direction effects (can be + or -)
            effects = {
                t: float(np.round(np.random.uniform(*fusion_range), 2))
                for t in targets
            }

            signatures[alt] = {"targets": targets, "effects": effects}
            continue

        # ------------------------------------------------------
        # 3. GOF / LOF mutations and CNA changes
        # ------------------------------------------------------
        # Skip if the gene isn’t in the expression dataset
        if base_gene not in gene_set:
            continue

        # Determine signature size
        n_targets = random.randint(min_size, max_size)

        # Avoid sampling the base gene twice
        pool = [g for g in genes if g != base_gene]

        # Signature always includes the gene itself
        targets = [base_gene] + random.sample(pool, k=n_targets - 1)

        # ------------------------------------------------------
        # Choose effect direction based on alteration type
        # ------------------------------------------------------
        if kind == "GOF":
            rng = gof_range
        elif kind == "LOF":
            rng = lof_range
        elif kind.endswith("_CNA"):
            # Interpret CNA as GOF-like or LOF-like
            if "gain" in alt.lower() or "amp" in alt.lower():
                rng = gof_range
            elif "loss" in alt.lower() or "del" in alt.lower():
                rng = lof_range
            else:
                rng = (-1, 1)
        else:
            rng = (-1, 1)

        # Draw per-gene effects
        effects = {
            t: float(np.round(np.random.uniform(*rng), 2))
            for t in targets
        }

        # ------------------------------------------------------
        # Strengthen the effect on the altered gene itself
        # ------------------------------------------------------
        if base_gene in effects:
            if rng == gof_range:
                # Make self-effect strongly upregulated
                effects[base_gene] = float(
                    np.round(np.random.uniform(1.2, gof_range[1]), 2)
                )
            elif rng == lof_range:
                # Make self-effect strongly downregulated
                effects[base_gene] = float(
                    np.round(np.random.uniform(lof_range[0], -1.2), 2)
                )

        signatures[alt] = {"targets": targets, "effects": effects}

    return signatures



def log2_to_ratio(cna_log2: pd.DataFrame) -> pd.DataFrame:
    """Convert log2(CN/2) → CN/2 ratio."""
    cna_ratio = np.power(2, cna_log2)
    cna_ratio = cna_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return cna_ratio

def inject_expression_effects_simplified(
    expr_df: pd.DataFrame,
    alteration_df: pd.DataFrame,
    signatures: dict,
    cna_df: pd.DataFrame | None = None,
    floor_zero: bool = True
):
    """
    Add alteration-driven expression effects onto a background RNA matrix.

    Effects are additive:
      - Each alteration contributes its own signature.
      - CNA values modulate the strength of the effect.
      - GOF + CNA gain amplify upregulation.
      - LOF + CNA loss amplify downregulation.

    If floor_zero=True, negative values are clipped to 0.
    """

    modified = expr_df.copy()

    # ------------------------------------------------------
    # 1. Convert CNA to ratio-scale if it looks log2-like
    #    (typical log2 CNA values rarely exceed ±3)
    # ------------------------------------------------------
    if cna_df is not None and cna_df.max().max() < 3:
        cna_ratio = log2_to_ratio(cna_df)
    else:
        cna_ratio = cna_df.copy() if cna_df is not None else None

    # ------------------------------------------------------
    # 2. Process each sample independently
    # ------------------------------------------------------
    for sample in expr_df.index:

        # Identify all alterations active in this sample
        active_alts = [
            alt for alt in alteration_df.columns
            if alteration_df.loc[sample, alt] == 1 and alt in signatures
        ]

        # ------------------------------------------------------
        # 3. Inject expression effects for each alteration
        # ------------------------------------------------------
        for alt in active_alts:
            sig = signatures[alt]
            kind, base_gene, partners = _parse_alt(alt)

            # Loop over all target genes in the signature
            for tgt, eff in sig["effects"].items():
                if tgt not in modified.columns:
                    continue

                total_effect = eff  # begin with signature effect

                # ------------------------------------------------------
                # 4. Modulate effect by CNA for the base gene (if available)
                # ------------------------------------------------------
                if cna_ratio is not None and base_gene in cna_ratio.columns:
                    cna_val = cna_ratio.loc[sample, base_gene]

                    if not np.isnan(cna_val):
                        # Base scaling proportional to copy number level
                        total_effect *= cna_val

                        # Reinforcement for consistent alteration + CNA direction
                        if kind == "GOF" and cna_val > 1:
                            total_effect += eff * (cna_val - 1)
                        elif kind == "LOF" and cna_val < 1:
                            total_effect += eff * (1 - cna_val)

                # Apply effect to the sample × gene entry
                modified.at[sample, tgt] += total_effect

    # ------------------------------------------------------
    # 5. Ensure values don’t drop below zero
    # ------------------------------------------------------
    if floor_zero:
        modified[modified < 0] = 0.0

    return modified


def simulate_rna_with_signatures(
    rna_df,
    alteration_df,
    cna_df=None,
    n_samples=600,
    min_sig_size=1,
    max_sig_size=150,
    n_genes_to_sim=10000,
    seed=44,
    cna_mode="ratio",
    add_cis_cna=True,
    cis_cna_scale=0.25,
    subtype=None,
    n_pcs=30,
    k_neighbors=30,
    mix_conc=1.0,
    residual_scale=0.5,
    metric="euclidean",
    use_deseq_size_factors=True
):
    """
    Simulates a realistic RNA-seq count matrix driven by genetic alterations.

    This function is the master orchestrator of the RNA simulation pipeline.

    Workflow:
    ----------
      1. Estimate DESeq2-style distribution parameters (means, dispersions, size factors)
         from a real RNA count matrix.
      2. Select genes to simulate (top expressed + altered genes).
      3. Generate subtype-aware background expression means using PCA + KNN mixing.
      4. Generate "true" alteration signatures (GOF/LOF/fusion/CNA-based).
      5. Inject additive alteration-driven expression effects into the background.
      6. Sample realistic RNA counts using a Gaussian copula Negative Binomial model
         to preserve gene–gene correlation.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Real RNA count matrix (samples x genes).
    alteration_df : pd.DataFrame
        Binary alteration matrix (0/1 for GOF, LOF, FUSION, etc.).
    cna_df : pd.DataFrame, optional
        CNA matrix (either in log2(CN/2) or ratio scale).
    n_samples : int
        Number of synthetic samples to simulate.
    min_sig_size, max_sig_size : int
        Minimum and maximum number of genes in each alteration signature.
    n_genes_to_sim : int
        Number of genes to simulate (typically top expressed + altered genes).
    seed : int
        Random seed for reproducibility.
    cna_mode : str
        Whether CNAs are handled as "ratio" or "log2" space.
    add_cis_cna : bool
        Whether to add gene-level (cis) CNA effects directly.
    cis_cna_scale : float
        Scaling factor for cis CNA contribution to expression.
    subtype : pd.Series, optional
        Subtype labels for stratified simulation.
    n_pcs : int
        Number of PCA components for expression manifold reconstruction.
    k_neighbors : int
        Number of neighbors for local mixing (PCA+KNN).
    mix_conc : float
        Dirichlet concentration parameter controlling neighbor weighting.
    residual_scale : float
        Scale of residual variance reintroduced post-PCA reconstruction.
    metric : str
        Distance metric for KNN (e.g. "euclidean").
    use_deseq_size_factors : bool
        Whether to sample and apply DESeq2 size factors to scaling simulated samples.

    Returns
    -------
    expr_sim : pd.DataFrame
        Simulated RNA-seq count matrix (samples x genes).
    true_signatures : dict
        Dictionary of generated alteration signatures with gene-level effects.
    sim_og : pd.DataFrame
        Baseline (pre-effect) expression means for each simulated sample.

    Notes
    -----
    • The resulting counts maintain realistic dispersion, variance,
      subtype-specific expression patterns, and correlation structure.
    • Alteration effects are additive and scaled by CNA copy number when relevant.
    """

    np.random.seed(seed)

    # ----------------------------------------------------------
    # STEP 1: Estimate DESeq2-like parameters from real RNA data
    # ----------------------------------------------------------
    # These include per-gene mean, variance, dispersion, and sample size factors.
    gene_means, gene_vars, dispersions, size_factors_real = estimate_deseq2_parameters(
        rna_df, seed=seed
    )

    # ----------------------------------------------------------
    # STEP 2: Select which genes to simulate
    # ----------------------------------------------------------
    # Include (a) the most highly expressed genes and
    #         (b) all genes involved in any alteration
    altered_genes = set()
    for alt in alteration_df.columns:
        if alt.endswith("_FUSION"):
            # Extract fusion partners (e.g., ETV6-RUNX1 → [ETV6, RUNX1])
            altered_genes.update(alt.replace("_FUSION", "").split("-"))
        elif alt.endswith(("_GOF", "_LOF")):
            altered_genes.add(alt.rsplit("_", 1)[0])

    top_genes = gene_means.sort_values(ascending=False).head(n_genes_to_sim).index
    genes_to_sim = sorted(
        set(top_genes).union(altered_genes).intersection(rna_df.columns)
    )

    rna_df_sub = rna_df[genes_to_sim]
    dispersions = dispersions[genes_to_sim]

    # ----------------------------------------------------------
    # STEP 3: Simulate subtype-aware RNA background (μ matrix)
    # ----------------------------------------------------------
    # Generate background (unperturbed) expression means using
    # PCA+KNN neighbor mixing, preserving subtype structure.
    sf_sim = None
    if use_deseq_size_factors:
        # Sample DESeq2-derived size factors to simulate library depth variation
        sf_sim = draw_size_factors_from_deseq(
            size_factors_real, n_samples, subtype=subtype, rng=seed
        )

    subseries = subtype.loc[rna_df_sub.index] if subtype is not None else None

    expr_bg_mu = simulate_background_knn(
        real_rna=rna_df_sub,
        n_samples=n_samples,
        n_pcs=n_pcs,
        k=k_neighbors,
        mix_conc=mix_conc,
        residual_scale=residual_scale,
        seed=seed,
        subtype=subseries,
        size_factors_to_apply=sf_sim,
        metric=metric,
    )

    sim_og = expr_bg_mu.copy()  # baseline pre-effect matrix

    # ----------------------------------------------------------
    # STEP 4: Prepare alteration and CNA matrices
    # ----------------------------------------------------------
    # Ensure alignment and proper column naming conventions
    alt = alteration_df.reindex(expr_bg_mu.index).fillna(0).clip(0, 1).astype(int)

    if cna_df is not None:
        cna = cna_df.copy()

        # Remove redundant "_CNA" suffix if present
        if all(col.endswith('_CNA') for col in cna.columns):
            cna.columns = cna.columns.str.replace('_CNA', '', regex=False)

        # Keep only genes that overlap with those being simulated
        cna = cna.loc[:, [g for g in genes_to_sim if g in cna.columns]]

        # Collapse duplicate gene columns and remove duplicates
        cna = cna.T.groupby(level=0).mean().T
        cna = cna[~cna.index.duplicated(keep="first")]
    else:
        cna = None

    # ----------------------------------------------------------
    # STEP 5: Generate and inject alteration signatures
    # ----------------------------------------------------------
    # Signatures encode per-gene log fold-change effects for each alteration.
    true_signatures = generate_signatures_simplified(
        genes=expr_bg_mu.columns.tolist(),
        alteration_features=alt.columns.tolist(),
        min_size=min_sig_size,
        max_size=max_sig_size,
        seed=seed,
    )

    # Inject additive GOF/LOF/fusion/CNA effects into baseline μ
    expr_effected = inject_expression_effects_simplified(
        expr_df=expr_bg_mu,
        alteration_df=alt,
        signatures=true_signatures,
        cna_df=cna,
        floor_zero=True,
    )

    # ----------------------------------------------------------
    # STEP 6: Sample realistic RNA-seq counts (Negative Binomial)
    # ----------------------------------------------------------
    # Uses Gaussian copula to retain correlation structure from real RNA data.
    disp = dispersions[expr_effected.columns]
    expr_counts = sample_nb_copula(
        mu_matrix=expr_effected.values,
        dispersions=disp,
        reference_expr=rna_df_sub,   # use real RNA for correlation estimation
        seed=seed,
    )

    expr_sim = pd.DataFrame(
        expr_counts,
        columns=expr_effected.columns,
        index=expr_effected.index,
    )

    # ----------------------------------------------------------
    # RETURN
    # ----------------------------------------------------------
    # expr_sim: final simulated counts
    # true_signatures: dictionary of alteration → signature effects
    # sim_og: background μ before alteration injection
    return expr_sim, true_signatures, sim_og
