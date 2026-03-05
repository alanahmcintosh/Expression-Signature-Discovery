"""
End-to-end RNA simulation with alteration-driven signatures.

Pipeline:
1) Estimate DESeq2-style dispersions + size factors from real RNA
2) Choose genes to simulate (top expressed + all altered genes)
3) Simulate subtype-aware background mean expression (mu) via KNN mixing in alteration space
4) Build (or accept) truth signatures and inject effects into mu (log2FC additive)
5) Sample counts with NB for signature genes (others rounded)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark_sigs.simulate.rna.deseq_params import (
    estimate_deseq2_parameters,
    draw_size_factors_from_deseq,
)
from benchmark_sigs.simulate.rna.background_rna_knn import (
    simulate_background_from_alterations_knn,
)
from benchmark_sigs.simulate.rna.signature_generation.signature_params import (
    build_alt_params_from_deseq2_summary,
)
from benchmark_sigs.simulate.rna.signature_generation.simulate_signatures import (
    generate_signatures_from_deseq2_params,
    induce_expression_effects,
)
from benchmark_sigs.simulate.rna.nb_sampling import (
    sample_nb_for_signature_genes,
)


def simulate_rna_with_signatures(
    rna_df,
    alteration_df,
    altertaions_real,
    cna_df=None,
    n_samples=600,
    n_genes_to_sim=10000,
    seed=44,
    subtype=None,
    k_neighbors=3,
    mix_conc=1.0,
    residual_scale=0.5,
    metric="euclidean",
    use_deseq_size_factors=True,
    deseq2_summary_path=None,
    deseq2_summary_df=None,
    deseq2_alteration_col="alteration",
    deseq2_n_sig_col="n_sig",
    deseq2_mean_abs_col="mean_abs_log2FC_sig",
    deseq2_median_abs_col="median_abs_log2FC_sig",
    deseq2_max_abs_col="abs_max_log2FC_sig",
    cap_k=8.0,
    global_cap=12.0,
    gof_p_pos=0.85,
    lof_p_pos=0.15,
    fusion_p_pos=0.50,
    true_signatures=None,
):
    """
    Simulate RNA-seq count matrix with alteration-driven expression signatures.

    Parameters
    ----------
    rna_df : pd.DataFrame
        Real RNA matrix (samples x genes), ideally counts (or pseudo-counts).
    alteration_df : pd.DataFrame
        Simulated alteration matrix (samples x alteration_features). Used as sim_alts for KNN background.
        Typically binary for GOF/LOF/FUSION/AMP/DEL, but any numeric is acceptable for neighbor search.
    altertaions_real : pd.DataFrame
        Real alteration matrix (samples x alteration_features) used to define KNN neighborhoods.
        Must overlap with rna_df index.
    cna_df : pd.DataFrame or None
        Real or simulated CNA GISTIC matrix (samples x genes) in [-2..2] for severity weighting of AMP/DEL.
        If provided, AMP/DEL weights become 0/1/2.
    n_samples : int
        Target number of simulated samples (used mainly for size-factor resampling; background uses alteration_df rows).
    n_genes_to_sim : int
        Start from top genes by mean expression then force-in altered genes.
    seed : int
        RNG seed.
    subtype : pd.Series or pd.DataFrame or None
        Optional subtype labels for size-factor resampling.
    k_neighbors, mix_conc, residual_scale, metric : background KNN mixing params
    use_deseq_size_factors : bool
        If True, resample DESeq2 size factors (currently computed but not applied inside this function).
    deseq2_summary_path / deseq2_summary_df :
        If provided AND true_signatures is None, use DE summary to generate alt_params -> signatures.
    true_signatures : dict or None
        If provided, used directly; otherwise generated if DESeq2 summary is given.

    Returns
    -------
    (pd.DataFrame, dict, pd.DataFrame)
        expr_sim       : simulated counts (samples x genes)
        true_signatures: dict of truth signatures
        sim_og         : background mu before signature injection
    """
    rng = np.random.default_rng(seed)

    if rna_df is None or rna_df.empty:
        raise ValueError("[simulate_rna_with_signatures] rna_df is empty.")
    if alteration_df is None or alteration_df.empty:
        raise ValueError("[simulate_rna_with_signatures] alteration_df is empty.")
    if altertaions_real is None or altertaions_real.empty:
        raise ValueError("[simulate_rna_with_signatures] altertaions_real is empty.")

    # ----------------------------------------------------------
    # STEP 1: Estimate DESeq2-like parameters from real RNA data
    # ----------------------------------------------------------
    gene_means, gene_vars, dispersions, size_factors_real = estimate_deseq2_parameters(
        rna_df, seed=seed
    )

    # ----------------------------------------------------------
    # STEP 2: Select which genes to simulate
    # ----------------------------------------------------------
    altered_genes = set()
    for alt_name in alteration_df.columns:
        alt_name = str(alt_name)
        if alt_name.endswith("_FUSION"):
            altered_genes.update(alt_name.replace("_FUSION", "").split("-"))
        elif alt_name.endswith(("_GOF", "_LOF", "_AMP", "_DEL")):
            altered_genes.add(alt_name.rsplit("_", 1)[0])

    top_genes = gene_means.sort_values(ascending=False).head(int(n_genes_to_sim)).index
    genes_to_sim = sorted(set(top_genes).union(altered_genes).intersection(rna_df.columns))

    rna_df_sub = rna_df.loc[:, genes_to_sim]
    dispersions = dispersions.reindex(genes_to_sim)

    # ----------------------------------------------------------
    # STEP 3: Simulate RNA background mean (μ matrix) via KNN mixing
    # ----------------------------------------------------------
    sf_sim = None
    if use_deseq_size_factors:
        # Keeping this for API parity / future use (e.g., scaling mu by sf_sim)
        sf_sim = draw_size_factors_from_deseq(
            size_factors_real, int(n_samples), subtype=subtype, rng=seed
        )

    # Note: background simulator returns mu for each row in sim_alts (alteration_df)
    # Ensure real_alts aligns to rna_df_sub.index
    real_alts_aligned = altertaions_real.loc[rna_df_sub.index]

    expr_bg_mu = simulate_background_from_alterations_knn(
        real_rna=rna_df_sub,
        real_alts=real_alts_aligned,
        sim_alts=alteration_df,
        k=k_neighbors,
        mix_conc=mix_conc,
        residual_scale=residual_scale,
        seed=seed,
        metric=metric,
    )

    # Keep a copy of background μ for QC / debugging
    sim_og = expr_bg_mu.copy()

    # ----------------------------------------------------------
    # STEP 4: Generate signatures (DESeq2-parameterised if provided)
    # ----------------------------------------------------------
    alt = alteration_df.reindex(expr_bg_mu.index).fillna(0).clip(0, 1).astype(int)
    truth_features = list(alt.columns)

    if true_signatures is None and (deseq2_summary_path is not None or deseq2_summary_df is not None):
        if deseq2_summary_df is None:
            deseq2_summary_df = pd.read_csv(deseq2_summary_path)

        alt_params = build_alt_params_from_deseq2_summary(
            deseq2_summary=deseq2_summary_df,
            alteration_col=deseq2_alteration_col,
            n_sig_col=deseq2_n_sig_col,
            mean_abs_col=deseq2_mean_abs_col,
            median_abs_col=deseq2_median_abs_col,
            max_abs_col=deseq2_max_abs_col,
            cap_k=cap_k,
            global_cap=global_cap,
        )

        true_signatures = generate_signatures_from_deseq2_params(
            genes=expr_bg_mu.columns.tolist(),
            alteration_features=truth_features,
            alt_params=alt_params,
            seed=seed,
            gof_p_pos=gof_p_pos,
            lof_p_pos=lof_p_pos,
            fusion_p_pos=fusion_p_pos,
        )

    if true_signatures is None:
        raise ValueError(
            "[simulate_rna_with_signatures] true_signatures is None and no DESeq2 summary was provided."
        )

    expr_effected = induce_expression_effects(
        expr_df=expr_bg_mu,
        alteration_df=alt,
        signatures=true_signatures,
        cna_gistic_df=cna_df,
        floor_zero=True,
    )

    # ----------------------------------------------------------
    # STEP 5: Sample RNA counts only for signature genes (NB)
    # ----------------------------------------------------------
    sig_genes = sorted(
        {
            str(tgt)
            for sig in true_signatures.values()
            for tgt in sig.get("targets", [])
            if str(tgt) in expr_effected.columns
        }
    )

    disp_subset = dispersions.reindex(expr_effected.columns)

    expr_sim = sample_nb_for_signature_genes(
        mu_df=expr_effected,
        dispersions=disp_subset,
        sig_genes=sig_genes,
        seed=seed,
    )

    return expr_sim, true_signatures, sim_og
