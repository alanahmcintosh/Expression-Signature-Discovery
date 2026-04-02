"""
Signature generation RNA simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from benchmark_sigs.simulate.rna.signature_generation.signature_utils import (
    parse_alt,
    sample_size,
    sample_abs,
    build_targets_default,
    build_targets_shared,
)


def generate_signatures_from_deseq2_params(
    genes,
    alteration_features,
    alt_params,
    seed=44,
    # direction bias defaults
    gof_p_pos=0.85,
    lof_p_pos=0.15,
    fusion_p_pos=0.50,
    # AMP/DEL target sharing with GOF/LOF
    share_frac=0.70,
    share_amp_with_gof=True,
    share_del_with_lof=True,
    # fallbacks if params missing/invalid
    fallback_abs_range=(1.0, 2.0),
    fallback_size_range=(10, 300),
):
    """
    Create truth signatures using DESeq2-derived parameters.

    For each alteration feature `alt`:
      - signature size sampled around size_mean (fallback if missing)
      - absolute magnitudes from lognormal(abs_mu, abs_sigma) capped at abs_cap (fallback if missing)
      - sign sampled from alteration-type bias p_pos
      - base gene is included as a target if present in gene universe

    Additional rule:
      - If base_gene_GOF exists and generating base_gene_AMP:
          AMP shares ~share_frac of its non-base targets with GOF targets
      - If base_gene_LOF exists and generating base_gene_DEL:
          DEL shares ~share_frac of its non-base targets with LOF targets
    """
    rng = np.random.default_rng(seed)

    genes = list(map(str, genes))
    gene_set = set(genes)

    # Ensure GOF/LOF created before AMP/DEL so sharing can happen
    alteration_features = sorted(
        map(str, alteration_features),
        key=lambda a: (a.endswith("_AMP") or a.endswith("_DEL"), a),
    )

    signatures = {}

    for alt in alteration_features:
        if alt not in alt_params:
            continue

        kind, base_gene, _partners = parse_alt(alt)
        p = alt_params[alt]

        L = sample_size(p.get("size_mean"), rng=rng, fallback_size_range=fallback_size_range)
        L = min(int(L), len(genes))

        # ----- TARGETS (with AMP/DEL sharing rules) -----
        targets = None

        if base_gene is not None and base_gene in gene_set:
            if share_amp_with_gof and alt.endswith("_AMP"):
                ref_key = f"{base_gene}_GOF"
                if ref_key in signatures:
                    targets = build_targets_shared(
                        base_gene=base_gene,
                        L=L,
                        ref_targets=signatures[ref_key]["targets"],
                        share_frac=share_frac,
                        gene_set=gene_set,
                        genes=genes,
                        rng=rng,
                    )

            elif share_del_with_lof and alt.endswith("_DEL"):
                ref_key = f"{base_gene}_LOF"
                if ref_key in signatures:
                    targets = build_targets_shared(
                        base_gene=base_gene,
                        L=L,
                        ref_targets=signatures[ref_key]["targets"],
                        share_frac=share_frac,
                        gene_set=gene_set,
                        genes=genes,
                        rng=rng,
                    )

        if targets is None:
            targets = build_targets_default(
                base_gene=base_gene,
                L=L,
                gene_set=gene_set,
                genes=genes,
                rng=rng,
            )

        # ----- SIGN BIAS -----
        if alt.endswith("_AMP"):
            p_pos = gof_p_pos
        elif alt.endswith("_DEL"):
            p_pos = lof_p_pos
        elif kind == "GOF":
            p_pos = gof_p_pos
        elif kind == "LOF":
            p_pos = lof_p_pos
        elif kind == "FUSION":
            p_pos = fusion_p_pos
        else:
            p_pos = 0.0

        # ----- EFFECTS -----
        effects = {}
        for t in targets:
            mag = sample_abs(
                p.get("abs_mu"),
                p.get("abs_sigma"),
                p.get("abs_cap"),
                rng=rng,
                fallback_abs_range=fallback_abs_range,
            )
            sign = 1.0 if rng.random() < float(p_pos) else -1.0

            # enforce self-direction for core driver
            if base_gene is not None and t == base_gene:
                if kind == "GOF" or alt.endswith("_AMP"):
                    sign = 1.0
                if kind == "LOF" or alt.endswith("_DEL"):
                    sign = -1.0

            effects[str(t)] = float(sign * mag)

        signatures[alt] = {
            "targets": list(map(str, targets)),
            "effects": effects,
            "effect_mode": "log2fc",
        }

    return signatures


def induce_expression_effects(
    expr_df,
    alteration_df,
    signatures,
    cna_gistic_df=None,
    floor_zero=True,
):
    """
    Inject signature effects into a mean-expression matrix (μ space).

    Effect model:
      - Each signature stores per-target signed log2FC values.
      - Effects add in log2-space:
            total_log2FC[sample,gene] = Σ_alt (alt_value * log2FC_alt_gene)
        where alt_value is:
            - 0/1 for GOF/LOF/FUSION (binary)
            - 0/1/2 for AMP/DEL if cna_gistic_df provided (severity weighted)
      - Apply in μ space:
            μ_new = μ_old * 2^(total_log2FC)

    Returns
    -------
    pd.DataFrame : modified μ matrix (samples × genes)
    """
    # -----------------------------
    # 0) Align samples
    # -----------------------------
    common_idx = expr_df.index.intersection(alteration_df.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping samples between expr_df and alteration_df.")

    mu_df = expr_df.loc[common_idx].copy()
    A_df = alteration_df.loc[common_idx].copy()

    # Align CNA severity if provided
    if cna_gistic_df is not None:
        cna_common = mu_df.index.intersection(cna_gistic_df.index)
        if len(cna_common) == 0:
            cna_gistic_df = None
        else:
            mu_df = mu_df.loc[cna_common]
            A_df = A_df.loc[cna_common]
            cna_gistic_df = cna_gistic_df.loc[cna_common]

    # -----------------------------
    # 1) Keep only alterations that have signatures
    # -----------------------------
    alts = [str(a) for a in A_df.columns if str(a) in signatures]
    if len(alts) == 0:
        return mu_df

    # numeric, keep as weights (still 0/1 for binaries)
    A_df = A_df[alts].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    A_df[A_df < 0] = 0.0

    # -----------------------------
    # 2) Replace AMP/DEL 0/1 with severity 0/1/2 from CNA gistic
    # -----------------------------
    if cna_gistic_df is not None:
        cna = cna_gistic_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cna = np.rint(cna).astype(int).clip(-2, 2)

        for col in A_df.columns:
            if col.endswith("_AMP"):
                gene = col[:-4]
                if gene in cna.columns:
                    A_df[col] = cna[gene].clip(lower=0).astype(float)  # 0/1/2
            elif col.endswith("_DEL"):
                gene = col[:-4]
                if gene in cna.columns:
                    A_df[col] = (-cna[gene]).clip(lower=0).astype(float)  # 0/1/2

    # -----------------------------
    # 3) Determine affected genes (targets ∩ mu columns)
    # -----------------------------
    mu_cols = list(map(str, mu_df.columns))
    mu_gene_set = set(mu_cols)

    affected_genes = set()
    for alt in alts:
        sig = signatures.get(alt, {})
        eff = sig.get("effects", {}) or {}
        for tgt in eff.keys():
            tgt = str(tgt)
            if tgt in mu_gene_set:
                affected_genes.add(tgt)

    if len(affected_genes) == 0:
        return mu_df

    affected_genes = sorted(affected_genes)
    mu_sub = mu_df[affected_genes].to_numpy(dtype=float, copy=True)

    # -----------------------------
    # 4) Build sparse A (samples × alts) with weights (0/1/2)
    # -----------------------------
    A = sparse.csr_matrix(A_df.to_numpy(dtype=float))

    # -----------------------------
    # 5) Build sparse E (alts × affected_genes) of per-target log2FC
    # -----------------------------
    alt_to_i = {alt: i for i, alt in enumerate(alts)}
    gene_to_j = {g: j for j, g in enumerate(affected_genes)}

    rows, cols, data = [], [], []
    for alt in alts:
        i = alt_to_i[alt]
        eff = signatures[alt].get("effects", {}) or {}
        for tgt, log2fc in eff.items():
            tgt = str(tgt)
            j = gene_to_j.get(tgt)
            if j is None:
                continue
            rows.append(i)
            cols.append(j)
            data.append(float(log2fc))

    if len(data) == 0:
        return mu_df

    E = sparse.coo_matrix(
        (np.array(data, dtype=float), (np.array(rows), np.array(cols))),
        shape=(len(alts), len(affected_genes)),
    ).tocsr()
    E.sum_duplicates()

    # -----------------------------
    # 6) Total log2FC per sample×gene and apply μ *= 2^L
    # -----------------------------
    L = (A @ E).tocoo()
    mu_sub[L.row, L.col] *= np.power(2.0, L.data)

    mu_df.loc[:, affected_genes] = mu_sub

    if floor_zero:
        mu_df[mu_df < 0] = 0.0

    return mu_df
