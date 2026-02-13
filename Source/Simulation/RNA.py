import numpy as np
import pandas as pd
import os
import random
from sklearn.neighbors import NearestNeighbors
from pydeseq2.dds import DeseqDataSet  
from scipy import sparse
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler


# ==============================================================
# RNA SIMULATION PIPELINE
# ==============================================================
# The goal is to simulate RNA-seq counts driven by alterations,
# preserving realistic structure (dispersion, correlation, subtype).
# ==============================================================
# ==============================================================
# 0 - HELPER FUNCTIONS
# ==============================================================


def sample_size(n_mean, seed=44, fallback_size_range=None):
    rng = np.random.default_rng(seed)
    #Using the DESeq2 signature size, sample this number and combat mam
    #Fallback
    if n_mean is None or n_mean <= 0 or not np.isfinite(n_mean):
        return int(rng.integers(fallback_size_range[0], fallback_size_range[1] + 1))
    
    L = int(round(n_mean))
    
    # Prevents a negative entry 
    return max(1, L)


def sample_abs(mu, sigma, cap, seed=44, fallback_abs_range=None):
    rng = np.random.default_rng(seed)
    # Sample absolute effect value from lognormal distribution
    # If params missing, use uniform fallback
    if mu is None or sigma is None or (not np.isfinite(mu)) or (not np.isfinite(sigma)):
        val = float(rng.uniform(*fallback_abs_range))
    else:
        val = float(rng.lognormal(mean=float(mu), sigma=float(sigma)))

    # Cap if provided and finite
    if cap is not None and np.isfinite(cap):
        val = min(val, float(cap))
    return float(val)

def build_targets_default(base_gene, L, seed=44, genes = None, gene_set = None):
    rng = np.random.default_rng(seed)
    # Choose targets without sharing rules ( exclusive GOFs/LOFs/AMPs/DELs)
    # If base_gene exists, force it in the signature
    # Fill remaining target slots with random (no replacement)
    if base_gene is not None and base_gene in gene_set:
        pool = [g for g in genes if g != base_gene]
        rest = rng.choice(pool, size=max(0, L - 1), replace=False).tolist()
        return [base_gene] + rest
    return rng.choice(genes, size=L, replace=False).tolist()

def build_targets_shared(
    base_gene,
    L,
    ref_targets,
    share_frac,
    seed=44,
    gene_set=None,
    genes = None
):
    """
    Choose targets with sharing rules. (GOF signatures approx = AMP signatures, LOF signatures approx= DEL signatures)
    If base_gene ecists, force it into the signatures
    """

    rng = np.random.default_rng(seed)
    forced = [base_gene] if base_gene in gene_set else []
    remaining_slots = max(0, L - len(forced))

    # remove base gene from reference list, and keep only genes in universe
    ref_targets = [g for g in ref_targets if g in gene_set]
    ref_wo_base = [g for g in ref_targets if g != base_gene]

    # how many shared?
    n_shared = int(round(float(share_frac) * float(remaining_slots)))
    n_shared = max(0, min(n_shared, len(ref_wo_base)))

    shared = (
        rng.choice(ref_wo_base, size=n_shared, replace=False).tolist()
        if n_shared > 0 else []
    )

    # fill the rest with "specific" targets not in ref_targets
    excluded = set(forced) | set(shared) | set(ref_targets)
    specific_pool = [g for g in genes if g not in excluded]
    n_specific = remaining_slots - len(shared)

    # if pool is too small, relax exclusion slightly (rare edge case)
    if n_specific > len(specific_pool):
        specific_pool = [g for g in genes if g not in (set(forced) | set(shared))]
    specific_pool = list(dict.fromkeys(specific_pool))  # unique preserving order

    specific = (
        rng.choice(specific_pool, size=n_specific, replace=False).tolist()
        if n_specific > 0 else []
    )

    return forced + shared + specific

def gistic_to_amp_del_binary(cna_gistic):
    x = cna_gistic.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = np.rint(x).astype(int).clip(-2, 2)

    amp = (x > 0).astype(int)
    dele = (x < 0).astype(int)

    amp.columns = [f"{c}_AMP" for c in amp.columns]
    dele.columns = [f"{c}_DEL" for c in dele.columns]

    return pd.concat([amp, dele], axis=1)

def apply_ampdel_severity_weights(alteration_df,
                                 cna_gistic):
    """
    Return a copy of alteration_df where *_AMP/*_DEL columns are replaced with
    severity (0/1/2) derived from cna_gistic (gene-level -2..2 states).
    Non-AMP/DEL columns are unchanged (remain 0/1).
    """
    A = alteration_df.copy()
    cna = cna_gistic.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cna = np.rint(cna).clip(-2, 2)

    # align samples
    common = A.index.intersection(cna.index)
    A = A.loc[common]
    cna = cna.loc[common]

    for col in A.columns:
        if col.endswith("_AMP"):
            gene = col[:-4]
            if gene in cna.columns:
                A[col] = cna[gene].clip(lower=0)   # 0,1,2
        elif col.endswith("_DEL"):
            gene = col[:-4]
            if gene in cna.columns:
                A[col] = (-cna[gene]).clip(lower=0)  # 0,1,2

    return A

def parse_alt(alt: str):
    alt = str(alt)

    if alt.endswith("_GOF"):
        return "GOF", alt[:-4], None
    if alt.endswith("_LOF"):
        return "LOF", alt[:-4], None
    if alt.endswith("_AMP"):
        return "AMP", alt[:-4], None
    if alt.endswith("_DEL"):
        return "DEL", alt[:-4], None
    if alt.endswith("_FUSION"):
        core = alt[:-7]
        partners = core.split("--") if "--" in core else core.split("-")
        # base gene can be first partner if you want
        base_gene = partners[0] if partners else None
        return "FUSION", base_gene, partners

    return "OTHER", None, None


def build_alt_params_from_deseq2_summary(
    deseq2_summary,
    alteration_col = "alteration",
    n_sig_col = "n_sig",
    mean_abs_col  = "mean_abs_log2FC_sig",
    median_abs_col = "median_abs_log2FC_sig",
    max_abs_col = "abs_max_log2FC_sig",
    cap_k = 8.0,
    global_cap = 12.0,
):
    """
    Given a dataframe containg summary statistics from a differential expression analysis, generate parameters to be used for defining paranetrs in simulated expression signatures. 
    Each altertaion gets a statistical model describing
        - Expected signature size
        - Distribution of effect magnitudes
        - Upper bound on effect size
    """
    df = deseq2_summary.copy()
    if alteration_col in df.columns:
        df = df.set_index(alteration_col)

    params = {}
    
    # Iterate over each altertaion
    for alt, row in df.iterrows():
        alt = str(alt)

        # Extract values from the table for that altertaion
        n_sig = row.get(n_sig_col, np.nan) # Signature size (number of genes who have an adj p-value <0.05)
        mean_abs = row.get(mean_abs_col, np.nan)# Mean |log2FC| - of the genes in signature, what is the mean log2FC
        median_abs = row.get(median_abs_col, np.nan)# Median |log2FC| - of the genes in signature, what is the median log2FC
        abs_max = row.get(max_abs_col, np.nan)# Largest |log2FC| in signature

        #Model |log2FC| as a lognormal distribution
        mu, sigma = fit_lognormal_from_mean_median(mean_abs, median_abs)
        
        #Determine realsitic max effect size
        cap = cap_abs_log2fc(mean_abs, abs_max, cap_k=cap_k, global_cap=global_cap)
    
        # Get signature size, allowed to be None if unavailable
        size_mean = None
        try:
            if np.isfinite(n_sig) and float(n_sig) > 0:
                size_mean = int(round(float(n_sig)))
        except Exception:
            size_mean = None
    
        params[alt] = {
            "size_mean": size_mean,
            "abs_mu": mu,
            "abs_sigma": sigma,
            "abs_cap": cap,
        }

    return params


def cap_abs_log2fc(mean_abs, abs_max, cap_k = 8.0, global_cap = 12.0):
    """
    Determine uper bound for simulated absolute log2 fold change. 
    Prevents biologically unrealistic outliers
    Your cap rule: cap at (mean + cap_k), also bounded by abs_max and global_cap.
    Abs_max - > Never exceed whats observed in real data
    Mean_abs + cap_k - > Relative cap based on genes typical effect
    Global_cap -> Prevents explosive values if summary stats are too noisy
    """
    try:
        mean_abs = float(mean_abs)
        abs_max = float(abs_max)
    except Exception:
        return float(global_cap)
    
    # If statistis are msiing/invalid, fallback to on global cap
    if not np.isfinite(mean_abs) or not np.isfinite(abs_max):
        return float(global_cap)

    return float(min(abs_max, mean_abs + cap_k, global_cap))


def fit_lognormal_from_mean_median(mean_val, median_val):
    """
    Estimate parameters of a lognormal distribution form its mean and median.
    Lognormal used for |log2FC| effect sizes as they are psoitive, right skewed and heavy tailed.
    Returns (mu, sigma) for np.random.lognormal(mean=mu, sigma=sigma).
    """
    try:
        mean_val = float(mean_val)
        median_val = float(median_val)
    except Exception:
        return None, None

    if not np.isfinite(mean_val) or not np.isfinite(median_val) or mean_val <= 0 or median_val <= 0:
        return None, None
    
    # Frome median = exp(mu)
    mu = np.log(median_val)
    
    # From mean = exp(mu+sigma^2/2 -> sigma^2 = 2* ln(mean) - ln(median))
    sig2 = 2.0 * (np.log(mean_val) - np.log(median_val))
    sig2 = max(sig2, 0.0)
    sigma = np.sqrt(sig2)
    return mu, sigma


# ==============================================================
# 1. ESTIMATE DESEQ2 LIKE PARAMETERS
# ==============================================================

def estimate_deseq2_parameters(rna_df, size_factor_sd=0.2, seed=44, condition='A'):
    """
    Estimate DESeq2-style parameters (mean, dispersion, and size factors)
    from a real RNA count matrix.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)

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

def draw_size_factors_from_deseq(size_factors, n_samples, subtype=None, rng=44):
    """
    Resample DESeq2 size factors (sample-level normalization coefficients)
    to assign realistic per-sample scaling in the simulated data.

    Optionally stratified by subtype proportions.
    """
    rng = np.random.default_rng(rng)
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


def sample_nb(mu, dispersions, rng=None):
    """
    Sample RNA-seq counts using a Negative Binomial distribution.
    variance = mu + mu^2 * dispersion
        where
            mu = expected expression level
            dispersion = gene specfic noise

    Handles edge cases and clips invalid values.
    """
    
    if rng is None:
        rng = np.random.default_rng()
        
    
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

    return rng.negative_binomial(n=r, p=p)

def sample_nb_for_signature_genes(
    mu_df,
    dispersions,
    sig_genes,
    seed = 44
):
    """
    Sample NB counts only for signature genes using sample_nb().
    Non-signature genes are rounded.
    """

    rng = np.random.default_rng(seed)

    counts = mu_df.copy()

    # keep only genes present in both
    sig_genes = [g for g in sig_genes if g in mu_df.columns and g in dispersions.index]
    if len(sig_genes) == 0:
        counts = counts.round().astype(int)
        counts[counts < 0] = 0
        return counts

    # --- sample NB only for signature genes ---
    mu_sub = mu_df[sig_genes].values
    disp_sub = dispersions.loc[sig_genes].values

    sampled = sample_nb(mu_sub, disp_sub)

    counts.loc[:, sig_genes] = sampled

    # --- non-signature genes → integer baseline ---
    counts = counts.round().astype(int)
    counts[counts < 0] = 0

    return counts
    
# ==============================================================
# 2. SIMULATE BACKGROUND
# ==============================================================


def simulate_background_from_alterations_knn(
    real_rna,
    real_alts,
    sim_alts,
    k = 5,
    mix_conc = 1.0,
    residual_scale = 0.0,
    seed = 44,
    metric = "euclidean",
):
    rng = np.random.default_rng(seed)

    # --- 1. Align alteration features ---
    common = real_alts.columns.intersection(sim_alts.columns)

    if len(common) == 0:
        raise ValueError(
            "[simulate_background_from_alterations_knn] "
            "No overlapping alteration features between real_alts and sim_alts."
        )

    # Use the same columns, same order
    Ar = real_alts.loc[:, common].copy()
    As = sim_alts.loc[:, common].copy()
    As = As[Ar.columns]   # enforce identical order

    # --- 2. Scale in alteration space, using .to_numpy() ---
    scaler = StandardScaler(with_mean=True, with_std=True)
    Ar_scaled = scaler.fit_transform(Ar.to_numpy())
    As_scaled = scaler.transform(As.to_numpy())

    # --- 3. KNN in alteration space (fit on REAL alterations) ---
    k_ = max(2, min(k, Ar_scaled.shape[0])) #Needs at least 2 neighbours and max neighbours of either defined k or number of available neighbours (whichevers smaller)
    nn = NearestNeighbors(n_neighbors=k_, metric=metric)
    nn.fit(Ar_scaled)

    # --- 4. Prepare RNA (log1p + z-score) ---
    R_counts = real_rna.loc[Ar.index].clip(lower=0)
    L = np.log1p(R_counts)

    mu = L.mean(axis=0)
    sd = L.std(axis=0, ddof=0).replace(0, 1.0)
    Z = (L - mu) / sd

    
    # --- 5. Generate simulated expression for each synthetuic sample ---
    mixes = []
    for i in range(As_scaled.shape[0]):
        
        # Find genetically similar real samples
        _, idxs = nn.kneighbors(As_scaled[i:i+1])
        nbr_idx = Ar.index[idxs[0]]
        nbrZ = Z.loc[nbr_idx]

        #Random convex mixture of neighbours
        #Drichlet ensures weights sum to 1
        w = rng.dirichlet(np.full(nbrZ.shape[0], mix_conc))
        z_mix = np.average(nbrZ.values, axis=0, weights=w)

        # Optional nois to avoid overfitting 
        if residual_scale and residual_scale > 0:
            z_mix += rng.normal(0, residual_scale, size=z_mix.shape[0])

        # Transform back to expression sclae
        L_mix = z_mix * sd.values + mu.values
        mu_expr = np.expm1(L_mix)
        
        # No negative values
        mu_expr[mu_expr < 0] = 0
        mixes.append(mu_expr)
        
    # Assemble simulated expression matrix
    sim_mu = pd.DataFrame(mixes, columns=real_rna.columns, index=sim_alts.index)
    return sim_mu

# ==============================================================
# 3. GENERATE SIGNATURES
# ==============================================================
import numpy as np


def generate_signatures_from_deseq2_params(
    genes,
    alteration_features,
    alt_params,
    seed = 44,

    # direction bias defaults (can later be empirical)
    gof_p_pos = 0.85,
    lof_p_pos = 0.15,
    fusion_p_pos = 0.50,

    # AMP/DEL target sharing with GOF/LOF
    share_frac =0.70,
    share_amp_with_gof = True,
    share_del_with_lof = True,

    # fallbacks if params missing/invalid
    fallback_abs_range=(1.0, 2.0),
    fallback_size_range=(10, 50),
):
    """
    Create truth signatures using DESeq2-derived parameters.

    For each alteration feature `alt`:
      - signature size ~ Normal(size_mean, size_jitter*size_mean) with floor at 1
      - absolute magnitudes sampled from lognormal(abs_mu, abs_sigma) then capped at abs_cap
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
        key=lambda a: (a.endswith("_AMP") or a.endswith("_DEL"), a)
    )

    signatures = {}

    for alt in alteration_features:
        # must exist in params to generate
        if alt not in alt_params:
            continue

        kind, base_gene, partners = parse_alt(alt)  # <-- you provide this helper

        p = alt_params[alt]
        L = sample_size(p.get("size_mean"), seed=44, fallback_size_range=fallback_size_range)
        L = min(L, len(genes))

        # ----- TARGETS (with AMP/DEL sharing rules) -----
        targets = None

        if base_gene is not None and base_gene in gene_set:
            if share_amp_with_gof and alt.endswith("_AMP"):
                ref_key = f"{base_gene}_GOF"
                if ref_key in signatures:
                    targets = build_targets_shared(base_gene, L, signatures[ref_key]["targets"], share_frac)

            elif share_del_with_lof and alt.endswith("_DEL"):
                ref_key = f"{base_gene}_LOF"
                if ref_key in signatures:
                    targets = build_targets_shared(base_gene, L, signatures[ref_key]["targets"], share_frac, gene_set=gene_set, genes=genes)

        if targets is None:
            targets = build_targets_default(base_gene, L, gene_set=gene_set, genes=genes)

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
            p_pos = 0

        # ----- EFFECTS -----
        effects = {}
        for t in targets:
            mag = sample_abs(p.get("abs_mu"), p.get("abs_sigma"), p.get("abs_cap"), fallback_abs_range=fallback_abs_range)
            sign = 1.0 if rng.random() < float(p_pos) else -1.0

            # enforce self-direction for core driver
            if base_gene is not None and t == base_gene:
                if kind == "GOF" or alt.endswith("_AMP"):
                    sign = 1.0
                if kind == "LOF" or alt.endswith("_DEL"):
                    sign = -1.0

            effects[t] = float(sign * mag)

        signatures[alt] = {
            "targets": targets,
            "effects": effects,
            "effect_mode": "log2fc",
        }

    return signatures



def induce_expression_effects(
    expr_df,
    alteration_df,
    signatures,
    cna_gistic_df= None,
    floor_zero = True,
):
    """
    Induction of alteration signature effects into a mean-expression matrix (μ space).
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



# ==============================================================
# 4. WRAPPER FUNCTION
# ==============================================================

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
    deseq2_summary_path = None,
    deseq2_summary_df = None,
    deseq2_alteration_col = "alteration",
    deseq2_n_sig_col = "n_sig",
    deseq2_mean_abs_col = "mean_abs_log2FC_sig",
    deseq2_median_abs_col = "median_abs_log2FC_sig",
    deseq2_max_abs_col = "abs_max_log2FC_sig",
    cap_k = 8.0,
    global_cap = 12.0,
    gof_p_pos = 0.85,
    lof_p_pos = 0.15,
    fusion_p_pos = 0.50,
    true_signatures = None
):

    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------
    # STEP 1: Estimate DESeq2-like parameters from real RNA data
    # ----------------------------------------------------------
    gene_means, gene_vars, dispersions, size_factors_real = estimate_deseq2_parameters(
        rna_df, seed=seed
    )

    # ----------------------------------------------------------
    # STEP 2: Select which genes to simulate
    # ----------------------------------------------------------
    # Selects gene names from altered genes in alterations_df
    altered_genes = set()
    for alt_name in alteration_df.columns:
        if alt_name.endswith("_FUSION"):
            altered_genes.update(alt_name.replace("_FUSION", "").split("-"))
        elif alt_name.endswith(("_GOF", "_LOF", "_AMP", "_DEL")):
            altered_genes.add(alt_name.rsplit("_", 1)[0])

    # Start from the top n_genes_to_sim genes by mean expression, then force-in altered genes
    top_genes = gene_means.sort_values(ascending=False).head(n_genes_to_sim).index
    genes_to_sim = sorted(
        set(top_genes).union(altered_genes).intersection(rna_df.columns)
    )

    rna_df_sub = rna_df[genes_to_sim]
    dispersions = dispersions[genes_to_sim]

    # ----------------------------------------------------------
    # STEP 3: Simulate subtype-aware RNA background (μ matrix)
    # ----------------------------------------------------------
    # Resample size factors (library sizes) to mimic sequencing depth variation.
    sf_sim = None
    if use_deseq_size_factors:
        sf_sim = draw_size_factors_from_deseq(
            size_factors_real, n_samples, subtype=subtype, rng=seed
     )
        
    #Build μ by finding real samples with similar alteration patterns and mixing their expression
    expr_bg_mu = simulate_background_from_alterations_knn(
        real_rna=rna_df_sub,
        real_alts=altertaions_real.loc[rna_df_sub.index],
        sim_alts=alteration_df,
        k=k_neighbors,
        mix_conc=mix_conc,
        residual_scale=residual_scale,
        seed=seed,
        metric=metric,
    )
    
    # Keep a copy of the background μ for QC / debugging
    sim_og = expr_bg_mu.copy()


    # ----------------------------------------------------------
    # STEP 4: Generate signatures (DESeq2-parameterised if provided)
    # ----------------------------------------------------------
    
    # Ensure binary alterations are 0/1 and aligned to simulated μ samples
    alt = alteration_df.reindex(expr_bg_mu.index).fillna(0).clip(0, 1).astype(int)
    
    truth_features = list(alt.columns)

    # If caller didn't pass precomputed truth_signatures, generate them
    if true_signatures is None and (deseq2_summary_path is not None or deseq2_summary_df is not None):
        # Load summary table from disk if needed
        if deseq2_summary_df is None:
            deseq2_summary_df = pd.read_csv(deseq2_summary_path)

            
        # Convert per-alteration DESeq2 summaries into generative parameters (size, magnitude distro, cap)
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

        # Generate a signature (targets + effects) for each alteration feature
        true_signatures = generate_signatures_from_deseq2_params(
            genes=expr_bg_mu.columns.tolist(),
            alteration_features=truth_features,
            alt_params=alt_params,
            seed=seed,
            gof_p_pos=gof_p_pos,
            lof_p_pos=lof_p_pos,
            fusion_p_pos=fusion_p_pos,
        )

    # Inject effects
    expr_effected = induce_expression_effects(
        expr_df=expr_bg_mu,
        alteration_df=alt,                 # your binary GOF/LOF/FUSION/AMP/DEL
        signatures=true_signatures,
        cna_gistic_df=cna_df,       # the severity matrix (-2..2)
        floor_zero=True,
    )

    # ----------------------------------------------------------
    # STEP 5: Sample RNA counts only for signature genes (NB)
    # ----------------------------------------------------------
    # Determine which genes are ever targeted by any signature (and exist in the simulated gene universe)
    sig_genes = sorted({
        tgt
        for sig in true_signatures.values()
        for tgt in sig["targets"]
        if tgt in expr_effected.columns
    })
    
    # Align dispersions to the μ matrix columns
    disp_subset = dispersions.reindex(expr_effected.columns)

    
    # Draw NB counts for signature genes (implementation may sample only sig_genes)
    expr_sim = sample_nb_for_signature_genes(
        mu_df=expr_effected,
        dispersions=disp_subset,
        sig_genes=sig_genes,
        seed=seed,
    )

    return expr_sim, true_signatures, sim_og

