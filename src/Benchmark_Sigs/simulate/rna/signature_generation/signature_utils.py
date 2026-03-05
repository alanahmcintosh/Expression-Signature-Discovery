"""
Helper utilities for RNA signature simulation.

Contains:
- alteration name parsing
- target gene selection (default + shared)
- sampling helpers for signature sizes and effect magnitudes
"""

from __future__ import annotations

import numpy as np


def parse_alt(alt: str):
    """
    Parse an alteration feature name into (kind, base_gene, partners).

    Supported suffixes:
      - _GOF, _LOF, _AMP, _DEL, _FUSION
    """
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
        base_gene = partners[0] if partners else None
        return "FUSION", base_gene, partners

    return "OTHER", None, None


def sample_size(size_mean, rng, fallback_size_range=(10, 50)):
    """
    Sample an integer signature size around size_mean. Falls back if missing/invalid.
    """
    if size_mean is None:
        lo, hi = fallback_size_range
        return int(rng.integers(int(lo), int(hi) + 1))

    try:
        m = float(size_mean)
        if not (np.isfinite(m) and m > 0):
            lo, hi = fallback_size_range
            return int(rng.integers(int(lo), int(hi) + 1))
    except Exception:
        lo, hi = fallback_size_range
        return int(rng.integers(int(lo), int(hi) + 1))

    mu = float(size_mean)
    sd = max(1.0, 0.25 * mu)  # mild jitter
    L = int(round(rng.normal(mu, sd)))
    return max(1, L)


def sample_abs(abs_mu, abs_sigma, abs_cap, rng, fallback_abs_range=(1.0, 2.0)):
    """
    Sample an absolute effect magnitude.
    - Primary: lognormal(abs_mu, abs_sigma), capped at abs_cap
    - Fallback: uniform(fallback_abs_range)
    """
    mag = None
    try:
        if abs_mu is not None and abs_sigma is not None:
            if np.isfinite(abs_mu) and np.isfinite(abs_sigma) and float(abs_sigma) > 0:
                mag = float(rng.lognormal(mean=float(abs_mu), sigma=float(abs_sigma)))
    except Exception:
        mag = None

    if mag is None or not np.isfinite(mag) or mag <= 0:
        lo, hi = fallback_abs_range
        mag = float(rng.uniform(float(lo), float(hi)))

    if abs_cap is not None and np.isfinite(abs_cap) and float(abs_cap) > 0:
        mag = float(min(mag, float(abs_cap)))

    return float(mag)


def build_targets_default(base_gene, L, gene_set, genes, rng):
    """
    Default target selection:
    - include base_gene if present in gene_set
    - fill remaining targets uniformly without replacement
    """
    targets = []

    if base_gene is not None and base_gene in gene_set:
        targets.append(base_gene)

    remaining = [g for g in genes if g != base_gene]
    need = max(0, int(L) - len(targets))

    if need > 0:
        if need >= len(remaining):
            targets += remaining
        else:
            targets += list(rng.choice(remaining, size=need, replace=False))

    return targets


def build_targets_shared(base_gene, L, ref_targets, share_frac, gene_set, genes, rng):
    """
    Target selection for AMP/DEL:
    - include base_gene if present
    - share share_frac of non-base targets with a reference signature
    - fill the rest from remaining genes
    """
    targets = []
    if base_gene is not None and base_gene in gene_set:
        targets.append(base_gene)

    ref = [g for g in list(ref_targets) if g != base_gene]
    want_total = max(0, int(L) - len(targets))
    share_n = int(round(float(share_frac) * want_total))
    share_n = max(0, min(share_n, len(ref)))

    if share_n > 0:
        targets += list(rng.choice(ref, size=share_n, replace=False))

    chosen = set(targets)
    pool = [g for g in genes if g not in chosen]
    need = max(0, int(L) - len(targets))

    if need > 0:
        if need >= len(pool):
            targets += pool
        else:
            targets += list(rng.choice(pool, size=need, replace=False))

    return targets
