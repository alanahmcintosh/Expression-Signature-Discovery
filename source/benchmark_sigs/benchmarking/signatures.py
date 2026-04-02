'''
Clean signatures, match target names between true and predicted, ensure signature dict orientation
'''

from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from benchmark_sigs.utils.list import as_list



def detect_signature_orientation(sig_dict, truth_alts=None):
    """
    Detect whether a signature dict is:
      - method_first: {method: {alt: genes}}
      - alt_first:    {alt: {method: genes}}
      - unknown
    """
    if not isinstance(sig_dict, dict) or not sig_dict:
        return "unknown"

    keys = [str(k) for k in sig_dict.keys()]

    if truth_alts is not None:
        truth_alts = set(map(str, truth_alts))
        n_in_truth = sum(k in truth_alts for k in keys)

        if n_in_truth >= max(1, len(keys) // 2):
            return "alt_first"
        return "method_first"

    # fallback heuristic
    first_val = next(iter(sig_dict.values()))
    if isinstance(first_val, dict):
        inner_keys = [str(k) for k in first_val.keys()]
        if truth_alts is not None:
            n_inner_in_truth = sum(k in truth_alts for k in inner_keys)
            if n_inner_in_truth >= max(1, len(inner_keys) // 2):
                return "method_first"

    return "method_first"


def to_method_first(sig_dict, truth_alts=None):
    """
    Normalize predicted signatures to:
        {method: {alt: [genes]}}

    Supports either input orientation:
        1) {alt: {method: genes}}
        2) {method: {alt: genes}}
    """
    if not isinstance(sig_dict, dict) or not sig_dict:
        return {}, "unknown"

    orientation = detect_signature_orientation(sig_dict, truth_alts=truth_alts)

    if orientation == "alt_first":
        out = defaultdict(dict)
        for alt, md in sig_dict.items():
            if not isinstance(md, dict):
                continue
            for method, genes in md.items():
                out[str(method)][str(alt)] = as_list(genes)
        return dict(out), orientation

    # assume already method-first
    out = {}
    for method, md in sig_dict.items():
        if not isinstance(md, dict):
            continue
        out[str(method)] = {str(alt): as_list(genes) for alt, genes in md.items()}
    return out, orientation


def restrict_methods_to_alts(methods, kept_alts):
    """
    Restrict normalized method-first signatures to the kept alteration set.
    """
    kept_alts = set(map(str, kept_alts))
    out = {}
    for method, alt_dict in methods.items():
        if not isinstance(alt_dict, dict):
            continue
        out[str(method)] = {
            str(alt): as_list(genes)
            for alt, genes in alt_dict.items()
            if str(alt) in kept_alts
        }
    return out

def summarize_truth_effects(truth_rich):
    """
    Convert truth dictionary into summary table.
    Gives size and strength of each alteration.
    """
    rows = []
    for alt, info in truth_rich.items():
        effects = info.get("effects", {}) or {}
        vals = np.array(list(effects.values()), dtype=float) if len(effects) else np.array([], dtype=float)

        rows.append({
            "alteration": alt,
            "truth_sig_size": int(len(info.get("targets", set()))),
            "truth_effect_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "truth_effect_mean_abs": float(np.mean(np.abs(vals))) if len(vals) else np.nan,
            "truth_effect_median_abs": float(np.median(np.abs(vals))) if len(vals) else np.nan,
            "truth_effect_max_abs": float(np.max(np.abs(vals))) if len(vals) else np.nan,
            "truth_effect_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            "truth_effect_n": int(len(vals)),
        })

    return pd.DataFrame(rows)

def extract_targets_and_effects(sig_obj):
    """
    Returns (targets_list, effects_list).
    Supports common signature dict styles.
    """
    if sig_obj is None:
        return [], []

    targets = None
    if isinstance(sig_obj, dict):
        for k in ["targets", "target_genes", "genes", "signature", "signature_genes"]:
            if k in sig_obj and isinstance(sig_obj[k], (list, tuple, set)):
                targets = list(sig_obj[k])
                break

    effects = None
    if isinstance(sig_obj, dict):
        for k in ["effects", "log2fc", "log2FC", "effect_sizes"]:
            if k in sig_obj:
                effects = sig_obj[k]
                break

    if effects is None:
        return (targets or []), []

    if isinstance(effects, dict):
        if targets is None:
            targets = list(effects.keys())
        eff_list = [effects.get(g, np.nan) for g in targets]
        return [str(t) for t in targets], eff_list

    if isinstance(effects, (list, tuple, np.ndarray)):
        eff_list = list(effects)
        if targets is None:
            return [], eff_list
        if len(eff_list) != len(targets):
            eff_list = eff_list[:len(targets)]
        return [str(t) for t in targets], eff_list

    return (targets or []), []