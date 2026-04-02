
import numpy as np
import pandas as pd

from benchmark_sigs.benchmarking import co_occurence, SNR, effective_snr, extract_targets_and_effects
from benchmark_sigs.utils import as_list


def compile_robustness_bundle(
    X,
    Y,
    true_signatures,
    min_group_n=1,
    fisher_min_co_count=0,
    fisher_p_thr=None,
    effective_kwargs=None,
):
    """
    Build a one-stop bundle for robustness analyses.

    Returns
    -------
    alt_df : pd.DataFrame
        One row per alteration with effect size, SNR, confounding, and frequency summaries.
    co_mats : dict
        Full co-occurrence matrices.
    fisher_by_alt : dict
        alt -> fisher sub-table.
    targets_by_alt : dict
    effects_by_alt : dict
    """
    effective_kwargs = effective_kwargs or {}

    co_occ, exp, co_occ_diff, fisher_df = co_occurence(X)
    co_mats = {"co_occ": co_occ, "exp": exp, "co_occ_diff": co_occ_diff}

    base_snr_df = SNR(X, Y, true_signatures, min_group_n=min_group_n).rename(columns={"snr": "base_snr"})

    eff_df = effective_snr(
        X=X,
        Y=Y,
        true_signatures=true_signatures,
        fisher_df=fisher_df,
        min_group_n=min_group_n,
        **effective_kwargs,
    )
    eff_df = eff_df.rename(columns={"snr_base": "base_snr"}).copy()

    rows = []
    targets_by_alt = {}
    effects_by_alt = {}

    freq = X.astype(int).sum(axis=0)

    for alt, sig in true_signatures.items():
        targets, effects = extract_targets_and_effects(sig)
        targets_by_alt[str(alt)] = targets
        effects_by_alt[str(alt)] = effects

        eff_arr = np.array([e for e in effects if e is not None], dtype=float) if effects else np.array([], dtype=float)
        eff_abs = np.abs(eff_arr) if eff_arr.size else eff_arr

        rows.append({
            "alteration": str(alt),
            "sig_size": len(targets),
            "targets": targets,
            "effects": effects,
            "mean_abs_effect": float(np.nanmean(eff_abs)) if eff_abs.size else np.nan,
            "median_abs_effect": float(np.nanmedian(eff_abs)) if eff_abs.size else np.nan,
            "mean_signed_effect": float(np.nanmean(eff_arr)) if eff_arr.size else np.nan,
            "median_signed_effect": float(np.nanmedian(eff_arr)) if eff_arr.size else np.nan,
            "alt_freq": int(freq.get(alt, 0)),
        })

    meta_df = pd.DataFrame(rows)

    alt_df = meta_df.merge(base_snr_df[["alteration", "base_snr"]], on="alteration", how="left")
    alt_df = alt_df.merge(
        eff_df[["alteration", "base_snr", "effective_snr", "confound_score", "n_partners_used"]],
        on="alteration",
        how="left",
        suffixes=("", "_eff"),
    )

    if "base_snr_eff" in alt_df.columns:
        alt_df["base_snr"] = alt_df["base_snr_eff"].combine_first(alt_df["base_snr"])
        alt_df = alt_df.drop(columns=["base_snr_eff"])

    fisher_by_alt = {}
    f = fisher_df.copy()
    if fisher_min_co_count > 0:
        f = f[f["co_count"] >= fisher_min_co_count]
    if fisher_p_thr is not None:
        f = f[f["p"] < fisher_p_thr]

    all_alts = alt_df["alteration"].tolist()
    for alt in all_alts:
        sub = f[(f["alt1"] == alt) | (f["alt2"] == alt)].copy()
        fisher_by_alt[alt] = sub.reset_index(drop=True)

    if not f.empty:
        tmp = f.copy()
        tmp["abs_log_or"] = np.log(tmp["odds_smooth"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).abs()

        def summ_for_alt(alt_name):
            s = tmp[(tmp["alt1"] == alt_name) | (tmp["alt2"] == alt_name)]
            return pd.Series({
                "n_pairs_fisher": len(s),
                "mean_abs_log_or": float(np.nanmean(s["abs_log_or"])) if len(s) else np.nan,
                "max_abs_log_or": float(np.nanmax(s["abs_log_or"])) if len(s) else np.nan,
                "mean_co_count_pairs": float(np.nanmean(s["co_count"])) if len(s) else np.nan,
                "max_co_count_pairs": float(np.nanmax(s["co_count"])) if len(s) else np.nan,
                "mean_co_percent_pairs": float(np.nanmean(s["co_percent"])) if len(s) else np.nan,
                "max_co_percent_pairs": float(np.nanmax(s["co_percent"])) if len(s) else np.nan,
            })

        pair_summ = pd.DataFrame({alt: summ_for_alt(alt) for alt in all_alts}).T.reset_index().rename(columns={"index": "alteration"})
        alt_df = alt_df.merge(pair_summ, on="alteration", how="left")

    return alt_df, co_mats, fisher_by_alt, targets_by_alt, effects_by_alt



def evaluate_signature(predicted, true_set, all_genes):
    pred_set = set(map(str, as_list(predicted)))
    true_set = set(map(str, true_set))
    all_genes = set(map(str, all_genes))

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = len(all_genes - (pred_set | true_set))

    total = tp + fp + fn + tn
    assert total == len(all_genes)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    jaccard_score = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard_score),
        "mcc": float(mcc),
    }

