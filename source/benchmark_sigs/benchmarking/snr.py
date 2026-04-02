from __future__ import annotations
import pandas as pd
import numpy as np

from benchmark_sigs.utils.list import as_list
from benchmark_sigs.methods.supervised import normalize_counts_log_cpm
from benchmark_sigs.benchmarking.metrics import co_occurence, jaccard



def snr_for_alt(Y_norm, X_bin, alt, true_targets, min_group_n=5):
    if alt not in X_bin.columns:
        return np.nan

    x = X_bin[alt].astype(int)
    idx1 = x.index[x == 1]
    idx0 = x.index[x == 0]
    if (len(idx1) < min_group_n) or (len(idx0) < min_group_n):
        return np.nan

    genes = [g for g in true_targets if g in Y_norm.columns]
    if len(genes) == 0:
        return np.nan

    Y1 = Y_norm.loc[idx1, genes]
    Y0 = Y_norm.loc[idx0, genes]

    def pooled_sd(x1, x0):
        v = (x1.var(axis=0, ddof=1) + x0.var(axis=0, ddof=1)) / 2.0
        return np.sqrt(v.replace(0, np.nan))

    diff = (Y1.mean(axis=0) - Y0.mean(axis=0)).abs()
    sd = pooled_sd(Y1, Y0)

    d = (diff / sd).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return np.nan

    return float(d.mean())


def SNR(X, Y, true_signatures, min_group_n=1):

    Y_norm = normalize_counts_log_cpm(Y)
    rows = []
    for alt, sig in true_signatures.items():
        targets = as_list(sig.get("targets", [])) if isinstance(sig, dict) else []
        snr = snr_for_alt(Y_norm, X, alt, targets, min_group_n=min_group_n)
        rows.append({"alteration": str(alt), "snr": snr, "n_targets": len(targets)})
    return pd.DataFrame(rows)


def effective_snr(
    X,
    Y,
    true_signatures,
    fisher_df=None,
    min_group_n=1,
    p_thr=0.05,
    use_only_sig_pairs=False,
    assoc_metric="log_odds_smooth",
    cap_log_or=6.0,
    min_co_count=0,
    mode="divide",
    lam=1.0,
):
    """
    Compute base SNR and an effective SNR that downweights signal for
    alterations that strongly co-occur with other alterations affecting
    overlapping target genes.
    """
    snr_df = SNR(X, Y, true_signatures, min_group_n=min_group_n).copy()
    snr_df = snr_df.rename(columns={"snr": "snr_base"})

    if fisher_df is None:
        _, _, _, fisher_df = co_occurence(X)

    df = fisher_df.copy()

    if min_co_count > 0 and "co_count" in df.columns:
        df = df[df["co_count"] >= min_co_count]

    if use_only_sig_pairs and "p" in df.columns:
        df = df[df["p"] < p_thr]

    if assoc_metric in ("abs_log_odds_smooth", "log_odds_smooth"):
        log_or = np.log(df["odds_smooth"].replace(0, np.nan))
        log_or = log_or.replace([np.inf, -np.inf], np.nan).clip(-cap_log_or, cap_log_or)
        if assoc_metric == "abs_log_odds_smooth":
            df["assoc"] = log_or.abs()
        else:
            df["assoc"] = log_or
    else:
        raise ValueError(f"Unknown assoc_metric: {assoc_metric}")

    df = df.dropna(subset=["assoc"])

    assoc_map = {}
    for a1, a2, assoc in df[["alt1", "alt2", "assoc"]].itertuples(index=False):
        assoc_map[(a1, a2)] = assoc
        assoc_map[(a2, a1)] = assoc

    target_sets = {
        str(alt): set(map(str, sig.get("targets", [])))
        for alt, sig in true_signatures.items()
        if isinstance(sig, dict)
    }

    conf_rows = []
    alts = list(target_sets.keys())

    for A in alts:
        setA = target_sets.get(A, set())
        conf = 0.0
        n_used = 0

        for B in alts:
            if B == A:
                continue
            assoc = assoc_map.get((A, B), None)
            if assoc is None:
                continue

            setB = target_sets.get(B, set())
            ov = jaccard(setA, setB)
            if np.isnan(ov) or ov == 0:
                continue

            conf += float(assoc) * float(ov)
            n_used += 1

        conf_rows.append({"alteration": A, "confound_score": conf, "n_partners_used": n_used})

    conf_df = pd.DataFrame(conf_rows)

    out = snr_df.merge(conf_df, on="alteration", how="left")
    out["confound_score"] = out["confound_score"].fillna(0.0)
    out["n_partners_used"] = out["n_partners_used"].fillna(0).astype(int)

    if mode == "divide":
        out["effective_snr"] = out["snr_base"] / (1.0 + out["confound_score"])
    elif mode == "subtract":
        out["effective_snr"] = out["snr_base"] - (lam * out["confound_score"])
    else:
        raise ValueError("mode must be 'divide' or 'subtract'")

    return out.sort_values("effective_snr", ascending=False).reset_index(drop=True)

