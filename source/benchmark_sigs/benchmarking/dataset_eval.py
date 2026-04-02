from __future__ import annotations
import os
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from benchmark_sigs.utils.list import as_list
from benchmark_sigs.benchmarking import compile_robustness_bundle, save_robustness_outputs, evaluate_dataset_outputs, summarize_truth_effects, read_dataset_for_evaluation, evaluate_signature


def evaluate_dataset_outputs(
    dataset,
    out_dir,
    truth,
    all_genes,
    methods,
    truth_alts,
    bundle,
    compute_robustness=True,
    robustness_min_group_n=5,
    fisher_min_co_count=0,
    fisher_p_thr=None,
    effective_kwargs=None,
):
    """
    Compares predicted gene signatures against the simulation ground truth
    for every (method × alteration) pair and produces:

        1) Per-signature metrics  -> df_individual
        2) Per-method summaries   -> df_overall
        3) Robustness/confounding outputs (optional)
        4) Saved prediction JSON  -> reproducibility
        5) Diagnostics JSON       -> sanity checks
    """
    individual_rows = []
    predicted_nested = defaultdict(dict)

    truth_alts = [alt for alt in truth_alts if alt in truth]

    robustness_df = None
    fisher_pairs_df = None
    robustness_status = "not_computed"
    robustness_error = None

    if compute_robustness:
        try:
            X_for_robust = bundle.get("alterations_clean")
            Y_for_robust = bundle.get("rna")
            if X_for_robust is None or Y_for_robust is None:
                raise ValueError("bundle must contain 'alterations_clean' and full 'rna' to compute robustness outputs.")

            robustness_df, co_mats, fisher_by_alt, _, _ = compile_robustness_bundle(
                X=X_for_robust,
                Y=Y_for_robust,
                true_signatures=truth,
                min_group_n=robustness_min_group_n,
                fisher_min_co_count=fisher_min_co_count,
                fisher_p_thr=fisher_p_thr,
                effective_kwargs=effective_kwargs or {},
            )
            fisher_pairs_df = save_robustness_outputs(dataset, out_dir, robustness_df, co_mats, fisher_by_alt)
            robustness_status = "computed"
        except Exception as e:
            robustness_status = "failed"
            robustness_error = str(e)
            print(f"[{dataset}] robustness bundle failed: {e}")

    for method, alt_dict in methods.items():
        if not isinstance(alt_dict, dict):
            continue

        for alt in truth_alts:
            predicted = as_list(alt_dict.get(alt, []))
            predicted_nested[str(method)][str(alt)] = predicted

            true_targets = set(map(str, truth[alt]["targets"]))
            metrics = evaluate_signature(predicted, true_targets, all_genes)

            individual_rows.append({
                "dataset": dataset,
                "method": str(method),
                "alteration": str(alt),
                "true_sig_size": int(len(true_targets)),
                "pred_sig_size": int(len(predicted)),
                **metrics,
            })

    df_individual = pd.DataFrame(individual_rows)

    if not df_individual.empty:
        truth_eff_df = summarize_truth_effects(truth)
        truth_eff_df["dataset"] = dataset
        df_individual = df_individual.merge(
            truth_eff_df,
            on=["dataset", "alteration"],
            how="left",
        )

        if robustness_df is not None and not robustness_df.empty:
            df_individual = df_individual.merge(
                robustness_df.drop(columns=[c for c in ["targets", "effects"] if c in robustness_df.columns]),
                on="alteration",
                how="left",
            )

    overall_rows = []
    if not df_individual.empty:
        for method in sorted(df_individual["method"].unique()):
            dm = df_individual[df_individual["method"] == method].copy()

            row = {
                "dataset": dataset,
                "method": method,
                "avg_f1": float(dm["f1"].mean()) if len(dm) else 0.0,
                "avg_precision": float(dm["precision"].mean()) if len(dm) else 0.0,
                "avg_recall": float(dm["recall"].mean()) if len(dm) else 0.0,
                "avg_accuracy": float(dm["accuracy"].mean()) if len(dm) else 0.0,
                "avg_jaccard": float(dm["jaccard"].mean()) if len(dm) else 0.0,
                "avg_mcc": float(dm["mcc"].mean()) if len(dm) else 0.0,
                "n_signatures_evaluated": int(len(dm)),
            }

            for col in [
                "base_snr",
                "effective_snr",
                "confound_score",
                "n_partners_used",
                "mean_abs_effect",
                "alt_freq",
                "mean_abs_log_or",
                "mean_co_count_pairs",
                "max_co_count_pairs",
                "mean_co_percent_pairs",
                "max_co_percent_pairs",
            ]:
                if col in dm.columns:
                    row[f"avg_{col}"] = float(dm[col].mean()) if len(dm) else np.nan

            overall_rows.append(row)

    df_overall = pd.DataFrame(overall_rows)

    # --- write outputs ---
    df_individual.to_csv(os.path.join(out_dir, f"individual_results_{dataset}.csv"), index=False)
    df_overall.to_csv(os.path.join(out_dir, f"overall_results_{dataset}.csv"), index=False)

    with open(os.path.join(out_dir, f"predicted_signatures_{dataset}.json"), "w") as f:
        json.dump(dict(predicted_nested), f, indent=2)

    diag = {
        "dataset": dataset,
        "methods_n": int(len(methods)),
        "truth_alterations_n": int(len(truth)),
        "kept_alterations_n": int(len(bundle.get("kept_alterations", []))),
        "evaluated_rows_n": int(len(df_individual)),
        "alterations_shape_raw": [
            int(bundle["alterations"].shape[0]),
            int(bundle["alterations"].shape[1]),
        ] if "alterations" in bundle and bundle["alterations"] is not None else None,
        "alterations_shape_clean": [
            int(bundle["alterations_clean"].shape[0]),
            int(bundle["alterations_clean"].shape[1]),
        ] if "alterations_clean" in bundle and bundle["alterations_clean"] is not None else None,
        "rna_gene_universe_n": int(len(all_genes)),
        "orientation_supervised": bundle.get("orientation_supervised"),
        "orientation_unsupervised": bundle.get("orientation_unsupervised"),
        "drop_info_counts": {
            k: len(v) if isinstance(v, list) else None
            for k, v in bundle.get("drop_info", {}).items()
        },
        "drop_info": bundle.get("drop_info", {}),
        "filtering_params": bundle.get("filtering_params", {}),
        "robustness_status": robustness_status,
        "robustness_error": robustness_error,
        "robustness_rows_n": int(len(robustness_df)) if robustness_df is not None else 0,
        "fisher_pairs_rows_n": int(len(fisher_pairs_df)) if fisher_pairs_df is not None else 0,
        "compute_robustness": bool(compute_robustness),
        "robustness_min_group_n": int(robustness_min_group_n),
        "fisher_min_co_count": int(fisher_min_co_count),
        "fisher_p_thr": fisher_p_thr,
        "effective_kwargs": effective_kwargs or {},
    }

    with open(os.path.join(out_dir, f"diagnostics_{dataset}.json"), "w") as f:
        json.dump(diag, f, indent=2)

    print(
        f"[{dataset}] methods={len(methods)} "
        f"truth_alts_kept={len(truth)} "
        f"evaluated_rows={len(df_individual)} "
        f"(sup_orient={bundle.get('orientation_supervised')}, robustness={robustness_status})"
    )

    return df_overall, df_individual, dict(predicted_nested), bundle


def evaluate_dataset(dataset, RESULTS_DIR):
    """
    Backwards-compatible wrapper 

    """
    ds = read_dataset_for_evaluation(dataset, RESULTS_DIR)
    return evaluate_dataset_outputs(
        dataset=ds["dataset"],
        out_dir=ds["out_dir"],
        truth=ds["truth"],
        all_genes=ds["all_genes"],
        methods=ds["methods"],
        truth_alts=ds["truth_alts"],
        bundle=ds["bundle"],
    )
