'''
Load RNA, altertaions, true signatures and predicted signatures for evaluation
'''

from __future__ import annotations
import os
import re
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import json
from benchmark_sigs.utils import as_list, sanitize_binary_design
from benchmark_sigs.benchmarking import restrict_methods_to_alts, to_method_first



def safe_read_csv(path, index_col=0, nrows=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, index_col=index_col, nrows=nrows)


def load_rna_gene_universe(dataset, BASE_DIR, LOAD_FULL_RNA):
    """
    Returns:
      all_genes: set of RNA gene names from columns
      rna_df: full RNA DataFrame if LOAD_FULL_RNA=True else None
    """
    rna_path = os.path.join(BASE_DIR, dataset, f"rna_simulated_{dataset}.csv")
    if not os.path.exists(rna_path):
        raise FileNotFoundError(f"Missing RNA file: {rna_path}")

    if LOAD_FULL_RNA:
        rna_df = pd.read_csv(rna_path, index_col=0)
        rna_df.columns = rna_df.columns.astype(str)
        rna_df.index = rna_df.index.astype(str)
        return set(rna_df.columns.astype(str)), rna_df

    cols = pd.read_csv(rna_path, index_col=0, nrows=1).columns.astype(str)
    return set(cols), None


def load_alterations(dataset, BASE_DIR):
    alt_path = os.path.join(BASE_DIR, dataset, f"alterations_{dataset}.csv")
    X = safe_read_csv(alt_path, index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    return X


def load_truth_rich(dataset, BASE_DIR):
    """
    Load ground truth signatures and convert them to a consistent representation.
    """
    truth_path = os.path.join(BASE_DIR, dataset, f"true_signatures_{dataset}.json")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Missing truth file: {truth_path}")

    with open(truth_path) as f:
        truth = json.load(f)

    out = {}

    for alt, payload in truth.items():
        alt = str(alt)

        if isinstance(payload, dict):
            targets = set(map(str, as_list(payload.get("targets", []))))
            effects_raw = payload.get("effects", {}) or {}
            effects = {str(g): float(v) for g, v in effects_raw.items()}

            out[alt] = {
                "targets": targets,
                "effects": effects,
                "n_targets": int(payload.get("n_targets", len(targets))),
            }

    return out


def load_signature_joblib(path):
    """
    Load raw predicted signature files for each dataset.
    """
    if not os.path.exists(path):
        return {}
    obj = joblib.load(path)
    return obj if isinstance(obj, dict) else {}

def load_dataset_bundle(
    dataset,
    BASE_DIR,
    LOAD_UNSUPERVISED=False,
    LOAD_FULL_RNA=True,
    FILTER_ALTERATIONS=True,
    min_group_n=5,
    priority_keywords=("GOF", "LOF", "FUSION"),
    min_residual_df=5,
):
    """
    Loads everything per cancer folder:
      - true_signatures_{dataset}.json
      - rna_simulated_{dataset}.csv
      - alterations_{dataset}.csv
      - combined_signatures_{dataset}.joblib
      - unsupervised_signatures.joblib (optional)

    Applies the same alteration filtering logic used for model fitting so
    evaluation only considers retained alterations.
    """
    folder = os.path.join(BASE_DIR, dataset)

    truth_full = load_truth_rich(dataset, BASE_DIR)
    all_genes, rna_df = load_rna_gene_universe(dataset, BASE_DIR, LOAD_FULL_RNA)
    X_alts_raw = load_alterations(dataset, BASE_DIR)

    if FILTER_ALTERATIONS:
        X_alts_clean, drop_info = sanitize_binary_design(
            X_alts_raw,
            priority_keywords=priority_keywords,
            return_drop_info=True,
        )
    else:
        X_alts_clean = X_alts_raw.copy()
        drop_info = {
            "constant": [],
            "imbalanced": [],
            "duplicate": [],
            "dependent": [],
            "capacity": [],
            "kept": list(X_alts_clean.columns.astype(str)),
        }

    kept_alts = list(map(str, X_alts_clean.columns))
    truth = {alt: truth_full[alt] for alt in kept_alts if alt in truth_full}

    # supervised signatures
    sup_path = os.path.join(folder, f"combined_signatures_{dataset}.joblib")
    sup_raw = load_signature_joblib(sup_path)
    sup, sup_orientation = to_method_first(sup_raw, truth_alts=kept_alts)
    sup = restrict_methods_to_alts(sup, kept_alts)

    # unsupervised signatures
    uns = {}
    uns_raw = {}
    uns_orientation = "not_loaded"
    if LOAD_UNSUPERVISED:
        uns_path = os.path.join(folder, "unsupervised_signatures.joblib")
        if os.path.exists(uns_path):
            uns_raw = load_signature_joblib(uns_path)
            uns, uns_orientation = to_method_first(uns_raw, truth_alts=kept_alts)
            uns = restrict_methods_to_alts(uns, kept_alts)

    all_methods = {}
    all_methods.update(sup)
    all_methods.update(uns)

    return {
        "dataset": dataset,
        "folder": folder,
        "truth": truth,
        "truth_full": truth_full,
        "all_genes": all_genes,
        "rna": rna_df,
        "alterations": X_alts_raw,
        "alterations_clean": X_alts_clean,
        "kept_alterations": kept_alts,
        "drop_info": drop_info,
        "supervised_raw": sup_raw,
        "unsupervised_raw": uns_raw,
        "methods": all_methods,
        "orientation_supervised": sup_orientation,
        "orientation_unsupervised": uns_orientation,
        "filtering_params": {
            "FILTER_ALTERATIONS": FILTER_ALTERATIONS,
            "min_group_n": min_group_n,
            "priority_keywords": list(priority_keywords),
            "min_residual_df": min_residual_df,
        },
    }


def read_dataset_for_evaluation(dataset, results_dir):
    """
    Read + prepare everything needed to evaluate one dataset.
    Keeps I/O and directory setup out of the evaluation logic.
    """
    out_dir = os.path.join(results_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    bundle = load_dataset_bundle(
        dataset=dataset,
        BASE_DIR=results_dir,
        LOAD_UNSUPERVISED=False,
        LOAD_FULL_RNA=True,
        FILTER_ALTERATIONS=True,
        min_group_n=5,
        priority_keywords=("GOF", "LOF", "FUSION"),
        min_residual_df=5,
    )

    truth = bundle["truth"]
    all_genes = bundle["all_genes"]
    methods = bundle["methods"]
    truth_alts = list(bundle["kept_alterations"])

    return {
        "dataset": dataset,
        "out_dir": out_dir,
        "bundle": bundle,
        "truth": truth,
        "all_genes": all_genes,
        "methods": methods,
        "truth_alts": truth_alts,
    }

def save_robustness_outputs(dataset, out_dir, alt_df, co_mats, fisher_by_alt):
    alt_df.to_csv(os.path.join(out_dir, f"robustness_by_alteration_{dataset}.csv"), index=False)
    co_mats["co_occ"].to_csv(os.path.join(out_dir, f"co_occurrence_counts_{dataset}.csv"))
    co_mats["exp"].to_csv(os.path.join(out_dir, f"co_occurrence_expected_{dataset}.csv"))
    co_mats["co_occ_diff"].to_csv(os.path.join(out_dir, f"co_occurrence_diff_{dataset}.csv"))

    fisher_all = []
    for alt, sub in fisher_by_alt.items():
        if sub is None or sub.empty:
            continue
        tmp = sub.copy()
        tmp["query_alteration"] = alt
        fisher_all.append(tmp)

    fisher_out = pd.concat(fisher_all, ignore_index=True) if fisher_all else pd.DataFrame()
    fisher_out.to_csv(os.path.join(out_dir, f"fisher_co_occurrence_pairs_{dataset}.csv"), index=False)
    return fisher_out

