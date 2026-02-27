import pandas as pd 
import numpy as np 
import json
import joblib
import os
from collections import defaultdict
from typing import Dict, Any, List, Set, Tuple, Optional


"""
HELPER
"""

# ==========================================================
# Small helpers
# ==========================================================
def as_list(x):
    """Normalize signature outputs to a list of strings."""
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x if str(t).strip()]
    if isinstance(x, (set, tuple)):
        return [str(t) for t in list(x) if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if ";" in s:
            return [t.strip() for t in s.split(";") if t.strip()]
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    try:
        return [str(t) for t in list(x)]
    except Exception:
        return [str(x)]


"""
==============================================================
DATA LOADER
==============================================================

Loads for each Dataset:
- Simulated RNA expression .csv file
- Simulated Altertaions .csv file
- True Signatures .json
- Predicted signatures .joblib
==============================================================
"""

def safe_read_csv(path, index_col=0, nrows=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, index_col=index_col, nrows=nrows)

  

def load_rna_gene_universe(dataset: str) -> Tuple[Set[str], Optional[pd.DataFrame]]:
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


def load_alterations(dataset):
    alt_path = os.path.join(BASE_DIR, dataset, f"alterations_{dataset}.csv")
    X = safe_read_csv(alt_path, index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    return X


def load_truth_rich(dataset):
    """
    Load gorund truth signaturesand convert them to a consistent representation
    """
    truth_path = os.path.join(BASE_DIR, dataset, f"true_signatures_{dataset}.json")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Missing truth file: {truth_path}")

    with open(truth_path) as f:
        truth = json.load(f)

    out = {}
    
    #Each key in the true signatures dict is an altertaion (eg TP53_LOF)
    for alt, payload in truth.items():
        alt = str(alt) #converts to string

        if isinstance(payload, dict): 
            # Targets may be a set/list/tuple - normalize them to a set of strings (set(str))
            targets = set(map(str, as_list(payload.get("targets", []))))
            
            #Standardise format of targets and effects  ( should be stored as str(Target) -> float(effect))
            effects_raw = payload.get("effects", {}) or {} 
            effects = {str(g): float(v) for g, v in effects_raw.items()}
            
            #store the standrdised truth signatures 
            out[alt] = {
                "targets": targets,
                "effects": effects,
                "n_targets": int(payload.get("n_targets", len(targets))), #if n_targets is stored use that, if not compute
            }


    return out


def load_signature_joblib(path):
    '''
    Loads raw predicted signature files for each datasets
    '''
    if not os.path.exists(path):
        return {}
    obj = joblib.load(path)
    return obj if isinstance(obj, dict) else {}


def to_method_first(sig_dict):
    """
    Organize predicted signature outpiuts into a single format:
    {alt:{method: [genes]}} to {method: {alt: [genes]}}
    """
    if not isinstance(sig_dict, dict) or not sig_dict:
        return {}

    out = defaultdict(dict)
    for alt, md in sig_dict.items():
        if not isinstance(md, dict):
            continue
        for method, genes in md.items():
            out[str(method)][str(alt)] = as_list(genes)
    return dict(out)


def load_dataset_bundle(dataset):
    """
    Loads everything per cancer folder:
      - true_signatures_{dataset}.json
      - rna_simulated_{dataset}.csv  (header only unless LOAD_FULL_RNA)
      - alterations_{dataset}.csv
      - supervised_signatures.joblib (+ unsupervised if enabled)
    """
    folder = os.path.join(BASE_DIR, dataset)

    truth = load_truth_rich(dataset)
    all_genes, rna_df = load_rna_gene_universe(dataset)
    X_alts = load_alterations(dataset)

    sup_path = os.path.join(folder, "supervised_signatures.joblib")
    sup_raw = load_signature_joblib(sup_path)
    sup = to_method_first(sup_raw)

    uns = {}
    uns_raw = {}
    if LOAD_UNSUPERVISED:
        uns_path = os.path.join(folder, "unsupervised_signatures.joblib")
        if os.path.exists(uns_path):
            uns_raw = load_signature_joblib(uns_path)
            uns = to_method_first(uns_raw)

    all_methods: Dict[str, Dict[str, List[str]]] = {}
    all_methods.update(sup)
    all_methods.update(uns)

    return {
        "dataset": dataset,
        "folder": folder,
        "truth": truth,
        "all_genes": all_genes,
        "rna": rna_df,               # may be None
        "alterations": X_alts,
        "supervised_raw": sup_raw,
        "unsupervised_raw": uns_raw,
        "methods": all_methods,
    }



def read_dataset_for_evaluation(
    dataset, results_dir):
    """
    Read + prepare everything needed to evaluate one dataset.
    Keeps I/O and directory setup out of the evaluation logic.
    """
    out_dir = os.path.join(results_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    bundle = load_dataset_bundle(dataset)

    truth = bundle["truth"]            # {alt: {"targets": set, "effects": dict, ...}}
    all_genes = bundle["all_genes"]    # set[str] universe for FP/FN definition
    methods = bundle["methods"]        # {method: {alt: [genes...]}} (already normalized in bundle)
    truth_alts = list(truth.keys())    # preserve ordering

    return {
        "dataset": dataset,
        "out_dir": out_dir,
        "bundle": bundle,
        "truth": truth,
        "all_genes": all_genes,
        "methods": methods,
        "truth_alts": truth_alts,
    }


"""
==============================================================
DATA SUMMARISATION/EVALUATION
==============================================================


==============================================================
"""

def summarize_truth_effects(truth_rich):
    '''
    Convert truth dictionary into summary table, gives size and strength of each alteration. Useful for later when we are computing simu;lation efficacy based off of sig size/effect
    '''
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

def evaluate_signature(predicted, true_set, all_genes):
    pred_set = set(map(str, as_list(predicted)))
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
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom else 0.0

    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "mcc": float(mcc)
    }



def evaluate_dataset_outputs(
    dataset,
    out_dir,
    truth,
    all_genes,
    methods,
    truth_alts,
    bundle
):
    """

    Compares predicted gene signatures against the simulation ground truth
    for every (method × alteration) pair and produces:

        1) Per-signature metrics  → df_individual
        2) Per-method summaries   → df_overall
        3) Saved prediction JSON  → reproducibility
        4) Diagnostics JSON       → sanity checks

    """
    
    #stores one row per evaluated signatures
    individual_rows = []
    
    # Nested structure: {method -> {alteration -> predicted_genes}}
    # defaultdict allows automatic creation of inner dicts
    predicted_nested = defaultdict(dict)

    # --- per (method, alteration) evaluation ---
    for method, alt_dict in methods.items():
        if not isinstance(alt_dict, dict):
            continue

        for alt in truth_alts:
            
            #Get predicted target genes as a list
            predicted = as_list(alt_dict.get(alt, []))
#             if len(predicted) == 0:
#                 continue

            predicted_nested[str(method)][str(alt)] = predicted
    
            # Get ground truth targetr genes
            true_targets = truth[alt]["targets"]
            
            #evaluate methods 
            metrics = evaluate_signature(predicted, true_targets, all_genes)

            #Record an evaluation per row
            individual_rows.append({
                "dataset": dataset,
                "method": str(method),
                "alteration": str(alt),
                "true_sig_size": int(len(true_targets)),
                "pred_sig_size": int(len(predicted)),
                **metrics,
            })

    #make into a dataframe
    df_individual = pd.DataFrame(individual_rows)

    # Add truth summaries (effect, mean , max ...)
    truth_eff_df = summarize_truth_effects(truth)
    truth_eff_df["dataset"] = dataset
    
    #Merge to individual df
    df_individual = df_individual.merge(
        truth_eff_df,
        on=["dataset", "alteration"],
        how="left",
    )

    # Summarises the metrics per method per dataset as per df_individual
    overall_rows = []
    for method in sorted(df_individual["method"].unique()):
        dm = df_individual[df_individual["method"] == method].copy()

        overall_rows.append({
            "dataset": dataset,
            "method": method,
            "avg_f1": float(dm["f1"].mean()) if len(dm) else 0.0,
            "avg_precision": float(dm["precision"].mean()) if len(dm) else 0.0,
            "avg_recall": float(dm["recall"].mean()) if len(dm) else 0.0,
            "n_signatures_evaluated": int(len(dm)),
        })

    df_overall = pd.DataFrame(overall_rows)

    # --- write outputs ---
    #Full per alteration evaluation df
    df_individual.to_csv(os.path.join(out_dir, f"individual_results_{dataset}.csv"), index=False)
    
    #Method Summary Dataframe
    df_overall.to_csv(os.path.join(out_dir, f"overall_results_{dataset}.csv"), index=False)

    with open(os.path.join(out_dir, f"predicted_signatures_{dataset}.json"), "w") as f:
        json.dump(dict(predicted_nested), f, indent=2)

    # --- diagnostics JSON (helps debug orientation / bundle integrity) ---
    diag = {
        "dataset": dataset,
        "methods_n": int(len(methods)),
        "truth_alterations_n": int(len(truth)),
        "evaluated_rows_n": int(len(df_individual)),
        "alterations_shape": [
            int(bundle["alterations"].shape[0]),
            int(bundle["alterations"].shape[1]),
        ] if "alterations" in bundle else None,
        "rna_gene_universe_n": int(len(all_genes)),
    }
    with open(os.path.join(out_dir, f"diagnostics_{dataset}.json"), "w") as f:
        json.dump(diag, f, indent=2)

    print(
        f"[{dataset}] methods={len(methods)} truth_alts={len(truth)} "
        f"evaluated_rows={len(df_individual)} "
        f"(sup_orient={bundle.get('orientation_supervised')})"
    )

    return df_overall, df_individual, dict(predicted_nested), bundle

"""
==============================================================
RUN ALL
==============================================================


==============================================================
"""

def evaluate_dataset(dataset: str):
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

