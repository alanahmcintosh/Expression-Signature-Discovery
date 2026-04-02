"""
Evaluation utilities for benchmarking predicted gene signatures
against simulated ground-truth signatures.
"""

from .io import (
    safe_read_csv,
    load_rna_gene_universe,
    load_alterations,
    load_truth_rich,
    load_signature_joblib,
    load_dataset_bundle,
    read_dataset_for_evaluation,
    save_robustness_outputs,
)

from .signatures import (
    detect_signature_orientation,
    to_method_first,
    restrict_methods_to_alts,
    extract_targets_and_effects,
    summarize_truth_effects,
)

from.cooccurence import (
    co_occurence, 
    jaccard,
)

from .snr import (
    snr_for_alt,
    SNR,
    effective_snr,
)

from .metrics import (
    evaluate_signature,
    compile_robustness_bundle,
)

from .dataset_eval import (
    evaluate_dataset_outputs,
    evaluate_dataset,
)

__all__ = [
    "safe_read_csv",
    "load_rna_gene_universe",
    "load_alterations",
    "load_truth_rich",
    "load_signature_joblib",
    "load_dataset_bundle",
    "read_dataset_for_evaluation",
    "save_robustness_outputs",
    "detect_signature_orientation",
    "to_method_first",
    "restrict_methods_to_alts",
    "extract_targets_and_effects",
    "summarize_truth_effects",
    "co_occurence",
    "snr_for_alt",
    "SNR",
    "jaccard",
    "effective_snr",
    "compile_robustness_bundle",
    "build_signature_cohesion_df",
    "evaluate_signature",
    "sanitize_binary_design",
    "evaluate_dataset_outputs",
    "evaluate_dataset",
]