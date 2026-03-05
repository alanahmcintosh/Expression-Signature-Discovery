# src/Benchmark_Sigs/simulate/rna/__init__.py

from .background_rna_knn import simulate_background_from_alterations_knn
from .deseq_params import estimate_deseq2_parameters, draw_size_factors_from_deseq
from .nb_sampling import sample_nb, sample_nb_for_signature_genes
from .simulate_rna import simulate_rna_with_signatures

from .signature_generation import (
    build_alt_params_from_deseq2_summary,
    generate_signatures_from_deseq2_params,
    induce_expression_effects,
)

__all__ = [
    "simulate_background_from_alterations_knn",
    "estimate_deseq2_parameters",
    "draw_size_factors_from_deseq",
    "sample_nb",
    "sample_nb_for_signature_genes",
    "simulate_rna_with_signatures",
    "build_alt_params_from_deseq2_summary",
    "generate_signatures_from_deseq2_params",
    "induce_expression_effects",
]
