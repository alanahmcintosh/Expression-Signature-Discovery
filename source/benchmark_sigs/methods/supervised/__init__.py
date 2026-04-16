
# methods/supervised/__init__.py

from .normalise import normalize_counts_log_cpm
from .feature_selection import select_features_by_stability

from .models import (
    fit_alt_to_expr_weights_lasso,
    fit_alt_to_expr_weights_elasticnet,
    fit_alt_to_expr_weights_ridge,
    fit_alt_to_expr_weights_svm,
    fit_alt_to_expr_weights_rf,
)

from .deseq2 import get_deseq2_signature_binary, precompute_deseq2_results

from .limma import get_limma_voom_signature_binary, precompute_limma_voom_results

from .wrappers import (
    precompute_supervised_weights,
    create_supervised_signatures,
)
from .multivariate import (
    precompute_limma_voom_results_multivariate, precompute_edger_results_multivariate
)

from .edgeR import (
    get_edger_signature_binary, precompute_edger_results
)


__all__ = [
    "normalize_counts_log_cpm",
    "select_features_by_stability",
    "fit_alt_to_expr_weights_lasso",
    "fit_alt_to_expr_weights_elasticnet",
    "fit_alt_to_expr_weights_ridge",
    "fit_alt_to_expr_weights_svm",
    "fit_alt_to_expr_weights_rf",
    "get_deseq2_signature_binary",
    "precompute_deseq2_results"
    "precompute_supervised_weights",
    "get_edger_signature_binary",
    "precompute_edger_results",
    "precompute_limma_voom_results_multivariate",
    "precompute_edger_results_multivariate",
    "create_supervised_signatures",
    "get_limma_voom_signature_binary",
    "precompute_limma_voom_results"
]
