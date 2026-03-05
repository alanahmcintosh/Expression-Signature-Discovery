# src/Benchmark_Sigs/simulate/__init__.py
from .alterations.simulate_alterations import (
    simulate_X,
    split_simulated_blocks,
)

from .rna.simulate_rna import simulate_rna_with_signatures

__all__ = [
    "simulate_X",
    "split_simulated_blocks",
    "simulate_rna_with_signatures",
]
