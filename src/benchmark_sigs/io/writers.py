import os
import json
import pandas as pd


def save_simulation_outputs(
    rna_real: pd.DataFrame,
    expr_sim: pd.DataFrame,
    alterations_df: pd.DataFrame,
    true_signatures: dict,
    output_dir: str,
    suffix: str,
    compress: bool = False,
    binary_alt_df: pd.DataFrame | None = None,
    alt_real: pd.DataFrame | None = None,
    gistic_alt: pd.DataFrame | None = None,
):
    """
    Save real and simulated RNA data, alteration data, and true signatures.

    Parameters
    ----------
    rna_real : pd.DataFrame
        Real expression matrix.
    expr_sim : pd.DataFrame
        Simulated RNA-seq expression matrix.
    alterations_df : pd.DataFrame
        Binary mutation/fusion/CNA alteration matrix.
    true_signatures : dict
        Dictionary of true signatures used in the simulation.
    output_dir : str
        Directory path where files will be saved.
    suffix : str
        Identifier for the dataset (e.g., "IBC", "COAD").
    compress : bool
        Whether to save CSVs as `.csv.gz`.
    binary_alt_df : pd.DataFrame, optional
        Binary alteration matrix (if separate from alterations_df).
    alt_real : pd.DataFrame, optional
        Real alteration matrix used to generate the simulation.
    gistic_alt : pd.DataFrame, optional
        CNA matrix in GISTIC format.
    """

    os.makedirs(output_dir, exist_ok=True)
    ext = ".csv.gz" if compress else ".csv"

    # Save RNA
    rna_real.to_csv(os.path.join(output_dir, f"rna_real_{suffix}{ext}"))
    expr_sim.to_csv(os.path.join(output_dir, f"rna_simulated_{suffix}{ext}"))

    # Save alteration matrices
    alterations_df.to_csv(os.path.join(output_dir, f"alterations_{suffix}{ext}"))

    if binary_alt_df is not None:
        binary_alt_df.to_csv(os.path.join(output_dir, f"binary_alt_{suffix}{ext}"))

    if alt_real is not None:
        alt_real.to_csv(os.path.join(output_dir, f"alt_real_{suffix}{ext}"))

    if gistic_alt is not None:
        gistic_alt.to_csv(os.path.join(output_dir, f"alt_gistic_{suffix}{ext}"))

    # Save true signatures
    sig_path = os.path.join(output_dir, f"true_signatures_{suffix}.json")
    with open(sig_path, "w") as f:
        json.dump(true_signatures, f, indent=2)

    print(f"[✓] Simulation outputs saved for '{suffix}' in: {output_dir}")
