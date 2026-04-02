import pandas as pd
import numpy as np


def gistic_to_amp_del_binary(cna_df):
    """
    Convert GISTIC CNA calls (-2,-1,0,1,2) into binary AMP and DEL matrices.

    Returns
    -------
    pd.DataFrame
        Columns:
            gene_AMP
            gene_DEL
    """

    if cna_df is None or cna_df.empty:
        return pd.DataFrame(index=cna_df.index)

    cna = cna_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    cna = np.rint(cna).astype(int).clip(-2, 2)

    amp = (cna > 0).astype(int)
    amp.columns = [f"{g}_AMP" for g in amp.columns]

    dele = (cna < 0).astype(int)
    dele.columns = [f"{g}_DEL" for g in dele.columns]

    return pd.concat([amp, dele], axis=1)
