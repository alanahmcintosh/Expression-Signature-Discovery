from __future__ import annotations

import pandas as pd

def nonempty(df: object) -> bool:
    """True if df is a non-empty DataFrame with at least 1 row and 1 column."""
    return isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0
