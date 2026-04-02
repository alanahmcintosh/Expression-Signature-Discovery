from __future__ import annotations

import pandas as pd
import numpy as np

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