from __future__ import annotations

import pandas as pd
import numpy as np

def align_XY(X, Y):
    common = X.index.intersection(Y.index)
    X2 = X.loc[common]
    Y2 = Y.loc[common]
    return X2, Y2