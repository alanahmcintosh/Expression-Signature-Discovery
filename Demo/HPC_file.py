import os
import pandas as pd
import joblib
import tarfile
from Benchmarking_functions import *


# ======== CONFIGURATION ========


X_PATH = "/home/alanah/Documents/alterations_OV.csv"
Y_PATH = "/home/alanah/Documents/rna_simulated_OV.csv"
OUTDIR = "test"

os.makedirs(OUTDIR, exist_ok=True)


# ===============================
# 1. Load Data
# ===============================
X = pd.read_csv(X_PATH, index_col=0)
Y = pd.read_csv(Y_PATH, index_col=0)


# Ensure gene names are str (important for indexing)
X.columns = X.columns.astype(str)
Y.columns = Y.columns.astype(str)

X_subset1 = X.iloc[0:50, 0:10]
X_subset2 = X.iloc[0:50, -10:]
X_subset = pd.concat((X_subset1, X_subset2), axis=1)
X_subset
Y_subset = Y.iloc[0:50, 0:500]

# # ===============================
# # 2. Run Deconfounder
# # ===============================
# global_results = precompute_global_results(X, Y)
# joblib.dump(global_results, os.path.join(OUTDIR, "global_results.pkl"))


# ===============================
# 3. Run Supervised Signature Discovery
# ===============================
supervised_sigs = {}
for gof in X_subset.columns:
    try:
        supervised_sigs[gof] = create_supervised_signatures(X_subset, Y_subset, gof, global_results=None)
        print(f"Finished supervised for {gof}")
    except Exception as e:
        print(f"Failed supervised for {gof}: {e}")


joblib.dump(supervised_sigs, os.path.join(OUTDIR, "supervised_signatures.joblib"))
