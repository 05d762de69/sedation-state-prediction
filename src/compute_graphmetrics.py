#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_graph_metrics.py
Compute graph-theoretical metrics from cached wPLI matrices.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from graph_metrics import compute_graph_metrics_epochs

#  CONFIGURATION 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONNECTIVITY_DIR = PROJECT_ROOT / "data/data_derivatives/connectivity_matrices"
MANIFEST_PATH = PROJECT_ROOT / "data/data_derivatives/manifests/manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/data_derivatives/features_per_epoch.csv"

SEDATION_MAP = {"Baseline": 1, "Mild": 2, "Moderate": 3}

#  LOAD MANIFEST 
manifest = pd.read_csv(MANIFEST_PATH)
manifest["SetPath"] = manifest["SetPath"].apply(lambda p: (PROJECT_ROOT / p).resolve())

records = []



# Helper

def compute_one_epoch(ep_idx, con_matrix, row, band):
    """Compute one epoch worth of graph metrics."""
    if np.isnan(con_matrix).all():
        return None

    metrics = compute_graph_metrics_epochs(con_matrix, n_rand=3)

    return {
        "Subject": row["Subject"],
        "Epoch": ep_idx,
        "SedationLabel": row["SedationLabel"],
        "SedationLevel": SEDATION_MAP.get(row["SedationLabel"], None),
        "Band": band,
        **metrics,
        "Propofol_ugL": row["Propofol_ugL"],
        "RT_ms": row["RT_ms"],
        "Correct": row["Correct"],
    }

for idx in range(len(manifest)):
    row = manifest.iloc[idx]
    label = row["SedationLabel"]

    # Skip states without connectivity matrices
    if label == "Recovery":
        continue

    # find all connectivity files for this subject & sedation level
    npy_files = sorted(CONNECTIVITY_DIR.glob(f"{row['Subject']}_{label}_*_wpli.npy"))

    for npy_file in npy_files:

        band = npy_file.stem.split("_")[-2]

        try:
            con_matrices = np.load(npy_file, allow_pickle=True)
        except Exception as e:
            print(f"Could not load {npy_file.name}: {e}")
            continue

        #  PARALLEL EPOCH PROCESSING 
        results = Parallel(n_jobs=16, backend="loky")(
            delayed(compute_one_epoch)(ep_idx, con_matrix, row, band)
            for ep_idx, con_matrix in enumerate(con_matrices)
        )

        # filter out empty results
        records.extend([r for r in results if r is not None])


# SAVE RESULTS

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)
