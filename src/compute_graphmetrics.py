#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_compute_graph_metrics.py
Compute graph-theoretical metrics from cached wPLI matrices.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from graph_metrics import compute_graph_metrics_epochs 

#  CONFIGURATION 
PROJECT_ROOT = Path(__file__).resolve().parent
CONNECTIVITY_DIR = PROJECT_ROOT / "data/data_derivatives/connectivity_matrices"
MANIFEST_PATH = PROJECT_ROOT / "data/data_derivatives/manifests/manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/data_derivatives/features_per_epoch.csv"

SEDATION_MAP = {"Baseline": 1, "Mild": 2, "Moderate": 3}

manifest = pd.read_csv(MANIFEST_PATH)
manifest["SetPath"] = manifest["SetPath"].apply(lambda p: (PROJECT_ROOT / p).resolve())

records = []

for idx in tqdm(range(len(manifest)), desc="Subjects"):
    row = manifest.iloc[idx]
    label = row["SedationLabel"]

    if label == "Recovery":
        continue

    for npy_file in sorted(CONNECTIVITY_DIR.glob(f"{row['Subject']}_{label}_*_wpli.npy")):
        band = npy_file.stem.split("_")[-2]

        try:
            con_matrices = np.load(npy_file, allow_pickle=True)
        except Exception as e:
            print(f"Could not load {npy_file.name}: {e}")
            continue

        for ep_idx, con_matrix in enumerate(con_matrices):
            if np.isnan(con_matrix).all():
                continue
            metrics = compute_graph_metrics_epochs(con_matrix, n_rand=3)

            record = {
                "Subject": row["Subject"],
                "Epoch": ep_idx,
                "SedationLabel": label,
                "SedationLevel": SEDATION_MAP.get(label, None),
                "Band": band,
                **metrics,
                "Propofol_ugL": row["Propofol_ugL"],
                "RT_ms": row["RT_ms"],
                "Correct": row["Correct"],
            }
            records.append(record)

#  Save results
df_metrics = pd.DataFrame(records)
df_metrics.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Graph metrics saved to: {OUTPUT_PATH}")
