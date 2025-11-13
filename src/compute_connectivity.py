#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_compute_connectivity.py
Compute wPLI connectivity matrices for each subject × frequency band
and cache them as .npy files for later metric computation.

Usage:
    python 01_compute_connectivity.py /path/to/folder_with_set_files
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os
from connectivity import compute_wpli_per_epoch

# ---------------- CLI ARGUMENTS ----------------
if len(sys.argv) < 2:
    print("Usage: python 01_compute_connectivity.py /path/to/folder_with_set_files")
    sys.exit(1)

DATA_DIR = Path(sys.argv[1]).resolve()
if not DATA_DIR.exists():
    raise FileNotFoundError(f"EEG data folder not found: {DATA_DIR}")

print(f"📂 Using EEG data from: {DATA_DIR}")

# ---------------- CONFIGURATION ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data/data_derivatives/manifests/manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/data_derivatives/connectivity_matrices"

FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD MANIFEST ----------------
manifest = pd.read_csv(MANIFEST_PATH)
print(f"Loaded manifest with {len(manifest)} entries.")

# ------------------------------------------------
for idx in tqdm(range(len(manifest)), desc="Subjects"):
    row = manifest.iloc[idx]
    label = row["SedationLabel"]
    if label == "Recovery":
        continue

    # Resolve full path to .set file relative to provided DATA_DIR
    set_path = DATA_DIR / Path(row["BaseName"]).with_suffix(".set")
    if not set_path.exists():
        print(f"⚠️ Missing file: {set_path}")
        continue

    try:
        epochs = mne.io.read_epochs_eeglab(set_path, verbose="error")
    except Exception as e:
        print(f"❌ Could not load {set_path.name}: {e}")
        continue

    # --- Compute wPLI for each frequency band ---
    for band, (fmin, fmax) in FREQ_BANDS.items():
        save_path = OUTPUT_DIR / f"{row['Subject']}_{label}_{band}_wpli.npy"

        if save_path.exists():
            continue  # Skip existing computations

        try:
            con_matrices = compute_wpli_per_epoch(epochs, fmin, fmax)
            np.save(save_path, con_matrices)
        except Exception as e:
            print(f"Error in {set_path.name} ({band}): {type(e).__name__}: {e}")
            continue

print("\n✅ Connectivity matrices computed and saved.")
