import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
import os


def compute_dwpli(epochs, fmin, fmax):
    """
    Compute debiased weighted Phase Lag Index (dwPLI) for a given frequency band. 
    The result is averaged across epochs (as per MNE's implementation).
    """
    conn = spectral_connectivity_epochs(
        data=epochs.get_data(),                   
        method="wpli2_debiased",         # dwPLI
        mode="multitaper",               
        fmin=fmin,
        fmax=fmax,
        faverage=True,                   # average across frequencies in band
        sfreq=epochs.info["sfreq"],
        verbose="error"
    )

    # Extract dense matrix (shape: n_nodes x n_nodes)
    con_matrix = conn.get_data(output="dense")
    con_matrix = np.squeeze(con_matrix)  
    np.fill_diagonal(con_matrix, 0)      
    return con_matrix

def compute_dwpli_per_epoch(epochs, fmin, fmax, n_cycles=7):
    """
    Compute dwPLI (via wPLI) for each epoch separately using spectral_connectivity_time.
    Returns array of shape (n_epochs, n_channels, n_channels).
    """

    # --- ensure stable environment for parallelism ---
    mne.set_config("MNE_MEMMAP_MIN_SIZE", "10M", set_env=True)
    mne.set_config("MNE_CACHE_DIR", os.path.expanduser("~/Library/Caches/mne_cache"), set_env=True)
    os.environ["OMP_NUM_THREADS"] = "1"

    # --- define frequency range ---
    freqs = np.linspace(fmin, fmax, num=5)
    sfreq = epochs.info["sfreq"]

    # --- compute per-epoch connectivity (vectorized, no averaging) ---
    con = spectral_connectivity_time(
        data=epochs.get_data(),             # (n_epochs, n_channels, n_times)
        freqs=freqs,
        method="wpli",                      # wPLI (dwPLI not supported in this API)
        mode="multitaper",
        average=False,                      # keep epoch-wise estimates
        fmin=fmin,
        fmax=fmax,
        faverage=True,                      # average across freqs within band
        sfreq=sfreq,
        n_cycles=n_cycles,
        n_jobs=-1,                          # use all available cores
        verbose="error",
    )

    # --- extract dense matrices and clean up ---
    con_data = con.get_data(output="dense")   # (n_epochs, n_channels, n_channels, n_freqs)
    con_data = np.nanmean(con_data, axis=-1)  # average over small freq grid (optional)

    # --- zero diagonals (self-connections) ---
    for i in range(con_data.shape[0]):
        np.fill_diagonal(con_data[i], 0)

    # --- replace NaNs with zeros to keep graph metrics stable ---
    con_data = np.nan_to_num(con_data, nan=0.0, posinf=0.0, neginf=0.0)

    return con_data