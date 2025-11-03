import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

def compute_dwpli(epochs, fmin, fmax):
    """
    Compute debiased weighted Phase Lag Index (dwPLI) for a given frequency band. 
    The result is averaged across epochs (as per MNE's implementation).
    """
    conn = spectral_connectivity_epochs(
        data=epochs,                     
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