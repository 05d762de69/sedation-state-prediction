import pandas as pd
import numpy as np

def normalize_within_subject(df, baseline_label="Baseline", suffix="_delta"):
    """
    Compute within-subject, within-band normalization of graph metrics.
    Each feature is expressed as a relative change from the subject's baseline.

    Δf = (f_level - f_baseline) / f_baseline

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['Subject', 'SedationLabel', 'Band', ...features...].
    baseline_label : str, default='Baseline'
        Label identifying the baseline condition.
    suffix : str, default='_delta'
        Suffix appended to normalized feature columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features and original identifiers.
    """
    df_norm = df.copy()
    
    # Automatically infer feature columns
    exclude_cols = ["Subject", "SedationLabel", "SedationLevel", "Band"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    norm_records = []

    for (subj, band), group in df.groupby(["Subject", "Band"]):
        base_row = group[group["SedationLabel"] == baseline_label]
        if base_row.empty:
            continue

        baseline_vals = base_row.iloc[0][feature_cols]

        for _, row in group.iterrows():
            new_row = row.copy()
            for col in feature_cols:
                base_val = baseline_vals[col]
                if base_val == 0 or pd.isna(base_val):
                    new_row[col + suffix] = np.nan
                else:
                    new_row[col + suffix] = (row[col] - base_val) / base_val
            norm_records.append(new_row)

    return pd.DataFrame(norm_records)
