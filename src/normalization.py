import pandas as pd

def normalize_within_subject(df, feature_cols, baseline_label="Baseline"):
    """
    Compute within-subject, within-band normalization of graph metrics.
    Each feature is expressed as a relative change from the subject's baseline.

    Δf = (f_level - f_baseline) / f_baseline

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['Subject', 'SedationLabel', 'Band', ...features...].
    feature_cols : list
        List of feature column names to normalize.
    baseline_label : str
        Label identifying the baseline condition (default: 'Baseline').

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame, same structure but with Δ-features.
    """
    df_norm = df.copy()
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
                    new_row[col] = np.nan
                else:
                    new_row[col] = (row[col] - base_val) / base_val
            norm_records.append(new_row)

    return pd.DataFrame(norm_records)
