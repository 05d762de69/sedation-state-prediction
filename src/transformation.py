import numpy as np
import pandas as pd

def compute_band_ratios(df, features=None, output_path=None):
    """
    Compute EEG graph metric ratios (e.g., theta/alpha, alpha/beta) for each subject and sedation level.
    Works with a MultiIndex (feature, band) structure as confirmed in df_wide.
    """
    if features is None:
        features = [
            'mean_degree', 'clustering', 'path_length',
            'global_efficiency', 'local_efficiency',
            'modularity', 'participation_coefficient',
            'small_worldness'
        ]

    ratio_pairs = [
        ('theta', 'alpha'),
        ('theta', 'beta'),
        ('alpha', 'beta'),
        ('delta', 'alpha')
    ]

    # Pivot the features × band structure
    df_wide = df.pivot_table(
        index=['Subject', 'SedationLevel'],
        columns='Band',
        values=features
    )

    # Flatten columns for easier access (convert MultiIndex to single strings)
    df_wide.columns = [f"{f}_{b}" for f, b in df_wide.columns]

    ratio_records = []
    for subj_sed, band_vals in df_wide.iterrows():
        ratios = {'Subject': subj_sed[0], 'SedationLevel': subj_sed[1]}
        for feat in features:
            for num, den in ratio_pairs:
                num_col = f"{feat}_{num}"
                den_col = f"{feat}_{den}"
                if num_col in band_vals and den_col in band_vals:
                    num_val = band_vals[num_col]
                    den_val = band_vals[den_col]
                    if pd.notna(num_val) and pd.notna(den_val) and den_val != 0:
                        ratios[f"{feat}_{num}_{den}_ratio"] = num_val / den_val
                    else:
                        ratios[f"{feat}_{num}_{den}_ratio"] = np.nan
                else:
                    ratios[f"{feat}_{num}_{den}_ratio"] = np.nan
        ratio_records.append(ratios)

    df_ratios = pd.DataFrame(ratio_records)
    if output_path:
        df_ratios.to_csv(output_path, index=False)
    return df_ratios
