import numpy as np
import pandas as pd

def compute_band_ratios(df, features=None, output_path=None):

    if features is None:
        features = [
            'mean_strength', 'clustering', 
            'path_length', 'global_efficiency', 'local_efficiency', 
            'modularity', 'participation_coefficient', 'small_worldness'
        ]

    ratio_pairs = [
        ('theta', 'alpha'),
        ('theta', 'beta'),
        ('alpha', 'beta'),
        ('delta', 'alpha')
    ]

    # average across epochs but keep bands distinct 
    df_agg = (
        df.groupby(['Subject', 'SedationLevel', 'Band'])[features]
        .mean()
        .reset_index()
    )

    # --- Step 2: pivot to wide format (features × bands) ---
    df_wide = df_agg.pivot_table(
        index=['Subject', 'SedationLevel'],
        columns='Band',
        values=features
    )

    # Flatten MultiIndex columns
    df_wide.columns = [f"{f}_{b}" for f, b in df_wide.columns]
    df_wide = df_wide.reset_index()

    # --- Step 3: compute ratios ---
    ratio_records = []
    for _, row in df_wide.iterrows():
        subj, sed = row['Subject'], row['SedationLevel']
        ratios = {'Subject': subj, 'SedationLevel': sed}

        for feat in features:
            for num, den in ratio_pairs:
                num_col = f"{feat}_{num}"
                den_col = f"{feat}_{den}"
                num_val = row.get(num_col, np.nan)
                den_val = row.get(den_col, np.nan)
                if pd.notna(num_val) and pd.notna(den_val) and den_val != 0:
                    ratios[f"{feat}_{num}_{den}_ratio"] = num_val / den_val
                else:
                    ratios[f"{feat}_{num}_{den}_ratio"] = np.nan
        ratio_records.append(ratios)

    # --- Step 4: finalize ---
    df_ratios = pd.DataFrame(ratio_records)
    if output_path:
        df_ratios.to_csv(output_path, index=False)
    return df_ratios
