import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import seaborn as sns

# OUTLIER HANDLING
def outlier_report(df, numeric_cols):
    report = []

    for col in numeric_cols:
        s = df[col].dropna()
        median = s.median()
        MAD = (s - median).abs().median()

        # 0.6745 scales MAD to be consistent with standard deviation under normality
        modified_z = 0.6745 * (s - median) / MAD

        threshold = 3.5  # standard cutoff recommended by Iglewicz & Hoaglin
        mask = modified_z.abs() > threshold

        lower = median - (threshold / 0.6745) * MAD
        upper = median + (threshold / 0.6745) * MAD

        z = zscore(s)
        Z_mask = abs(z) > 3

        report.append({
            "feature": col,

            "n": len(s),

            "IQR_n_outliers": mask.sum(),
            "IQR_outlier_%": 100 * mask.mean(),
            "IQR_min": s.min(),
            "IQR_max": s.max(),
            "IQR_lower_bound": lower,
            "IQR_upper_bound": upper,

            "Z_n_outliers": Z_mask.sum(),
            "Z_outlier_%": 100 * Z_mask.mean(),
            "Z_min": s.min(),
            "Z_max": s.max(),
            "Z_lower_bound": -3,
            "Z_upper_bound": 3
        })

    outlier_report = pd.DataFrame(report)

    # IRQ is preferred as it doesnt assume a specific data distribution
    outlier_report = outlier_report.sort_values(
        "IQR_outlier_%",
        ascending=False
    )

    return outlier_report


def outlier_sc(features, df, numeric_cols):
    outlier_scores = pd.DataFrame(
        0.0,
        index=features.index,
        columns=features.columns
    )
    for col in numeric_cols:
        s = df[col].dropna()
        median = s.median()
        MAD = (s - median).abs().median()

        modified_z = 0.6745 * (s - median) / MAD

        threshold = 3.5  # standard cutoff recommended by Iglewicz & Hoaglin
        mask = modified_z.abs() > threshold

        lower = median - (threshold / 0.6745) * MAD
        upper = median + (threshold / 0.6745) * MAD

        lower_mask = features[col] < lower

        outlier_scores.loc[lower_mask, col] = (
                (lower - features.loc[lower_mask, col]) / IQR
        )

        upper_mask = features[col] > upper

        outlier_scores.loc[upper_mask, col] = (
                (features.loc[upper_mask, col] - upper) / IQR
        )

    sample_outlier_score = outlier_scores.sum(axis=1)
    ood_scores = sample_outlier_score.sort_values(ascending=False)
    ood_df = features.loc[sample_outlier_score.nlargest(197).index]
    id_df = features.loc[~features.index.isin(ood_df.index)]
    return id_df, ood_df




# MISSING VALUES HANDLING
def missing_data(features, df):

    start_time = {}
    for i in features:
        valid = df[i].notna()
        if valid.any():
            start_time[i] = df.loc[valid, "DATE"].iloc[0]
        else:
            start_time[i] = pd.NaT
    start_times = pd.Series(start_time, name="start")
    return start_times

def plot(features, df, availability):

    for feature in features:
        plt.figure(figsize=(14, 4))

        # availability = df[feature].notna().astype(int)

        plt.plot(df["DATE"], df[feature])

        plt.title(f"Recording feature: {feature}")
        plt.xlabel("Timestamp")
        plt.ylabel(f"{feature}")

        plt.tight_layout()

        plt.savefig(f"plot{feature}.png")