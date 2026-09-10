import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation


df = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Niveaux_journaliers")
df_metadata = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Metadonnees_puits")
df_diag = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Couverture_series")

features = df.drop("DATE", axis=1)
availability = features.notna().astype(int)

# OUTLIER HANDLING
def IQR():
    numeric_cols = df.select_dtypes(include="number").columns

    outlier_counts = {}

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()

    outlier_counts = pd.Series(outlier_counts).sort_values(ascending=False)

    return outlier_counts





# MISSING VALUES HANDLING
def missing_data():

    start_time = {}
    for i in features:
        valid = df[i].notna()
        if valid.any():
            start_time[i] = df.loc[valid, "DATE"].iloc[0]
        else:
            start_time[i] = pd.NaT
    start_times = pd.Series(start_time, name="start")
    return start_times

def plot():

    for feature in features:
        plt.figure(figsize=(14, 4))

        # availability = df[feature].notna().astype(int)

        plt.plot(df["DATE"], df[feature])

        plt.title(f"Recording feature: {feature}")
        plt.xlabel("Timestamp")
        plt.ylabel(f"{feature}")

        plt.tight_layout()

        plt.savefig(f"plot{feature}.png")



# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(features)
# df_scaled = pd.DataFrame(df_scaled, columns=features.columns)
# df_all = pd.concat([df["DATE"], df_scaled], axis=1)
