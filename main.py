import pandas as pd



df = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Niveaux_journaliers")
df_metadata = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Metadonnees_puits")
df_diag = pd.read_excel("001_RSESQ_niveaux_eau_2026-08-04.xlsx", sheet_name="Couverture_series")

features = df.drop("DATE", axis=1)
availability = features.notna().astype(int)
numeric_cols = df.select_dtypes(include="number").columns



