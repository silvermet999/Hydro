import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(df):
    for i in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(df[i], ax=axes[0])
        plot_pacf(df[i], ax=axes[1])
        plt.title(f"Recording ACF/PACF for: {i}")
        plt.savefig(f"acf_pacf{i}.png")
