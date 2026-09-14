import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_models(df, features, period):
    results = {}
    future_dates = pd.date_range(start=df["DATE"].iloc[-1], periods=period + 1, freq='D')[1:]

    for i in features.columns:
        sarima_model = SARIMAX(features[i], order=(5,1,0), seasonal_order=(1,1,1,12)).fit()
        sarima_forecast = sarima_model.forecast(steps=period)

        plt.figure(figsize=(10, 5))
        plt.plot(df["DATE"], features[i], label="Actual Data", color="blue")
        plt.plot(future_dates, sarima_forecast, label="SARIMA Forecast", color="green")
        plt.xlabel("Date")
        plt.ylabel(f"{i}")
        plt.title("SARIMA")
        plt.savefig(f"SARIMA_{i}")

    return results
