
# Baseline forecasting models for time series prediction.

import numpy as np
import pandas as pd


# A naive last value forecast is a forecast that repeats the last observed value for all forecast steps. Useful for testing and as a baseline.
# horizon: Number of steps to forecast ahead
def naive_last_value_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    if len(series) == 0:
        raise ValueError("Series cannot be empty for forecasting")
    
    last_value = series.iloc[-1]
    return np.full(horizon, last_value)

# A moving average forecast is a forecast that uses the mean of the last window values for all forecast steps. Useful for smoothing out noise and as a baseline.
# window: Number of recent values to average
def moving_average_forecast(
    series: pd.Series, window: int, horizon: int
) -> np.ndarray:
    if len(series) == 0:
        raise ValueError("Series cannot be empty for forecasting")
    
    # Use shorter window if series is too short
    actual_window = min(window, len(series))
    
    # Calculate mean of last actual_window values
    mean_value = series.iloc[-actual_window:].mean()
    
    # Return the same mean value for all forecast steps
    return np.full(horizon, mean_value)

