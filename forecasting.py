import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_forecast(df, country, item, metric, forecast_periods=5):
    """
    Generate a time series forecast for the specified country, item, and metric.
    Uses Holt-Winters if sufficient data; falls back to linear regression if not.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        country (str): Selected country
        item (str): Selected item
        metric (str): Selected metric (Element)
        forecast_periods (int): Number of periods to forecast
    
    Returns:
        pd.DataFrame: Forecast DataFrame or None if generation fails
    """
    try:
        
        df_filtered = df[(df["Country"] == country) & 
                         (df["Item"] == item) & 
                         (df["Element"] == metric)][["Year", "Value"]].copy()

        if df_filtered.empty:
            logging.warning(f"No data found for Country: {country}, Item: {item}, Metric: {metric}")
            return None

        df_filtered = df_filtered.dropna()
        df_filtered = df_filtered.groupby("Year").mean().reset_index()  

        
        df_filtered["Year"] = df_filtered["Year"].astype(int)
        df_filtered.sort_values("Year", inplace=True)

        
        logging.info(f"Filtered data shape: {df_filtered.shape}")
        logging.info(f"Years available: {df_filtered['Year'].tolist()}")

        
        ts = df_filtered.set_index("Year")["Value"]

        if len(ts) < 2:     
            X = ts.index.values.reshape(-1, 1)
            y = ts.values
            model = LinearRegression().fit(X, y)
            forecast_years = np.array(range(ts.index.max() + 1, ts.index.max() + 1 + forecast_periods)).reshape(-1, 1)
            forecast_values = model.predict(forecast_years)

            forecast_df = pd.DataFrame({
                "Year": forecast_years.flatten(),
                "Forecast": forecast_values
            })
        else:
            
            try:
                model = ExponentialSmoothing(ts, trend="add", seasonal=None)
                fitted_model = model.fit()
                forecast_years = list(range(ts.index.max() + 1, ts.index.max() + 1 + forecast_periods))
                forecast_values = fitted_model.forecast(forecast_periods)

                forecast_df = pd.DataFrame({
                    "Year": forecast_years,
                    "Forecast": forecast_values
                })
            except Exception as e:
                logging.warning(f"Holt-Winters failed: {e}. Falling back to linear regression.")
                X = ts.index.values.reshape(-1, 1)
                y = ts.values
                model = LinearRegression().fit(X, y)
                forecast_years = np.array(range(ts.index.max() + 1, ts.index.max() + 1 + forecast_periods)).reshape(-1, 1)
                forecast_values = model.predict(forecast_years)

                forecast_df = pd.DataFrame({
                    "Year": forecast_years.flatten(),
                    "Forecast": forecast_values
                })

        logging.info(f"Forecast generated for {country}, {item}, {metric}")
        return forecast_df

    except Exception as e:
        logging.error(f"Error generating forecast: {e}")
        return None