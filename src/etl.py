import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from datetime import datetime, timedelta
import pytz
from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)

class WeatherProvider:
    """
    Handles fetching weather data from Open-Meteo API and generating
    synthetic historical data for model training.
    """

    def __init__(self):
        # Setup the Open-Meteo client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.url = "https://api.open-meteo.com/v1/forecast"

    def get_forecast(self, lat: float, lon: float) -> pd.DataFrame:
        """
        Fetches 3-day hourly forecast for specified coordinates.
        
        Args:
            lat: Latitude.
            lon: Longitude.
            
        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'temperature_2m', 
                          'shortwave_radiation', 'precipitation', 'wind_speed_10m']
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "shortwave_radiation", "precipitation", "wind_speed_10m"],
            "timezone": "Europe/Madrid",
            "forecast_days": 3
        }
        
        try:
            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]
            
            # Process hourly data
            hourly = response.Hourly()
            hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
            hourly_shortwave_radiation = hourly.Variables(1).ValuesAsNumpy()
            hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
            hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
            
            hourly_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert('Europe/Madrid'),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert('Europe/Madrid'),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            hourly_data["temperature_2m"] = hourly_temperature_2m
            hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
            hourly_data["precipitation"] = hourly_precipitation
            hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
            
            df = pd.DataFrame(data=hourly_data)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            raise e

    def get_historical_training_data(self) -> pd.DataFrame:
        """
        Generates a synthetic dataset representing 3 years of hourly data
        for training the solar generation model.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['date', 'temperature_2m', 
                          'shortwave_radiation', 'power_mw']
        """
        logger.info("Generating synthetic historical data...")
        
        # Date range: 2020-01-01 to 2023-01-01
        dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="H", tz="Europe/Madrid")
        n = len(dates)
        
        # Synthetic features
        # 1. Solar Radiation: Based on hour of day and season (simplified)
        # Day of year for seasonality
        doy = dates.dayofyear
        hour = dates.hour
        
        # Max radiation varies by season (higher in summer)
        seasonal_factor = 0.5 + 0.5 * np.sin((doy - 80) / 365.0 * 2 * np.pi)
        
        # Daily curve (gaussian-ish) centered at 14:00
        daily_curve = np.maximum(0, np.sin((hour - 6) / 14.0 * np.pi))
        daily_curve[hour < 6] = 0
        daily_curve[hour > 20] = 0
        
        # Add some noise/clouds
        cloud_cover = np.random.beta(2, 5, n) # Skewed towards clear days
        radiation = 1000 * seasonal_factor * daily_curve * (1 - cloud_cover * 0.8)
        
        # 2. Temperature: Correlated with radiation + seasonal lag
        temp_seasonal = 15 + 10 * np.sin((doy - 100) / 365.0 * 2 * np.pi)
        temp_daily = 5 * np.sin((hour - 10) / 24.0 * 2 * np.pi)
        temperature = temp_seasonal + temp_daily + np.random.normal(0, 2, n)
        
        # 3. Power Generation (Target): Linear with radiation, efficiency drops with heat
        # Efficiency loss: -0.4% per degree above 25°C
        efficiency_temp_coeff = -0.004
        base_efficiency = 0.20 # 20% efficiency
        panel_area_m2 = 5000 # Arbitrary large plant
        
        temp_correction = 1 + np.minimum(0, (25 - temperature) * efficiency_temp_coeff)
        power_mw = (radiation * panel_area_m2 * base_efficiency * temp_correction) / 1e6
        
        # Add some random noise to power
        power_mw = power_mw * np.random.normal(1.0, 0.05, n)
        power_mw = np.maximum(0, power_mw)
        
        df = pd.DataFrame({
            "date": dates,
            "temperature_2m": temperature,
            "shortwave_radiation": radiation,
            "power_mw": power_mw
        })
        
        return df
