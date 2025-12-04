import xgboost as xgb
import pandas as pd
import numpy as np
import os
import streamlit as st
from typing import Optional
from src.etl import WeatherProvider
from src.utils import setup_logger

logger = setup_logger(__name__)

class SolarPredictor:
    """
    Handles training and prediction for solar power generation.
    """
    
    def __init__(self, model_path: str = "model.json"):
        self.model_path = model_path
        self.model = None

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds time-based features for the model.
        """
        df = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        df['hour'] = df['date'].dt.hour
        df['month'] = df['date'].dt.month
        
        # Cyclic features for hour
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df

    def train_and_save(self, lat: float, lon: float):
        """
        Trains the XGBoost model on historical data for the specific location.
        """
        logger.info(f"Starting model training for {lat}, {lon}...")
        provider = WeatherProvider()
        # Fetch real historical data for this location
        df = provider.get_historical_training_data(lat, lon)
        
        df = self._feature_engineering(df)
        
        features = ['shortwave_radiation', 'temperature_2m', 'sin_hour', 'cos_hour', 'month']
        target = 'power_mw'
        
        X = df[features]
        y = df[target]
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        
        model.fit(X, y)
        
        model.save_model(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        self.model = model

    def load_model(self, lat: float = None, lon: float = None):
        """
        Loads the model from disk. If lat/lon provided, ensures model is trained for it.
        For this demo, we'll retrain if lat/lon is provided to ensure accuracy.
        """
        if lat is not None and lon is not None:
            # Always retrain for new location to ensure "location specific" accuracy
            # In production, we would check if a model for this location exists
            self.train_and_save(lat, lon)
        elif os.path.exists(self.model_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
        else:
            logger.warning("Model file not found. Please provide lat/lon to train.")

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts solar generation for the forecast dataframe.
        
        Args:
            forecast_df: DataFrame from WeatherProvider.get_forecast
            
        Returns:
            pd.DataFrame: Original dataframe with 'pred_solar_mw' column.
        """
        if self.model is None:
            self.load_model()
            
        df_features = self._feature_engineering(forecast_df)
        features = ['shortwave_radiation', 'temperature_2m', 'sin_hour', 'cos_hour', 'month']
        
        X = df_features[features]
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)
        
        result_df = forecast_df.copy()
        result_df['pred_solar_mw'] = predictions
        
        return result_df

@st.cache_resource
def get_base_predictor() -> SolarPredictor:
    """
    Singleton provider for the SolarPredictor, cached by Streamlit.
    """
    predictor = SolarPredictor()
    # Ensure model is loaded/trained on startup
    predictor.load_model()
    return predictor
