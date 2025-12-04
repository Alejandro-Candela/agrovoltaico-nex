import pandas as pd
import numpy as np

def calculate_frost_risk(df: pd.DataFrame) -> str:
    """
    Analyzes the minimum temperature in the forecast dataframe to determine frost risk.
    
    Args:
        df: DataFrame containing 'temperature_2m' column.
        
    Returns:
        str: Risk level ('CRÍTICO', 'ALTO', 'MEDIO', 'BAJO').
    """
    if 'temperature_2m' not in df.columns:
        raise ValueError("DataFrame must contain 'temperature_2m' column")
        
    min_temp = df['temperature_2m'].min()
    
    if min_temp < -1:
        return 'CRÍTICO'
    elif min_temp < 0:
        return 'ALTO'
    elif min_temp < 2:
        return 'MEDIO'
    else:
        return 'BAJO'

def calculate_gdd(df: pd.DataFrame, base_temp: float = 10.0) -> float:
    """
    Calculates Growing Degree Days (GDD) using the simple average method (Winkler simplified).
    
    Args:
        df: DataFrame containing 'temperature_2m' and 'date'.
        base_temp: Base temperature for the crop (default 10 for vines).
        
    Returns:
        float: Accumulated GDD over the period.
    """
    if 'temperature_2m' not in df.columns or 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'temperature_2m' and 'date' columns")
    
    # Resample to daily frequency to calculate daily mean
    # Ensure date is index for resampling
    temp_df = df.set_index('date')
    daily_temps = temp_df['temperature_2m'].resample('D').mean()
    
    # Calculate GDD for each day: max(0, T_avg - T_base)
    gdd = (daily_temps - base_temp).clip(lower=0)
    
    return float(gdd.sum())
