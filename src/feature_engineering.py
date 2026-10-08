"""
Feature Engineering Engine for HealthTwin AI.
Computes multi-scale time-series features, rolling windows, and physiological indices.
"""

import pandas as pd
import numpy as np

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes temporal rolling-window metrics (mean, std, min, max, slope)
    for continuous glucose, heart rate, HRV, and activity level.
    Assumes 15-minute sampling frequency (1 step = 15m).
    """
    df = df.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
    
    # Define rolling window step lengths
    # 15m = 1 step, 30m = 2 steps, 1h = 4 steps, 2h = 8 steps, 6h = 24 steps, 24h = 96 steps
    windows = {
        '30m': 2,
        '1h': 4,
        '2h': 8,
        '6h': 24,
        '24h': 96
    }
    
    engineered = []
    
    for pid, group in df.groupby('patient_id'):
        group = group.copy()
        
        # 1. Continuous Glucose Statistics across windows
        for w_name, w_steps in windows.items():
            group[f'glucose_mean_{w_name}'] = group['continuous_glucose'].rolling(window=w_steps, min_periods=1).mean()
            group[f'glucose_std_{w_name}'] = group['continuous_glucose'].rolling(window=w_steps, min_periods=1).std().fillna(0.0)
            group[f'glucose_max_{w_name}'] = group['continuous_glucose'].rolling(window=w_steps, min_periods=1).max()
            group[f'glucose_min_{w_name}'] = group['continuous_glucose'].rolling(window=w_steps, min_periods=1).min()
            
            group[f'hr_mean_{w_name}'] = group['heart_rate'].rolling(window=w_steps, min_periods=1).mean()
            group[f'hrv_mean_{w_name}'] = group['hrv'].rolling(window=w_steps, min_periods=1).mean()
            group[f'stress_mean_{w_name}'] = group['stress_level'].rolling(window=w_steps, min_periods=1).mean()
            
        # 2. Slopes & Rate of Change (Velocity)
        # 15m change (1 step)
        group['recent_glucose_change'] = group['continuous_glucose'].diff(1).fillna(0.0)
        group['glucose_slope'] = group['continuous_glucose'].diff(4).fillna(0.0) / 4.0  # mg/dL per 15 min (1hr window)
        group['heart_rate_slope'] = group['heart_rate'].diff(4).fillna(0.0) / 4.0
        group['hrv_slope'] = group['hrv'].diff(4).fillna(0.0) / 4.0
        
        # 3. Glucose Variability Index (GVI) & CV
        group['glucose_mean'] = group['glucose_mean_2h']
        group['glucose_std'] = group['glucose_std_2h']
        group['glucose_variability'] = np.where(group['glucose_mean'] > 0, group['glucose_std'] / group['glucose_mean'], 0.0)
        
        # 4. Aggregated Daily Metrics
        group['avg_heart_rate'] = group['hr_mean_24h']
        group['resting_heart_rate'] = group['heart_rate'].rolling(window=96, min_periods=1).min()
        group['avg_hrv'] = group['hrv_mean_24h']
        group['steps_24h'] = group['steps'].rolling(window=96, min_periods=1).sum()
        group['sleep_24h'] = group['sleep_hours']
        
        # 5. Composite Risk Scores & Behavioral Ratios
        group['sleep_debt'] = np.maximum(0.0, 8.0 - group['sleep_hours'])
        group['activity_score'] = (group['steps_24h'] / 10000.0).clip(0.0, 2.0)
        group['stress_score'] = (group['stress_mean_2h'] / 100.0).clip(0.0, 1.0)
        group['activity_change'] = group['steps'].diff(4).fillna(0.0)
        
        engineered.append(group)
        
    df_engineered = pd.concat(engineered, ignore_index=True)
    return df_engineered

def get_feature_columns():
    """
    Returns the complete list of predictive features used by Machine Learning models.
    """
    return [
        'age', 'bmi', 'fasting_glucose', 'hba1c', 'diabetes_history', 'hypertension',
        'continuous_glucose', 'recent_glucose_change', 'glucose_slope',
        'glucose_mean_1h', 'glucose_std_1h', 'glucose_mean_2h', 'glucose_std_2h',
        'glucose_mean_6h', 'glucose_variability',
        'heart_rate', 'avg_heart_rate', 'resting_heart_rate', 'heart_rate_slope',
        'hrv', 'avg_hrv', 'hrv_slope',
        'steps_24h', 'sleep_24h', 'sleep_quality', 'sleep_debt',
        'activity_score', 'stress_level', 'stress_score', 'activity_change'
    ]
