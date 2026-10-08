"""
Glucose Time-Series Forecasting Module for HealthTwin AI.
Predicts continuous glucose concentration 2 hours (120 minutes) into the future.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import get_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def train_forecasting_model(df_events: pd.DataFrame):
    """
    Trains Ridge Regression forecaster for 2-hour continuous glucose concentration.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    feature_cols = get_feature_columns()
    df_clean = df_events.dropna(subset=['future_glucose_2h']).copy()
    
    X = df_clean[feature_cols].copy().fillna(df_clean[feature_cols].median())
    y = df_clean['future_glucose_2h'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    regressor = Ridge(alpha=1.0)
    regressor.fit(X_scaled, y)
    
    joblib.dump(regressor, os.path.join(MODELS_DIR, 'forecasting_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'forecasting_scaler.pkl'))
    
    print("Forecasting Regressor trained successfully.")
    return regressor, scaler

def load_forecasting_model():
    """
    Loads saved glucose forecasting regressor artifact.
    """
    model_path = os.path.join(MODELS_DIR, 'forecasting_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'forecasting_scaler.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        from src.data_loader import load_all_data
        from src.feature_engineering import compute_rolling_features
        
        _, _, df_events = load_all_data()
        df_eng = compute_rolling_features(df_events)
        return train_forecasting_model(df_eng)
        
    regressor = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return regressor, scaler

def forecast_glucose_2h(features_df: pd.DataFrame):
    """
    Computes 2-hour future glucose forecast, predicted change delta, and confidence interval.
    """
    regressor, scaler = load_forecasting_model()
    feature_cols = get_feature_columns()
    
    X_input = features_df[feature_cols].copy().fillna(0.0)
    X_scaled = scaler.transform(X_input)
    
    predictions = regressor.predict(X_scaled)
    
    forecasts = []
    for idx, row in features_df.reset_index(drop=True).iterrows():
        curr_g = float(row.get('continuous_glucose', 110.0))
        pred_g = float(np.round(max(50.0, min(380.0, predictions[idx])), 1))
        change = float(np.round(pred_g - curr_g, 1))
        
        forecasts.append({
            'current_glucose': curr_g,
            'predicted_glucose_2h': pred_g,
            'predicted_change': change,
            'confidence_lower': float(np.round(max(40.0, pred_g - 12.5), 1)),
            'confidence_upper': float(np.round(min(400.0, pred_g + 12.5), 1))
        })
        
    return forecasts
