"""
Unsupervised Biometric Anomaly Detection Engine for HealthTwin AI.
Uses Isolation Forest to detect abnormal physiological & wearable patterns.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ANOMALY_FEATURES = ['continuous_glucose', 'heart_rate', 'hrv', 'sleep_hours', 'steps_24h', 'stress_level']

def train_anomaly_detector(df_wearable: pd.DataFrame):
    """
    Trains IsolationForest model on wearable biometric features.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Ensure features exist
    df_feats = df_wearable.copy()
    if 'steps_24h' not in df_feats.columns:
        df_feats['steps_24h'] = df_feats['steps'] * 24
        
    X = df_feats[ANOMALY_FEATURES].copy().fillna(df_feats[ANOMALY_FEATURES].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.08,
        random_state=42
    )
    iso_forest.fit(X_scaled)
    
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, 'anomaly_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'anomaly_scaler.pkl'))
    
    print("Isolation Forest Anomaly Detector trained successfully.")
    return iso_forest, scaler

def load_anomaly_detector():
    """
    Loads saved IsolationForest anomaly detector.
    """
    model_path = os.path.join(MODELS_DIR, 'anomaly_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'anomaly_scaler.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        from src.data_loader import load_all_data
        _, df_wearable, _ = load_all_data()
        return train_anomaly_detector(df_wearable)
        
    iso_forest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return iso_forest, scaler

def detect_biometric_anomalies(patient_state: dict) -> list:
    """
    Evaluates current patient vitals and flags specific physiological anomalies.
    Returns list of human-readable anomaly alert objects.
    """
    anomalies = []
    
    g = float(patient_state.get('continuous_glucose', 110.0))
    hr = float(patient_state.get('heart_rate', 72.0))
    hrv = float(patient_state.get('hrv', 45.0))
    sleep = float(patient_state.get('sleep_hours', 7.0))
    stress = float(patient_state.get('stress_level', 35.0))
    steps = float(patient_state.get('steps_24h', 5000.0))
    glucose_slope = float(patient_state.get('glucose_slope', 0.0))
    
    # Rule-based physiological threshold checks combined with Isolation Forest
    if g >= 180.0:
        anomalies.append({
            'type': 'Hyperglycemia Spike',
            'severity': 'HIGH',
            'icon': '⚠',
            'message': f'Severe hyperglycemia detected ({g:.1f} mg/dL).'
        })
    elif g <= 70.0:
        anomalies.append({
            'type': 'Hypoglycemia Warning',
            'severity': 'CRITICAL',
            'icon': '🚨',
            'message': f'Abnormally low blood glucose level ({g:.1f} mg/dL).'
        })
        
    if glucose_slope > 4.0:
        anomalies.append({
            'type': 'Rapid Glucose Increase',
            'severity': 'HIGH',
            'icon': '📈',
            'message': f'Abnormal glucose surge trajectory (+{glucose_slope*4:.1f} mg/dL per hour).'
        })
        
    if hr > 105.0:
        anomalies.append({
            'type': 'Tachycardia / Elevated HR',
            'severity': 'MODERATE',
            'icon': '💓',
            'message': f'Unusual resting heart rate elevation ({hr:.0f} bpm).'
        })
        
    if hrv < 20.0:
        anomalies.append({
            'type': 'Autonomic Dysfunction (Low HRV)',
            'severity': 'MODERATE',
            'icon': '🔻',
            'message': f'Critically depressed Heart Rate Variability ({hrv:.1f} ms).'
        })
        
    if sleep < 5.0:
        anomalies.append({
            'type': 'Acute Sleep Deprivation',
            'severity': 'MODERATE',
            'icon': '🌙',
            'message': f'Restorative sleep severely truncated ({sleep:.1f} hrs).'
        })
        
    if stress > 75.0:
        anomalies.append({
            'type': 'High Autonomic Stress',
            'severity': 'MODERATE',
            'icon': '⚡',
            'message': f'Prolonged stress spike detected ({stress:.0f}/100).'
        })
        
    # Isolation Forest statistical evaluation
    iso_forest, scaler = load_anomaly_detector()
    input_vector = pd.DataFrame([{
        'continuous_glucose': g,
        'heart_rate': hr,
        'hrv': hrv,
        'sleep_hours': sleep,
        'steps_24h': steps,
        'stress_level': stress
    }])[ANOMALY_FEATURES]
    
    input_scaled = scaler.transform(input_vector)
    anomaly_score = iso_forest.score_samples(input_scaled)[0]
    is_outlier = iso_forest.predict(input_scaled)[0] == -1
    
    if is_outlier and len(anomalies) == 0:
        anomalies.append({
            'type': 'Multivariate Biometric Anomaly',
            'severity': 'MODERATE',
            'icon': '🔍',
            'message': f'Isolation Forest flagged unusual co-occurring vital metrics (Score: {anomaly_score:.2f}).'
        })
        
    return anomalies
