"""
Glucose Spike Risk Classification Engine for HealthTwin AI.
Trains XGBoost binary classifier to predict 2-hour forward glucose spikes with full clinical metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import xgboost as xgb

from src.feature_engineering import get_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def train_risk_model(df_events: pd.DataFrame):
    """
    Trains XGBoost Classifier for 2-hour glucose spike prediction with group-aware train/test split.
    Saves model and scaler artifacts to models/ directory.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    feature_cols = get_feature_columns()
    X = df_events[feature_cols].copy()
    y = df_events['glucose_spike'].astype(int).values
    groups = df_events['patient_id'].values
    
    # Fill missing feature values with column medians
    X = X.fillna(X.median())
    
    # Patient-grouped split to prevent data leakage across timestamps of same patient
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Class imbalance weight
    scale_pos = (len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train))
    
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.45).astype(int)  # Slightly lowered decision threshold prioritizing recall
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    p_curve, r_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(r_curve, p_curve)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist()
    }
    
    # Save artifacts
    model_path = os.path.join(MODELS_DIR, 'risk_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    
    print("Risk Classifier trained successfully:")
    print(f" - ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print(f" - Recall: {rec:.4f}, Precision: {prec:.4f}")
    
    return model, scaler, metrics

def load_risk_model():
    """
    Loads saved model, scaler, and feature names.
    If missing, prompts training on demand.
    """
    model_path = os.path.join(MODELS_DIR, 'risk_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        from src.data_loader import load_all_data
        from src.preprocessing import merge_ehr_and_wearable
        from src.feature_engineering import compute_rolling_features
        
        _, _, df_events = load_all_data()
        df_eng = compute_rolling_features(df_events)
        model, scaler, _ = train_risk_model(df_eng)
        return model, scaler
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_glucose_spike_risk(features_df: pd.DataFrame):
    """
    Predicts glucose spike probability and risk category for input features.
    """
    model, scaler = load_risk_model()
    feature_cols = get_feature_columns()
    
    X_input = features_df[feature_cols].copy().fillna(0.0)
    X_scaled = scaler.transform(X_input)
    
    probas = model.predict_proba(X_scaled)[:, 1]
    
    results = []
    for prob in probas:
        pct = float(np.round(prob * 100, 1))
        if pct < 25.0:
            category = "LOW"
            color = "green"
        elif pct < 55.0:
            category = "MODERATE"
            color = "yellow"
        elif pct < 78.0:
            category = "HIGH"
            color = "orange"
        else:
            category = "CRITICAL"
            color = "red"
            
        results.append({
            'spike_probability': pct,
            'risk_level': category,
            'color': color,
            'raw_probability': float(prob)
        })
        
    return results
