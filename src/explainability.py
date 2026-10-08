"""
Explainable AI (XAI) Engine for HealthTwin AI.
Leverages SHAP (SHapley Additive exPlanations) to quantify individual feature contributions
and generate clinician-friendly natural language interpretations.
"""

import numpy as np
import pandas as pd
import shap

from src.prediction import load_risk_model
from src.feature_engineering import get_feature_columns

def get_shap_explanation(features_df: pd.DataFrame):
    """
    Computes SHAP feature importance values for a patient's current feature vector.
    Returns sorted top risk factors, percentage impact, raw SHAP values, and natural language summary.
    """
    model, scaler = load_risk_model()
    feature_cols = get_feature_columns()
    
    X_input = features_df[feature_cols].copy().fillna(0.0)
    X_scaled = scaler.transform(X_input)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)
    
    # Check shape of shap_vals
    if isinstance(shap_vals, list):
        shap_vec = shap_vals[1][0] # Positive class for binary classifier
    elif len(shap_vals.shape) == 2:
        shap_vec = shap_vals[0]
    else:
        shap_vec = shap_vals[0, :, 1]
        
    base_val = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
    
    # Format readable feature labels
    feature_label_map = {
        'continuous_glucose': 'Recent Glucose Level',
        'recent_glucose_change': 'Glucose Trend (15m)',
        'glucose_slope': 'Glucose Velocity (1h)',
        'glucose_mean_2h': '2-Hour Avg Glucose',
        'glucose_std_2h': 'Glucose Fluctuations (2h)',
        'glucose_variability': 'Glucose Variability Index',
        'bmi': 'Body Mass Index (BMI)',
        'hba1c': 'Baseline HbA1c',
        'fasting_glucose': 'Fasting Glucose Baseline',
        'sleep_debt': 'Sleep Deprivation Debt',
        'sleep_quality': 'Sleep Quality Score',
        'sleep_24h': 'Sleep Duration (24h)',
        'activity_score': 'Daily Physical Activity',
        'steps_24h': 'Steps Count (24h)',
        'stress_level': 'Autonomic Stress Level',
        'stress_score': 'Stress Index',
        'heart_rate': 'Current Heart Rate',
        'avg_heart_rate': 'Average Heart Rate',
        'hrv': 'Heart Rate Variability (HRV)',
        'avg_hrv': '24-Hour Avg HRV',
        'age': 'Patient Age',
        'diabetes_history': 'Prior Diabetes History',
        'hypertension': 'Hypertension Status'
    }
    
    contributions = []
    tot_positive_impact = np.sum(np.maximum(0.0, shap_vec)) + 1e-6
    
    for feat_name, val, s_val in zip(feature_cols, X_input.iloc[0].values, shap_vec):
        readable_name = feature_label_map.get(feat_name, feat_name.replace('_', ' ').title())
        pct_contrib = (max(0.0, s_val) / tot_positive_impact) * 100.0
        
        contributions.append({
            'feature': feat_name,
            'label': readable_name,
            'actual_value': float(np.round(val, 2)),
            'shap_value': float(np.round(s_val, 4)),
            'impact_pct': float(np.round(pct_contrib, 1))
        })
        
    df_contrib = pd.DataFrame(contributions)
    df_contrib_sorted = df_contrib.sort_values(by='shap_value', ascending=False).reset_index(drop=True)
    
    # Build Natural Language Explanation
    top_positive = df_contrib_sorted[df_contrib_sorted['shap_value'] > 0.0].head(4)
    
    if len(top_positive) > 0:
        reasons = []
        for _, row in top_positive.iterrows():
            reasons.append(f"{row['label']} ({row['impact_pct']:.0f}% contribution)")
        summary_text = (
            "Risk is elevated primarily due to: " + ", ".join(reasons) + ". "
            "Modifying these parameters in the What-If Simulator can help lower predicted glucose spike risk."
        )
    else:
        summary_text = (
            "Patient's vitals are currently within stable baseline parameters. "
            "Low risk of glucose spike over the next 2 hours."
        )
        
    return {
        'base_value': base_val,
        'contributions': df_contrib_sorted,
        'summary_text': summary_text
    }
