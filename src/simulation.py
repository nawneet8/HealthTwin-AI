"""
What-If Counterfactual Simulation Engine for HealthTwin AI.
Allows clinicians and researchers to perturb sleep, activity, stress, and physiological variables
to quantify predicted risk reduction and target physiological states.
"""

import copy
import pandas as pd
import numpy as np

from src.feature_engineering import compute_rolling_features
from src.prediction import predict_glucose_spike_risk
from src.forecasting import forecast_glucose_2h

def run_what_if_simulation(baseline_df: pd.DataFrame, overrides: dict) -> dict:
    """
    Performs counterfactual prediction by overriding baseline parameters.
    
    overrides can contain:
    - 'sleep_hours': float (e.g. 7.5)
    - 'steps_24h': int (e.g. 8000)
    - 'stress_level': float (e.g. 25.0)
    - 'continuous_glucose': float (e.g. 115.0)
    - 'heart_rate': float (e.g. 68.0)
    - 'bmi': float (e.g. 27.5)
    """
    df_base = baseline_df.copy()
    df_sim = baseline_df.copy()
    
    # Apply direct perturbations
    for param, val in overrides.items():
        if param in df_sim.columns:
            df_sim[param] = float(val)
            
    # Recalculate derived features for simulated vector
    if 'sleep_hours' in overrides:
        df_sim['sleep_24h'] = float(overrides['sleep_hours'])
        df_sim['sleep_debt'] = max(0.0, 8.0 - float(overrides['sleep_hours']))
        
    if 'steps_24h' in overrides:
        df_sim['steps'] = float(overrides['steps_24h']) / 24.0
        df_sim['activity_score'] = min(2.0, float(overrides['steps_24h']) / 10000.0)
        
    if 'stress_level' in overrides:
        df_sim['stress_score'] = float(overrides['stress_level']) / 100.0
        df_sim['stress_mean_2h'] = float(overrides['stress_level'])
        
    if 'continuous_glucose' in overrides:
        df_sim['glucose_mean_1h'] = float(overrides['continuous_glucose'])
        df_sim['glucose_mean_2h'] = float(overrides['continuous_glucose'])
        
    # Run predictions on baseline vs simulated
    base_risk = predict_glucose_spike_risk(df_base)[0]
    sim_risk = predict_glucose_spike_risk(df_sim)[0]
    
    base_fc = forecast_glucose_2h(df_base)[0]
    sim_fc = forecast_glucose_2h(df_sim)[0]
    
    prob_base = base_risk['spike_probability']
    prob_sim = sim_risk['spike_probability']
    
    abs_risk_reduction = float(np.round(prob_base - prob_sim, 1))
    rel_risk_reduction = float(np.round((abs_risk_reduction / max(0.1, prob_base)) * 100.0, 1)) if prob_base > 0 else 0.0
    
    param_changes = {}
    for param, val in overrides.items():
        orig_val = float(df_base[param].iloc[0]) if param in df_base.columns else 0.0
        param_changes[param] = {
            'baseline': orig_val,
            'simulated': float(val),
            'delta': float(np.round(float(val) - orig_val, 1))
        }
        
    return {
        'baseline_risk_score': prob_base,
        'baseline_risk_level': base_risk['risk_level'],
        'baseline_predicted_glucose': base_fc['predicted_glucose_2h'],
        
        'simulated_risk_score': prob_sim,
        'simulated_risk_level': sim_risk['risk_level'],
        'simulated_predicted_glucose': sim_fc['predicted_glucose_2h'],
        
        'absolute_risk_reduction': abs_risk_reduction,
        'relative_risk_reduction_pct': rel_risk_reduction,
        'glucose_reduction_mgdl': float(np.round(base_fc['predicted_glucose_2h'] - sim_fc['predicted_glucose_2h'], 1)),
        
        'parameter_changes': param_changes,
        'disclaimer': 'RESEARCH PROOF-OF-CONCEPT ONLY. Counterfactual simulations demonstrate model sensitivity and are NOT medical prescriptions.'
    }
