import sys
import os

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="What-If Simulation | HealthTwin AI", page_icon="🧪", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="Counterfactual What-If Simulation Engine", subtitle="Simulate Behavioral & Physiological Interventions to Quantify Risk Reduction")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

st.markdown(f"### 🧪 Counterfactual Simulator: `{selected_pid}`")

st.markdown("""
<div style="background-color: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
    ℹ️ <strong>HOW IT WORKS:</strong> Adjust the interactive sliders below to test hypothetical lifestyle modifications (e.g. increasing sleep duration, stepping count, reducing stress, or weight loss). The Digital Twin re-runs machine learning inference to calculate predicted risk reduction.
</div>
""", unsafe_allow_html=True)

# Baseline metrics
base_g = twin.current_vitals['continuous_glucose']
base_hr = twin.current_vitals['heart_rate']
base_sleep = twin.wearable_state['sleep_hours']
base_steps = twin.wearable_state['steps'] * 24
base_stress = twin.wearable_state['stress_level']
base_bmi = twin.demographics['bmi']

col_ctrl, col_res = st.columns([2, 3])

with col_ctrl:
    st.markdown("#### 🎛️ Intervention Parameter Sliders")
    
    sim_sleep = st.slider("Sleep Duration (hours):", min_value=4.0, max_value=10.0, value=float(base_sleep), step=0.5)
    sim_steps = st.slider("Daily Steps Count:", min_value=1000, max_value=15000, value=int(max(1000, base_steps)), step=500)
    sim_stress = st.slider("Autonomic Stress Level (0-100):", min_value=5.0, max_value=95.0, value=float(base_stress), step=5.0)
    sim_glucose = st.slider("Baseline Glucose (mg/dL):", min_value=70.0, max_value=250.0, value=float(base_g), step=5.0)
    sim_hr = st.slider("Resting Heart Rate (bpm):", min_value=50.0, max_value=110.0, value=float(base_hr), step=2.0)
    sim_bmi = st.slider("Body Mass Index - BMI (kg/m²):", min_value=18.5, max_value=45.0, value=float(base_bmi), step=0.5)
    
    overrides = {
        'sleep_hours': sim_sleep,
        'steps_24h': sim_steps,
        'stress_level': sim_stress,
        'continuous_glucose': sim_glucose,
        'heart_rate': sim_hr,
        'bmi': sim_bmi
    }
    
    sim_results = twin.simulate(overrides)

with col_res:
    st.markdown("#### 📊 Baseline vs Simulated Patient Comparison")
    
    b_prob = sim_results['baseline_risk_score']
    s_prob = sim_results['simulated_risk_score']
    b_lvl = sim_results['baseline_risk_level']
    s_lvl = sim_results['simulated_risk_level']
    
    b_pred_g = sim_results['baseline_predicted_glucose']
    s_pred_g = sim_results['simulated_predicted_glucose']
    
    abs_red = sim_results['absolute_risk_reduction']
    rel_red = sim_results['relative_risk_reduction_pct']
    
    # Delta Callout Box
    if abs_red > 0:
        st.markdown(f"""
        <div style="background-color: rgba(34, 197, 94, 0.15); border: 2px solid #22c55e; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;">
            <h3 style="color: #4ade80; margin: 0;">🎉 PREDICTED RISK REDUCTION</h3>
            <h1 style="color: #4ade80; font-size: 3.2rem; margin: 10px 0;">-{abs_red:.1f}% Absolute Risk</h1>
            <p style="color: #cbd5e1; margin: 0;">Relative Risk Reduction: <strong>-{rel_red:.1f}%</strong> | Predicted Glucose Reduction: <strong>{b_pred_g - s_pred_g:+.1f} mg/dL</strong></p>
        </div>
        """, unsafe_allow_html=True)
    elif abs_red < 0:
        st.markdown(f"""
        <div style="background-color: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;">
            <h3 style="color: #f87171; margin: 0;">⚠️ PREDICTED RISK INCREASE</h3>
            <h1 style="color: #f87171; font-size: 3.2rem; margin: 10px 0;">+{abs(abs_red):.1f}% Absolute Risk</h1>
            <p style="color: #cbd5e1; margin: 0;">Perturbations increased predicted glucose spike risk!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #1e293b; border: 1px solid #475569; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px;">
            <h4 style="color: #94a3b8; margin: 0;">No Change in Predicted Risk Category</h4>
        </div>
        """, unsafe_allow_html=True)
        
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"""
        <div class="health-card" style="border: 1px solid #475569;">
            <h4 style="color: #94a3b8; margin: 0;">BASELINE PATIENT</h4>
            <h2 style="color: #f8fafc; margin: 10px 0;">{b_prob:.1f}% Risk</h2>
            <p style="color: #94a3b8;">Stratum: <strong>{b_lvl}</strong></p>
            <p style="color: #94a3b8;">2h Pred Glucose: <strong>{b_pred_g:.1f} mg/dL</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="health-card" style="border: 2px solid #38bdf8;">
            <h4 style="color: #38bdf8; margin: 0;">SIMULATED PATIENT</h4>
            <h2 style="color: #38bdf8; margin: 10px 0;">{s_prob:.1f}% Risk</h2>
            <p style="color: #94a3b8;">Stratum: <strong>{s_lvl}</strong></p>
            <p style="color: #94a3b8;">2h Pred Glucose: <strong>{s_pred_g:.1f} mg/dL</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### 🔄 Parameter Modification Matrix")
    change_rows = []
    for p, data in sim_results['parameter_changes'].items():
        change_rows.append({
            'Parameter': p.replace('_', ' ').title(),
            'Baseline Value': f"{data['baseline']:.1f}",
            'Simulated Value': f"{data['simulated']:.1f}",
            'Delta Change': f"{data['delta']:+.1f}"
        })
    st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)
