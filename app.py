import sys
import os

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="HealthTwin AI | Virtual Patient Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="HealthTwin AI", subtitle="An Intelligent Virtual Patient Digital Twin for Predictive Healthcare")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

st.markdown("---")

# Quick Overview & Command Summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Monitored Patients", f"{len(df_ehr)} Patients", delta="Active Twins")
with col2:
    high_risk_count = len(df_eng[df_eng['glucose_spike'] == 1]['patient_id'].unique())
    st.metric("High Glucose Spike Risk", f"{high_risk_count} Patients", delta="Elevated Risk", delta_color="inverse")
with col3:
    avg_glucose = float(np.round(df_wearable['continuous_glucose'].mean(), 1))
    st.metric("Population Avg Glucose", f"{avg_glucose} mg/dL", delta="Target: 70-140")
with col4:
    total_telemetry = len(df_wearable)
    st.metric("Wearable Telemetry Stream", f"{total_telemetry:,}", delta="Continuous 15m")

st.markdown("---")

# Feature Highlights Grid
st.markdown("### 🧬 Core Digital Twin Architecture & Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    #### 1. 🔄 Real-Time Digital Twin Engine
    - **Continuous State Assimilation:** Synchronizes dynamic wearable IoT telemetry (HR, HRV, Spo2, Steps, Sleep, Stress) with static EHR clinical baseline.
    - **Live State Transitions:** Visualizes OLD STATE ➔ NEW STREAMING DATA ➔ UPDATED VIRTUAL TWIN.
    """)

with feat_col2:
    st.markdown("""
    #### 2. 🔮 2-Hour Forward Predictive AI
    - **Glucose Spike Probability:** XGBoost machine learning model trained on multi-scale temporal rolling features.
    - **Continuous Forecast:** 2-hour trajectory prediction for early clinical intervention.
    """)

with feat_col3:
    st.markdown("""
    #### 3. 🧪 Counterfactual What-If Simulator
    - **Interactive Intervention Testing:** Simulate how changing lifestyle habits (sleep, steps, stress reduction) alters predicted glucose risk.
    - **Explainable AI (SHAP):** Transparent feature contributions and natural language logic summaries.
    """)

st.markdown("---")

# Active Digital Twin Snapshot Card
st.markdown(f"### 👤 Active Virtual Patient Digital Twin: `{selected_pid}`")

snap_col1, snap_col2, snap_col3 = st.columns([1, 1, 1])

with snap_col1:
    st.markdown("##### 📊 Demographics & EHR Profile")
    st.json(twin.demographics)
    st.markdown("##### 🩺 Medical History")
    st.json(twin.medical_history)

with snap_col2:
    st.markdown("##### 💓 Current Biometric Vitals")
    st.json(twin.current_vitals)
    st.markdown("##### 🏃 Wearable Telemetry State")
    st.json(twin.wearable_state)

with snap_col3:
    st.markdown("##### 🔮 AI Predictive Risk Engine")
    
    risk_color = twin.risk_color
    risk_score = twin.risk_score
    risk_lvl = twin.risk_level
    pred_outcome = twin.predicted_outcome
    
    st.markdown(f"""
    <div style="background-color: #1e293b; border: 2px solid {risk_color}; border-radius: 10px; padding: 16px; text-align: center;">
        <h4 style="color: #94a3b8; margin: 0;">2-Hour Glucose Spike Probability</h4>
        <h1 style="color: {risk_color}; margin: 10px 0; font-size: 3rem;">{risk_score:.1f}%</h1>
        <p><span class="risk-badge-{risk_lvl.lower()}">{risk_lvl} RISK</span></p>
        <hr style="border-color: #334155; margin: 12px 0;">
        <p style="color: #cbd5e1; margin: 4px 0;">Current Glucose: <strong>{twin.current_vitals['continuous_glucose']:.1f} mg/dL</strong></p>
        <p style="color: #cbd5e1; margin: 4px 0;">Predicted in 2h: <strong>{pred_outcome.get('predicted_glucose_2h', 0.0):.1f} mg/dL</strong></p>
        <p style="color: #cbd5e1; margin: 4px 0;">Predicted Delta: <strong>{pred_outcome.get('predicted_change', 0.0):+.1f} mg/dL</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(twin.anomalies) > 0:
        st.warning(f"⚠️ {len(twin.anomalies)} Biometric Anomalies Detected!")
    else:
        st.success("✅ Vitals within normal physiological bounds.")

st.sidebar.info("💡 Use the sidebar navigation menu above to explore interactive pages!")
