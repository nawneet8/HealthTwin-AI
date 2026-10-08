import sys
import os

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Anomaly Detection | HealthTwin AI", page_icon="⚠", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)
from src.anomaly_detection import detect_biometric_anomalies

inject_custom_css()
render_header(title="Unsupervised Anomaly Detection Engine", subtitle="Isolation Forest & Multimodal Biometric Outlier Monitoring")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

st.markdown(f"### ⚠ Detected Biometric Anomalies: `{selected_pid}`")

anomalies = twin.anomalies

if len(anomalies) > 0:
    st.markdown(f"**Found `{len(anomalies)}` active physiological anomalies for Digital Twin `{selected_pid}`:**")
    
    col_a1, col_a2 = st.columns([2, 1])
    
    with col_a1:
        for idx, item in enumerate(anomalies):
            severity = item.get('severity', 'MODERATE')
            color_map = {'LOW': '#22c55e', 'MODERATE': '#eab308', 'HIGH': '#f97316', 'CRITICAL': '#ef4444'}
            b_color = color_map.get(severity, '#eab308')
            
            st.markdown(f"""
            <div style="background-color: #1e293b; border-left: 5px solid {b_color}; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #f8fafc;">{item['icon']} {item['type']}</h4>
                    <span style="background-color: {b_color}; color: #000; font-weight: bold; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;">{severity}</span>
                </div>
                <p style="color: #cbd5e1; margin-top: 8px; margin-bottom: 0;">{item['message']}</p>
                <small style="color: #64748b;">Detected at: {twin.timestamp}</small>
            </div>
            """, unsafe_allow_html=True)
            
    with col_a2:
        st.markdown("#### 🌲 Isolation Forest Score")
        st.info("Isolation Forest isolates anomalies by randomly selecting a feature and splitting values. Shorter paths indicate high multivariate outlier probability.")
        st.metric("Contamination Rate", "8.0%", "Outlier Sensitivity")
else:
    st.success(f"✅ All biometric indicators for Digital Twin `{selected_pid}` are currently within expected physiological ranges.")

st.markdown("---")

# Historical Anomaly Scanning Across Patient Telemetry
st.markdown("### 📜 Historical Anomaly Log (Past 48 Hours)")

p_wearable = df_wearable[df_wearable['patient_id'] == selected_pid].sort_values('timestamp').tail(48).copy()

history_anomalies = []
for idx, row in p_wearable.iterrows():
    query_state = {
        'continuous_glucose': row['continuous_glucose'],
        'heart_rate': row['heart_rate'],
        'hrv': row['hrv'],
        'sleep_hours': row['sleep_hours'],
        'steps_24h': row['steps'] * 24,
        'stress_level': row['stress_level'],
        'glucose_slope': 0.0
    }
    anom_list = detect_biometric_anomalies(query_state)
    for a in anom_list:
        history_anomalies.append({
            'Timestamp': row['timestamp'],
            'Anomaly Type': a['type'],
            'Severity': a['severity'],
            'Glucose (mg/dL)': row['continuous_glucose'],
            'Heart Rate (bpm)': row['heart_rate'],
            'HRV (ms)': row['hrv'],
            'Stress': row['stress_level']
        })

df_anom_hist = pd.DataFrame(history_anomalies)

if len(df_anom_hist) > 0:
    st.dataframe(df_anom_hist.sort_values(by='Timestamp', ascending=False), use_container_width=True, hide_index=True)
else:
    st.write("No historical anomalies detected in the past 48 hours.")
