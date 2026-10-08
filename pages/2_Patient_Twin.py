import sys
import os

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Patient Digital Twin | HealthTwin AI", page_icon="🧬", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="Patient Digital Twin Engine", subtitle="Continuous Multimodal Virtual Patient Representation & Stream Assimilation")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

st.markdown(f"### 🧬 Digital Twin Virtual State Representation: `{selected_pid}`")

# Patient Profile Header Card
col_dem, col_med, col_status = st.columns(3)

with col_dem:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Patient Demographics</div>
        <div style="margin-top: 10px; color: #cbd5e1;">
            <p><strong>Patient ID:</strong> {}</p>
            <p><strong>Age / Gender:</strong> {} yrs | {}</p>
            <p><strong>BMI:</strong> {} kg/m²</p>
        </div>
    </div>
    """.format(twin.patient_id, twin.demographics['age'], twin.demographics['gender'], twin.demographics['bmi']), unsafe_allow_html=True)

with col_med:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Clinical History & EHR</div>
        <div style="margin-top: 10px; color: #cbd5e1;">
            <p><strong>T2D History:</strong> {}</p>
            <p><strong>Fasting Glucose:</strong> {} mg/dL</p>
            <p><strong>HbA1c Baseline:</strong> {}%</p>
            <p><strong>Medication:</strong> {}</p>
        </div>
    </div>
    """.format('Yes' if twin.medical_history['diabetes_history']==1 else 'No', twin.medical_history['fasting_glucose'], twin.medical_history['hba1c'], twin.medical_history['medication']), unsafe_allow_html=True)

with col_status:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Digital Twin Engine Status</div>
        <div style="margin-top: 10px; color: #cbd5e1;">
            <p><strong>Sync Status:</strong> <span style="color: #4ade80;">● ONLINE</span></p>
            <p><strong>Last Telemetry:</strong> {}</p>
            <p><strong>Risk Stratum:</strong> <span class="risk-badge-{}">{}</span></p>
        </div>
    </div>
    """.format(twin.timestamp, twin.risk_level.lower(), twin.risk_level), unsafe_allow_html=True)

st.markdown("---")

# Stream Telemetry Simulator Section (OLD STATE -> NEW TELEMETRY -> UPDATED DIGITAL TWIN -> NEW PREDICTION)
st.markdown("### 🔄 Interactive Telemetry Stream Assimilation Pipeline")
st.info("Simulate receiving new streaming wearable telemetry to observe real-time Digital Twin state transition and prediction recalculation.")

sim_col1, sim_col2 = st.columns([1, 2])

with sim_col1:
    st.markdown("#### ⚡ Generate Inbound Telemetry Event")
    event_type = st.radio(
        "Choose Simulated Event Scenario:",
        ["High Carb Meal Intake (+40 mg/dL Glucose)", "Stress Surge (+30 Stress, +20 HR)", "Post-Meal Exercise (-25 mg/dL Glucose, +150 Steps)", "Restorative Recovery (+Sleep, +HRV)"],
        index=0
    )
    
    if st.button("🚀 Push Inbound Wearable Telemetry Packet", use_container_width=True):
        # Create perturbed telemetry copy
        cur_v = twin.current_vitals
        cur_w = twin.wearable_state
        
        new_g = cur_v['continuous_glucose']
        new_hr = cur_v['heart_rate']
        new_hrv = cur_v['hrv']
        new_stress = cur_w['stress_level']
        new_steps = cur_w['steps']
        
        if "High Carb" in event_type:
            new_g = min(350.0, new_g + 45.0)
            new_stress += 10.0
        elif "Stress Surge" in event_type:
            new_stress = min(95.0, new_stress + 30.0)
            new_hr += 18.0
            new_hrv = max(12.0, new_hrv - 15.0)
            new_g += 15.0
        elif "Exercise" in event_type:
            new_g = max(65.0, new_g - 25.0)
            new_steps += 250
            new_hr += 25.0
            new_stress = max(10.0, new_stress - 15.0)
        elif "Restorative" in event_type:
            new_hrv += 20.0
            new_hr = max(55.0, new_hr - 10.0)
            new_stress = max(5.0, new_stress - 20.0)
            
        new_packet = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'continuous_glucose': new_g,
            'heart_rate': new_hr,
            'hrv': new_hrv,
            'spo2': cur_v['spo2'],
            'steps': new_steps,
            'sleep_hours': cur_w['sleep_hours'],
            'sleep_quality': cur_w['sleep_quality'],
            'calories': cur_w['calories'] + 1.5,
            'activity_level': 'Active' if "Exercise" in event_type else cur_w['activity_level'],
            'stress_level': new_stress
        }
        
        # Update Digital Twin State
        twin.update_state(new_packet)
        st.success(f"Telemetry assimilated into Digital Twin `{selected_pid}`!")

# Display Pipeline Transition Flow
if len(twin.state_history) > 0:
    latest_hist = twin.state_history[-1]
    old_s = latest_hist['old_state']
    new_packet = latest_hist['incoming_telemetry']
    new_s = latest_hist['new_state']
    
    st.markdown("#### 🌊 State Transition Visual Pipeline")
    
    t_col1, arrow1, t_col2, arrow2, t_col3, arrow3, t_col4 = st.columns([2, 0.4, 2, 0.4, 2, 0.4, 2])
    
    with t_col1:
        st.markdown(f"""
        <div style="background-color: #1e293b; border: 1px solid #475569; padding: 14px; border-radius: 8px;">
            <h5 style="color: #94a3b8; margin:0;">1. OLD STATE</h5>
            <small style="color: #64748b;">TS: {old_s['timestamp']}</small>
            <hr style="margin: 8px 0; border-color: #334155;">
            <p style="margin: 2px 0;">Glucose: <strong>{old_s['continuous_glucose']:.1f} mg/dL</strong></p>
            <p style="margin: 2px 0;">HR / HRV: <strong>{old_s['heart_rate']:.0f} / {old_s['hrv']:.0f}</strong></p>
            <p style="margin: 2px 0;">Risk Prob: <strong>{old_s['risk_score']:.1f}% ({old_s['risk_level']})</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with arrow1:
        st.markdown("<h2 style='text-align: center; color: #38bdf8; margin-top: 30px;'>➔</h2>", unsafe_allow_html=True)
        
    with t_col2:
        st.markdown(f"""
        <div style="background-color: #0f172a; border: 1px solid #38bdf8; padding: 14px; border-radius: 8px;">
            <h5 style="color: #38bdf8; margin:0;">2. INBOUND WEARABLE</h5>
            <small style="color: #64748b;">TS: {new_packet['timestamp']}</small>
            <hr style="margin: 8px 0; border-color: #38bdf8;">
            <p style="margin: 2px 0;">New Glucose: <strong>{new_packet['continuous_glucose']:.1f} mg/dL</strong></p>
            <p style="margin: 2px 0;">New HR: <strong>{new_packet['heart_rate']:.0f} bpm</strong></p>
            <p style="margin: 2px 0;">New Stress: <strong>{new_packet['stress_level']:.0f}/100</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with arrow2:
        st.markdown("<h2 style='text-align: center; color: #38bdf8; margin-top: 30px;'>➔</h2>", unsafe_allow_html=True)
        
    with t_col3:
        st.markdown(f"""
        <div style="background-color: #1e293b; border: 1px solid #475569; padding: 14px; border-radius: 8px;">
            <h5 style="color: #a855f7; margin:0;">3. UPDATED TWIN</h5>
            <small style="color: #64748b;">TS: {new_s['timestamp']}</small>
            <hr style="margin: 8px 0; border-color: #334155;">
            <p style="margin: 2px 0;">Glucose: <strong>{new_s['current_vitals']['continuous_glucose']:.1f} mg/dL</strong></p>
            <p style="margin: 2px 0;">HR / HRV: <strong>{new_s['current_vitals']['heart_rate']:.0f} / {new_s['current_vitals']['hrv']:.0f}</strong></p>
            <p style="margin: 2px 0;">Anomalies: <strong>{len(new_s['anomalies'])} detected</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    with arrow3:
        st.markdown("<h2 style='text-align: center; color: #38bdf8; margin-top: 30px;'>➔</h2>", unsafe_allow_html=True)
        
    with t_col4:
        risk_c = new_s['risk_color']
        st.markdown(f"""
        <div style="background-color: #1e293b; border: 2px solid {risk_c}; padding: 14px; border-radius: 8px;">
            <h5 style="color: {risk_c}; margin:0;">4. NEW PREDICTION</h5>
            <small style="color: #64748b;">2-Hour Lookahead</small>
            <hr style="margin: 8px 0; border-color: #334155;">
            <p style="margin: 2px 0;">Spike Prob: <strong style="color:{risk_c};">{new_s['risk_score']:.1f}%</strong></p>
            <p style="margin: 2px 0;">Pred Glucose: <strong>{new_s['predicted_outcome'].get('predicted_glucose_2h',0.0):.1f} mg/dL</strong></p>
            <p style="margin: 2px 0;">Risk Level: <strong>{new_s['risk_level']}</strong></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Full Current Vitals Grid
st.markdown("### 📊 Comprehensive Telemetry & Vitals Monitor")

v1, v2, v3, v4, v5, v6 = st.columns(6)

with v1:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Continuous Glucose</div>
        <div class="card-value">{:.1f}</div>
        <div class="card-unit">mg/dL</div>
    </div>
    """.format(twin.current_vitals['continuous_glucose']), unsafe_allow_html=True)

with v2:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Heart Rate</div>
        <div class="card-value">{:.0f}</div>
        <div class="card-unit">bpm</div>
    </div>
    """.format(twin.current_vitals['heart_rate']), unsafe_allow_html=True)

with v3:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">HRV (SDNN)</div>
        <div class="card-value">{:.1f}</div>
        <div class="card-unit">ms</div>
    </div>
    """.format(twin.current_vitals['hrv']), unsafe_allow_html=True)

with v4:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Blood Oxygen (SpO2)</div>
        <div class="card-value">{:.1f}</div>
        <div class="card-unit">%</div>
    </div>
    """.format(twin.current_vitals['spo2']), unsafe_allow_html=True)

with v5:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Stress Level</div>
        <div class="card-value">{:.0f}</div>
        <div class="card-unit">/ 100</div>
    </div>
    """.format(twin.wearable_state['stress_level']), unsafe_allow_html=True)

with v6:
    st.markdown("""
    <div class="health-card">
        <div class="card-label">Sleep Duration</div>
        <div class="card-value">{:.1f}</div>
        <div class="card-unit">hours</div>
    </div>
    """.format(twin.wearable_state['sleep_hours']), unsafe_allow_html=True)
