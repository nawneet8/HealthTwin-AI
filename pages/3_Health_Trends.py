import sys
import os

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Health Trends | HealthTwin AI", page_icon="📈", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="Interactive Health Trends & Telemetry Analytics", subtitle="Multi-Parametric Physiological Time-Series Trajectory Analysis")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

p_wearable = df_wearable[df_wearable['patient_id'] == selected_pid].sort_values('timestamp').copy()
p_wearable['timestamp'] = pd.to_datetime(p_wearable['timestamp'])

st.markdown(f"### 📈 Biometric Time-Series Visualizer: `{selected_pid}`")

# Timeframe Selector Filter
timeframe = st.selectbox("Select Time Range:", ["Last 24 Hours", "Last 3 Days", "Full Dataset (5 Days)"], index=1)

if timeframe == "Last 24 Hours":
    p_data = p_wearable.tail(96)
elif timeframe == "Last 3 Days":
    p_data = p_wearable.tail(288)
else:
    p_data = p_wearable

# 1. Continuous Glucose Trajectory
st.markdown("#### 🩸 1. Continuous Glucose Monitoring (CGM)")
fig_g = go.Figure()
fig_g.add_trace(go.Scatter(
    x=p_data['timestamp'], y=p_data['continuous_glucose'],
    mode='lines+markers', name='Glucose (mg/dL)',
    line=dict(color='#38bdf8', width=2), marker=dict(size=4)
))
fig_g.add_hline(y=180, line_dash="dash", line_color="#ef4444", annotation_text="Hyperglycemia (>180 mg/dL)")
fig_g.add_hline(y=70, line_dash="dash", line_color="#f59e0b", annotation_text="Hypoglycemia (<70 mg/dL)")
fig_g.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a', margin=dict(l=40, r=40, t=30, b=30), height=320)
st.plotly_chart(fig_g, use_container_width=True)

# 2. Heart Rate & HRV Autonomic Balance
st.markdown("#### 💓 2. Heart Rate & Heart Rate Variability (HRV)")
col_hr1, col_hr2 = st.columns(2)

with col_hr1:
    fig_hr = px.line(p_data, x='timestamp', y='heart_rate', title="Heart Rate (bpm)", color_discrete_sequence=['#f87171'])
    fig_hr.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a', height=280)
    st.plotly_chart(fig_hr, use_container_width=True)

with col_hr2:
    fig_hrv = px.line(p_data, x='timestamp', y='hrv', title="Heart Rate Variability - HRV (ms)", color_discrete_sequence=['#a855f7'])
    fig_hrv.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a', height=280)
    st.plotly_chart(fig_hrv, use_container_width=True)

# 3. Activity, Steps, and Stress Profile
st.markdown("#### 🏃 3. Activity, Steps & Stress Index")
col_st1, col_st2 = st.columns(2)

with col_st1:
    fig_steps = px.bar(p_data, x='timestamp', y='steps', title="Step Accumulation per 15m Interval", color_discrete_sequence=['#4ade80'])
    fig_steps.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a', height=280)
    st.plotly_chart(fig_steps, use_container_width=True)

with col_st2:
    fig_stress = px.area(p_data, x='timestamp', y='stress_level', title="Autonomic Stress Level (0-100)", color_discrete_sequence=['#fb923c'])
    fig_stress.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a', height=280)
    st.plotly_chart(fig_stress, use_container_width=True)

# 4. Correlation Matrix Heatmap
st.markdown("#### 🔗 4. Physiological Multi-Parametric Correlation Matrix")
corr_cols = ['continuous_glucose', 'heart_rate', 'hrv', 'steps', 'sleep_hours', 'stress_level', 'calories']
corr_df = p_wearable[corr_cols].corr()

fig_corr = px.imshow(
    corr_df, text_auto=".2f", color_continuous_scale="Blues",
    title="Biometric & Behavioral Feature Correlation Matrix"
)
fig_corr.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a', height=400)
st.plotly_chart(fig_corr, use_container_width=True)
