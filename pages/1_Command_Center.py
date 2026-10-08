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
import plotly.graph_objects as go

st.set_page_config(page_title="Command Center | HealthTwin AI", page_icon="🏥", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="Command Center Dashboard", subtitle="Real-Time Clinical Overview & Patient Population Risk Radar")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()
twin = get_or_create_digital_twin(selected_pid)

# Top Key Performance Indicators (KPIs)
st.markdown(f"### 📊 Clinical Command Metrics for `{selected_pid}`")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.metric("Current Glucose", f"{twin.current_vitals['continuous_glucose']:.1f} mg/dL")
with kpi2:
    st.metric("Predicted (2h)", f"{twin.predicted_outcome.get('predicted_glucose_2h', 0.0):.1f} mg/dL", delta=f"{twin.predicted_outcome.get('predicted_change', 0.0):+.1f}")
with kpi3:
    st.metric("Spike Risk Prob", f"{twin.risk_score:.1f}%", delta=twin.risk_level)
with kpi4:
    st.metric("Heart Rate", f"{twin.current_vitals['heart_rate']:.0f} bpm")
with kpi5:
    st.metric("HRV (ms)", f"{twin.current_vitals['hrv']:.1f} ms")
with kpi6:
    st.metric("Sleep / Steps", f"{twin.wearable_state['sleep_hours']:.1f}h | {twin.wearable_state['steps']}")

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📈 Live Glucose & 2-Hour Prediction Trajectory")
    
    # Patient trajectory data
    p_data = df_wearable[df_wearable['patient_id'] == selected_pid].sort_values('timestamp').tail(48).copy()
    p_data['timestamp'] = pd.to_datetime(p_data['timestamp'])
    
    fig = go.Figure()
    
    # Historical trajectory
    fig.add_trace(go.Scatter(
        x=p_data['timestamp'],
        y=p_data['continuous_glucose'],
        mode='lines+markers',
        name='Continuous Glucose (CGM)',
        line=dict(color='#38bdf8', width=3),
        marker=dict(size=5)
    ))
    
    # Hyperglycemia & Hypoglycemia risk thresholds
    fig.add_hline(y=180, line_dash="dash", line_color="#ef4444", annotation_text="Hyperglycemia Threshold (180 mg/dL)", annotation_position="top right")
    fig.add_hline(y=70, line_dash="dash", line_color="#f59e0b", annotation_text="Hypoglycemia Threshold (70 mg/dL)", annotation_position="bottom right")
    
    # Future projection point
    last_ts = p_data['timestamp'].iloc[-1]
    future_ts = last_ts + pd.Timedelta(hours=2)
    pred_g = twin.predicted_outcome.get('predicted_glucose_2h', twin.current_vitals['continuous_glucose'])
    
    fig.add_trace(go.Scatter(
        x=[last_ts, future_ts],
        y=[p_data['continuous_glucose'].iloc[-1], pred_g],
        mode='lines+markers',
        name='2-Hour AI Forecast',
        line=dict(color='#f97316', width=4, dash='dot'),
        marker=dict(size=10, symbol='star', color='#f97316')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1e293b',
        plot_bgcolor='#0f172a',
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### 🚨 Active Alerts & Risk Factors")
    
    if len(twin.anomalies) > 0:
        for anomaly in twin.anomalies:
            st.error(f"{anomaly['icon']} **{anomaly['type']}** ({anomaly['severity']})\n\n{anomaly['message']}")
    else:
        st.success("✅ No critical biometric anomalies detected.")
        
    st.markdown("#### 🔬 Top Risk Contributing Factors")
    explanation = twin.explanation
    if 'contributions' in explanation and len(explanation['contributions']) > 0:
        top_contribs = explanation['contributions'].head(4)
        for _, row in top_contribs.iterrows():
            st.markdown(f"- **{row['label']}:** `{row['actual_value']}` (+{row['impact_pct']:.0f}% impact)")
    
    st.markdown(f"""
    <div style="background-color: #0f172a; border-left: 4px solid #38bdf8; padding: 12px; border-radius: 6px; margin-top: 14px;">
        <small style="color: #94a3b8;"><strong>AI Summary:</strong> {twin.explanation.get('summary_text', '')}</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Population Risk Radar Overview
st.markdown("### 🌐 Population Risk Radar (All Virtual Twins)")

pop_df = []
for pid in sorted(df_ehr['patient_id'].unique()):
    p_twin = get_or_create_digital_twin(pid)
    pop_df.append({
        'Patient ID': pid,
        'Age': p_twin.demographics['age'],
        'Gender': p_twin.demographics['gender'],
        'BMI': p_twin.demographics['bmi'],
        'Current Glucose (mg/dL)': p_twin.current_vitals['continuous_glucose'],
        '2h Forecast (mg/dL)': p_twin.predicted_outcome.get('predicted_glucose_2h', 0.0),
        'Spike Risk Prob (%)': p_twin.risk_score,
        'Risk Level': p_twin.risk_level,
        'Anomalies': len(p_twin.anomalies)
    })

df_pop = pd.DataFrame(pop_df)

pop_col1, pop_col2 = st.columns([1, 2])

with pop_col1:
    fig_risk_dist = px.pie(
        df_pop, names='Risk Level', title="Patient Risk Stratification",
        color='Risk Level',
        color_discrete_map={'LOW': '#22c55e', 'MODERATE': '#eab308', 'HIGH': '#f97316', 'CRITICAL': '#ef4444'},
        hole=0.4
    )
    fig_risk_dist.update_layout(template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a')
    st.plotly_chart(fig_risk_dist, use_container_width=True)

with pop_col2:
    st.dataframe(
        df_pop.sort_values(by='Spike Risk Prob (%)', ascending=False),
        use_container_width=True,
        hide_index=True
    )
