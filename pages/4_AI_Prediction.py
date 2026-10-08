import sys
import os

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Prediction | HealthTwin AI", page_icon="🔮", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="AI Predictive Risk & Glucose Forecasting Engine", subtitle="Machine Learning Classification & Time-Series 2-Hour Trajectory Prediction")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

st.markdown(f"### 🔮 Predictive Inference Summary: `{selected_pid}`")

p_outcome = twin.predicted_outcome
curr_g = twin.current_vitals['continuous_glucose']
pred_g = p_outcome.get('predicted_glucose_2h', curr_g)
change_g = p_outcome.get('predicted_change', 0.0)
prob_spike = twin.risk_score
risk_lvl = twin.risk_level
risk_color = twin.risk_color

# Primary Risk Display Card
card_col1, card_col2, card_col3, card_col4 = st.columns(4)

with card_col1:
    st.markdown(f"""
    <div class="health-card" style="border-left: 5px solid #38bdf8;">
        <div class="card-label">Current Glucose Level</div>
        <div class="card-value">{curr_g:.1f}</div>
        <div class="card-unit">mg/dL</div>
    </div>
    """, unsafe_allow_html=True)

with card_col2:
    st.markdown(f"""
    <div class="health-card" style="border-left: 5px solid #f97316;">
        <div class="card-label">Predicted in 2 Hours</div>
        <div class="card-value">{pred_g:.1f}</div>
        <div class="card-unit">mg/dL (Delta: {change_g:+.1f})</div>
    </div>
    """, unsafe_allow_html=True)

with card_col3:
    st.markdown(f"""
    <div class="health-card" style="border-left: 5px solid {risk_color};">
        <div class="card-label">Spike Probability</div>
        <div class="card-value" style="color:{risk_color};">{prob_spike:.1f}%</div>
        <div class="card-unit">XGBoost Class Prob</div>
    </div>
    """, unsafe_allow_html=True)

with card_col4:
    st.markdown(f"""
    <div class="health-card" style="border-left: 5px solid {risk_color};">
        <div class="card-label">Risk Category</div>
        <div class="card-value" style="color:{risk_color};">{risk_lvl}</div>
        <div class="card-unit">Stratum</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### 📉 2-Hour Forward Glucose Trajectory Projection")
    
    # Generate past 2 hours + 2 hours forecast
    p_data = df_wearable[df_wearable['patient_id'] == selected_pid].sort_values('timestamp').tail(16).copy()
    p_data['timestamp'] = pd.to_datetime(p_data['timestamp'])
    
    last_ts = p_data['timestamp'].iloc[-1]
    future_ts = last_ts + pd.Timedelta(hours=2)
    
    fig = go.Figure()
    
    # Past 2 hours
    fig.add_trace(go.Scatter(
        x=p_data['timestamp'], y=p_data['continuous_glucose'],
        mode='lines+markers', name='Observed CGM',
        line=dict(color='#38bdf8', width=3)
    ))
    
    # 2-hour forecast line
    fig.add_trace(go.Scatter(
        x=[last_ts, future_ts],
        y=[curr_g, pred_g],
        mode='lines+markers', name='Predicted 2h Glucose',
        line=dict(color=risk_color, width=4, dash='dash')
    ))
    
    # Confidence Interval Shading
    conf_low = p_outcome.get('confidence_lower', pred_g - 12.0)
    conf_high = p_outcome.get('confidence_upper', pred_g + 12.0)
    
    fig.add_trace(go.Scatter(
        x=[future_ts, future_ts],
        y=[conf_low, conf_high],
        mode='lines',
        name='95% Confidence Interval',
        line=dict(color='#94a3b8', width=6)
    ))
    
    fig.add_hline(y=180, line_dash="dash", line_color="#ef4444", annotation_text="Spike Threshold (180 mg/dL)")
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a',
        xaxis_title="Time", yaxis_title="Glucose Concentration (mg/dL)",
        margin=dict(l=40, r=40, t=30, b=30), height=380
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("#### 🎛️ Clinical Decision Sensitivity Tuning")
    st.write("Adjust decision probability threshold for glucose spike early warning trigger:")
    
    threshold = st.slider("Trigger Decision Threshold (%):", min_value=10, max_value=90, value=45, step=5)
    
    triggered = prob_spike >= threshold
    
    st.markdown(f"""
    <div style="background-color: #0f172a; border: 1px solid #334155; padding: 18px; border-radius: 8px;">
        <p><strong>Configured Threshold:</strong> {threshold}%</p>
        <p><strong>Model Calculated Probability:</strong> {prob_spike:.1f}%</p>
        <p><strong>Alert Trigger Status:</strong> {'<span style="color:#ef4444; font-weight:bold;">🚨 SPIKE ALERT TRIGGERED</span>' if triggered else '<span style="color:#22c55e; font-weight:bold;">✅ NO ALERT TRIGGERED</span>'}</p>
        <hr style="border-color:#334155;">
        <small style="color: #94a3b8;">* Lowering decision thresholds increases sensitivity (Recall), prioritizing early clinical detection over precision.</small>
    </div>
    """, unsafe_allow_html=True)
