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

st.set_page_config(page_title="Explainable AI | HealthTwin AI", page_icon="🧠", layout="wide")

from src.ui_components import (
    inject_custom_css, render_header, render_disclaimer,
    get_cached_dataset, render_patient_sidebar_selector, get_or_create_digital_twin
)

inject_custom_css()
render_header(title="Explainable AI (SHAP) & Clinical Transparency", subtitle="Quantifying Feature Contributions & Explainable Machine Learning Decision Logic")
render_disclaimer()

selected_pid = render_patient_sidebar_selector()
df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()

twin = get_or_create_digital_twin(selected_pid)

st.markdown(f"### 🧠 SHAP Risk Feature Attribution: `{selected_pid}`")

explanation = twin.explanation
summary_text = explanation.get('summary_text', '')
df_contribs = explanation.get('contributions', pd.DataFrame())

# Render Natural Language Clinician Summary Card
st.markdown(f"""
<div style="background-color: #0f172a; border-left: 5px solid #38bdf8; padding: 20px; border-radius: 10px; margin-bottom: 24px;">
    <h4 style="color: #38bdf8; margin-top:0;">💬 Natural Language Clinical Interpretation</h4>
    <p style="color: #f8fafc; font-size: 1.1rem; line-height: 1.6; margin: 0;">"{summary_text}"</p>
</div>
""", unsafe_allow_html=True)

col_x1, col_x2 = st.columns([3, 2])

with col_x1:
    st.markdown("#### 📊 Top Feature Impact Breakdown (SHAP Values)")
    
    if len(df_contribs) > 0:
        top_df = df_contribs.head(10).sort_values(by='shap_value', ascending=True)
        
        colors = ['#ef4444' if x > 0 else '#22c55e' for x in top_df['shap_value']]
        
        fig = go.Figure(go.Bar(
            x=top_df['shap_value'],
            y=top_df['label'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f} ({p:.0f}%)" for v, p in zip(top_df['shap_value'], top_df['impact_pct'])],
            textposition='auto'
        ))
        
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#1e293b', plot_bgcolor='#0f172a',
            xaxis_title="SHAP Value (Impact on Log-Odds of Glucose Spike)",
            yaxis_title="Feature",
            margin=dict(l=40, r=40, t=30, b=30), height=420
        )
        st.plotly_chart(fig, use_container_width=True)

with col_x2:
    st.markdown("#### 📋 Feature Attribution Matrix")
    if len(df_contribs) > 0:
        display_df = df_contribs[['label', 'actual_value', 'shap_value', 'impact_pct']].head(8).copy()
        display_df.columns = ['Feature Label', 'Observed Value', 'SHAP Value', 'Impact (%)']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    st.markdown("""
    <div style="background-color: #1e293b; border: 1px solid #334155; padding: 14px; border-radius: 8px; margin-top: 14px;">
        <small style="color: #94a3b8;">
        <strong>Understanding SHAP Values:</strong><br>
        - <span style="color: #ef4444;">Red Bars (Positive SHAP):</span> Increase predicted risk of glucose spike.<br>
        - <span style="color: #22c55e;">Green Bars (Negative SHAP):</span> Protective factors lowering predicted glucose risk.
        </small>
    </div>
    """, unsafe_allow_html=True)
