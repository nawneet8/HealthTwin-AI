"""
UI Components and Theme Styling Helper for HealthTwin AI Streamlit Dashboard.
Provides modern, polished healthcare component blocks, CSS styling, and disclaimer banners.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_all_data
from src.preprocessing import merge_ehr_and_wearable
from src.feature_engineering import compute_rolling_features
from src.digital_twin import DigitalTwin

def inject_custom_css():
    """
    Injects custom healthcare theme styling, glassmorphism card containers,
    and responsive font styles.
    """
    css = """
    <style>
    /* Main Theme Overrides */
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Disclaimer Bar */
    .disclaimer-bar {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 8px;
        padding: 10px 16px;
        color: #fbbf24;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Healthcare Metric Card */
    .health-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    .card-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
    }
    .card-unit {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 400;
    }
    
    /* Risk Badges */
    .risk-badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-moderate {
        background-color: rgba(234, 179, 8, 0.2);
        color: #facc15;
        border: 1px solid #eab308;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #fb923c;
        border: 1px solid #f97316;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_disclaimer():
    """
    Renders research & PoC disclaimer callout.
    """
    st.markdown("""
    <div class="disclaimer-bar">
        <span>⚠️ <strong>RESEARCH PROOF-OF-CONCEPT DISCLAIMER:</strong> HealthTwin AI is designed for research, educational, and computational demonstration purposes only. It does not provide medical diagnoses, treatment advice, or clinical predictions for live medical decisions. All patient data is synthetic.</span>
    </div>
    """, unsafe_allow_html=True)

def render_header(title="HealthTwin AI", subtitle="An Intelligent Virtual Patient Digital Twin for Predictive Healthcare"):
    """
    Renders top application header banner.
    """
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">🩺 {title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def get_cached_dataset():
    """
    Loads and caches EHR, Wearables, and Glucose Events dataframes.
    """
    df_ehr, df_wearable, df_events = load_all_data()
    df_eng = compute_rolling_features(df_events)
    return df_ehr, df_wearable, df_events, df_eng

def get_or_create_digital_twin(patient_id: str) -> DigitalTwin:
    """
    Instantiates or retrieves a cached DigitalTwin object for a specified patient_id.
    """
    df_ehr, df_wearable, df_events, df_eng = get_cached_dataset()
    
    # Filter for selected patient
    patient_ehr = df_ehr[df_ehr['patient_id'] == patient_id].iloc[0].to_dict()
    patient_wearables = df_wearable[df_wearable['patient_id'] == patient_id].sort_values('timestamp')
    
    latest_wearable = patient_wearables.iloc[-1].to_dict()
    
    # Instantiate Digital Twin
    twin = DigitalTwin(patient_id=patient_id, ehr_data=patient_ehr, initial_wearable=latest_wearable)
    return twin

def render_patient_sidebar_selector():
    """
    Render global patient selection sidebar widget.
    """
    df_ehr, _, _, _ = get_cached_dataset()
    patient_ids = sorted(df_ehr['patient_id'].unique())
    
    st.sidebar.markdown("### 👤 Select Virtual Patient")
    selected_pid = st.sidebar.selectbox("Patient ID", patient_ids, index=0)
    
    p_info = df_ehr[df_ehr['patient_id'] == selected_pid].iloc[0]
    
    st.sidebar.markdown(f"""
    **Demographics & Clinical Profile:**
    - **Age / Gender:** {p_info['age']} yrs | {p_info['gender']}
    - **BMI:** {p_info['bmi']} kg/m²
    - **T2D History:** {'Yes' if p_info['diabetes_history']==1 else 'No'}
    - **Fasting Glucose:** {p_info['fasting_glucose']} mg/dL
    - **HbA1c:** {p_info['hba1c']}%
    - **Medication:** {p_info['medication']}
    """)
    
    return selected_pid
