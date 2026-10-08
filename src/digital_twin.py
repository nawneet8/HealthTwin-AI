"""
Digital Twin Engine for HealthTwin AI.
Maintains continuous state representation of virtual patient and handles live updates,
predictive inference, anomaly detection, counterfactual simulations, and state transitions.
"""

from datetime import datetime
import pandas as pd
import numpy as np

from src.feature_engineering import compute_rolling_features, get_feature_columns
from src.prediction import predict_glucose_spike_risk
from src.forecasting import forecast_glucose_2h
from src.anomaly_detection import detect_biometric_anomalies
from src.explainability import get_shap_explanation

class DigitalTwin:
    """
    Virtual representation of a patient continuously assimilating EHR demographics
    and dynamic wearable time-series telemetry to maintain state and drive predictive analytics.
    """
    def __init__(self, patient_id: str, ehr_data: dict, initial_wearable: dict = None):
        self.patient_id = patient_id
        self.demographics = {
            'age': ehr_data.get('age', 50),
            'gender': ehr_data.get('gender', 'Unknown'),
            'bmi': ehr_data.get('bmi', 25.0)
        }
        self.medical_history = {
            'diabetes_history': ehr_data.get('diabetes_history', 0),
            'hypertension': ehr_data.get('hypertension', 0),
            'cholesterol': ehr_data.get('cholesterol', 200),
            'fasting_glucose': ehr_data.get('fasting_glucose', 100.0),
            'hba1c': ehr_data.get('hba1c', 5.7),
            'family_history': ehr_data.get('family_history', 0),
            'medication': ehr_data.get('medication', 'None'),
            'smoking_status': ehr_data.get('smoking_status', 'Never')
        }
        
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.current_vitals = {}
        self.wearable_state = {}
        self.risk_score = 0.0
        self.risk_level = "LOW"
        self.predicted_outcome = {}
        self.anomalies = []
        self.explanation = {}
        self.state_history = []
        
        if initial_wearable:
            self.update_state(initial_wearable)

    def update_state(self, new_wearable: dict):
        """
        Assimilates incoming wearable streaming telemetry, re-computes feature space,
        updates internal virtual state, and executes pipeline predictions.
        """
        # Save snapshot of previous state for transition visualization
        old_state_snapshot = {
            'timestamp': self.timestamp,
            'continuous_glucose': self.current_vitals.get('continuous_glucose', 100.0),
            'heart_rate': self.current_vitals.get('heart_rate', 70.0),
            'hrv': self.current_vitals.get('hrv', 45.0),
            'stress_level': self.current_vitals.get('stress_level', 30.0),
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'predicted_glucose_2h': self.predicted_outcome.get('predicted_glucose_2h', 100.0)
        }
        
        self.timestamp = new_wearable.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Update current vitals and wearable state
        self.current_vitals = {
            'heart_rate': float(new_wearable.get('heart_rate', 72.0)),
            'hrv': float(new_wearable.get('hrv', 45.0)),
            'spo2': float(new_wearable.get('spo2', 98.0)),
            'continuous_glucose': float(new_wearable.get('continuous_glucose', 110.0))
        }
        
        self.wearable_state = {
            'steps': int(new_wearable.get('steps', 0)),
            'sleep_hours': float(new_wearable.get('sleep_hours', 7.0)),
            'sleep_quality': float(new_wearable.get('sleep_quality', 80.0)),
            'calories': float(new_wearable.get('calories', 2.0)),
            'activity_level': new_wearable.get('activity_level', 'Sedentary'),
            'stress_level': float(new_wearable.get('stress_level', 35.0))
        }
        
        # Build composite feature vector for model ingestion
        combined_dict = {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp,
            **self.demographics,
            **self.medical_history,
            **self.current_vitals,
            **self.wearable_state
        }
        
        df_single = pd.DataFrame([combined_dict])
        df_feat = compute_rolling_features(df_single)
        
        self._current_features_df = df_feat
        
        # Execute Pipeline Operations
        self.calculate_risk()
        self.predict_future()
        self.detect_anomalies()
        self.get_explanation()
        
        # Append transition record
        self.state_history.append({
            'old_state': old_state_snapshot,
            'incoming_telemetry': new_wearable,
            'new_state': self.to_dict()
        })

    def calculate_risk(self):
        """
        Executes binary risk prediction model to determine glucose spike probability.
        """
        risk_res = predict_glucose_spike_risk(self._current_features_df)[0]
        self.risk_score = risk_res['spike_probability']
        self.risk_level = risk_res['risk_level']
        self.risk_color = risk_res['color']
        self.raw_risk_prob = risk_res['raw_probability']

    def predict_future(self):
        """
        Executes time-series forecasting engine for 2-hour glucose level.
        """
        fc = forecast_glucose_2h(self._current_features_df)[0]
        self.predicted_outcome = {
            'current_glucose': fc['current_glucose'],
            'predicted_glucose_2h': fc['predicted_glucose_2h'],
            'predicted_change': fc['predicted_change'],
            'confidence_lower': fc['confidence_lower'],
            'confidence_upper': fc['confidence_upper'],
            'spike_probability': self.risk_score,
            'risk_level': self.risk_level
        }

    def detect_anomalies(self):
        """
        Runs Isolation Forest & clinical rule anomaly detection.
        """
        state_query = {
            **self.current_vitals,
            **self.wearable_state,
            'steps_24h': self._current_features_df.get('steps_24h', pd.Series([5000.0])).iloc[0],
            'glucose_slope': self._current_features_df.get('glucose_slope', pd.Series([0.0])).iloc[0]
        }
        self.anomalies = detect_biometric_anomalies(state_query)

    def get_explanation(self):
        """
        Generates SHAP explainability insights and natural language summary.
        """
        self.explanation = get_shap_explanation(self._current_features_df)

    def simulate(self, parameter_overrides: dict) -> dict:
        """
        Executes counterfactual What-If simulation by perturbing specific parameters
        and re-evaluating risk & 2-hour forecasts against baseline.
        """
        from src.simulation import run_what_if_simulation
        return run_what_if_simulation(self._current_features_df, parameter_overrides)

    def to_dict(self) -> dict:
        """
        Serializes current Digital Twin state into dictionary view.
        """
        return {
            'patient_id': self.patient_id,
            'timestamp': self.timestamp,
            'demographics': self.demographics,
            'medical_history': self.medical_history,
            'current_vitals': self.current_vitals,
            'wearable_state': self.wearable_state,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'risk_color': getattr(self, 'risk_color', 'green'),
            'predicted_outcome': self.predicted_outcome,
            'anomalies': self.anomalies,
            'explanation_summary': self.explanation.get('summary_text', '')
        }
