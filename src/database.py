"""
Database Access Layer for HealthTwin AI.
Uses SQLite for local persistence with clean abstraction enabling seamless migration to PostgreSQL.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "healthtwin.db")

def get_connection():
    """
    Establishes connection to SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes SQL database tables for patients, EHR, wearables, predictions, anomalies, and simulations.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Patients Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. EHR Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ehr_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        age INTEGER,
        gender TEXT,
        bmi REAL,
        diabetes_history INTEGER,
        hypertension INTEGER,
        cholesterol INTEGER,
        fasting_glucose REAL,
        hba1c REAL,
        family_history INTEGER,
        medication TEXT,
        smoking_status TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """)
    
    # 3. Wearable Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS wearable_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        timestamp TEXT,
        heart_rate REAL,
        hrv REAL,
        steps INTEGER,
        sleep_hours REAL,
        sleep_quality REAL,
        spo2 REAL,
        calories REAL,
        activity_level TEXT,
        stress_level REAL,
        continuous_glucose REAL,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """)
    
    # 4. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        timestamp TEXT,
        current_glucose REAL,
        predicted_glucose_2h REAL,
        spike_probability REAL,
        risk_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """)
    
    # 5. Anomalies Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        timestamp TEXT,
        anomaly_type TEXT,
        severity TEXT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """)
    
    # 6. Simulation Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        baseline_risk REAL,
        simulated_risk REAL,
        risk_reduction REAL,
        parameters_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    );
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DB_PATH)

def sync_dataframes_to_db(df_ehr: pd.DataFrame, df_wearable: pd.DataFrame):
    """
    Populates SQLite database with generated or loaded EHR and wearable DataFrames.
    """
    init_db()
    conn = get_connection()
    
    # Patients table
    patients_df = pd.DataFrame({'patient_id': df_ehr['patient_id'].unique()})
    patients_df.to_sql('patients', conn, if_exists='replace', index=False)
    
    # EHR table
    df_ehr.to_sql('ehr_records', conn, if_exists='replace', index=False)
    
    # Wearables table
    df_wearable.to_sql('wearable_records', conn, if_exists='replace', index=False)
    
    conn.close()
    print("DataFrame records synced to database.")

def log_prediction(patient_id: str, timestamp: str, current_g: float, pred_g: float, prob: float, risk_lvl: str):
    """
    Logs an inference prediction into the predictions table.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (patient_id, timestamp, current_glucose, predicted_glucose_2h, spike_probability, risk_level)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (patient_id, timestamp, current_g, pred_g, prob, risk_lvl))
    conn.commit()
    conn.close()
