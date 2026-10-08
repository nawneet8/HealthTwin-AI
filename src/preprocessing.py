"""
Data Preprocessing and Pipeline Validation for HealthTwin AI.
Handles schema validation, missing data imputation, outlier trimming, and dataset merging.
"""

import pandas as pd
import numpy as np

EHR_REQUIRED_COLUMNS = [
    'patient_id', 'age', 'gender', 'bmi', 'diabetes_history',
    'hypertension', 'cholesterol', 'fasting_glucose', 'hba1c',
    'family_history', 'medication', 'smoking_status'
]

WEARABLE_REQUIRED_COLUMNS = [
    'patient_id', 'timestamp', 'heart_rate', 'hrv', 'steps',
    'sleep_hours', 'sleep_quality', 'spo2', 'calories',
    'activity_level', 'stress_level', 'continuous_glucose'
]

def validate_schemas(df_ehr: pd.DataFrame, df_wearable: pd.DataFrame) -> bool:
    """
    Validate that incoming DataFrames conform to expected clinical schemas.
    """
    missing_ehr = set(EHR_REQUIRED_COLUMNS) - set(df_ehr.columns)
    missing_wearable = set(WEARABLE_REQUIRED_COLUMNS) - set(df_wearable.columns)
    
    if missing_ehr:
        raise ValueError(f"EHR data missing required columns: {missing_ehr}")
    if missing_wearable:
        raise ValueError(f"Wearable data missing required columns: {missing_wearable}")
        
    return True

def clean_ehr_data(df_ehr: pd.DataFrame) -> pd.DataFrame:
    """
    Clean EHR data: remove duplicates, impute missing values, clip extreme outliers.
    """
    df = df_ehr.copy()
    df = df.drop_duplicates(subset=['patient_id'])
    
    # Impute missing numericals with median
    num_cols = ['age', 'bmi', 'cholesterol', 'fasting_glucose', 'hba1c']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # Impute categoricals with mode
    cat_cols = ['gender', 'medication', 'smoking_status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Clip physiological limits
    df['age'] = df['age'].clip(18, 110)
    df['bmi'] = df['bmi'].clip(15.0, 60.0)
    df['fasting_glucose'] = df['fasting_glucose'].clip(50.0, 400.0)
    df['hba1c'] = df['hba1c'].clip(4.0, 16.0)
    
    return df

def clean_wearable_data(df_wearable: pd.DataFrame) -> pd.DataFrame:
    """
    Clean wearable time-series data: normalize timestamps, remove duplicates,
    impute forward/backward, and handle biometric outliers.
    """
    df = df_wearable.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop duplicates per patient and timestamp
    df = df.drop_duplicates(subset=['patient_id', 'timestamp'])
    df = df.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
    
    # Clip extreme physiological outlier values
    df['heart_rate'] = df['heart_rate'].clip(40.0, 210.0)
    df['hrv'] = df['hrv'].clip(5.0, 200.0)
    df['spo2'] = df['spo2'].clip(80.0, 100.0)
    df['continuous_glucose'] = df['continuous_glucose'].clip(40.0, 450.0)
    df['stress_level'] = df['stress_level'].clip(0.0, 100.0)
    
    # Forward fill missing physiological signals within each patient block
    physio_cols = ['heart_rate', 'hrv', 'steps', 'sleep_hours', 'sleep_quality', 'spo2', 'calories', 'stress_level', 'continuous_glucose']
    df[physio_cols] = df.groupby('patient_id')[physio_cols].transform(lambda group: group.ffill().bfill())
    
    return df

def merge_ehr_and_wearable(df_ehr: pd.DataFrame, df_wearable: pd.DataFrame) -> pd.DataFrame:
    """
    Merge static EHR clinical demographics with dynamic wearable time-series data.
    """
    df_ehr_clean = clean_ehr_data(df_ehr)
    df_wearable_clean = clean_wearable_data(df_wearable)
    
    validate_schemas(df_ehr_clean, df_wearable_clean)
    
    df_merged = pd.merge(df_wearable_clean, df_ehr_clean, on='patient_id', how='left')
    return df_merged
