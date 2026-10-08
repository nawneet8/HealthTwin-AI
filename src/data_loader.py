"""
Data Loader and Synthetic Healthcare Data Generator for HealthTwin AI.
Generates realistic, statistically correlated EHR and wearable time-series datasets.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def generate_synthetic_data(num_patients=30, days=5, freq_minutes=15, seed=42):
    """
    Generate realistic synthetic EHR, Wearable time-series, and Glucose Event datasets
    with physiologically grounded correlations.
    """
    np.random.seed(seed)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    ehr_records = []
    wearable_records = []
    event_records = []
    
    genders = ['Male', 'Female']
    medications = ['None', 'Metformin', 'Sulfonylurea', 'Insulin', 'Combination']
    smoking_statuses = ['Never', 'Former', 'Current']
    
    start_base_date = datetime(2026, 10, 1, 0, 0, 0)
    timestamps = [start_base_date + timedelta(minutes=i * freq_minutes) for i in range((24 * 60 // freq_minutes) * days)]
    
    for pid in range(101, 101 + num_patients):
        patient_id = f"PAT-{pid}"
        
        # EHR Demographics & Baseline Clinical Metrics
        age = int(np.random.normal(54, 12))
        age = max(25, min(82, age))
        gender = np.random.choice(genders, p=[0.52, 0.48])
        
        # Risk factor baseline weighting
        has_diabetes = np.random.choice([0, 1], p=[0.4, 0.6])
        family_hist = np.random.choice([0, 1], p=[0.45, 0.55])
        
        # BMI correlated with diabetes history & age
        bmi_mean = 31.5 if has_diabetes else 26.2
        bmi = float(np.round(np.random.normal(bmi_mean, 4.5), 1))
        bmi = max(18.5, min(48.0, bmi))
        
        hypertension = 1 if (bmi > 30 or age > 60 or np.random.rand() < 0.35) else 0
        cholesterol = int(np.random.normal(210 if hypertension else 185, 30))
        
        fasting_glucose = float(np.round(np.random.normal(135 if has_diabetes else 95, 20), 1))
        fasting_glucose = max(70.0, min(240.0, fasting_glucose))
        
        hba1c = float(np.round(fasting_glucose / 20.0 + np.random.normal(0.2, 0.3), 1))
        hba1c = max(4.8, min(11.5, hba1c))
        
        med = np.random.choice(medications, p=[0.2, 0.4, 0.2, 0.1, 0.1] if has_diabetes else [0.85, 0.1, 0.03, 0.01, 0.01])
        smoking = np.random.choice(smoking_statuses, p=[0.6, 0.25, 0.15])
        
        ehr_records.append({
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'diabetes_history': has_diabetes,
            'hypertension': hypertension,
            'cholesterol': cholesterol,
            'fasting_glucose': fasting_glucose,
            'hba1c': hba1c,
            'family_history': family_hist,
            'medication': med,
            'smoking_status': smoking
        })
        
        # Wearable time-series generation
        # Patient specific baselines
        base_sleep = float(np.round(np.random.normal(6.8, 1.2), 1))
        base_sleep = max(4.0, min(9.5, base_sleep))
        sleep_quality = float(np.round(max(40, min(98, 75 + (base_sleep - 7.5) * 10 + np.random.normal(0, 5))), 1))
        
        base_hr = float(np.round(np.random.normal(72 + (bmi - 25) * 0.8, 6), 1))
        base_hrv = float(np.round(max(15, 65 - (age * 0.4) - (bmi * 0.5) + np.random.normal(0, 5)), 1))
        
        # Generate temporal trajectory
        c_glucose = fasting_glucose
        step_accumulator = 0
        
        for idx, ts in enumerate(timestamps):
            hour = ts.hour
            minute = ts.minute
            
            # Circadian rhythm & sleep pattern
            is_sleeping = 23 <= hour or hour < 6
            
            if is_sleeping:
                hr = max(48.0, base_hr - 12 + np.random.normal(0, 2))
                hrv = max(10.0, base_hrv + 12 + np.random.normal(0, 3))
                steps = 0
                spo2 = float(np.round(np.random.normal(97.8, 0.8), 1))
                activity = 'Resting'
                stress = max(5.0, np.random.normal(15, 5))
                cals = float(np.round(np.random.normal(1.2, 0.2), 2))
            else:
                # Meals effect at 8 AM (Breakfast), 1 PM (Lunch), 7 PM (Dinner)
                is_meal_time = (hour == 8 and minute <= 30) or (hour == 13 and minute <= 30) or (hour == 19 and minute <= 30)
                is_post_meal = (hour == 9 or hour == 14 or hour == 20)
                
                # Activity bursts around commute / exercise
                is_active = (hour == 7 and minute >= 30) or (hour == 17 and minute <= 45) or (np.random.rand() < 0.1)
                
                if is_active:
                    steps = int(np.random.randint(120, 350))
                    hr = base_hr + np.random.normal(25, 8)
                    hrv = max(10.0, base_hrv - 15 + np.random.normal(0, 4))
                    activity = 'Active'
                    stress = float(np.round(np.random.normal(45, 10), 1))
                    cals = float(np.round(np.random.normal(6.5, 1.5), 2))
                else:
                    steps = int(np.random.randint(0, 40))
                    hr = base_hr + np.random.normal(4, 4)
                    hrv = base_hrv + np.random.normal(0, 4)
                    activity = 'Sedentary' if steps < 10 else 'Light'
                    stress = float(np.round(np.random.normal(35, 12), 1))
                    cals = float(np.round(np.random.normal(2.1, 0.4), 2))
                
                spo2 = float(np.round(max(92.0, min(100.0, np.random.normal(97.2, 1.0))), 1))
            
            step_accumulator += steps
            
            # Glucose dynamics (Physiological modeling)
            # Meal shock + Poor sleep penalty + High stress penalty - Physical activity reduction
            sleep_penalty = (7.5 - base_sleep) * 3.5 if base_sleep < 7.0 else 0
            stress_impact = (stress - 30) * 0.15 if stress > 30 else 0
            bmi_impact = (bmi - 25.0) * 0.2
            
            glucose_drift = 0.0
            if (hour == 8 or hour == 13 or hour == 19) and minute == 15:
                # Meal intake spike trigger
                meal_carb_magnitude = np.random.normal(45 if has_diabetes else 25, 10)
                glucose_drift += meal_carb_magnitude
            elif (hour in [9, 10, 14, 15, 20, 21]):
                # Postprandial elevation tail
                glucose_drift += np.random.normal(4, 2)
            else:
                # Return toward baseline
                glucose_drift -= (c_glucose - fasting_glucose) * 0.08
            
            # Dampened by physical activity
            activity_mitigation = (steps / 100.0) * 2.5
            c_glucose = c_glucose + glucose_drift + sleep_penalty * 0.05 + stress_impact * 0.1 + bmi_impact * 0.05 - activity_mitigation + np.random.normal(0, 2)
            c_glucose = float(np.round(max(65.0, min(320.0, c_glucose)), 1))
            
            wearable_records.append({
                'patient_id': patient_id,
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate': float(np.round(hr, 1)),
                'hrv': float(np.round(hrv, 1)),
                'steps': steps,
                'sleep_hours': base_sleep,
                'sleep_quality': sleep_quality,
                'spo2': spo2,
                'calories': cals,
                'activity_level': activity,
                'stress_level': float(np.round(max(0, min(100, stress)), 1)),
                'continuous_glucose': c_glucose
            })

    df_ehr = pd.DataFrame(ehr_records)
    df_wearable = pd.DataFrame(wearable_records)
    df_wearable['timestamp'] = pd.to_datetime(df_wearable['timestamp'])
    
    # Calculate 2-hour future glucose events per patient
    # 2 hours in 15-min intervals = 8 steps lookahead
    steps_lookahead = 8
    
    for pid, group in df_wearable.groupby('patient_id'):
        group = group.sort_values('timestamp').copy()
        group['future_glucose_2h'] = group['continuous_glucose'].shift(-steps_lookahead)
        group['spike_magnitude'] = group['future_glucose_2h'] - group['continuous_glucose']
        # Glucose spike defined as reaching >= 175 mg/dL OR an increase >= 35 mg/dL within 2 hours
        group['glucose_spike'] = ((group['future_glucose_2h'] >= 175.0) | (group['spike_magnitude'] >= 35.0)).astype(int)
        
        # Drop rows where future_glucose_2h is NaN (end of time series)
        group_clean = group.dropna(subset=['future_glucose_2h'])
        event_records.append(group_clean)
        
    df_events_wearable = pd.concat(event_records, ignore_index=True)
    df_events = pd.merge(df_events_wearable, df_ehr, on='patient_id', how='left')
    
    df_wearable['timestamp'] = df_wearable['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save CSV files
    df_ehr.to_csv(os.path.join(DATA_DIR, 'ehr_data.csv'), index=False)
    df_wearable.to_csv(os.path.join(DATA_DIR, 'wearable_data.csv'), index=False)
    df_events.to_csv(os.path.join(DATA_DIR, 'glucose_events.csv'), index=False)
    
    print(f"Synthetic data successfully generated in {DATA_DIR}:")
    print(f" - ehr_data.csv: {len(df_ehr)} patients")
    print(f" - wearable_data.csv: {len(df_wearable)} timestamp records")
    print(f" - glucose_events.csv: {len(df_events)} event records")
    
    return df_ehr, df_wearable, df_events

def load_all_data():
    """
    Loads EHR, Wearable, and Glucose Events data.
    Automatically generates synthetic dataset if files are missing.
    """
    ehr_path = os.path.join(DATA_DIR, 'ehr_data.csv')
    wearable_path = os.path.join(DATA_DIR, 'wearable_data.csv')
    events_path = os.path.join(DATA_DIR, 'glucose_events.csv')
    
    if not (os.path.exists(ehr_path) and os.path.exists(wearable_path) and os.path.exists(events_path)):
        return generate_synthetic_data()
        
    df_ehr = pd.read_csv(ehr_path)
    df_wearable = pd.read_csv(wearable_path)
    df_events = pd.read_csv(events_path)
    
    return df_ehr, df_wearable, df_events

if __name__ == "__main__":
    generate_synthetic_data()
