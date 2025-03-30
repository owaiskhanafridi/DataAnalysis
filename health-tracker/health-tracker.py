import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import datetime as dt

# Step 2: Load and Analyze Data
def analyze_health_data(filename='health_data.csv'):
    data = pd.read_csv(filename)
    
    # Prepare data for anomaly detection
    features = data[['heart_rate', 'oxygen_level', 'systolic_bp', 'diastolic_bp']]
    
    # Isolation Forest for anomaly detection
    model = IsolationForest(contamination=0.05)  # Assume 5% anomalies
    data['anomaly'] = model.fit_predict(features)
    
    # Mark anomalies (-1 is an anomaly in Isolation Forest)
    anomalies = data[data['anomaly'] == -1]
    
    # Step 3: Generate Summary
    heart_rate_stats = data["heart_rate"].agg(['mean', 'min', 'max'])
    oxygen_level_stats = data["oxygen_level"].agg(['mean', 'min', 'max'])
    systolic_bp_stats = data["systolic_bp"].agg(['mean', 'min', 'max'])
    diastolic_bp_stats = data["diastolic_bp"].agg(['mean', 'min', 'max'])

    summary = {
    "heart_rate_avg": heart_rate_stats['mean'],
    "heart_rate_min": heart_rate_stats['min'],
    "heart_rate_max": heart_rate_stats['max'],
    "oxygen_level_avg": oxygen_level_stats['mean'],
    "oxygen_level_min": oxygen_level_stats['min'],
    "oxygen_level_max": oxygen_level_stats['max'],
    "systolic_bp_avg": systolic_bp_stats['mean'],
    "systolic_bp_min": systolic_bp_stats['min'],
    "systolic_bp_max": systolic_bp_stats['max'],
    "diastolic_bp_avg": diastolic_bp_stats['mean'],
    "diastolic_bp_min": diastolic_bp_stats['min'],
    "diastolic_bp_max": diastolic_bp_stats['max'],
    "total_anomalies": len(anomalies),
    "anomaly_instances": anomalies[['timestamp', 'heart_rate', 'oxygen_level', 'systolic_bp', 'diastolic_bp']].values.tolist()
}

    # Display Summary
    print("Health Data Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save anomalies to a separate file if needed
    anomalies.to_csv('anomalies.csv', index=False)
    print("Anomalies saved to anomalies.csv")

# Generate dummy data and analyze it
analyze_health_data()
