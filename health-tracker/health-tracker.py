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
    summary = {
        "heart_rate_avg": data["heart_rate"].mean(),
        "heart_rate_min": data["heart_rate"].min(),
        "heart_rate_max": data["heart_rate"].max(),
        "oxygen_level_avg": data["oxygen_level"].mean(),
        "oxygen_level_min": data["oxygen_level"].min(),
        "oxygen_level_max": data["oxygen_level"].max(),
        "systolic_bp_avg": data["systolic_bp"].mean(),
        "systolic_bp_min": data["systolic_bp"].min(),
        "systolic_bp_max": data["systolic_bp"].max(),
        "diastolic_bp_avg": data["diastolic_bp"].mean(),
        "diastolic_bp_min": data["diastolic_bp"].min(),
        "diastolic_bp_max": data["diastolic_bp"].max(),
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
