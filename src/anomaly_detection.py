import pandas as pd
import numpy as np


def detect_anomalies(df, window=24, z_threshold=2.0):
    """
    Detects abnormally high energy consumption using rolling Z-Score statistics.
    Returns DataFrame containing flagged anomaly events with severity and potential reasons.
    """
    data = df[['Energy_kWh', 'Hour', 'DayOfWeek', 'IsWeekend']].copy()
    
    # 24-hour rolling mean and standard deviation
    data['rolling_mean'] = data['Energy_kWh'].rolling(window=window, min_periods=4).mean()
    data['rolling_std'] = data['Energy_kWh'].rolling(window=window, min_periods=4).std().fillna(0.001)
    
    # Z-Score calculation
    data['z_score'] = (data['Energy_kWh'] - data['rolling_mean']) / data['rolling_std']
    data['z_score'] = data['z_score'].fillna(0)

    # Flag positive anomalies (unusually HIGH consumption)
    anomaly_mask = data['z_score'] >= z_threshold
    anomalies = data[anomaly_mask].copy()

    # Determine Severity and Possible Reason
    def get_severity(z):
        if z >= 3.5:
            return 'Critical'
        elif z >= 3.0:
            return 'High'
        elif z >= 2.5:
            return 'Medium'
        else:
            return 'Low'

    def get_possible_reason(row):
        hour = row['Hour']
        actual = row['Energy_kWh']
        expected = row['rolling_mean']
        ratio = actual / max(expected, 0.1)

        if hour >= 0 and hour <= 5:
            return "Unusual late-night/overnight power surge (possible heating, cooling, or left-on high power appliance)."
        elif row['IsWeekend'] == 1:
            return f"Weekend heavy consumption spike ({round(ratio, 1)}x expected baseline)."
        elif hour >= 18 and hour <= 22:
            return f"Severe peak-hour usage stack (multiple major appliances running simultaneously)."
        else:
            return f"Sudden power draw burst ({round(actual - expected, 2)} kWh above expected 24-hour trend)."

    anomalies['Severity'] = anomalies['z_score'].apply(get_severity)
    anomalies['Possible_Reason'] = anomalies.apply(get_possible_reason, axis=1)

    # Format result output
    anomalies = anomalies.reset_index()
    anomalies['Timestamp'] = anomalies['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    anomalies['Actual_kWh'] = anomalies['Energy_kWh'].round(3)
    anomalies['Expected_kWh'] = anomalies['rolling_mean'].round(3)
    anomalies['Z_Score'] = anomalies['z_score'].round(2)

    result_cols = ['Timestamp', 'Actual_kWh', 'Expected_kWh', 'Z_Score', 'Severity', 'Possible_Reason']
    
    summary = {
        'total_anomalies': len(anomalies),
        'critical_count': int((anomalies['Severity'] == 'Critical').sum()),
        'high_count': int((anomalies['Severity'] == 'High').sum()),
        'medium_count': int((anomalies['Severity'] == 'Medium').sum()),
        'low_count': int((anomalies['Severity'] == 'Low').sum()),
        'max_z_score': round(float(anomalies['z_score'].max()), 2) if len(anomalies) > 0 else 0.0
    }

    return {
        'summary': summary,
        'anomalies': anomalies[result_cols].to_dict(orient='records')
    }
