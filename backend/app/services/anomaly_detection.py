import pandas as pd
import numpy as np


def detect_anomalies(df, window=24, z_threshold=2.0):
    """
    Detects abnormally high energy consumption using rolling Z-Score statistics combined with Weather & 3-Tier Occupancy telemetry.
    Returns flagged anomaly events with severity classifications and root-cause diagnoses.
    """
    cols = ['Energy_kWh', 'Hour', 'DayOfWeek', 'IsWeekend']
    if 'Temperature_C' in df.columns:
        cols.append('Temperature_C')
    if 'Occupancy_Level' in df.columns:
        cols.append('Occupancy_Level')
    if 'Occupancy_Ratio' in df.columns:
        cols.append('Occupancy_Ratio')
        
    data = df[cols].copy()
    
    # 24-hour rolling mean and standard deviation
    data['rolling_mean'] = data['Energy_kWh'].rolling(window=window, min_periods=4).mean()
    data['rolling_std'] = data['Energy_kWh'].rolling(window=window, min_periods=4).std().fillna(0.001)
    
    # Z-Score calculation
    data['z_score'] = (data['Energy_kWh'] - data['rolling_mean']) / data['rolling_std']
    data['z_score'] = data['z_score'].fillna(0)

    # Flag positive anomalies (unusually HIGH consumption)
    anomaly_mask = data['z_score'] >= z_threshold
    anomalies = data[anomaly_mask].copy()

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
        temp = row.get('Temperature_C', 20.0)
        occ_level = row.get('Occupancy_Level', 'Medium')

        if occ_level == 'Low' and actual > expected * 1.3:
            return f"Wasted Energy Alert: High power draw ({round(actual, 2)} kWh) during Low occupancy period. Equipment or HVAC left running unattended."
        elif temp > 28.0 or temp < 10.0:
            return f"Extreme Weather Impact: HVAC thermal load during ambient temperature of {round(temp, 1)}°C."
        elif hour >= 0 and hour <= 5:
            return "Unusual overnight power surge (unattended appliances or baseload leak)."
        elif row['IsWeekend'] == 1:
            return f"Weekend heavy consumption spike ({round(ratio, 1)}x expected baseline)."
        elif hour >= 18 and hour <= 22:
            return f"Peak demand coincidental surge ({round(actual - expected, 2)} kWh above trend)."
        else:
            return f"Sudden power draw burst ({round(actual - expected, 2)} kWh above 24-hour moving average)."

    anomalies['Severity'] = anomalies['z_score'].apply(get_severity)
    anomalies['Possible_Reason'] = anomalies.apply(get_possible_reason, axis=1)

    anomalies = anomalies.reset_index()
    dt_col = 'Datetime' if 'Datetime' in anomalies.columns else anomalies.columns[0]
    anomalies['Timestamp'] = pd.to_datetime(anomalies[dt_col]).dt.strftime('%Y-%m-%d %H:%M')
    anomalies['Actual_kWh'] = anomalies['Energy_kWh'].round(3)
    anomalies['Expected_kWh'] = anomalies['rolling_mean'].round(3)
    anomalies['Z_Score'] = anomalies['z_score'].round(2)
    anomalies['Temperature_C'] = anomalies.get('Temperature_C', 20.0).round(1)
    anomalies['Occupancy_Level'] = anomalies.get('Occupancy_Level', 'Medium')
    anomalies['Occupancy_Ratio'] = anomalies.get('Occupancy_Ratio', 0.5).round(2)

    # Sort descending so newest detected anomalies appear first
    anomalies = anomalies.sort_values(by=dt_col, ascending=False)

    result_cols = ['Timestamp', 'Actual_kWh', 'Expected_kWh', 'Z_Score', 'Severity', 'Temperature_C', 'Occupancy_Level', 'Occupancy_Ratio', 'Possible_Reason']
    
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
