import pandas as pd
import numpy as np


def analyze_peak_usage(df):
    """
    Analyzes peak energy consumption hours and compares against overall average baseline.
    """
    overall_avg_kWh = float(df['Energy_kWh'].mean())

    # Hourly distribution (0-23)
    hourly = df.groupby('Hour')['Energy_kWh'].agg(['mean', 'max', 'count']).reset_index()
    hourly['mean'] = hourly['mean'].round(3)
    hourly['max'] = hourly['max'].round(3)
    
    # Calculate percentage increase over average
    hourly['pct_above_avg'] = (((hourly['mean'] - overall_avg_kWh) / overall_avg_kWh) * 100).round(1)

    # Sort to find top peak hours
    top_peak_hours = hourly.sort_values(by='mean', ascending=False).head(10).copy()
    top_peak_hours['HourLabel'] = top_peak_hours['Hour'].apply(lambda h: f"{h:02d}:00 - {h+1:02d}:00")

    # High load window detection (consecutive hours with highest load)
    evening_peak = hourly[(hourly['Hour'] >= 17) & (hourly['Hour'] <= 22)]['mean'].mean()
    morning_peak = hourly[(hourly['Hour'] >= 7) & (hourly['Hour'] <= 10)]['mean'].mean()
    overnight_base = hourly[(hourly['Hour'] >= 0) & (hourly['Hour'] <= 5)]['mean'].mean()

    return {
        'overall_avg_kWh': round(overall_avg_kWh, 3),
        'hourly_breakdown': hourly.to_dict(orient='records'),
        'top_peak_hours': top_peak_hours[['Hour', 'HourLabel', 'mean', 'max', 'pct_above_avg']].to_dict(orient='records'),
        'period_averages': {
            'morning_peak_7_to_10_kWh': round(float(morning_peak), 3),
            'evening_peak_17_to_22_kWh': round(float(evening_peak), 3),
            'overnight_baseline_0_to_5_kWh': round(float(overnight_base), 3)
        }
    }
