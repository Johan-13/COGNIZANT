import pandas as pd
import numpy as np


def analyze_peak_usage(df):
    """
    Analyzes peak energy consumption hours, weather & occupancy correlation,
    and isolates optimal load-shifting opportunities.
    """
    overall_avg_kWh = float(df['Energy_kWh'].mean())

    cols = {'Energy_kWh': ['mean', 'max']}
    if 'Temperature_C' in df.columns:
        cols['Temperature_C'] = 'mean'
    if 'Occupancy_Ratio' in df.columns:
        cols['Occupancy_Ratio'] = 'mean'

    hourly = df.groupby('Hour').agg(cols)
    hourly.columns = ['_'.join(c).strip('_') for c in hourly.columns]
    hourly = hourly.reset_index()

    hourly['Energy_kWh_mean'] = hourly['Energy_kWh_mean'].round(3)
    hourly['Energy_kWh_max'] = hourly['Energy_kWh_max'].round(3)
    hourly['pct_above_avg'] = (((hourly['Energy_kWh_mean'] - overall_avg_kWh) / overall_avg_kWh) * 100).round(1)

    top_peak_hours = hourly.sort_values(by='Energy_kWh_mean', ascending=False).head(8).copy()
    top_peak_hours['HourLabel'] = top_peak_hours['Hour'].apply(lambda h: f"{h:02d}:00 - {(h+1)%24:02d}:00")

    evening_peak = hourly[(hourly['Hour'] >= 17) & (hourly['Hour'] <= 22)]['Energy_kWh_mean'].mean()
    morning_peak = hourly[(hourly['Hour'] >= 7) & (hourly['Hour'] <= 10)]['Energy_kWh_mean'].mean()
    overnight_base = hourly[(hourly['Hour'] >= 0) & (hourly['Hour'] <= 5)]['Energy_kWh_mean'].mean()

    # Load shifting window recommendations
    peak_hours_list = hourly[hourly['pct_above_avg'] > 15]['Hour'].tolist()
    offpeak_hours_list = hourly[hourly['pct_above_avg'] < -10]['Hour'].tolist()

    # Calculate shiftable load (kWh per day)
    peak_hours_count = len(peak_hours_list)
    daily_peak_kWh = df[df['Hour'].isin(peak_hours_list)]['Energy_kWh'].resample('D').sum().mean()
    shiftable_kWh_daily = round(float(daily_peak_kWh * 0.20), 2)  # 20% shift target

    return {
        'overall_avg_kWh': round(overall_avg_kWh, 3),
        'hourly_breakdown': hourly.to_dict(orient='records'),
        'top_peak_hours': top_peak_hours.to_dict(orient='records'),
        'period_averages': {
            'morning_peak_7_to_10_kWh': round(float(morning_peak), 3),
            'evening_peak_17_to_22_kWh': round(float(evening_peak), 3),
            'overnight_baseline_0_to_5_kWh': round(float(overnight_base), 3)
        },
        'load_shifting_opportunity': {
            'peak_windows': [f"{h:02d}:00" for h in peak_hours_list],
            'offpeak_windows': [f"{h:02d}:00" for h in offpeak_hours_list],
            'daily_shiftable_kWh': shiftable_kWh_daily,
            'recommended_strategy': f"Shift approximately {shiftable_kWh_daily} kWh of flexible equipment load (cooling pre-chill, thermal storage, battery charging) from peak window ({peak_hours_list[0] if peak_hours_list else 18}:00-{peak_hours_list[-1] if peak_hours_list else 21}:00) to off-peak overnight hours."
        }
    }
