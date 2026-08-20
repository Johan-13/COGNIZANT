import pandas as pd
import numpy as np


def get_summary_metrics(df):
    """
    Computes overall key metrics for the dataset including energy, real weather, and 3-tier occupancy.
    """
    total_kWh = float(df['Energy_kWh'].sum())
    avg_hourly_kWh = float(df['Energy_kWh'].mean())
    avg_daily_kWh = float(df['Energy_kWh'].resample('D').sum().mean())
    max_hourly_kWh = float(df['Energy_kWh'].max())
    min_hourly_kWh = float(df['Energy_kWh'].min())
    peak_hour = int(df.groupby('Hour')['Energy_kWh'].mean().idxmax())
    
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    total_days = int((df.index.max() - df.index.min()).days) + 1

    # Weather & Occupancy summary metrics
    avg_temp = float(df['Temperature_C'].mean()) if 'Temperature_C' in df.columns else 20.0
    avg_humidity = float(df['Humidity_pct'].mean()) if 'Humidity_pct' in df.columns else 55.0
    avg_occupancy = float(df['Occupancy_Score'].mean()) if 'Occupancy_Score' in df.columns else (
        float(df['Occupancy_Ratio'].mean()) if 'Occupancy_Ratio' in df.columns else 0.5
    )

    # 3-tier Occupancy counts
    occ_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    if 'Occupancy_Level' in df.columns:
        counts = df['Occupancy_Level'].value_counts().to_dict()
        for k, v in counts.items():
            if k in occ_counts:
                occ_counts[k] = int(v)

    # Correlations
    temp_corr = float(df['Energy_kWh'].corr(df['Temperature_C'])) if 'Temperature_C' in df.columns else 0.0
    occ_score_col = 'Occupancy_Score' if 'Occupancy_Score' in df.columns else ('Occupancy_Ratio' if 'Occupancy_Ratio' in df.columns else None)
    occ_corr = float(df['Energy_kWh'].corr(df[occ_score_col])) if occ_score_col and occ_score_col in df.columns else 0.0

    return {
        'total_consumption_kWh': round(total_kWh, 2),
        'avg_hourly_kWh': round(avg_hourly_kWh, 3),
        'avg_daily_kWh': round(avg_daily_kWh, 2),
        'max_hourly_kWh': round(max_hourly_kWh, 3),
        'min_hourly_kWh': round(min_hourly_kWh, 3),
        'peak_hour_of_day': peak_hour,
        'start_date': start_date,
        'end_date': end_date,
        'total_days': total_days,
        'total_records': len(df),
        'avg_temperature_C': round(avg_temp, 1),
        'avg_humidity_pct': round(avg_humidity, 1),
        'avg_occupancy_ratio': round(avg_occupancy, 2),
        'occupancy_counts': occ_counts,
        'temp_energy_correlation': round(temp_corr, 3),
        'occupancy_energy_correlation': round(occ_corr, 3)
    }


def get_hourly_profile(df):
    """
    Returns average consumption, temperature, and occupancy by hour of day (0-23).
    """
    cols = {'Energy_kWh': 'mean'}
    if 'Temperature_C' in df.columns:
        cols['Temperature_C'] = 'mean'
    if 'Occupancy_Score' in df.columns:
        cols['Occupancy_Score'] = 'mean'
    elif 'Occupancy_Ratio' in df.columns:
        cols['Occupancy_Ratio'] = 'mean'
        
    hourly_avg = df.groupby('Hour').agg(cols).reset_index().round(3)
    
    # Add dominant Occupancy_Level per hour
    if 'Occupancy_Level' in df.columns:
        mode_occ = df.groupby('Hour')['Occupancy_Level'].agg(lambda x: x.mode()[0] if not x.empty else 'Medium').reset_index()
        hourly_avg['Occupancy_Level'] = mode_occ['Occupancy_Level']
        
    return hourly_avg.to_dict(orient='records')


def get_day_of_week_profile(df):
    """
    Returns average daily consumption by day of week (Monday to Sunday).
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('DayOfWeek')['Energy_kWh'].mean().reset_index()
    daily['DayName'] = daily['DayOfWeek'].apply(lambda x: days[x])
    daily = daily.round(3)
    return daily.to_dict(orient='records')


def get_monthly_profile(df):
    """
    Returns total and average energy consumption by month.
    """
    monthly = df.resample('ME')['Energy_kWh'].agg(['sum', 'mean']).reset_index()
    monthly['MonthStr'] = monthly['Datetime'].dt.strftime('%b %Y')
    monthly = monthly.round(2)
    return monthly.to_dict(orient='records')


def get_consumption_time_series(df, period='daily', limit=90):
    """
    Returns time series data aggregated by daily, weekly, or hourly for charts with real weather & 3-tier occupancy context.
    """
    if period == 'hourly':
        sub = df.tail(limit * 24 if limit else len(df)).copy()
        sub['Timestamp'] = sub.index.strftime('%Y-%m-%d %H:%M')
        res_cols = ['Timestamp', 'Energy_kWh']
        if 'Temperature_C' in sub.columns:
            res_cols.append('Temperature_C')
        if 'Occupancy_Level' in sub.columns:
            res_cols.append('Occupancy_Level')
        if 'Occupancy_Score' in sub.columns:
            res_cols.append('Occupancy_Score')
        elif 'Occupancy_Ratio' in sub.columns:
            res_cols.append('Occupancy_Ratio')
        return sub[res_cols].to_dict(orient='records')
    elif period == 'weekly':
        sub = df.resample('W').agg({
            'Energy_kWh': 'sum',
            'Temperature_C': 'mean',
            'Occupancy_Score': 'mean' if 'Occupancy_Score' in df.columns else 'first'
        }).reset_index()
        sub = sub.tail(limit if limit else len(sub))
        sub['Timestamp'] = sub['Datetime'].dt.strftime('%Y-%W')
        sub['Energy_kWh'] = sub['Energy_kWh'].round(2)
        sub['Temperature_C'] = sub['Temperature_C'].round(1)
        sub['Occupancy_Score'] = sub['Occupancy_Score'].round(2) if 'Occupancy_Score' in sub.columns else 0.5
        return sub.to_dict(orient='records')
    else:  # daily default
        sub = df.resample('D').agg({
            'Energy_kWh': 'sum',
            'Temperature_C': 'mean',
            'Occupancy_Score': 'mean' if 'Occupancy_Score' in df.columns else 'first'
        }).reset_index()
        sub = sub.tail(limit if limit else len(sub))
        sub['Timestamp'] = sub['Datetime'].dt.strftime('%Y-%m-%d')
        sub['Energy_kWh'] = sub['Energy_kWh'].round(2)
        sub['Temperature_C'] = sub['Temperature_C'].round(1)
        sub['Occupancy_Score'] = sub['Occupancy_Score'].round(2) if 'Occupancy_Score' in sub.columns else 0.5
        return sub.to_dict(orient='records')
