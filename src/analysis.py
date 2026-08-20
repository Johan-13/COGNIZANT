import pandas as pd
import numpy as np


def get_summary_metrics(df):
    """
    Computes overall key metrics for the dataset.
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
        'total_records': len(df)
    }


def get_hourly_profile(df):
    """
    Returns average consumption by hour of day (0-23).
    """
    hourly_avg = df.groupby('Hour')['Energy_kWh'].agg(['mean', 'std', 'max', 'min']).reset_index()
    hourly_avg = hourly_avg.round(3)
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
    monthly = df.resample('M')['Energy_kWh'].agg(['sum', 'mean']).reset_index()
    monthly['MonthStr'] = monthly['Datetime'].dt.strftime('%b %Y')
    monthly = monthly.round(2)
    return monthly.to_dict(orient='records')


def get_consumption_time_series(df, period='daily', limit=90):
    """
    Returns time series data aggregated by daily, weekly, or hourly for charts.
    limit: number of recent periods to return.
    """
    if period == 'hourly':
        sub = df.tail(limit * 24 if limit else len(df))
        resample_df = sub[['Energy_kWh', 'Global_active_power_kW']].reset_index()
        resample_df['Timestamp'] = resample_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
    elif period == 'weekly':
        sub = df.resample('W')['Energy_kWh'].sum().reset_index()
        sub = sub.tail(limit if limit else len(sub))
        resample_df = sub
        resample_df['Timestamp'] = resample_df['Datetime'].dt.strftime('%Y-%W')
    else:  # daily default
        sub = df.resample('D')['Energy_kWh'].sum().reset_index()
        sub = sub.tail(limit if limit else len(sub))
        resample_df = sub
        resample_df['Timestamp'] = resample_df['Datetime'].dt.strftime('%Y-%m-%d')

    resample_df['Energy_kWh'] = resample_df['Energy_kWh'].round(3)
    return resample_df[['Timestamp', 'Energy_kWh']].to_dict(orient='records')
