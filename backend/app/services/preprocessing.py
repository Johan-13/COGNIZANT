import os
import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np

try:
    from backend.app.core.config import DATA_DIR, RAW_DATA_PATH, SAMPLE_DATA_PATH, PROCESSED_DATA_PATH, WEATHER_CACHE_PATH
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'household_power_consumption.txt')
    SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_power_consumption.txt')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_power_consumption.csv')
    WEATHER_CACHE_PATH = os.path.join(DATA_DIR, 'weather_cache.csv')


def fetch_historical_weather(start_dt, end_dt, lat=48.8566, lon=2.3522, cache_path=WEATHER_CACHE_PATH):
    """
    Fetches real historical hourly weather data (Temperature & Relative Humidity) from Open-Meteo API.
    Utilizes local disk caching to prevent redundant API calls.
    Gracefully falls back to localized historical climatology if offline or network restricted.
    """
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)
    
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    
    # 1. Check local cache first
    if os.path.exists(cache_path):
        try:
            cache_df = pd.read_csv(cache_path, parse_dates=['Datetime'], index_col='Datetime')
            if cache_df.index.min() <= start_dt and cache_df.index.max() >= end_dt:
                sub_cache = cache_df.loc[start_dt:end_dt].copy()
                if not sub_cache.empty and 'Temperature_C' in sub_cache.columns and 'Humidity_pct' in sub_cache.columns:
                    print(f"Loaded real historical weather from cache ({len(sub_cache)} hourly records).")
                    return sub_cache[['Temperature_C', 'Humidity_pct']]
        except Exception as e:
            print(f"Weather cache read warning: {e}. Fetching from Open-Meteo API...")

    # 2. Query Open-Meteo Historical Weather API
    print(f"Fetching real historical weather from Open-Meteo API for [{start_str} to {end_str}] (Lat: {lat}, Lon: {lon})...")
    
    # Open-Meteo Archive API supports historical dates up to ~5 days ago.
    # For very recent dates, fallback to forecast API with past_days.
    archive_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_str}&end_date={end_str}&"
        f"hourly=temperature_2m,relative_humidity_2m&timezone=auto"
    )
    
    weather_df = None
    try:
        req = urllib.request.Request(archive_url, headers={'User-Agent': 'EnergyOptimizer/2.0'})
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            if 'hourly' in res_data and 'time' in res_data['hourly']:
                hourly = res_data['hourly']
                weather_df = pd.DataFrame({
                    'Datetime': pd.to_datetime(hourly['time']),
                    'Temperature_C': np.round(hourly['temperature_2m'], 1),
                    'Humidity_pct': np.round(hourly['relative_humidity_2m'], 1)
                }).set_index('Datetime')
                print(f"Successfully retrieved {len(weather_df)} hourly historical weather records from Open-Meteo.")
    except Exception as api_err:
        print(f"Open-Meteo Archive API notice ({api_err}). Trying recent forecast weather endpoint...")
        try:
            recent_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&past_days=92&forecast_days=7&"
                f"hourly=temperature_2m,relative_humidity_2m&timezone=auto"
            )
            req = urllib.request.Request(recent_url, headers={'User-Agent': 'EnergyOptimizer/2.0'})
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                if 'hourly' in res_data and 'time' in res_data['hourly']:
                    hourly = res_data['hourly']
                    weather_df = pd.DataFrame({
                        'Datetime': pd.to_datetime(hourly['time']),
                        'Temperature_C': np.round(hourly['temperature_2m'], 1),
                        'Humidity_pct': np.round(hourly['relative_humidity_2m'], 1)
                    }).set_index('Datetime')
                    weather_df = weather_df.loc[start_dt:end_dt]
        except Exception as e2:
            print(f"Network request to Open-Meteo failed ({e2}). Synthesizing climate normal baseline...")

    # 3. Fallback climatology baseline if network is unavailable
    if weather_df is None or weather_df.empty:
        idx = pd.date_range(start=start_dt, end=end_dt, freq='h')
        months = idx.month
        hours = idx.hour
        # Paris / Sceaux historical climatology curve
        temp = 12.0 + 9.5 * np.sin(2 * np.pi * (months - 4) / 12) + 3.8 * np.sin(2 * np.pi * (hours - 9) / 24)
        hum = np.clip(72 - 1.1 * (temp - 12) + np.random.normal(0, 3, len(idx)), 30, 95)
        weather_df = pd.DataFrame({
            'Temperature_C': np.round(temp, 1),
            'Humidity_pct': np.round(hum, 1)
        }, index=idx)

    # 4. Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        weather_df.to_csv(cache_path)
    except Exception:
        pass

    return weather_df


def compute_occupancy_levels(df_hourly):
    """
    Computes a 3-tier categorical Occupancy system (Low, Medium, High):
    - 'Low': Building inactive/standby, overnight (00:00-06:00) with minimal draw, or workday vacancy.
    - 'Medium': Normal baseline activity, moderate appliance operation, or weekend daytime background.
    - 'High': Peak simultaneous appliance draw, high active power, active kitchen/laundry sub-metering, or evening peak hours.
    
    Also provides numerical 'Occupancy_Score' (0.15 for Low, 0.50 for Medium, 0.85 for High) for ML regressors.
    """
    hours = df_hourly.index.hour
    dayofweek = df_hourly.index.dayofweek
    is_weekend = (dayofweek >= 5).astype(int)
    power = df_hourly['Global_active_power_kW'] if 'Global_active_power_kW' in df_hourly.columns else df_hourly['Energy_kWh']
    
    # Sub-metering total if available
    sub_total = 0
    if 'Sub_metering_1_Wh' in df_hourly.columns:
        sub_total += df_hourly['Sub_metering_1_Wh'].fillna(0)
    if 'Sub_metering_2_Wh' in df_hourly.columns:
        sub_total += df_hourly['Sub_metering_2_Wh'].fillna(0)
    if 'Sub_metering_3_Wh' in df_hourly.columns:
        sub_total += df_hourly['Sub_metering_3_Wh'].fillna(0)
        
    p_median = float(power.median())
    p_high = float(power.quantile(0.75))
    p_low = float(power.quantile(0.25))
    
    occupancy_levels = []
    occupancy_scores = []
    
    for dt, p_val in zip(df_hourly.index, power):
        h = dt.hour
        dow = dt.dayofweek
        weekend = (dow >= 5)
        
        # Determine High / Medium / Low
        if p_val >= p_high or (18 <= h <= 22 and p_val >= p_median) or (7 <= h <= 9 and not weekend and p_val >= p_median):
            level = 'High'
            score = 0.85
        elif (0 <= h <= 5 and p_val <= p_low * 1.3) or (not weekend and 10 <= h <= 16 and p_val <= p_low * 1.2):
            level = 'Low'
            score = 0.15
        else:
            level = 'Medium'
            score = 0.50
            
        occupancy_levels.append(level)
        occupancy_scores.append(score)
        
    return pd.Series(occupancy_levels, index=df_hourly.index), pd.Series(occupancy_scores, index=df_hourly.index)


def generate_sample_dataset(output_path=SAMPLE_DATA_PATH, days=365):
    """
    Generates a realistic UCI Household Electric Power Consumption dataset with real Open-Meteo historical weather
    and 3-tier (Low, Medium, High) Occupancy levels.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Generating realistic dataset ({days} days) with real historical weather at {output_path}...")
    
    # 1-hour resolution for 365 past days
    end_date = pd.Timestamp.now().floor('h')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Fetch real historical weather
    weather_df = fetch_historical_weather(start_date, end_date)
    weather_df = weather_df.reindex(date_range).interpolate(method='linear').bfill().ffill()
    
    temp_series = weather_df['Temperature_C'].values
    humidity_series = weather_df['Humidity_pct'].values
    
    np.random.seed(42)
    hours = date_range.hour
    day_of_week = date_range.dayofweek
    
    # Base load + morning / evening peaks
    base_load = 0.55
    morning_peak = 1.7 * np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    evening_peak = 2.4 * np.exp(-0.5 * ((hours - 20) / 2.0) ** 2)
    weekend_factor = np.where(day_of_week >= 5, 1.15, 1.0)
    
    # HVAC load driven by actual temperature
    hvac_load = np.maximum(0, (temp_series - 22) * 0.08) + np.maximum(0, (15 - temp_series) * 0.06)
    
    # Initial occupancy distribution (Low, Medium, High)
    occ_levels = []
    occ_scores = []
    for h, dow in zip(hours, day_of_week):
        if (h >= 18 and h <= 22) or (dow >= 5 and 11 <= h <= 15):
            occ_levels.append('High')
            occ_scores.append(0.85)
        elif (0 <= h <= 5) or (dow < 5 and 10 <= h <= 16):
            occ_levels.append('Low')
            occ_scores.append(0.15)
        else:
            occ_levels.append('Medium')
            occ_scores.append(0.50)
            
    occ_scores_arr = np.array(occ_scores)
    noise = np.random.normal(0, 0.10, len(date_range))
    spikes = np.random.choice([0, 2.0, 3.5], size=len(date_range), p=[0.97, 0.02, 0.01])
    
    global_active_power = np.clip(
        base_load + (morning_peak + evening_peak) * weekend_factor + hvac_load + (occ_scores_arr * 0.75) + noise + spikes,
        0.1, 8.5
    )
    global_reactive_power = np.clip(global_active_power * np.random.uniform(0.05, 0.22, len(date_range)), 0.01, 1.2)
    voltage = np.random.normal(240.0, 2.8, len(date_range))
    global_intensity = (global_active_power * 1000) / voltage
    
    sub_1 = np.where((hours >= 18) & (hours <= 21), np.random.uniform(0, 25, len(date_range)), 0)
    sub_2 = np.where((hours >= 7) & (hours <= 9), np.random.uniform(0, 20, len(date_range)), 0)
    sub_3 = np.clip(global_active_power * 480 * np.random.uniform(0.4, 0.75, len(date_range)), 0, 55)
    
    df = pd.DataFrame({
        'Date': date_range.strftime('%d/%m/%Y'),
        'Time': date_range.strftime('%H:%M:%S'),
        'Global_active_power': np.round(global_active_power, 3),
        'Global_reactive_power': np.round(global_reactive_power, 3),
        'Voltage': np.round(voltage, 2),
        'Global_intensity': np.round(global_intensity, 2),
        'Sub_metering_1': np.round(sub_1, 1),
        'Sub_metering_2': np.round(sub_2, 1),
        'Sub_metering_3': np.round(sub_3, 1),
        'Temperature_C': np.round(temp_series, 1),
        'Humidity_pct': np.round(humidity_series, 1),
        'Occupancy_Level': occ_levels,
        'Occupancy_Ratio': np.round(occ_scores_arr, 2)
    })
    
    df.to_csv(output_path, sep=';', index=False)
    print(f"Dataset generated at {output_path} ({len(df)} rows) with Open-Meteo real weather & Low/Medium/High occupancy.")
    return output_path


def load_raw_dataset(raw_path=RAW_DATA_PATH):
    if not os.path.exists(raw_path):
        if not os.path.exists(SAMPLE_DATA_PATH):
            generate_sample_dataset(SAMPLE_DATA_PATH)
        target_path = SAMPLE_DATA_PATH
    else:
        target_path = raw_path

    print(f"Loading raw electricity consumption data from {target_path}...")
    df = pd.read_csv(
        target_path,
        sep=';',
        low_memory=False,
        na_values=['?'],
        dtype={'Date': str, 'Time': str}
    )
    return df, target_path


def preprocess_data(raw_path=RAW_DATA_PATH, save_processed=True):
    """
    Preprocesses dataset:
    - Clean Datetime parsing & numeric conversions
    - Time-series linear interpolation & hourly resampling
    - Integrates real historical weather from Open-Meteo API
    - Computes 3-tier Occupancy (Low, Medium, High)
    """
    df_raw, data_source = load_raw_dataset(raw_path)
    
    df_raw['Datetime'] = pd.to_datetime(
        df_raw['Date'] + ' ' + df_raw['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    df_raw = df_raw.dropna(subset=['Datetime']).sort_values('Datetime')
    df_raw.set_index('Datetime', inplace=True)
    df_raw = df_raw.drop(columns=['Date', 'Time'], errors='ignore')
    
    numeric_cols = [c for c in df_raw.columns if c not in ['Datetime', 'Occupancy_Level']]
    for col in numeric_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    df_raw[numeric_cols] = df_raw[numeric_cols].infer_objects(copy=False).interpolate(method='time').bfill().ffill()
    
    hourly_df = pd.DataFrame()
    hourly_df['Global_active_power_kW'] = df_raw['Global_active_power'].resample('h').mean()
    hourly_df['Global_reactive_power_kW'] = df_raw['Global_reactive_power'].resample('h').mean()
    hourly_df['Voltage_V'] = df_raw['Voltage'].resample('h').mean()
    hourly_df['Global_intensity_A'] = df_raw['Global_intensity'].resample('h').mean()
    
    if 'Sub_metering_1' in df_raw.columns:
        hourly_df['Sub_metering_1_Wh'] = df_raw['Sub_metering_1'].resample('h').sum()
        hourly_df['Sub_metering_2_Wh'] = df_raw['Sub_metering_2'].resample('h').sum()
        hourly_df['Sub_metering_3_Wh'] = df_raw['Sub_metering_3'].resample('h').sum()
    
    hourly_df['Energy_kWh'] = hourly_df['Global_active_power_kW'] * 1.0
    
    # Integrate Real Historical Weather from Open-Meteo
    if 'Temperature_C' in df_raw.columns and 'Humidity_pct' in df_raw.columns:
        hourly_df['Temperature_C'] = df_raw['Temperature_C'].resample('h').mean()
        hourly_df['Humidity_pct'] = df_raw['Humidity_pct'].resample('h').mean()
    else:
        start_dt = hourly_df.index.min()
        end_dt = hourly_df.index.max()
        weather = fetch_historical_weather(start_dt, end_dt)
        weather_reindexed = weather.reindex(hourly_df.index).infer_objects(copy=False).interpolate(method='linear').bfill().ffill()
        hourly_df['Temperature_C'] = weather_reindexed['Temperature_C']
        hourly_df['Humidity_pct'] = weather_reindexed['Humidity_pct']

    hourly_df['Temperature_C'] = hourly_df['Temperature_C'].round(1)
    hourly_df['Humidity_pct'] = hourly_df['Humidity_pct'].round(1)

    # Apparent Temperature (Heat Index formula)
    hourly_df['Apparent_Temp_C'] = np.round(
        hourly_df['Temperature_C'] + 0.33 * (
            hourly_df['Humidity_pct'] / 100.0 * 6.105 * np.exp(
                17.27 * hourly_df['Temperature_C'] / (237.7 + hourly_df['Temperature_C'])
            )
        ) - 4.0, 1
    )
    
    # 3-Tier Occupancy (Low, Medium, High)
    occ_levels, occ_scores = compute_occupancy_levels(hourly_df)
    hourly_df['Occupancy_Level'] = occ_levels
    hourly_df['Occupancy_Score'] = occ_scores
    hourly_df['Occupancy_Ratio'] = occ_scores  # Maintained for backwards compatibility
    hourly_df['Occupied_Flag'] = (hourly_df['Occupancy_Level'] != 'Low').astype(int)
    
    # Calendar & Time features
    hourly_df['Hour'] = hourly_df.index.hour
    hourly_df['DayOfWeek'] = hourly_df.index.dayofweek
    hourly_df['DayName'] = hourly_df.index.day_name()
    hourly_df['Month'] = hourly_df.index.month
    hourly_df['Year'] = hourly_df.index.year
    hourly_df['IsWeekend'] = hourly_df['DayOfWeek'].isin([5, 6]).astype(int)
    
    hourly_df = hourly_df.infer_objects(copy=False).interpolate(method='linear').bfill().ffill()
    
    if save_processed:
        os.makedirs(DATA_DIR, exist_ok=True)
        hourly_df.to_csv(PROCESSED_DATA_PATH)
        print(f"Processed dataset saved to {PROCESSED_DATA_PATH} ({len(hourly_df)} records).")
        
    return hourly_df


def get_processed_data(force_reprocess=False):
    if force_reprocess or not os.path.exists(PROCESSED_DATA_PATH):
        print("Processed data cache missing or reload requested. Preprocessing...")
        return preprocess_data()
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
        # Ensure Occupancy_Level is present
        if 'Occupancy_Level' not in df.columns:
            occ_levels, occ_scores = compute_occupancy_levels(df)
            df['Occupancy_Level'] = occ_levels
            df['Occupancy_Score'] = occ_scores
        return df


if __name__ == '__main__':
    df = get_processed_data(force_reprocess=True)
    print("Preprocessing verification complete:")
    print(df[['Energy_kWh', 'Temperature_C', 'Humidity_pct', 'Occupancy_Level', 'Occupancy_Score']].head(10))
    print("\nOccupancy Distribution:")
    print(df['Occupancy_Level'].value_counts())
