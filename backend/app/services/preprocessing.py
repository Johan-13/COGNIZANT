import os
import pandas as pd
import numpy as np

try:
    from backend.app.core.config import DATA_DIR, RAW_DATA_PATH, SAMPLE_DATA_PATH, PROCESSED_DATA_PATH
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'household_power_consumption.txt')
    SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_power_consumption.txt')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_power_consumption.csv')


def generate_sample_dataset(output_path=SAMPLE_DATA_PATH, days=365):
    """
    Generates a realistic mock UCI Household Electric Power Consumption dataset if raw dataset is missing.
    Includes base columns: Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
    Also generates synthetic Weather and Occupancy telemetry.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Generating realistic sample dataset ({days} days) at {output_path}...")
    
    # 1-hour resolution for synthetic dataset
    date_range = pd.date_range(end=pd.Timestamp.now().floor('h'), periods=days * 24, freq='h')
    
    np.random.seed(42)
    
    hours = date_range.hour
    day_of_week = date_range.dayofweek
    months = date_range.month
    
    base_load = 0.6  # Base load in kW
    morning_peak = 1.8 * np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    evening_peak = 2.5 * np.exp(-0.5 * ((hours - 20) / 2.0) ** 2)
    weekend_factor = np.where(day_of_week >= 5, 1.15, 1.0)
    
    # Seasonal temperature simulation (°C)
    seasonal_temp = 18 + 12 * np.sin(2 * np.pi * (months - 4) / 12)
    diurnal_temp = 5 * np.sin(2 * np.pi * (hours - 9) / 24)
    temperature = seasonal_temp + diurnal_temp + np.random.normal(0, 1.5, len(date_range))
    
    # HVAC load driven by extreme high or low temperatures
    hvac_load = np.maximum(0, (temperature - 22) * 0.08) + np.maximum(0, (15 - temperature) * 0.06)
    
    # Occupancy simulation (ratio 0.0 - 1.0)
    work_occupancy = np.where((hours >= 8) & (hours <= 18) & (day_of_week < 5), 0.75 + np.random.uniform(-0.1, 0.15, len(date_range)), 0.1)
    home_occupancy = np.where((hours >= 6) & (hours <= 23), 0.5 + np.random.uniform(-0.1, 0.2, len(date_range)), 0.15)
    occupancy_ratio = np.clip((work_occupancy * 0.6 + home_occupancy * 0.4) * weekend_factor, 0.05, 1.0)
    
    # Humidity simulation (%)
    humidity = np.clip(60 - 0.8 * (temperature - 18) + np.random.normal(0, 5, len(date_range)), 25, 95)
    
    # Random noise and occasional load spikes
    noise = np.random.normal(0, 0.12, len(date_range))
    spikes = np.random.choice([0, 2.2, 3.8], size=len(date_range), p=[0.97, 0.02, 0.01])
    
    global_active_power = np.clip(
        base_load + (morning_peak + evening_peak) * weekend_factor + hvac_load + (occupancy_ratio * 0.8) + noise + spikes,
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
        'Temperature_C': np.round(temperature, 1),
        'Humidity_pct': np.round(humidity, 1),
        'Occupancy_Ratio': np.round(occupancy_ratio, 2)
    })
    
    df.to_csv(output_path, sep=';', index=False)
    print(f"Sample dataset with Weather and Occupancy generated at {output_path} ({len(df)} rows).")
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
    - Clean Datetime parsing
    - Numeric conversions
    - Time-series linear interpolation
    - Resampling to hourly aggregates
    - Feature engineering: Weather (Temp, Humidity, Apparent Temp) & Occupancy (Ratio, Flag, Activity)
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
    
    numeric_cols = [c for c in df_raw.columns if c != 'Datetime']
    for col in numeric_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(method='time').bfill().ffill()
    
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
    
    # Weather features (resample if existing, or engineer realistic values if missing in raw)
    if 'Temperature_C' in df_raw.columns:
        hourly_df['Temperature_C'] = df_raw['Temperature_C'].resample('h').mean()
    else:
        months = hourly_df.index.month
        hours = hourly_df.index.hour
        hourly_df['Temperature_C'] = np.round(
            18 + 10 * np.sin(2 * np.pi * (months - 4) / 12) + 4 * np.sin(2 * np.pi * (hours - 9) / 24) + np.random.normal(0, 1.0, len(hourly_df)), 1
        )
        
    if 'Humidity_pct' in df_raw.columns:
        hourly_df['Humidity_pct'] = df_raw['Humidity_pct'].resample('h').mean()
    else:
        hourly_df['Humidity_pct'] = np.round(
            np.clip(60 - 0.7 * (hourly_df['Temperature_C'] - 18) + np.random.normal(0, 3, len(hourly_df)), 25, 95), 1
        )

    # Apparent Temperature (Heat Index approximation)
    hourly_df['Apparent_Temp_C'] = np.round(
        hourly_df['Temperature_C'] + 0.33 * (hourly_df['Humidity_pct'] / 100.0 * 6.105 * np.exp(17.27 * hourly_df['Temperature_C'] / (237.7 + hourly_df['Temperature_C']))) - 4.0, 1
    )
    
    # Occupancy features
    if 'Occupancy_Ratio' in df_raw.columns:
        hourly_df['Occupancy_Ratio'] = df_raw['Occupancy_Ratio'].resample('h').mean()
    else:
        hours = hourly_df.index.hour
        dayofweek = hourly_df.index.dayofweek
        weekend_mult = np.where(dayofweek >= 5, 0.7, 1.0)
        base_occ = np.where((hours >= 8) & (hours <= 18), 0.7, 0.25)
        hourly_df['Occupancy_Ratio'] = np.round(np.clip(base_occ * weekend_mult + np.random.normal(0, 0.05, len(hourly_df)), 0.05, 1.0), 2)
        
    hourly_df['Occupied_Flag'] = (hourly_df['Occupancy_Ratio'] >= 0.3).astype(int)
    
    # Zone Activity Level
    hourly_df['Zone_Activity_Level'] = pd.cut(
        hourly_df['Occupancy_Ratio'],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
        labels=['Low', 'Moderate', 'High', 'Peak']
    ).astype(str)
    
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
        print(f"Processed dataset saved successfully to {PROCESSED_DATA_PATH} ({len(hourly_df)} hourly records).")
        
    return hourly_df


def get_processed_data(force_reprocess=False):
    if force_reprocess or not os.path.exists(PROCESSED_DATA_PATH):
        print("Processed data cache missing or reload requested. Preprocessing...")
        return preprocess_data()
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
        return df


if __name__ == '__main__':
    df = get_processed_data(force_reprocess=True)
    print("Preprocessing verification complete:")
    print(df.info())
    print(df[['Energy_kWh', 'Temperature_C', 'Humidity_pct', 'Occupancy_Ratio', 'Zone_Activity_Level']].head())
