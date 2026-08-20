import os
import pandas as pd
import numpy as np

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'household_power_consumption.txt')
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, 'sample_power_consumption.txt')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_power_consumption.csv')


def generate_sample_dataset(output_path=SAMPLE_DATA_PATH, days=365):
    """
    Generates a realistic mock UCI Household Electric Power Consumption dataset if raw dataset is missing.
    Format matching UCI: Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Generating realistic sample dataset ({days} days) at {output_path}...")
    
    # 1-hour resolution for synthetic dataset to keep generation fast while realistic
    date_range = pd.date_range(end=pd.Timestamp.now().floor('h'), periods=days * 24, freq='h')
    
    np.random.seed(42)
    
    # Diurnal pattern (morning peak around 7-9am, evening peak around 18-22pm)
    hours = date_range.hour
    day_of_week = date_range.dayofweek
    
    base_load = 0.5  # Base load in kW
    morning_peak = 1.8 * np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    evening_peak = 2.5 * np.exp(-0.5 * ((hours - 20) / 2.0) ** 2)
    weekend_factor = np.where(day_of_week >= 5, 1.2, 1.0)
    
    # Random variation and occasional spikes (anomalies)
    noise = np.random.normal(0, 0.15, len(date_range))
    spikes = np.random.choice([0, 2.5, 4.0], size=len(date_range), p=[0.97, 0.02, 0.01])
    
    global_active_power = np.clip(base_load + (morning_peak + evening_peak) * weekend_factor + noise + spikes, 0.1, 8.0)
    global_reactive_power = np.clip(global_active_power * np.random.uniform(0.05, 0.25, len(date_range)), 0.01, 1.2)
    voltage = np.random.normal(240.0, 3.0, len(date_range))
    global_intensity = (global_active_power * 1000) / voltage
    
    # Sub-meterings (in Wh)
    sub_1 = np.where((hours >= 18) & (hours <= 21), np.random.uniform(0, 25, len(date_range)), 0)
    sub_2 = np.where((hours >= 7) & (hours <= 9), np.random.uniform(0, 20, len(date_range)), 0)
    sub_3 = np.clip(global_active_power * 500 * np.random.uniform(0.4, 0.7, len(date_range)), 0, 50)
    
    df = pd.DataFrame({
        'Date': date_range.strftime('%d/%m/%Y'),
        'Time': date_range.strftime('%H:%M:%S'),
        'Global_active_power': np.round(global_active_power, 3),
        'Global_reactive_power': np.round(global_reactive_power, 3),
        'Voltage': np.round(voltage, 2),
        'Global_intensity': np.round(global_intensity, 2),
        'Sub_metering_1': np.round(sub_1, 1),
        'Sub_metering_2': np.round(sub_2, 1),
        'Sub_metering_3': np.round(sub_3, 1)
    })
    
    # Inject a few missing values '?' to test robust missing value handling
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
    for col in ['Global_active_power', 'Voltage', 'Sub_metering_3']:
        df[col] = df[col].astype(object)
        df.loc[missing_indices, col] = '?'
        
    df.to_csv(output_path, sep=';', index=False)
    print(f"Sample dataset successfully generated with {len(df)} rows.")
    return output_path


def load_raw_dataset(raw_path=RAW_DATA_PATH):
    """
    Loads raw UCI dataset or sample dataset if raw file is not present.
    Returns raw dataframe.
    """
    if not os.path.exists(raw_path):
        print(f"Raw dataset not found at {raw_path}.")
        if not os.path.exists(SAMPLE_DATA_PATH):
            generate_sample_dataset(SAMPLE_DATA_PATH)
        target_path = SAMPLE_DATA_PATH
    else:
        target_path = raw_path

    print(f"Loading electricity consumption data from {target_path}...")
    
    # Read semicolon separated file with '?' missing values
    df = pd.read_csv(
        target_path,
        sep=';',
        low_memory=False,
        na_values=['?'],
        dtype={
            'Date': str,
            'Time': str
        }
    )
    return df, target_path


def preprocess_data(raw_path=RAW_DATA_PATH, save_processed=True):
    """
    Preprocesses raw dataset:
    - Parses Date and Time into datetime index
    - Converts columns to numeric
    - Handles missing values via interpolation
    - Resamples minute-level or hourly data into clean hourly aggregates
    - Saves processed dataframe to CSV
    """
    df_raw, data_source = load_raw_dataset(raw_path)
    
    # Clean datetime
    df_raw['Datetime'] = pd.to_datetime(
        df_raw['Date'] + ' ' + df_raw['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    # Drop rows where Datetime could not be parsed
    df_raw = df_raw.dropna(subset=['Datetime']).sort_values('Datetime')
    df_raw.set_index('Datetime', inplace=True)
    
    # Drop string Date & Time columns
    df_raw = df_raw.drop(columns=['Date', 'Time'], errors='ignore')
    
    # Numeric conversion for electricity columns
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
    # Interpolate missing values smoothly
    df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(method='time').bfill().ffill()
    
    # Determine input sampling rate (minute vs hourly)
    freq = pd.infer_freq(df_raw.index[:100])
    print(f"Inferred data frequency: {freq}")
    
    # Resample to Hourly aggregates
    hourly_df = pd.DataFrame()
    hourly_df['Global_active_power_kW'] = df_raw['Global_active_power'].resample('h').mean()
    hourly_df['Global_reactive_power_kW'] = df_raw['Global_reactive_power'].resample('h').mean()
    hourly_df['Voltage_V'] = df_raw['Voltage'].resample('h').mean()
    hourly_df['Global_intensity_A'] = df_raw['Global_intensity'].resample('h').mean()
    
    # If minute-level, sub-metering is in Wh per minute -> sum over hour gives total Wh for hour
    # If hourly-level sample data, sub-metering is already per hour -> sum/mean
    if 'Sub_metering_1' in df_raw.columns:
        hourly_df['Sub_metering_1_Wh'] = df_raw['Sub_metering_1'].resample('h').sum()
        hourly_df['Sub_metering_2_Wh'] = df_raw['Sub_metering_2'].resample('h').sum()
        hourly_df['Sub_metering_3_Wh'] = df_raw['Sub_metering_3'].resample('h').sum()
    
    # Active Energy in kWh for each hour = average kW * 1h
    hourly_df['Energy_kWh'] = hourly_df['Global_active_power_kW'] * 1.0
    
    # Time features
    hourly_df['Hour'] = hourly_df.index.hour
    hourly_df['DayOfWeek'] = hourly_df.index.dayofweek
    hourly_df['DayName'] = hourly_df.index.day_name()
    hourly_df['Month'] = hourly_df.index.month
    hourly_df['Year'] = hourly_df.index.year
    hourly_df['IsWeekend'] = hourly_df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Ensure no NaN remains after resampling
    hourly_df = hourly_df.infer_objects(copy=False).interpolate(method='linear').bfill().ffill()
    
    if save_processed:
        os.makedirs(DATA_DIR, exist_ok=True)
        hourly_df.to_csv(PROCESSED_DATA_PATH)
        print(f"Processed dataset saved successfully to {PROCESSED_DATA_PATH} ({len(hourly_df)} hourly records).")
        
    return hourly_df


def get_processed_data(force_reprocess=False):
    """
    Returns preprocessed hourly data as pandas DataFrame.
    Uses cached CSV if present unless force_reprocess is True.
    """
    if force_reprocess or not os.path.exists(PROCESSED_DATA_PATH):
        print("Processed data cache not found or reload requested. Preprocessing raw data...")
        return preprocess_data()
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
        return df


if __name__ == '__main__':
    df = get_processed_data(force_reprocess=True)
    print("Preprocessing verification complete:")
    print(df.info())
    print(df.head())
