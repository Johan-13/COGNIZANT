import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))

def get_current_time() -> pd.Timestamp:
    """Returns current real-world timestamp in Indian Standard Time (IST, UTC+5:30) as naive timestamp floored to current hour."""
    now_ist = datetime.now(IST_TIMEZONE).replace(tzinfo=None)
    return pd.Timestamp(now_ist).floor('h')

try:
    from backend.app.core.config import PROCESSED_DATA_PATH, DATA_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'app', 'data')
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_power_consumption.csv')


class LiveDataStreamer:
    """
    Asynchronous continuous data streaming engine.
    Generates and appends realistic hourly power telemetry as time progresses.
    """
    def __init__(self):
        self.is_streaming = True
        self.interval_seconds = 5.0  # 1 hour of simulated time every 5 seconds
        self.records_streamed = 0
        self.last_streamed_record = None
        self._task = None

    def get_status(self):
        return {
            'is_streaming': self.is_streaming,
            'interval_seconds': self.interval_seconds,
            'records_streamed': self.records_streamed,
            'last_record': self.last_streamed_record
        }

    def set_streaming(self, active: bool):
        self.is_streaming = active
        return self.get_status()

    def set_interval(self, seconds: float):
        self.interval_seconds = max(1.0, float(seconds))
        return self.get_status()

    def generate_next_record(self, df_current: pd.DataFrame) -> dict:
        """
        Generates the next sequential hourly record based on diurnal curve, real weather patterns,
        and 3-tier occupancy schedule.
        """
        if df_current.empty:
            next_dt = pd.Timestamp.now().floor('h')
            last_temp = 20.0
            last_power = 1.8
        else:
            last_dt = df_current.index.max()
            next_dt = last_dt + pd.Timedelta(hours=1)
            last_temp = float(df_current['Temperature_C'].iloc[-1]) if 'Temperature_C' in df_current.columns else 20.0
            last_power = float(df_current['Energy_kWh'].iloc[-1])

        h = next_dt.hour
        dow = next_dt.dayofweek
        weekend = (dow >= 5)

        # 1. Diurnal temperature variation
        temp_diurnal = 22.0 + 5.5 * np.sin(2 * np.pi * (h - 9) / 24)
        next_temp = round(float(0.7 * last_temp + 0.3 * temp_diurnal + np.random.normal(0, 0.4)), 1)
        next_humidity = round(float(np.clip(70.0 - (next_temp - 15.0) * 1.8 + np.random.normal(0, 2.0), 30.0, 95.0)), 1)

        # 2. 3-Tier Occupancy determination
        if (18 <= h <= 22) or (weekend and 11 <= h <= 15):
            occ_level = 'High'
            occ_score = 0.85
        elif (0 <= h <= 5) or (not weekend and 10 <= h <= 16):
            occ_level = 'Low'
            occ_score = 0.15
        else:
            occ_level = 'Medium'
            occ_score = 0.50

        # 3. Electrical load components
        base_load = 0.55
        morning_peak = 1.7 * np.exp(-0.5 * ((h - 8) / 1.5) ** 2)
        evening_peak = 2.4 * np.exp(-0.5 * ((h - 20) / 2.0) ** 2)
        weekend_mult = 1.15 if weekend else 1.0
        hvac_load = max(0.0, (next_temp - 22) * 0.08) + max(0.0, (15 - next_temp) * 0.06)
        noise = np.random.normal(0, 0.08)
        spike = np.random.choice([0.0, 2.0, 3.5], p=[0.97, 0.02, 0.01])

        global_active_power = float(np.clip(
            base_load + (morning_peak + evening_peak) * weekend_mult + hvac_load + (occ_score * 0.75) + noise + spike,
            0.1, 8.5
        ))
        global_reactive_power = float(np.clip(global_active_power * np.random.uniform(0.05, 0.22), 0.01, 1.2))
        voltage = round(float(np.random.normal(240.0, 2.5)), 2)
        global_intensity = round(float((global_active_power * 1000) / voltage), 2)

        sub_1 = float(np.random.uniform(0, 25)) if 18 <= h <= 21 else 0.0
        sub_2 = float(np.random.uniform(0, 20)) if 7 <= h <= 9 else 0.0
        sub_3 = float(np.clip(global_active_power * 480 * np.random.uniform(0.4, 0.75), 0, 55))

        apparent_temp = round(float(next_temp + 0.33 * (next_humidity / 100.0 * 6.105 * np.exp(17.27 * next_temp / (237.7 + next_temp))) - 4.0), 1)

        record = {
            'Datetime': next_dt,
            'Global_active_power_kW': round(global_active_power, 3),
            'Global_reactive_power_kW': round(global_reactive_power, 3),
            'Voltage_V': voltage,
            'Global_intensity_A': global_intensity,
            'Sub_metering_1_Wh': round(sub_1, 1),
            'Sub_metering_2_Wh': round(sub_2, 1),
            'Sub_metering_3_Wh': round(sub_3, 1),
            'Energy_kWh': round(global_active_power, 3),
            'Hour': h,
            'DayOfWeek': dow,
            'Month': next_dt.month,
            'IsWeekend': int(weekend),
            'Temperature_C': next_temp,
            'Humidity_pct': next_humidity,
            'Apparent_Temp_C': apparent_temp,
            'Occupancy_Level': occ_level,
            'Occupancy_Score': occ_score,
            'Occupancy_Ratio': occ_score
        }
        return record

    def step(self, df_current: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Appends 1 new hourly record to the dataframe.
        """
        record = self.generate_next_record(df_current)
        new_row = pd.DataFrame([record]).set_index('Datetime')
        updated_df = pd.concat([df_current, new_row])
        
        self.records_streamed += 1
        self.last_streamed_record = {
            'Timestamp': record['Datetime'].strftime('%Y-%m-%d %H:%M'),
            'Energy_kWh': record['Energy_kWh'],
            'Temperature_C': record['Temperature_C'],
            'Humidity_pct': record['Humidity_pct'],
            'Occupancy_Level': record['Occupancy_Level'],
            'Occupancy_Ratio': record['Occupancy_Ratio']
        }
        return updated_df, self.last_streamed_record

    def catch_up_to_now(self, df_current: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Fills any hourly gap between the last timestamp in df_current and the current real-world hour (IST).
        """
        now = get_current_time()
        if df_current.empty:
            return df_current, 0

        last_dt = df_current.index.max() if isinstance(df_current.index, pd.DatetimeIndex) else pd.to_datetime(df_current['Datetime']).max()
        added = 0
        while last_dt < now:
            df_current, last_record = self.step(df_current)
            last_dt = df_current.index.max() if isinstance(df_current.index, pd.DatetimeIndex) else pd.to_datetime(df_current['Datetime']).max()
            added += 1

        if added > 0:
            try:
                df_current.to_csv(PROCESSED_DATA_PATH)
                print(f"[LiveDataStreamer] Synchronized {added} missing hourly records up to {now} (IST).")
            except Exception as e:
                print(f"[LiveDataStreamer] Note: could not save backfilled CSV: {e}")
        return df_current, added


# Global streamer instance
streamer_engine = LiveDataStreamer()
