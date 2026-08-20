import os
import json
import joblib
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from backend.app.core.config import MODELS_DIR, PROPHET_MODEL_PATH, METADATA_PATH
    MODEL_DIR = MODELS_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_DIR = os.path.join(BASE_DIR, 'app', 'models')
    PROPHET_MODEL_PATH = os.path.join(MODEL_DIR, 'prophet_model.pkl')
    METADATA_PATH = os.path.join(MODEL_DIR, 'forecasting_meta.json')


def calculate_metrics(y_true, y_pred):
    """
    Computes MAE, RMSE, and MAPE.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true > 0.01
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    else:
        mape = 0.0

    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'MAPE': round(float(mape), 2)
    }


class EnergyForecaster:
    """
    High-Performance Prophet Forecasting Engine:
    - Multiplicative seasonality modeling
    - High-resolution daily Fourier harmonics (fourier_order=10)
    - 24-Hour Autoregressive Lag Regressor (lag_24)
    - Real Weather (Temperature_C) & 3-Tier Occupancy regressors
    - Overnight Baseload Floor Calibration (00:00 - 05:00 AM) to beat Naive Baseline on 24h MAPE
    """
    def __init__(self):
        self.model = None
        self.meta = {}
        self.baseload_median = 0.55

    def prepare_prophet_df(self, df):
        """
        Formats dataframe to Prophet standard format with lag_24, weather, and occupancy regressors.
        """
        p_df = pd.DataFrame()
        p_df['ds'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Datetime'])
        p_df['y'] = df['Energy_kWh'].values
        
        # 24-hour autoregressive lag
        p_df['lag_24'] = pd.Series(p_df['y']).shift(24).bfill().values
        
        if 'Temperature_C' in df.columns:
            p_df['Temperature_C'] = df['Temperature_C'].values
        else:
            p_df['Temperature_C'] = 20.0

        if 'Occupancy_Score' in df.columns:
            p_df['Occupancy_Ratio'] = df['Occupancy_Score'].values
        elif 'Occupancy_Ratio' in df.columns:
            p_df['Occupancy_Ratio'] = df['Occupancy_Ratio'].values
        elif 'Occupancy_Level' in df.columns:
            mapping = {'Low': 0.15, 'Medium': 0.50, 'High': 0.85}
            p_df['Occupancy_Ratio'] = df['Occupancy_Level'].map(mapping).fillna(0.50).values
        else:
            p_df['Occupancy_Ratio'] = 0.5
            
        return p_df

    def train(self, df, test_ratio=0.15, save_model=True):
        """
        Trains High-Performance Prophet Model with multiplicative seasonality, Fourier daily harmonics,
        and lag_24 + weather + occupancy regressors.
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        p_df = self.prepare_prophet_df(df)
        n_total = len(p_df)
        split_idx = int(n_total * (1 - test_ratio))
        
        train_p = p_df.iloc[:split_idx].copy()
        test_p = p_df.iloc[split_idx:].copy()
        
        # Calculate stationary overnight baseload median from training data
        train_night = train_p[train_p['ds'].dt.hour.isin([0, 1, 2, 3, 4, 5])]
        if not train_night.empty:
            self.baseload_median = float(train_night['y'].median())
        else:
            self.baseload_median = float(train_p['y'].quantile(0.20))
            
        print(f"Training Tuned Prophet Model (Total samples: {n_total}, Train: {len(train_p)}, Test: {len(test_p)})...")
        print(f"Overnight Baseload Median: {round(self.baseload_median, 3)} kWh")
        
        # 1. Naive Baseline Evaluation (24h seasonal lag)
        y_test_actual = test_p['y'].values
        naive_pred = p_df['y'].shift(24).iloc[split_idx:].values
        valid_mask = ~np.isnan(naive_pred)
        baseline_metrics = calculate_metrics(y_test_actual[valid_mask], naive_pred[valid_mask])
        print(f"Naive Baseline Metrics (24h lag): {baseline_metrics}")

        # 2. Fit Tuned Prophet Model on Train set for validation evaluation
        prophet_val = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.15,
            interval_width=0.95
        )
        prophet_val.add_seasonality(name='daily', period=1, fourier_order=10)
        prophet_val.add_regressor('Temperature_C')
        prophet_val.add_regressor('Occupancy_Ratio')
        prophet_val.add_regressor('lag_24')
        
        prophet_val.fit(train_p)
        
        val_forecast = prophet_val.predict(test_p[['ds', 'Temperature_C', 'Occupancy_Ratio', 'lag_24']])
        yhat_val_raw = val_forecast['yhat'].values
        val_hours = test_p['ds'].dt.hour.values
        
        # Apply Overnight Baseload Floor Calibration: yhat_overnight = min(yhat_raw, self.baseload_median)
        is_overnight = np.isin(val_hours, [0, 1, 2, 3, 4, 5])
        yhat_val_calibrated = np.where(is_overnight, np.minimum(yhat_val_raw, self.baseload_median), yhat_val_raw)
        yhat_val_calibrated = np.maximum(yhat_val_calibrated, 0.05)
        
        prophet_metrics = calculate_metrics(y_test_actual, yhat_val_calibrated)
        print(f"Tuned Prophet Model Metrics: {prophet_metrics}")

        # 3. Fit Full Prophet Model on entire historical dataset for deployment forecasting
        full_prophet = Prophet(
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.15,
            interval_width=0.95
        )
        full_prophet.add_seasonality(name='daily', period=1, fourier_order=10)
        full_prophet.add_regressor('Temperature_C')
        full_prophet.add_regressor('Occupancy_Ratio')
        full_prophet.add_regressor('lag_24')
        full_prophet.fit(p_df)

        # Improvement metric over baseline (positive indicates Prophet beats Baseline)
        mape_improvement = round(float(baseline_metrics['MAPE'] - prophet_metrics['MAPE']), 2)

        self.meta = {
            'trained_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': n_total,
            'model_type': 'Facebook Prophet (Multiplicative + lag_24 Regressor + Overnight Calibration)',
            'baseload_median_kWh': round(self.baseload_median, 3),
            'prophet_metrics': prophet_metrics,
            'baseline_metrics': baseline_metrics,
            'mape_improvement_vs_baseline': mape_improvement,
            'features_used': ['ds', 'y', 'lag_24', 'Temperature_C', 'Occupancy_Ratio']
        }

        if save_model:
            self.model = full_prophet
            joblib.dump(full_prophet, PROPHET_MODEL_PATH)
            with open(METADATA_PATH, 'w') as f:
                json.dump(self.meta, f, indent=2)

            # Sync to other models directories
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alt_dirs = [
                os.path.join(current_dir, '..', 'models'),
                os.path.join(current_dir, '..', '..', 'models'),
                os.path.join(current_dir, '..', '..', '..', 'models')
            ]
            for ad in alt_dirs:
                norm_ad = os.path.normpath(ad)
                if os.path.exists(norm_ad) and norm_ad != os.path.normpath(MODEL_DIR):
                    try:
                        joblib.dump(full_prophet, os.path.join(norm_ad, 'prophet_model.pkl'))
                        with open(os.path.join(norm_ad, 'forecasting_meta.json'), 'w') as f:
                            json.dump(self.meta, f, indent=2)
                    except Exception:
                        pass

        return self.meta

    def predict(self, df, horizon_hours=24):
        """
        Generates future Prophet forecast for specified horizon_hours.
        Uses recursive lag_24 estimation, diurnal weather curve, 3-tier Occupancy, and overnight calibration.
        """
        if self.model is None:
            if os.path.exists(PROPHET_MODEL_PATH):
                self.model = joblib.load(PROPHET_MODEL_PATH)
            else:
                self.train(df)

        if os.path.exists(METADATA_PATH) and not self.meta:
            try:
                with open(METADATA_PATH, 'r') as f:
                    self.meta = json.load(f)
                    self.baseload_median = float(self.meta.get('baseload_median_kWh', 0.55))
            except Exception:
                pass

        last_timestamp = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Datetime']).max()
        future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq='h')
        
        future_hours = future_dates.hour
        future_dayofweek = future_dates.dayofweek
        
        # Estimate future weather curve
        recent_temp = df['Temperature_C'].iloc[-24:].mean() if 'Temperature_C' in df.columns else 20.0
        future_temp = np.round(
            recent_temp + 3.5 * np.sin(2 * np.pi * (future_hours - 9) / 24) + np.random.normal(0, 0.3, len(future_dates)), 1
        )
        
        # Estimate 3-tier occupancy schedule
        future_occ_levels = []
        future_occ_scores = []
        for h, dow in zip(future_hours, future_dayofweek):
            if (18 <= h <= 22) or (dow >= 5 and 11 <= h <= 15):
                future_occ_levels.append('High')
                future_occ_scores.append(0.85)
            elif (0 <= h <= 5) or (dow < 5 and 10 <= h <= 16):
                future_occ_levels.append('Low')
                future_occ_scores.append(0.15)
            else:
                future_occ_levels.append('Medium')
                future_occ_scores.append(0.50)
                
        # Recursive multi-step prediction with lag_24
        predictions = []
        recent_history_y = df['Energy_kWh'].iloc[-24:].tolist()
        predicted_y = []
        
        for i in range(horizon_hours):
            dt_step = future_dates[i]
            h_step = future_hours[i]
            temp_step = future_temp[i]
            occ_score_step = future_occ_scores[i]
            occ_level_step = future_occ_levels[i]
            
            # Lag 24: from historical actuals or earlier forecast predictions
            if i < 24:
                lag_val = recent_history_y[i]
            else:
                lag_val = predicted_y[i - 24]
                
            step_df = pd.DataFrame({
                'ds': [dt_step],
                'Temperature_C': [temp_step],
                'Occupancy_Ratio': [occ_score_step],
                'lag_24': [lag_val]
            })
            
            fc_step = self.model.predict(step_df)
            yhat = round(float(fc_step['yhat'].iloc[0]), 3)
            
            # Apply Overnight Baseload Floor Calibration (00:00 - 05:00 AM)
            if h_step in [0, 1, 2, 3, 4, 5]:
                yhat = min(yhat, round(self.baseload_median, 3))
                
            yhat = max(0.05, yhat)
            predicted_y.append(yhat)
            
            yhat_lower = max(0.01, round(float(fc_step['yhat_lower'].iloc[0]), 3))
            yhat_upper = max(yhat, round(float(fc_step['yhat_upper'].iloc[0]), 3))
            
            predictions.append({
                'Timestamp': dt_step.strftime('%Y-%m-%d %H:%M'),
                'Forecast_kWh': yhat,
                'yhat_lower': yhat_lower,
                'yhat_upper': yhat_upper,
                'Trend': round(float(fc_step['trend'].iloc[0]), 3),
                'Weekly_Seasonality': round(float(fc_step.get('weekly', [0.0]).iloc[0]), 3) if 'weekly' in fc_step else 0.0,
                'Daily_Seasonality': round(float(fc_step.get('daily', [0.0]).iloc[0]), 3) if 'daily' in fc_step else 0.0,
                'Temperature_C': float(temp_step),
                'Occupancy_Level': occ_level_step,
                'Occupancy_Ratio': float(occ_score_step)
            })
            
        return predictions


def get_forecast_results(df, horizon_hours=24, force_retrain=False):
    forecaster = EnergyForecaster()
    
    if force_retrain or not os.path.exists(METADATA_PATH) or not os.path.exists(PROPHET_MODEL_PATH):
        meta = forecaster.train(df)
    else:
        with open(METADATA_PATH, 'r') as f:
            meta = json.load(f)

    predictions = forecaster.predict(df, horizon_hours=horizon_hours)

    # Actual test set records (last 48 hours)
    recent_historical = df.tail(48).copy().reset_index()
    historical_actuals = []
    
    for _, row in recent_historical.iterrows():
        dt_val = row['Datetime'] if 'Datetime' in row else row.name
        occ_lvl = row.get('Occupancy_Level', 'Medium')
        historical_actuals.append({
            'Timestamp': pd.to_datetime(dt_val).strftime('%Y-%m-%d %H:%M'),
            'Actual_kWh': round(float(row['Energy_kWh']), 3),
            'Temperature_C': round(float(row.get('Temperature_C', 20.0)), 1),
            'Occupancy_Level': occ_lvl,
            'Occupancy_Ratio': round(float(row.get('Occupancy_Score', row.get('Occupancy_Ratio', 0.5))), 2)
        })

    return {
        'horizon_hours': horizon_hours,
        'metadata': meta,
        'predictions': predictions,
        'historical_test_actuals': historical_actuals
    }


if __name__ == '__main__':
    try:
        from backend.app.services.preprocessing import get_processed_data
    except ImportError:
        from preprocessing import get_processed_data
    df = get_processed_data()
    results = get_forecast_results(df, horizon_hours=24, force_retrain=True)
    print("Prophet Forecasting Verification:")
    print("Metadata:", json.dumps(results['metadata'], indent=2))
    print("First 3 predictions:", results['predictions'][:3])
