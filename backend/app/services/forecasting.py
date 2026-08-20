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
    Forecasting Engine utilizing Facebook Prophet with Weather & Occupancy regressors.
    """
    def __init__(self):
        self.model = None
        self.meta = {}

    def prepare_prophet_df(self, df):
        """
        Formats dataframe to Prophet standard format (ds, y) with extra regressors.
        """
        p_df = pd.DataFrame()
        p_df['ds'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Datetime'])
        p_df['y'] = df['Energy_kWh'].values
        
        if 'Temperature_C' in df.columns:
            p_df['Temperature_C'] = df['Temperature_C'].values
        else:
            p_df['Temperature_C'] = 20.0

        if 'Occupancy_Ratio' in df.columns:
            p_df['Occupancy_Ratio'] = df['Occupancy_Ratio'].values
        else:
            p_df['Occupancy_Ratio'] = 0.5
            
        return p_df

    def train(self, df, test_ratio=0.15):
        """
        Trains Facebook Prophet forecasting model with daily/weekly seasonality & weather/occupancy regressors.
        Benchmarks Prophet performance against a Naive 24h seasonal baseline.
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        p_df = self.prepare_prophet_df(df)
        n_total = len(p_df)
        split_idx = int(n_total * (1 - test_ratio))
        
        train_p = p_df.iloc[:split_idx].copy()
        test_p = p_df.iloc[split_idx:].copy()
        
        print(f"Training Prophet Model (Total samples: {n_total}, Train: {len(train_p)}, Test: {len(test_p)})...")
        
        # 1. Naive Baseline Evaluation (24h seasonal lag)
        y_test_actual = test_p['y'].values
        naive_pred = p_df['y'].shift(24).iloc[split_idx:].values
        
        # Remove initial NaNs in naive prediction if any
        valid_mask = ~np.isnan(naive_pred)
        baseline_metrics = calculate_metrics(y_test_actual[valid_mask], naive_pred[valid_mask])
        print(f"Naive Baseline Metrics (24h lag): {baseline_metrics}")

        # 2. Fit Prophet Model on Train set for validation evaluation
        prophet_val = Prophet(
            growth='linear',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )
        prophet_val.add_regressor('Temperature_C')
        prophet_val.add_regressor('Occupancy_Ratio')
        
        prophet_val.fit(train_p)
        
        val_forecast = prophet_val.predict(test_p[['ds', 'Temperature_C', 'Occupancy_Ratio']])
        prophet_metrics = calculate_metrics(y_test_actual, val_forecast['yhat'].values)
        print(f"Prophet Model Metrics: {prophet_metrics}")

        # 3. Fit Full Prophet Model on entire historical dataset for deployment forecasting
        full_prophet = Prophet(
            growth='linear',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )
        full_prophet.add_regressor('Temperature_C')
        full_prophet.add_regressor('Occupancy_Ratio')
        full_prophet.fit(p_df)
        
        self.model = full_prophet
        joblib.dump(full_prophet, PROPHET_MODEL_PATH)

        # Improvement metric over baseline
        mape_improvement = round(float(baseline_metrics['MAPE'] - prophet_metrics['MAPE']), 2)

        self.meta = {
            'trained_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': n_total,
            'model_type': 'Facebook Prophet (Additive Seasonality + Regressors)',
            'prophet_metrics': prophet_metrics,
            'baseline_metrics': baseline_metrics,
            'mape_improvement_vs_baseline': mape_improvement,
            'features_used': ['ds', 'y', 'Temperature_C', 'Occupancy_Ratio']
        }

        with open(METADATA_PATH, 'w') as f:
            json.dump(self.meta, f, indent=2)

        return self.meta

    def predict(self, df, horizon_hours=24):
        """
        Generates future Prophet forecast for specified horizon_hours.
        Includes forecast lower/upper confidence bounds, trend, and seasonal components.
        """
        if self.model is None:
            if os.path.exists(PROPHET_MODEL_PATH):
                self.model = joblib.load(PROPHET_MODEL_PATH)
            else:
                self.train(df)

        last_timestamp = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Datetime']).max()
        future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq='h')
        
        # Build future dataframe for Prophet with regressors
        future_p = pd.DataFrame({'ds': future_dates})
        
        # Estimate regressors for future (diurnal temperature curve & occupancy pattern)
        future_hours = future_dates.hour
        future_dayofweek = future_dates.dayofweek
        future_months = future_dates.month
        
        recent_temp = df['Temperature_C'].iloc[-24:].mean() if 'Temperature_C' in df.columns else 20.0
        future_p['Temperature_C'] = np.round(
            recent_temp + 3.5 * np.sin(2 * np.pi * (future_hours - 9) / 24) + np.random.normal(0, 0.5, len(future_p)), 1
        )
        
        future_weekend_factor = np.where(future_dayofweek >= 5, 0.7, 1.0)
        future_base_occ = np.where((future_hours >= 8) & (future_hours <= 18), 0.7, 0.25)
        future_p['Occupancy_Ratio'] = np.round(np.clip(future_base_occ * future_weekend_factor, 0.05, 1.0), 2)
        
        forecast = self.model.predict(future_p)
        
        predictions = []
        for i, row in forecast.iterrows():
            yhat = max(0.05, round(float(row['yhat']), 3))
            yhat_lower = max(0.01, round(float(row['yhat_lower']), 3))
            yhat_upper = round(float(row['yhat_upper']), 3)
            
            predictions.append({
                'Timestamp': row['ds'].strftime('%Y-%m-%d %H:%M'),
                'Forecast_kWh': yhat,
                'yhat_lower': yhat_lower,
                'yhat_upper': yhat_upper,
                'Trend': round(float(row['trend']), 3),
                'Weekly_Seasonality': round(float(row.get('weekly', 0.0)), 3),
                'Daily_Seasonality': round(float(row.get('daily', 0.0)), 3),
                'Temperature_C': float(future_p['Temperature_C'].iloc[i]),
                'Occupancy_Ratio': float(future_p['Occupancy_Ratio'].iloc[i])
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
        historical_actuals.append({
            'Timestamp': pd.to_datetime(dt_val).strftime('%Y-%m-%d %H:%M'),
            'Actual_kWh': round(float(row['Energy_kWh']), 3),
            'Temperature_C': round(float(row.get('Temperature_C', 20.0)), 1),
            'Occupancy_Ratio': round(float(row.get('Occupancy_Ratio', 0.5)), 2)
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
