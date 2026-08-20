import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SARIMA_MODEL_PATH = os.path.join(MODEL_DIR, 'sarima_model.pkl')
ML_MODEL_PATH = os.path.join(MODEL_DIR, 'ml_forecast_model.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'forecasting_meta.json')


def calculate_metrics(y_true, y_pred):
    """
    Computes MAE, RMSE, and MAPE.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid division by zero in MAPE
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'MAPE': round(float(mape), 2)
    }


def create_lag_features(df, target_col='Energy_kWh', max_lags=24):
    """
    Generates lag and seasonal calendar features for ML forecasting.
    """
    data = df[[target_col]].copy()
    data['Hour'] = data.index.hour
    data['DayOfWeek'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)

    # Lags: 1h, 2h, 24h (1 day), 168h (1 week)
    lags = [1, 2, 3, 24, 48, 168]
    for lag in lags:
        if len(data) > lag:
            data[f'lag_{lag}'] = data[target_col].shift(lag)

    # Rolling statistics
    data['rolling_mean_24'] = data[target_col].shift(1).rolling(window=24, min_periods=1).mean()
    data['rolling_std_24'] = data[target_col].shift(1).rolling(window=24, min_periods=1).std().fillna(0)

    data = data.dropna()
    return data


class EnergyForecaster:
    """
    Forecasting Engine using SARIMA with robust Ridge/ML regression fallback.
    """
    def __init__(self):
        self.model = None
        self.ml_model = None
        self.meta = {}

    def train(self, df, model_type='sarima', test_ratio=0.15):
        """
        Trains the forecasting model on historical hourly dataframe.
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Prepare daily or hourly series for SARIMA
        y = df['Energy_kWh'].copy()
        
        n_total = len(y)
        split_idx = int(n_total * (1 - test_ratio))
        train_y, test_y = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Training forecasting model (Total samples: {n_total}, Train: {len(train_y)}, Test: {len(test_y)})...")
        
        # 1. Naive Baseline Evaluation on Test Set (Same hour previous week lag_168 or previous day lag_24)
        naive_pred = y.shift(24).iloc[split_idx:]
        # Align test set for valid baseline comparison
        valid_indices = naive_pred.dropna().index
        test_y_aligned = test_y.loc[valid_indices]
        naive_pred_aligned = naive_pred.loc[valid_indices]
        
        baseline_metrics = calculate_metrics(test_y_aligned, naive_pred_aligned)
        print(f"Naive Baseline Metrics: {baseline_metrics}")

        # 2. Train ML Lag Model (Always fit for high-speed multi-step forecasting)
        ml_data = create_lag_features(df, target_col='Energy_kWh')
        feature_cols = [c for c in ml_data.columns if c != 'Energy_kWh']
        
        X_train = ml_data.iloc[:int(len(ml_data) * (1 - test_ratio))][feature_cols]
        y_train_ml = ml_data.iloc[:int(len(ml_data) * (1 - test_ratio))]['Energy_kWh']
        X_test = ml_data.iloc[int(len(ml_data) * (1 - test_ratio)):][feature_cols]
        y_test_ml = ml_data.iloc[int(len(ml_data) * (1 - test_ratio)):]['Energy_kWh']

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train_ml)
        ml_pred = ridge.predict(X_test)
        ml_metrics = calculate_metrics(y_test_ml, ml_pred)
        print(f"ML Ridge Model Metrics: {ml_metrics}")
        
        self.ml_model = ridge
        joblib.dump({'model': ridge, 'feature_cols': feature_cols}, ML_MODEL_PATH)

        # 3. Train SARIMA Model (resampled to daily or recent hourly window for performance)
        sarima_metrics = ml_metrics
        sarima_fitted = False
        
        try:
            # Use daily aggregates or last 90 days of hourly data for fast SARIMA fitting
            sarima_input = df['Energy_kWh'].tail(90 * 24).resample('D').sum() if len(df) > 90 * 24 else df['Energy_kWh'].resample('D').sum()
            
            s_split = int(len(sarima_input) * (1 - test_ratio))
            s_train, s_test = sarima_input.iloc[:s_split], sarima_input.iloc[s_split:]
            
            sarima_model = SARIMAX(
                s_train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = sarima_model.fit(disp=False)
            
            if len(s_test) > 0:
                s_pred = res.forecast(steps=len(s_test))
                sarima_metrics = calculate_metrics(s_test, s_pred)
                print(f"SARIMA Daily Metrics: {sarima_metrics}")
                
            # Fit full SARIMA on available series
            full_sarima = SARIMAX(
                sarima_input,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            
            self.model = full_sarima
            joblib.dump(full_sarima, SARIMA_MODEL_PATH)
            sarima_fitted = True
        except Exception as e:
            print(f"SARIMA fitting warning (falling back to ML model): {e}")

        # Metadata output
        self.meta = {
            'trained_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points_trained': n_total,
            'model_used': 'SARIMA (1,1,1)x(1,0,1)_7 + Ridge' if sarima_fitted else 'ML Ridge Regressor',
            'sarima_fitted': sarima_fitted,
            'metrics': sarima_metrics if sarima_fitted else ml_metrics,
            'baseline_metrics': baseline_metrics,
            'ml_metrics': ml_metrics
        }

        with open(METADATA_PATH, 'w') as f:
            json.dump(self.meta, f, indent=2)

        return self.meta

    def predict(self, df, horizon_hours=24):
        """
        Generates forecast for the specified horizon (in hours).
        Returns DataFrame with Actuals (if available) and Forecast values.
        """
        # Load models if not already in memory
        if self.ml_model is None and os.path.exists(ML_MODEL_PATH):
            ml_data_loaded = joblib.load(ML_MODEL_PATH)
            self.ml_model = ml_data_loaded['model']
            self.feature_cols = ml_data_loaded['feature_cols']

        last_timestamp = df.index.max()
        future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq='h')

        # Multi-step autoregressive prediction using feature engineering
        history_df = df[['Energy_kWh']].copy()
        predictions = []

        for future_date in future_dates:
            # Build temporary row at future_date
            temp_df = pd.DataFrame(index=[future_date], data={'Energy_kWh': np.nan})
            combined = pd.concat([history_df, temp_df])
            
            feat = create_lag_features(combined, target_col='Energy_kWh')
            if len(feat) > 0 and hasattr(self, 'feature_cols'):
                X_feat = feat.iloc[-1:][self.feature_cols]
                pred_val = float(self.ml_model.predict(X_feat)[0])
            else:
                # Fallback to mean of recent same hour if feature creation incomplete
                same_hours = history_df[history_df.index.hour == future_date.hour]['Energy_kWh']
                pred_val = float(same_hours.mean()) if len(same_hours) > 0 else float(history_df['Energy_kWh'].mean())

            pred_val = max(0.05, round(pred_val, 3))
            predictions.append(pred_val)
            
            # Append predicted value to history for next step lag creation
            history_df.loc[future_date] = pred_val

        forecast_df = pd.DataFrame({
            'Timestamp': future_dates.strftime('%Y-%m-%d %H:%M'),
            'Forecast_kWh': predictions
        })

        return forecast_df.to_dict(orient='records')


def get_forecast_results(df, horizon_hours=24, force_retrain=False):
    """
    Wrapper function to get or generate forecast predictions & metrics.
    """
    forecaster = EnergyForecaster()
    
    # Train if models don't exist or retrain requested
    if force_retrain or not os.path.exists(METADATA_PATH):
        meta = forecaster.train(df)
    else:
        with open(METADATA_PATH, 'r') as f:
            meta = json.load(f)

    predictions = forecaster.predict(df, horizon_hours=horizon_hours)

    # Actual vs Predicted comparison on recent historical test data (last 48h)
    recent_historical = df.tail(48).copy().reset_index()
    historical_actual_vs_pred = []
    
    # Simple evaluation on historical recent segment
    for _, row in recent_historical.iterrows():
        historical_actual_vs_pred.append({
            'Timestamp': row['Datetime'].strftime('%Y-%m-%d %H:%M'),
            'Actual_kWh': round(float(row['Energy_kWh']), 3)
        })

    return {
        'horizon_hours': horizon_hours,
        'metadata': meta,
        'predictions': predictions,
        'historical_test_actuals': historical_actual_vs_pred
    }
