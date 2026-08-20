import sys
import os
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.preprocessing import get_processed_data
from backend.app.services.forecasting import get_forecast_results

client = TestClient(app)


def test_preprocessing_columns():
    df = get_processed_data()
    assert 'Energy_kWh' in df.columns
    assert 'Temperature_C' in df.columns
    assert 'Humidity_pct' in df.columns
    assert 'Occupancy_Level' in df.columns
    assert 'Occupancy_Ratio' in df.columns
    assert set(df['Occupancy_Level'].unique()).issubset({'Low', 'Medium', 'High'})
    assert len(df) > 100


def test_prophet_forecasting():
    df = get_processed_data()
    res = get_forecast_results(df, horizon_hours=24, force_retrain=False)
    assert 'metadata' in res
    assert 'predictions' in res
    assert len(res['predictions']) == 24
    meta = res['metadata']
    assert 'prophet_metrics' in meta
    assert 'baseline_metrics' in meta
    assert 'MAPE' in meta['prophet_metrics']


def test_fastapi_endpoints():
    r_summary = client.get('/api/summary')
    assert r_summary.status_code == 200
    assert r_summary.json()['status'] == 'success'

    r_consumption = client.get('/api/consumption?period=daily&limit=30')
    assert r_consumption.status_code == 200
    assert len(r_consumption.json()['data']) > 0

    r_hourly = client.get('/api/hourly')
    assert r_hourly.status_code == 200
    assert 'hourly_profile' in r_hourly.json()

    r_forecast = client.get('/api/forecast?horizon=24')
    assert r_forecast.status_code == 200
    assert len(r_forecast.json()['data']['predictions']) == 24

    r_anomalies = client.get('/api/anomalies?threshold=2.0')
    assert r_anomalies.status_code == 200

    r_peaks = client.get('/api/peaks')
    assert r_peaks.status_code == 200

    r_recs = client.get('/api/recommendations')
    assert r_recs.status_code == 200

    r_savings = client.post('/api/calculate-savings', json={
        'monthly_kWh': 500,
        'tariff_rate': 0.18,
        'currency_symbol': '$'
    })
    assert r_savings.status_code == 200
    assert r_savings.json()['data']['current_estimates']['flat_monthly_cost'] == 90.0


if __name__ == '__main__':
    pytest.main(['-v', __file__])
