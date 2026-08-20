"""
Services module package containing domain logic and ML engines.
"""
from backend.app.services.preprocessing import get_processed_data, preprocess_data, generate_sample_dataset
from backend.app.services.analysis import (
    get_summary_metrics, get_hourly_profile, get_day_of_week_profile,
    get_monthly_profile, get_consumption_time_series
)
from backend.app.services.forecasting import get_forecast_results, EnergyForecaster, calculate_metrics
from backend.app.services.anomaly_detection import detect_anomalies
from backend.app.services.peak_analysis import analyze_peak_usage
from backend.app.services.savings import generate_recommendations, calculate_cost_and_savings
from backend.app.services.streamer import LiveDataStreamer, streamer_engine
