import os
from flask import Flask, render_template, jsonify, request

from src.preprocessing import get_processed_data, RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.analysis import (
    get_summary_metrics, get_hourly_profile, get_day_of_week_profile,
    get_monthly_profile, get_consumption_time_series
)
from src.forecasting import get_forecast_results, EnergyForecaster
from src.anomaly_detection import detect_anomalies
from src.peak_analysis import analyze_peak_usage
from src.savings import generate_recommendations, calculate_cost_and_savings

app = Flask(__name__)

# Global cache for dataset to prevent repeated disk re-reads on fast navigation
_DATA_CACHE = None


def load_dataset(force_reload=False):
    global _DATA_CACHE
    if _DATA_CACHE is None or force_reload:
        _DATA_CACHE = get_processed_data(force_reprocess=force_reload)
    return _DATA_CACHE


# --- HTML Page Routes ---

@app.route('/')
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')


@app.route('/forecast')
def forecast():
    return render_template('forecast.html', active_page='forecast')


@app.route('/anomalies')
def anomalies():
    return render_template('anomalies.html', active_page='anomalies')


@app.route('/peaks')
def peaks():
    return render_template('peaks.html', active_page='peaks')


@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html', active_page='recommendations')


@app.route('/savings')
def savings():
    return render_template('savings.html', active_page='savings')


# --- REST API Endpoints ---

@app.route('/api/summary', methods=['GET'])
def api_summary():
    try:
        df = load_dataset()
        summary = get_summary_metrics(df)
        
        # Add basic default tariff estimation
        avg_monthly_kwh = summary['avg_daily_kWh'] * 30
        cost_info = calculate_cost_and_savings(avg_monthly_kwh, tariff_rate=0.15)
        summary['estimated_monthly_cost'] = cost_info['current_estimates']['monthly_cost']
        summary['potential_monthly_savings'] = cost_info['savings_scenarios'][2]['saved_cost_monthly']  # 15% scenario
        
        anom_res = detect_anomalies(df)
        summary['anomaly_count'] = anom_res['summary']['total_anomalies']
        
        return jsonify({'status': 'success', 'data': summary})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/consumption', methods=['GET'])
def api_consumption():
    try:
        df = load_dataset()
        period = request.args.get('period', 'daily')
        limit = int(request.args.get('limit', 90))
        
        series_data = get_consumption_time_series(df, period=period, limit=limit)
        return jsonify({'status': 'success', 'period': period, 'data': series_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/hourly', methods=['GET'])
def api_hourly():
    try:
        df = load_dataset()
        hourly_prof = get_hourly_profile(df)
        dow_prof = get_day_of_week_profile(df)
        return jsonify({
            'status': 'success',
            'hourly_profile': hourly_prof,
            'day_of_week_profile': dow_prof
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    try:
        df = load_dataset()
        horizon = int(request.args.get('horizon', 24))
        force_retrain = request.args.get('retrain', 'false').lower() == 'true'
        
        forecast_data = get_forecast_results(df, horizon_hours=horizon, force_retrain=force_retrain)
        return jsonify({'status': 'success', 'data': forecast_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/anomalies', methods=['GET'])
def api_anomalies():
    try:
        df = load_dataset()
        z_threshold = float(request.args.get('threshold', 2.0))
        res = detect_anomalies(df, z_threshold=z_threshold)
        return jsonify({'status': 'success', 'data': res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/peaks', methods=['GET'])
def api_peaks():
    try:
        df = load_dataset()
        peak_res = analyze_peak_usage(df)
        return jsonify({'status': 'success', 'data': peak_res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    try:
        df = load_dataset()
        anom_res = detect_anomalies(df)
        peak_res = analyze_peak_usage(df)
        recs = generate_recommendations(df, anomalies_summary=anom_res['summary'], peak_info=peak_res)
        return jsonify({'status': 'success', 'data': recs})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/calculate-savings', methods=['POST'])
def api_calculate_savings():
    try:
        data = request.get_json(silent=True) or request.form
        tariff_rate = float(data.get('tariff_rate', 0.15))
        
        if 'monthly_kWh' in data and data['monthly_kWh']:
            monthly_kWh = float(data['monthly_kWh'])
        else:
            df = load_dataset()
            avg_daily = float(df['Energy_kWh'].resample('D').sum().mean())
            monthly_kWh = avg_daily * 30.0

        res = calculate_cost_and_savings(monthly_kWh=monthly_kWh, tariff_rate=tariff_rate)
        return jsonify({'status': 'success', 'data': res})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    try:
        df = load_dataset(force_reload=True)
        forecaster = EnergyForecaster()
        meta = forecaster.train(df)
        return jsonify({'status': 'success', 'message': 'Model successfully retrained.', 'metadata': meta})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Initial data caching on boot
    print("Initializing Smart Energy Consumption Optimizer server...")
    load_dataset()
    app.run(host='0.0.0.0', port=5000, debug=True)
