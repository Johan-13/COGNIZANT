import pandas as pd
import numpy as np


def analyze_peak_usage(df):
    """
    Performs Peak Load Analysis and Demand-Side Management (DSM) Load Shifting Optimization.
    Computes original vs optimized load curves, Peak-to-Average Ratio (PAR), Load Factor,
    and appliance dispatch rescheduling strategies with financial savings in Indian Rupees (₹).
    """
    overall_avg_kWh = float(df['Energy_kWh'].mean())

    cols = {'Energy_kWh': ['mean', 'max', 'min', 'std']}
    if 'Temperature_C' in df.columns:
        cols['Temperature_C'] = 'mean'
    if 'Occupancy_Ratio' in df.columns:
        cols['Occupancy_Ratio'] = 'mean'

    hourly = df.groupby('Hour').agg(cols)
    hourly.columns = ['_'.join(c).strip('_') for c in hourly.columns]
    hourly = hourly.reset_index()

    hourly['HourLabel'] = hourly['Hour'].apply(lambda h: f"{h:02d}:00")
    hourly['TimeRange'] = hourly['Hour'].apply(lambda h: f"{h:02d}:00 - {(h+1)%24:02d}:00")
    hourly['Energy_kWh_mean'] = hourly['Energy_kWh_mean'].round(3)
    hourly['Energy_kWh_max'] = hourly['Energy_kWh_max'].round(3)
    hourly['Energy_kWh_min'] = hourly['Energy_kWh_min'].round(3)
    hourly['pct_above_avg'] = (((hourly['Energy_kWh_mean'] - overall_avg_kWh) / overall_avg_kWh) * 100).round(1)

    # 1. Classify diurnal windows
    morning_peak = float(hourly[(hourly['Hour'] >= 7) & (hourly['Hour'] <= 10)]['Energy_kWh_mean'].mean())
    evening_peak = float(hourly[(hourly['Hour'] >= 17) & (hourly['Hour'] <= 22)]['Energy_kWh_mean'].mean())
    overnight_base = float(hourly[(hourly['Hour'] >= 0) & (hourly['Hour'] <= 5)]['Energy_kWh_mean'].mean())

    # 2. Mathematical Load Shifting Model (Minimizing Peak-to-Average Ratio)
    # Target: Shift flexible load from top peak hours (17:00 - 22:00) to off-peak valley (01:00 - 05:00)
    original_curve = hourly['Energy_kWh_mean'].values.copy()
    optimized_curve = original_curve.copy()

    peak_mask = (hourly['Hour'] >= 17) & (hourly['Hour'] <= 22)
    offpeak_mask = (hourly['Hour'] >= 1) & (hourly['Hour'] <= 5)

    # Calculate total shiftable load: 35% of demand above daily mean during peak window
    peak_excess = np.maximum(0, original_curve[peak_mask] - overall_avg_kWh)
    shiftable_kwh_per_peak_hour = peak_excess * 0.40  # 40% shift efficiency
    total_shifted_daily_kWh = float(np.sum(shiftable_kwh_per_peak_hour))

    optimized_curve[peak_mask] -= shiftable_kwh_per_peak_hour
    # Distribute the shifted load evenly across valley hours
    offpeak_count = np.sum(offpeak_mask)
    if offpeak_count > 0:
        optimized_curve[offpeak_mask] += (total_shifted_daily_kWh / offpeak_count)

    hourly['Optimized_kWh'] = np.round(optimized_curve, 3)
    hourly['Shifted_kWh'] = np.round(original_curve - optimized_curve, 3)

    # 3. Peak-to-Average Ratio (PAR) & Load Factor Metrics
    orig_peak = float(np.max(original_curve))
    opt_peak = float(np.max(optimized_curve))

    orig_par = round(orig_peak / max(overall_avg_kWh, 0.01), 2)
    opt_par = round(opt_peak / max(overall_avg_kWh, 0.01), 2)
    par_reduction_pct = round(((orig_par - opt_par) / max(orig_par, 0.01)) * 100, 1)

    orig_load_factor = round((overall_avg_kWh / max(orig_peak, 0.01)) * 100, 1)
    opt_load_factor = round((overall_avg_kWh / max(opt_peak, 0.01)) * 100, 1)

    # Financial Rupee (₹) Savings from Peak Shift (Tariff differential ₹8.50 peak vs ₹3.35 off-peak = ₹5.15/kWh saved)
    monthly_shifted_kWh = round(total_shifted_daily_kWh * 30.0, 1)
    monthly_peak_savings_inr = round(monthly_shifted_kWh * 5.15, 2)

    # 4. Top Peak Hours Table with Contextual Strategy
    top_peak_hours = hourly.sort_values(by='Energy_kWh_mean', ascending=False).head(6).copy()

    # 5. Appliance Rescheduling Dispatch Table
    appliance_schedules = [
        {
            'appliance': 'EV Charger / Vehicle Charging',
            'power_rating': '3.3 kW',
            'current_window': '19:00 – 22:00 (Peak)',
            'optimal_window': '01:00 – 04:00 (Valley)',
            'daily_kwh': round(total_shifted_daily_kWh * 0.45, 2),
            'monthly_savings_inr': round(total_shifted_daily_kWh * 0.45 * 30 * 5.15, 2),
            'status': 'Recommended'
        },
        {
            'appliance': 'Electric Water Heater / Geyser',
            'power_rating': '2.0 kW',
            'current_window': '07:00 – 09:00 (Morning Peak)',
            'optimal_window': '04:30 – 06:30 (Early Off-Peak)',
            'daily_kwh': round(total_shifted_daily_kWh * 0.25, 2),
            'monthly_savings_inr': round(total_shifted_daily_kWh * 0.25 * 30 * 5.15, 2),
            'status': 'Scheduled'
        },
        {
            'appliance': 'Dishwasher & Laundry Wash Cycle',
            'power_rating': '1.5 kW',
            'current_window': '20:00 – 22:00 (Peak)',
            'optimal_window': '23:00 – 01:00 (Night Off-Peak)',
            'daily_kwh': round(total_shifted_daily_kWh * 0.18, 2),
            'monthly_savings_inr': round(total_shifted_daily_kWh * 0.18 * 30 * 5.15, 2),
            'status': 'Automated'
        },
        {
            'appliance': 'HVAC Thermal Pre-Cooling',
            'power_rating': '2.5 kW',
            'current_window': '18:00 – 21:00 (Coincident Peak)',
            'optimal_window': '14:00 – 16:30 (Solar / Pre-Peak)',
            'daily_kwh': round(total_shifted_daily_kWh * 0.12, 2),
            'monthly_savings_inr': round(total_shifted_daily_kWh * 0.12 * 30 * 5.15, 2),
            'status': 'Active'
        }
    ]

    return {
        'overall_avg_kWh': round(overall_avg_kWh, 3),
        'hourly_averages': hourly.to_dict(orient='records'),  # supports both keys
        'hourly_breakdown': hourly.to_dict(orient='records'),
        'top_peak_hours': top_peak_hours.to_dict(orient='records'),
        'period_averages': {
            'morning_peak_7_to_10_kWh': round(morning_peak, 3),
            'evening_peak_17_to_22_kWh': round(evening_peak, 3),
            'overnight_baseline_0_to_5_kWh': round(overnight_base, 3)
        },
        'optimization_metrics': {
            'original_peak_kW': round(orig_peak, 2),
            'optimized_peak_kW': round(opt_peak, 2),
            'original_par': orig_par,
            'optimized_par': opt_par,
            'par_reduction_pct': par_reduction_pct,
            'original_load_factor_pct': orig_load_factor,
            'optimized_load_factor_pct': opt_load_factor,
            'daily_shiftable_kWh': round(total_shifted_daily_kWh, 2),
            'monthly_shiftable_kWh': monthly_shifted_kWh,
            'monthly_savings_inr': monthly_peak_savings_inr
        },
        'appliance_schedules': appliance_schedules
    }
