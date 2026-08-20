import pandas as pd
import numpy as np


def generate_recommendations(df, anomalies_summary=None, peak_info=None):
    """
    Generates rule-based energy saving and load-shifting recommendations based on time-series, weather, and occupancy analysis.
    """
    recommendations = []

    overall_avg = float(df['Energy_kWh'].mean())
    hourly_avg = df.groupby('Hour')['Energy_kWh'].mean()

    # Rule 1: Evening Peak Load Shifting (18:00 - 22:00)
    evening_peak = hourly_avg[(hourly_avg.index >= 18) & (hourly_avg.index <= 22)].mean()
    if evening_peak > overall_avg * 1.2:
        pct_higher = round(((evening_peak - overall_avg) / overall_avg) * 100, 1)
        recommendations.append({
            'id': 'rec_peak_shift',
            'category': 'Load Shifting',
            'title': 'Shift Flexible Loads Away from Peak Window (18:00 - 22:00)',
            'description': f"Peak period power consumption is {pct_higher}% above your daily baseline. Shifting heavy appliances (washers, pumps, thermal pre-cooling) to off-peak hours cuts peak demand charges.",
            'potential_impact': 'High',
            'estimated_kwh_saving_monthly': round((evening_peak - overall_avg) * 4 * 30 * 0.45, 1),
            'actionable_step': 'Program smart timer delays or pre-cool/pre-heat spaces prior to 17:00.'
        })

    # Rule 2: Low Occupancy Power Waste
    if 'Occupancy_Level' in df.columns:
        low_occ_df = df[df['Occupancy_Level'] == 'Low']
    elif 'Occupancy_Ratio' in df.columns:
        low_occ_df = df[df['Occupancy_Ratio'] < 0.3]
    else:
        low_occ_df = pd.DataFrame()

    if len(low_occ_df) > 0:
        unattended_draw = float(low_occ_df['Energy_kWh'].mean())
        if unattended_draw > overall_avg * 0.6:
            recommendations.append({
                'id': 'rec_occupancy_waste',
                'category': 'Occupancy Automation',
                'title': 'Automate HVAC and Lighting Shutdown During Low Occupancy Hours',
                'description': f"Average energy draw during Low Occupancy periods is {round(unattended_draw, 2)} kWh/hr. Automated occupancy sensors can eliminate unneeded cooling/lighting.",
                'potential_impact': 'High',
                'estimated_kwh_saving_monthly': round(unattended_draw * 0.4 * 8 * 30, 1),
                'actionable_step': 'Install PIR motion sensors and smart thermostat occupancy setback schedules.'
            })

    # Rule 3: Overnight Standby/Vampire Draw (01:00 - 05:00)
    night_load = hourly_avg[(hourly_avg.index >= 1) & (hourly_avg.index <= 5)].mean()
    if night_load > 0.25:
        recommendations.append({
            'id': 'rec_phantom_load',
            'category': 'Standby Reduction',
            'title': 'Eliminate Overnight Standby Power (Vampire Draw)',
            'description': f"Overnight baseline draw averages {round(night_load, 2)} kWh/hr when building activity is minimal.",
            'potential_impact': 'Medium',
            'estimated_kwh_saving_monthly': round(night_load * 0.35 * 6 * 30, 1),
            'actionable_step': 'Use smart power strips that automatically cut standby power to non-essential loads overnight.'
        })

    # Rule 4: High Anomaly Frequency Inspection
    if anomalies_summary and anomalies_summary.get('total_anomalies', 0) > 8:
        recommendations.append({
            'id': 'rec_appliance_check',
            'category': 'Maintenance & Audit',
            'title': 'Inspect Equipment for Abnormal Power Draw Spikes',
            'description': f"{anomalies_summary['total_anomalies']} power draw spikes were flagged by rolling Z-score anomaly detection.",
            'potential_impact': 'High',
            'estimated_kwh_saving_monthly': round(anomalies_summary['total_anomalies'] * 1.8, 1),
            'actionable_step': 'Inspect HVAC compressor coils, refrigeration seals, and water heater elements for degradation.'
        })

    # Rule 5: Weather-Driven HVAC Pre-Conditioning
    if 'Temperature_C' in df.columns:
        hot_hours = df[df['Temperature_C'] > 28.0]
        if len(hot_hours) > 0:
            recommendations.append({
                'id': 'rec_hvac_precool',
                'category': 'Thermal Management',
                'title': 'Leverage Pre-Cooling Prior to High Ambient Temperature Peaks',
                'description': f"Detected {len(hot_hours)} hours with ambient temperatures above 28°C. Pre-cooling spaces during early morning off-peak hours reduces peak HVAC compressor strain.",
                'potential_impact': 'Medium',
                'estimated_kwh_saving_monthly': round(overall_avg * 24 * 30 * 0.08, 1),
                'actionable_step': 'Set cooling target 2°C lower between 06:00-09:00 AM, then let temperature coast during peak heat hours.'
            })

    return recommendations


SLAB_TARIFF_STRUCTURE = [
    {'min': 0, 'max': 50, 'rate': 3.35, 'label': '0 – 50 units (₹3.35/kWh)'},
    {'min': 50, 'max': 100, 'rate': 4.25, 'label': '51 – 100 units (₹4.25/kWh)'},
    {'min': 100, 'max': 150, 'rate': 5.35, 'label': '101 – 150 units (₹5.35/kWh)'},
    {'min': 150, 'max': 200, 'rate': 7.20, 'label': '151 – 200 units (₹7.20/kWh)'},
    {'min': 200, 'max': 250, 'rate': 8.50, 'label': '201 – 250 units (₹8.50/kWh)'},
    {'min': 250, 'max': 999999999, 'rate': 8.50, 'label': '> 250 units (₹8.50/kWh)'}
]


def calculate_slab_cost(units_kWh):
    """
    Computes energy cost using Indian tiered slab tariff:
    - 0 to 50 kWh: ₹3.35 / kWh
    - 51 to 100 kWh: ₹4.25 / kWh
    - 101 to 150 kWh: ₹5.35 / kWh
    - 151 to 200 kWh: ₹7.20 / kWh
    - 201 to 250 kWh: ₹8.50 / kWh
    - > 250 kWh: ₹8.50 / kWh
    """
    u = max(0.0, float(units_kWh))
    total_cost = 0.0
    slab_breakdown = []
    remaining = u

    for slab in SLAB_TARIFF_STRUCTURE:
        slab_size = slab['max'] - slab['min']
        if remaining > 0:
            units = min(remaining, slab_size)
            cost = units * slab['rate']
            total_cost += cost
            remaining -= units
            slab_breakdown.append({
                'slab': slab['label'],
                'units': round(units, 1),
                'rate': slab['rate'],
                'cost': round(cost, 2)
            })
        else:
            break

    return round(total_cost, 2), slab_breakdown


def calculate_cost_and_savings(monthly_kWh=None, tariff_rate=None, peak_tariff_multiplier=1.4, currency_symbol='₹', df=None):
    """
    Calculates estimated monthly/yearly energy costs using Indian Rupee (₹) slab tariffs,
    Time-Of-Use peak tariff breakdown, and quantified total savings referencing both
    Actual historical values and Prophet Predicted future values.
    """
    is_flat_override = (tariff_rate is not None and float(tariff_rate) > 0 and float(tariff_rate) not in (0.15, 0.0))
    if not is_flat_override:
        currency_symbol = '₹'
    else:
        tariff_rate = float(tariff_rate)

    # 1. Determine Actual Monthly kWh
    if monthly_kWh is not None and monthly_kWh > 0:
        actual_monthly_kWh = float(monthly_kWh)
    elif df is not None and not df.empty:
        avg_daily = float(df['Energy_kWh'].resample('D').sum().mean())
        actual_monthly_kWh = round(avg_daily * 30.0, 2)
    else:
        actual_monthly_kWh = 1430.0

    # 2. Determine Prophet Predicted Monthly kWh
    predicted_monthly_kWh = actual_monthly_kWh
    if df is not None and not df.empty:
        try:
            from backend.app.services.forecasting import EnergyForecaster
            forecaster = EnergyForecaster()
            preds = forecaster.predict(df, horizon_hours=168)
            if preds and len(preds) > 0:
                avg_pred_daily = sum(p['Forecast_kWh'] for p in preds) / (len(preds) / 24.0)
                predicted_monthly_kWh = round(avg_pred_daily * 30.0, 2)
        except Exception:
            predicted_monthly_kWh = round(actual_monthly_kWh * 1.05, 2)

    # 3. Baseline Financials
    if is_flat_override:
        actual_flat_monthly = round(actual_monthly_kWh * tariff_rate, 2)
        actual_slabs = []
        predicted_flat_monthly = round(predicted_monthly_kWh * tariff_rate, 2)
        predicted_slabs = []
    else:
        actual_flat_monthly, actual_slabs = calculate_slab_cost(actual_monthly_kWh)
        predicted_flat_monthly, predicted_slabs = calculate_slab_cost(predicted_monthly_kWh)

    actual_flat_yearly = round(actual_flat_monthly * 12, 2)
    predicted_flat_yearly = round(predicted_flat_monthly * 12, 2)

    variance_kwh = round(predicted_monthly_kWh - actual_monthly_kWh, 2)
    variance_cost = round(predicted_flat_monthly - actual_flat_monthly, 2)
    variance_pct = round((variance_kwh / max(actual_monthly_kWh, 1)) * 100, 1)

    # 4. Itemized Total Savings Strategies referencing Actual vs Predicted
    eff_rate = tariff_rate if is_flat_override else 8.50

    # Strategy 1: Peak Load Shifting (18:00 - 22:00)
    actual_peak_shift_kwh = round(actual_monthly_kWh * 0.30 * 0.40, 1)
    actual_peak_shift_savings = round(actual_peak_shift_kwh * eff_rate * (peak_tariff_multiplier - 1.0), 2)
    pred_peak_shift_kwh = round(predicted_monthly_kWh * 0.30 * 0.40, 1)
    pred_peak_shift_savings = round(pred_peak_shift_kwh * eff_rate * (peak_tariff_multiplier - 1.0), 2)

    # Strategy 2: Occupancy Automation & Low-Activity Waste
    actual_occ_kwh = round(actual_monthly_kWh * 0.15, 1)
    if is_flat_override:
        actual_occ_savings = round(actual_occ_kwh * tariff_rate, 2)
        pred_occ_kwh = round(predicted_monthly_kWh * 0.15, 1)
        pred_occ_savings = round(pred_occ_kwh * tariff_rate, 2)
    else:
        actual_occ_savings = round(actual_flat_monthly - calculate_slab_cost(actual_monthly_kWh - actual_occ_kwh)[0], 2)
        pred_occ_kwh = round(predicted_monthly_kWh * 0.15, 1)
        pred_occ_savings = round(predicted_flat_monthly - calculate_slab_cost(predicted_monthly_kWh - pred_occ_kwh)[0], 2)

    # Strategy 3: Vampire / Standby Power Elimination (01:00 - 05:00)
    actual_vampire_kwh = round(actual_monthly_kWh * 0.08, 1)
    if is_flat_override:
        actual_vampire_savings = round(actual_vampire_kwh * tariff_rate, 2)
        pred_vampire_kwh = round(predicted_monthly_kWh * 0.08, 1)
        pred_vampire_savings = round(pred_vampire_kwh * tariff_rate, 2)
    else:
        actual_vampire_savings = round(actual_flat_monthly - calculate_slab_cost(actual_monthly_kWh - actual_vampire_kwh)[0], 2)
        pred_vampire_kwh = round(predicted_monthly_kWh * 0.08, 1)
        pred_vampire_savings = round(predicted_flat_monthly - calculate_slab_cost(predicted_monthly_kWh - pred_vampire_kwh)[0], 2)

    # Strategy 4: Weather Pre-Cooling Optimization
    actual_weather_kwh = round(actual_monthly_kWh * 0.06, 1)
    if is_flat_override:
        actual_weather_savings = round(actual_weather_kwh * tariff_rate, 2)
        pred_weather_kwh = round(predicted_monthly_kWh * 0.06, 1)
        pred_weather_savings = round(pred_weather_kwh * tariff_rate, 2)
    else:
        actual_weather_savings = round(actual_flat_monthly - calculate_slab_cost(actual_monthly_kWh - actual_weather_kwh)[0], 2)
        pred_weather_kwh = round(predicted_monthly_kWh * 0.06, 1)
        pred_weather_savings = round(predicted_flat_monthly - calculate_slab_cost(predicted_monthly_kWh - pred_weather_kwh)[0], 2)

    total_actual_kwh_saved = round(actual_peak_shift_kwh + actual_occ_kwh + actual_vampire_kwh + actual_weather_kwh, 1)
    if is_flat_override:
        total_actual_savings_monthly = round(actual_peak_shift_savings + actual_occ_savings + actual_vampire_savings + actual_weather_savings, 2)
        optimized_actual_monthly_cost = round(max(0, actual_flat_monthly - total_actual_savings_monthly), 2)
    else:
        optimized_actual_monthly_cost = calculate_slab_cost(max(0, actual_monthly_kWh - (actual_occ_kwh + actual_vampire_kwh + actual_weather_kwh)))[0] - actual_peak_shift_savings
        optimized_actual_monthly_cost = round(max(0, optimized_actual_monthly_cost), 2)
        total_actual_savings_monthly = round(actual_flat_monthly - optimized_actual_monthly_cost, 2)
    total_actual_savings_yearly = round(total_actual_savings_monthly * 12, 2)

    total_pred_kwh_saved = round(pred_peak_shift_kwh + pred_occ_kwh + pred_vampire_kwh + pred_weather_kwh, 1)
    if is_flat_override:
        total_pred_savings_monthly = round(pred_peak_shift_savings + pred_occ_savings + pred_vampire_savings + pred_weather_savings, 2)
        optimized_pred_monthly_cost = round(max(0, predicted_flat_monthly - total_pred_savings_monthly), 2)
    else:
        optimized_pred_monthly_cost = calculate_slab_cost(max(0, predicted_monthly_kWh - (pred_occ_kwh + pred_vampire_kwh + pred_weather_kwh)))[0] - pred_peak_shift_savings
        optimized_pred_monthly_cost = round(max(0, optimized_pred_monthly_cost), 2)
        total_pred_savings_monthly = round(predicted_flat_monthly - optimized_pred_monthly_cost, 2)
    total_pred_savings_yearly = round(total_pred_savings_monthly * 12, 2)

    strategies = [
        {
            'strategy': 'Peak Load Shifting (18:00 - 22:00)',
            'description': 'Shift 40% of flexible peak load away from peak tariff window',
            'actual_kwh_saved': actual_peak_shift_kwh,
            'actual_savings_monthly': actual_peak_shift_savings,
            'predicted_kwh_saved': pred_peak_shift_kwh,
            'predicted_savings_monthly': pred_peak_shift_savings
        },
        {
            'strategy': 'Occupancy-Based Setback Automation',
            'description': 'Automated setback during low occupancy periods',
            'actual_kwh_saved': actual_occ_kwh,
            'actual_savings_monthly': actual_occ_savings,
            'predicted_kwh_saved': pred_occ_kwh,
            'predicted_savings_monthly': pred_occ_savings
        },
        {
            'strategy': 'Eliminate Standby & Vampire Loads',
            'description': 'Smart power strips cutting overnight standby power',
            'actual_kwh_saved': actual_vampire_kwh,
            'actual_savings_monthly': actual_vampire_savings,
            'predicted_kwh_saved': pred_vampire_kwh,
            'predicted_savings_monthly': pred_vampire_savings
        },
        {
            'strategy': 'Weather Pre-Cooling Management',
            'description': 'Pre-cool during morning off-peak hours before heat peaks',
            'actual_kwh_saved': actual_weather_kwh,
            'actual_savings_monthly': actual_weather_savings,
            'predicted_kwh_saved': pred_weather_kwh,
            'predicted_savings_monthly': pred_weather_savings
        }
    ]

    # 5. Standard Percentage Scenarios
    percentage_levels = [5, 10, 15, 20, 25]
    savings_breakdown = []

    for pct in percentage_levels:
        act_saved_kwh_m = round(actual_monthly_kWh * (pct / 100.0), 1)
        if is_flat_override:
            act_saved_cost_m = round(act_saved_kwh_m * tariff_rate, 2)
            act_new_monthly = round(actual_flat_monthly - act_saved_cost_m, 2)
            pred_saved_kwh_m = round(predicted_monthly_kWh * (pct / 100.0), 1)
            pred_saved_cost_m = round(pred_saved_kwh_m * tariff_rate, 2)
            pred_new_monthly = round(predicted_flat_monthly - pred_saved_cost_m, 2)
        else:
            act_new_monthly, _ = calculate_slab_cost(actual_monthly_kWh - act_saved_kwh_m)
            act_saved_cost_m = round(actual_flat_monthly - act_new_monthly, 2)
            pred_saved_kwh_m = round(predicted_monthly_kWh * (pct / 100.0), 1)
            pred_new_monthly, _ = calculate_slab_cost(predicted_monthly_kWh - pred_saved_kwh_m)
            pred_saved_cost_m = round(predicted_flat_monthly - pred_new_monthly, 2)

        savings_breakdown.append({
            'percentage': pct,
            'actual_saved_kwh_monthly': act_saved_kwh_m,
            'actual_saved_cost_monthly': act_saved_cost_m,
            'actual_saved_cost_yearly': round(act_saved_cost_m * 12, 2),
            'actual_new_monthly_cost': act_new_monthly,
            'predicted_saved_kwh_monthly': pred_saved_kwh_m,
            'predicted_saved_cost_monthly': pred_saved_cost_m,
            'predicted_saved_cost_yearly': round(pred_saved_cost_m * 12, 2),
            'predicted_new_monthly_cost': pred_new_monthly,
            'saved_cost_monthly': act_saved_cost_m,
            'saved_cost_yearly': round(act_saved_cost_m * 12, 2),
            'formatted_saved_monthly': f"{currency_symbol}{act_saved_cost_m:,.2f}",
            'formatted_saved_yearly': f"{currency_symbol}{round(act_saved_cost_m * 12, 2):,.2f}"
        })

    return {
        'input': {
            'monthly_kWh': actual_monthly_kWh,
            'tariff_rate_per_kWh': f"{tariff_rate:.2f}" if is_flat_override else 'Slab Rates (₹3.35 - ₹8.50)',
            'currency_symbol': currency_symbol
        },
        'actual_metrics': {
            'monthly_kWh': actual_monthly_kWh,
            'flat_monthly_cost': actual_flat_monthly,
            'flat_yearly_cost': actual_flat_yearly,
            'total_savings_monthly': total_actual_savings_monthly,
            'total_savings_yearly': total_actual_savings_yearly,
            'optimized_monthly_cost': optimized_actual_monthly_cost,
            'total_kwh_saved_monthly': total_actual_kwh_saved,
            'savings_pct': round((total_actual_savings_monthly / max(actual_flat_monthly, 0.01)) * 100, 1)
        },
        'predicted_metrics': {
            'monthly_kWh': predicted_monthly_kWh,
            'flat_monthly_cost': predicted_flat_monthly,
            'flat_yearly_cost': predicted_flat_yearly,
            'total_savings_monthly': total_pred_savings_monthly,
            'total_savings_yearly': total_pred_savings_yearly,
            'optimized_monthly_cost': optimized_pred_monthly_cost,
            'total_kwh_saved_monthly': total_pred_kwh_saved,
            'savings_pct': round((total_pred_savings_monthly / max(predicted_flat_monthly, 0.01)) * 100, 1)
        },
        'variance': {
            'kwh_difference': variance_kwh,
            'cost_difference': variance_cost,
            'percent_difference': variance_pct
        },
        'strategies': strategies,
        'slab_tariff_rates': SLAB_TARIFF_STRUCTURE,
        'actual_slab_breakdown': actual_slabs,
        'current_estimates': {  # backwards compatibility
            'flat_monthly_cost': actual_flat_monthly,
            'flat_yearly_cost': actual_flat_yearly,
            'formatted_monthly_cost': f"{currency_symbol}{actual_flat_monthly:,.2f}",
            'formatted_yearly_cost': f"{currency_symbol}{actual_flat_yearly:,.2f}"
        },
        'savings_scenarios': savings_breakdown
    }


