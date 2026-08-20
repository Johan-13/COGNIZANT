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
    if 'Occupancy_Ratio' in df.columns:
        low_occ_df = df[df['Occupancy_Ratio'] < 0.3]
        if len(low_occ_df) > 0:
            unattended_draw = float(low_occ_df['Energy_kWh'].mean())
            if unattended_draw > overall_avg * 0.6:
                recommendations.append({
                    'id': 'rec_occupancy_waste',
                    'category': 'Occupancy Automation',
                    'title': 'Automate HVAC and Lighting Shutdown During Unoccupied Hours',
                    'description': f"Average energy draw during low occupancy (<30% capacity) is {round(unattended_draw, 2)} kWh/hr. Automated occupancy sensors can eliminate unneeded cooling/lighting.",
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


def calculate_cost_and_savings(monthly_kWh, tariff_rate=0.15, peak_tariff_multiplier=1.4, currency_symbol='$'):
    """
    Calculates estimated monthly/yearly energy costs, Time-Of-Use peak tariff breakdown,
    and quantified savings across optimization scenarios.
    """
    tariff_rate = float(tariff_rate)
    monthly_kWh = float(monthly_kWh)
    peak_multiplier = float(peak_tariff_multiplier)

    # Standard flat billing estimate
    flat_monthly_cost = monthly_kWh * tariff_rate
    flat_yearly_cost = flat_monthly_cost * 12

    # Time-Of-Use billing estimate (30% peak, 70% off-peak)
    peak_kWh = monthly_kWh * 0.30
    offpeak_kWh = monthly_kWh * 0.70
    tou_monthly_cost = (peak_kWh * tariff_rate * peak_multiplier) + (offpeak_kWh * tariff_rate)
    tou_yearly_cost = tou_monthly_cost * 12

    percentage_levels = [5, 10, 15, 20, 25]
    savings_breakdown = []

    for pct in percentage_levels:
        saved_kwh_monthly = monthly_kWh * (pct / 100.0)
        saved_kwh_yearly = saved_kwh_monthly * 12
        saved_cost_monthly = saved_kwh_monthly * tariff_rate
        saved_cost_yearly = saved_cost_monthly * 12
        new_monthly_cost = flat_monthly_cost - saved_cost_monthly

        savings_breakdown.append({
            'percentage': pct,
            'saved_kwh_monthly': round(saved_kwh_monthly, 1),
            'saved_kwh_yearly': round(saved_kwh_yearly, 1),
            'saved_cost_monthly': round(saved_cost_monthly, 2),
            'saved_cost_yearly': round(saved_cost_yearly, 2),
            'new_monthly_cost': round(new_monthly_cost, 2),
            'formatted_saved_monthly': f"{currency_symbol}{round(saved_cost_monthly, 2):,.2f}",
            'formatted_saved_yearly': f"{currency_symbol}{round(saved_cost_yearly, 2):,.2f}"
        })

    return {
        'input': {
            'monthly_kWh': round(monthly_kWh, 2),
            'tariff_rate_per_kWh': tariff_rate,
            'currency_symbol': currency_symbol
        },
        'current_estimates': {
            'flat_monthly_cost': round(flat_monthly_cost, 2),
            'flat_yearly_cost': round(flat_yearly_cost, 2),
            'tou_monthly_cost': round(tou_monthly_cost, 2),
            'tou_yearly_cost': round(tou_yearly_cost, 2),
            'formatted_monthly_cost': f"{currency_symbol}{round(flat_monthly_cost, 2):,.2f}",
            'formatted_yearly_cost': f"{currency_symbol}{round(flat_yearly_cost, 2):,.2f}"
        },
        'savings_scenarios': savings_breakdown
    }
