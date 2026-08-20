import pandas as pd
import numpy as np


def generate_recommendations(df, anomalies_summary=None, peak_info=None):
    """
    Generates rule-based energy saving recommendations based on household data analysis.
    """
    recommendations = []

    overall_avg = float(df['Energy_kWh'].mean())
    hourly_avg = df.groupby('Hour')['Energy_kWh'].mean()

    # Rule 1: Evening Peak Load Shifting (18:00 - 22:00)
    evening_peak = hourly_avg[(hourly_avg.index >= 18) & (hourly_avg.index <= 22)].mean()
    if evening_peak > overall_avg * 1.25:
        pct_higher = round(((evening_peak - overall_avg) / overall_avg) * 100, 1)
        recommendations.append({
            'id': 'rec_peak_shift',
            'category': 'Load Shifting',
            'title': 'Shift Flexible Appliance Usage Away from Evening Peak (18:00 - 22:00)',
            'description': f"Evening power consumption is {pct_higher}% above your daily average. Delay high-wattage appliances like washing machines, dishwashers, and EV chargers to off-peak hours (after 22:00 or mid-day).",
            'potential_impact': 'High',
            'estimated_kwh_saving_monthly': round((evening_peak - overall_avg) * 4 * 30 * 0.4, 1),
            'actionable_step': 'Program timer delays on washers/dryers to start after 10:00 PM.'
        })

    # Rule 2: Overnight Standby/Phantom Load Reduction (01:00 - 05:00)
    night_load = hourly_avg[(hourly_avg.index >= 1) & (hourly_avg.index <= 5)].mean()
    if night_load > 0.3:  # Baseline threshold
        recommendations.append({
            'id': 'rec_phantom_load',
            'category': 'Standby Reduction',
            'title': 'Eliminate Overnight Standby Power (Vampire Draw)',
            'description': f"Your overnight baseline draw averages {round(night_load, 2)} kWh/hr while the household is asleep. Idle electronics, chargers, set-top boxes, and gaming consoles draw continuous power.",
            'potential_impact': 'Medium',
            'estimated_kwh_saving_monthly': round(night_load * 0.35 * 6 * 30, 1),
            'actionable_step': 'Use smart power strips that automatically cut power to peripheral devices overnight.'
        })

    # Rule 3: High Anomaly Frequency Inspection
    if anomalies_summary and anomalies_summary.get('total_anomalies', 0) > 10:
        recommendations.append({
            'id': 'rec_appliance_check',
            'category': 'Maintenance & Audit',
            'title': 'Inspect Heavy Appliances for High Power Spikes',
            'description': f"{anomalies_summary['total_anomalies']} power spikes were detected. Unusually high consumption bursts often indicate degraded refrigerator coils, faulty water heaters, or HVAC compressor issues.",
            'potential_impact': 'High',
            'estimated_kwh_saving_monthly': round(anomalies_summary['total_anomalies'] * 1.5, 1),
            'actionable_step': 'Schedule maintenance check for HVAC filters, water heater thermostats, and refrigerator gaskets.'
        })

    # Rule 4: Weekend Consumption Management
    weekend_avg = df[df['IsWeekend'] == 1]['Energy_kWh'].mean()
    weekday_avg = df[df['IsWeekend'] == 0]['Energy_kWh'].mean()
    if weekend_avg > weekday_avg * 1.15:
        pct_weekend_high = round(((weekend_avg - weekday_avg) / weekday_avg) * 100, 1)
        recommendations.append({
            'id': 'rec_weekend_opt',
            'category': 'Behavioral Adjustment',
            'title': 'Optimize Weekend Energy Habit Patterns',
            'description': f"Weekend electricity usage is {pct_weekend_high}% higher than weekdays. Stagger cooking, laundry, and heating schedules to avoid simultaneous multi-appliance operation.",
            'potential_impact': 'Medium',
            'estimated_kwh_saving_monthly': round((weekend_avg - weekday_avg) * 24 * 8 * 0.3, 1),
            'actionable_step': 'Use smart thermostat Eco modes when relaxing at home during weekends.'
        })

    # General Efficiency Rule
    recommendations.append({
        'id': 'rec_led_lighting',
        'category': 'Equipment Upgrade',
        'title': 'Upgrade High-Usage Lighting and HVAC Controls',
        'description': 'Replace remaining incandescent bulbs with ENERGY STAR LEDs and install programmable/smart thermostats to auto-adjust temperatures.',
        'potential_impact': 'Medium',
        'estimated_kwh_saving_monthly': round(overall_avg * 24 * 30 * 0.08, 1),
        'actionable_step': 'Switch to LED bulbs and set cooling target to 24°C / heating to 20°C.'
    })

    return recommendations


def calculate_cost_and_savings(monthly_kWh, tariff_rate=0.15):
    """
    Calculates estimated monthly/yearly costs and dynamic potential savings at key reduction percentages.
    """
    tariff_rate = float(tariff_rate)
    monthly_kWh = float(monthly_kWh)

    monthly_cost = monthly_kWh * tariff_rate
    yearly_cost = monthly_cost * 12

    percentage_levels = [5, 10, 15, 20, 25]
    savings_breakdown = []

    for pct in percentage_levels:
        saved_kwh_monthly = monthly_kWh * (pct / 100.0)
        saved_kwh_yearly = saved_kwh_monthly * 12
        saved_cost_monthly = saved_kwh_monthly * tariff_rate
        saved_cost_yearly = saved_cost_monthly * 12
        new_monthly_cost = monthly_cost - saved_cost_monthly

        savings_breakdown.append({
            'percentage': pct,
            'saved_kwh_monthly': round(saved_kwh_monthly, 1),
            'saved_kwh_yearly': round(saved_kwh_yearly, 1),
            'saved_cost_monthly': round(saved_cost_monthly, 2),
            'saved_cost_yearly': round(saved_cost_yearly, 2),
            'new_monthly_cost': round(new_monthly_cost, 2)
        })

    return {
        'input': {
            'monthly_kWh': round(monthly_kWh, 2),
            'tariff_rate_per_kWh': tariff_rate
        },
        'current_estimates': {
            'monthly_cost': round(monthly_cost, 2),
            'yearly_cost': round(yearly_cost, 2)
        },
        'savings_scenarios': savings_breakdown
    }
