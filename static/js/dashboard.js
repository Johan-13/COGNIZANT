// Dashboard JavaScript Engine
let timeSeriesChartInstance = null;
let dowChartInstance = null;
let hourlyProfileChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  fetchSummaryMetrics();
  loadTimeSeriesChart('daily', 90);
  fetchHourlyAndDowProfiles();
});

function refreshDashboard() {
  fetchSummaryMetrics();
  loadTimeSeriesChart('daily', 90);
  fetchHourlyAndDowProfiles();
}

function fetchSummaryMetrics() {
  fetch('/api/summary')
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        const d = data.data;
        document.getElementById('metric-total-kwh').innerText = d.total_consumption_kWh.toLocaleString();
        document.getElementById('metric-daily-avg').innerText = d.avg_daily_kWh;
        document.getElementById('metric-peak-hour').innerText = `${d.peak_hour_of_day}:00`;
        document.getElementById('metric-monthly-cost').innerText = `$${d.estimated_monthly_cost}`;
        document.getElementById('metric-anomalies').innerText = d.anomaly_count;
        document.getElementById('metric-potential-savings').innerText = `$${d.potential_monthly_savings}`;
      }
    })
    .catch(err => console.error('Error fetching summary:', err));
}

function loadTimeSeriesChart(period = 'daily', limit = 90) {
  // Update button active state
  ['daily', 'hourly', 'weekly'].forEach(p => {
    const btn = document.getElementById(`btn-ts-${p}`);
    if (btn) {
      if (p === period) btn.classList.add('active');
      else btn.classList.remove('active');
    }
  });

  fetch(`/api/consumption?period=${period}&limit=${limit}`)
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        const timestamps = data.data.map(item => item.Timestamp);
        const values = data.data.map(item => item.Energy_kWh);

        const ctx = document.getElementById('timeSeriesChart').getContext('2d');
        if (timeSeriesChartInstance) timeSeriesChartInstance.destroy();

        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(6, 182, 212, 0.4)');
        gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

        timeSeriesChartInstance = new Chart(ctx, {
          type: 'line',
          data: {
            labels: timestamps,
            datasets: [{
              label: 'Energy Consumption (kWh)',
              data: values,
              borderColor: '#06b6d4',
              backgroundColor: gradient,
              borderWidth: 2,
              fill: true,
              tension: 0.3,
              pointRadius: period === 'hourly' ? 0 : 2,
              pointHoverRadius: 5
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { display: false },
              tooltip: {
                backgroundColor: '#1e293b',
                titleColor: '#f8fafc',
                bodyColor: '#06b6d4',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1
              }
            },
            scales: {
              x: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8', maxTicksLimit: 12 }
              },
              y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#94a3b8' }
              }
            }
          }
        });
      }
    })
    .catch(err => console.error('Error fetching time series:', err));
}

function fetchHourlyAndDowProfiles() {
  fetch('/api/hourly')
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        renderHourlyProfile(data.hourly_profile);
        renderDowProfile(data.day_of_week_profile);
      }
    })
    .catch(err => console.error('Error fetching profiles:', err));
}

function renderHourlyProfile(hourlyData) {
  const hours = hourlyData.map(d => `${d.Hour}:00`);
  const averages = hourlyData.map(d => d.mean);

  const ctx = document.getElementById('hourlyProfileChart').getContext('2d');
  if (hourlyProfileChartInstance) hourlyProfileChartInstance.destroy();

  hourlyProfileChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hours,
      datasets: [{
        label: 'Average Energy (kWh)',
        data: averages,
        backgroundColor: averages.map(v => v > 1.5 ? '#f59e0b' : '#3b82f6'),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

function renderDowProfile(dowData) {
  const days = dowData.map(d => d.DayName);
  const averages = dowData.map(d => d.Energy_kWh);

  const ctx = document.getElementById('dowChart').getContext('2d');
  if (dowChartInstance) dowChartInstance.destroy();

  dowChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: days,
      datasets: [{
        label: 'Avg Daily kWh',
        data: averages,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.2)',
        borderWidth: 2,
        pointBackgroundColor: '#10b981'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          grid: { color: 'rgba(255,255,255,0.1)' },
          angleLines: { color: 'rgba(255,255,255,0.1)' },
          pointLabels: { color: '#f8fafc', font: { size: 11 } },
          ticks: { backdropColor: 'transparent', color: '#94a3b8' }
        }
      }
    }
  });
}
