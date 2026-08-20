// Forecasting View JavaScript Engine
let forecastChartInstance = null;
let currentHorizon = 24;

document.addEventListener('DOMContentLoaded', () => {
  loadForecast(24);
});

function loadForecast(horizonHours = 24) {
  currentHorizon = horizonHours;

  // Active button formatting
  [24, 48, 72, 168].forEach(h => {
    const btn = document.getElementById(`btn-horizon-${h}`);
    if (btn) {
      if (h === horizonHours) btn.classList.add('active');
      else btn.classList.remove('active');
    }
  });

  fetch(`/api/forecast?horizon=${horizonHours}`)
    .then(res => res.json())
    .then(res => {
      if (res.status === 'success') {
        const data = res.data;
        updateForecastMetrics(data.metadata);
        renderForecastChart(data.historical_test_actuals, data.predictions);
        renderForecastTable(data.predictions);
      }
    })
    .catch(err => console.error('Error loading forecast:', err));
}

function updateForecastMetrics(meta) {
  if (!meta || !meta.metrics) return;

  const m = meta.metrics;
  const b = meta.baseline_metrics || {};

  document.getElementById('metric-mae').innerText = m.MAE;
  document.getElementById('metric-rmse').innerText = m.RMSE;
  document.getElementById('metric-mape').innerText = `${m.MAPE}%`;
  document.getElementById('metric-model-name').innerText = meta.model_used || 'SARIMA + Ridge';
  document.getElementById('metric-trained-date').innerText = meta.trained_at ? `Trained: ${meta.trained_at.split(' ')[0]}` : '--';

  if (b.MAE) {
    const diff = round((b.MAE - m.MAE), 3);
    const sign = diff >= 0 ? 'Better' : 'Worse';
    document.getElementById('baseline-mae-comparison').innerText = `${Math.abs(diff)} ${sign} than Naive`;
  }
}

function renderForecastChart(historicalActuals, predictions) {
  // Combine Historical Recent (48h) and Future Predictions
  const labels = [];
  const actualValues = [];
  const forecastValues = [];

  historicalActuals.forEach(item => {
    labels.push(item.Timestamp);
    actualValues.push(item.Actual_kWh);
    forecastValues.push(null);
  });

  // Connect last actual point to first forecast point for smooth line graph transition
  if (actualValues.length > 0 && predictions.length > 0) {
    forecastValues[forecastValues.length - 1] = actualValues[actualValues.length - 1];
  }

  predictions.forEach(item => {
    labels.push(item.Timestamp);
    actualValues.push(null);
    forecastValues.push(item.Forecast_kWh);
  });

  const ctx = document.getElementById('forecastChart').getContext('2d');
  if (forecastChartInstance) forecastChartInstance.destroy();

  forecastChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Historical Actual (Recent)',
          data: actualValues,
          borderColor: '#3b82f6',
          borderWidth: 2,
          pointRadius: 1,
          tension: 0.2
        },
        {
          label: `Forecast Prediction (${currentHorizon}h)`,
          data: forecastValues,
          borderColor: '#06b6d4',
          borderDash: [5, 5],
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          borderWidth: 2.5,
          pointRadius: 2,
          pointBackgroundColor: '#06b6d4',
          fill: true,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#f8fafc' }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: '#1e293b',
          borderColor: 'rgba(255,255,255,0.1)'
        }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', maxTicksLimit: 14 } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

function renderForecastTable(predictions) {
  const tbody = document.getElementById('forecast-table-body');
  tbody.innerHTML = '';

  predictions.slice(0, 15).forEach(item => {
    const kwh = item.Forecast_kWh;
    let category = 'Normal';
    let badgeClass = 'bg-success-subtle text-success';

    if (kwh > 2.0) {
      category = 'Peak Load';
      badgeClass = 'bg-danger-subtle text-danger';
    } else if (kwh > 1.2) {
      category = 'Moderate Load';
      badgeClass = 'bg-warning-subtle text-warning';
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.Timestamp}</td>
      <td class="fw-bold">${kwh} kWh</td>
      <td>${category}</td>
      <td><span class="badge ${badgeClass}">${category}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function retrainModel() {
  const btn = event.target;
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Retraining...';

  fetch('/api/retrain', { method: 'POST' })
    .then(res => res.json())
    .then(res => {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-arrows-rotate me-1"></i> Retrain Model';
      if (res.status === 'success') {
        alert('Forecasting model retrained successfully!');
        loadForecast(currentHorizon);
      } else {
        alert(`Retraining failed: ${res.message}`);
      }
    })
    .catch(err => {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-arrows-rotate me-1"></i> Retrain Model';
      alert(`Error during retraining: ${err}`);
    });
}

function round(val, decimals) {
  return Number(Math.round(val + 'e' + decimals) + 'e-' + decimals);
}
