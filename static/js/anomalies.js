// Anomaly Detection View JavaScript Engine
let anomalyChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  loadAnomalies();
});

function loadAnomalies() {
  const threshold = document.getElementById('z-threshold-select').value;

  fetch(`/api/anomalies?threshold=${threshold}`)
    .then(res => res.json())
    .then(res => {
      if (res.status === 'success') {
        const data = res.data;
        updateAnomalySummary(data.summary);
        renderAnomalyChart(data.anomalies);
        renderAnomalyTable(data.anomalies);
      }
    })
    .catch(err => console.error('Error loading anomalies:', err));
}

function updateAnomalySummary(summary) {
  if (!summary) return;
  document.getElementById('anom-total').innerText = summary.total_anomalies;
  document.getElementById('anom-critical').innerText = summary.critical_count;
  document.getElementById('anom-high').innerText = summary.high_count;
  document.getElementById('anom-max-z').innerText = summary.max_z_score;
}

function renderAnomalyChart(anomalies) {
  // Sort anomalies by timestamp
  const sorted = anomalies.slice().sort((a, b) => new Date(a.Timestamp) - new Date(b.Timestamp));
  const timestamps = sorted.map(a => a.Timestamp);
  const actuals = sorted.map(a => a.Actual_kWh);
  const expecteds = sorted.map(a => a.Expected_kWh);

  const ctx = document.getElementById('anomalyChart').getContext('2d');
  if (anomalyChartInstance) anomalyChartInstance.destroy();

  anomalyChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: timestamps,
      datasets: [
        {
          label: 'Actual Anomaly Load (kWh)',
          data: actuals,
          borderColor: '#ef4444',
          backgroundColor: '#ef4444',
          pointRadius: 5,
          pointHoverRadius: 8,
          showLine: false
        },
        {
          label: 'Expected Rolling Baseline (kWh)',
          data: expecteds,
          borderColor: '#3b82f6',
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#f8fafc' } },
        tooltip: { backgroundColor: '#1e293b' }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', maxTicksLimit: 12 } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

function renderAnomalyTable(anomalies) {
  const tbody = document.getElementById('anomalies-table-body');
  tbody.innerHTML = '';

  if (anomalies.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No anomalies detected for the selected Z-Score threshold.</td></tr>';
    return;
  }

  anomalies.forEach(item => {
    let badgeClass = 'badge-low';
    if (item.Severity === 'Critical') badgeClass = 'badge-critical';
    else if (item.Severity === 'High') badgeClass = 'badge-high';
    else if (item.Severity === 'Medium') badgeClass = 'badge-medium';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw-bold">${item.Timestamp}</td>
      <td class="text-danger fw-bold">${item.Actual_kWh} kWh</td>
      <td class="text-muted">${item.Expected_kWh} kWh</td>
      <td><span class="badge bg-secondary">${item.Z_Score}</span></td>
      <td><span class="badge ${badgeClass}">${item.Severity}</span></td>
      <td class="small text-light">${item.Possible_Reason}</td>
    `;
    tbody.appendChild(tr);
  });
}
