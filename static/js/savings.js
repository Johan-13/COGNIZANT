// Savings Calculator JavaScript Engine
let savingsChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  calculateSavings();
});

function calculateSavings() {
  const tariff = document.getElementById('input-tariff').value || 0.15;
  const monthlyKwh = document.getElementById('input-monthly-kwh').value;

  fetch('/api/calculate-savings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tariff_rate: tariff,
      monthly_kWh: monthlyKwh
    })
  })
  .then(res => res.json())
  .then(res => {
    if (res.status === 'success') {
      const data = res.data;
      updateCostDisplay(data.current_estimates, data.input);
      renderSavingsChart(data.savings_scenarios, data.current_estimates.monthly_cost);
      renderSavingsTable(data.savings_scenarios);
    }
  })
  .catch(err => console.error('Error calculating savings:', err));
}

function updateCostDisplay(estimates, inputs) {
  document.getElementById('calc-monthly-cost').innerText = `$${estimates.monthly_cost}`;
  document.getElementById('calc-yearly-cost').innerText = `$${estimates.yearly_cost}`;
  document.getElementById('calc-monthly-sub').innerText = `${inputs.monthly_kWh} kWh @ $${inputs.tariff_rate_per_kWh}/kWh`;
  
  if (!document.getElementById('input-monthly-kwh').value) {
    document.getElementById('input-monthly-kwh').value = inputs.monthly_kWh;
  }
}

function renderSavingsChart(scenarios, currentMonthlyCost) {
  const labels = ['Current', ...scenarios.map(s => `-${s.percentage}%`)];
  const costs = [currentMonthlyCost, ...scenarios.map(s => s.new_monthly_cost)];

  const ctx = document.getElementById('savingsChart').getContext('2d');
  if (savingsChartInstance) savingsChartInstance.destroy();

  savingsChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Monthly Electricity Cost ($)',
        data: costs,
        backgroundColor: costs.map((v, i) => i === 0 ? '#3b82f6' : '#10b981'),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

function renderSavingsTable(scenarios) {
  const tbody = document.getElementById('savings-table-body');
  tbody.innerHTML = '';

  scenarios.forEach(item => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="fw-bold text-success">${item.percentage}% Efficiency Gain</td>
      <td>${item.saved_kwh_monthly} kWh</td>
      <td>${item.saved_kwh_yearly} kWh</td>
      <td class="text-success fw-bold">-$${item.saved_cost_monthly} / mo</td>
      <td class="text-success fw-bold">-$${item.saved_cost_yearly} / yr</td>
      <td class="fw-bold text-light">$${item.new_monthly_cost}</td>
    `;
    tbody.appendChild(tr);
  });
}
