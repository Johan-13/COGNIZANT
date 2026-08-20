# Smart Energy Consumption Optimizer

An enterprise-grade, full-stack Data Science and Machine Learning Web Application built with **FastAPI**, **Facebook Prophet**, **Pandas**, **Scikit-learn**, and **Vanilla JavaScript / Plotly.js / Chart.js**. The system analyzes building and household electricity usage patterns, syncs with real-world clock time in a continuous 1:1 live telemetry stream, forecasts future hourly energy consumption using multi-step additive seasonality models with Weather & Occupancy regressors, detects abnormal consumption spikes (anomalies), and computes financial savings using an authentic **Indian Rupee (₹) Telescopic Slab Tariff Engine**.

---

## 🌟 Key Features

### 1. ⏱️ Real-Time 1:1 Live Data Streaming & Clock Synchronization
- **Real-World Clock Sync:** When the application starts, it computes any missing time intervals and automatically generates historical data up to the **current real-world hour** (`catch_up_to_now()`).
- **Continuous 1:1 Streaming Worker:** A background daemon thread records 1 new data hour for every 3,600 seconds (1 real-world hour) of elapsed time with synchronized ambient temperature, humidity, and occupancy telemetry.
- **Live Stream Endpoints & Dashboard Watcher:** Provides `/api/stream/status` and `/api/stream/latest` for real-time monitoring and a client-side streaming watcher bar.

### 2. 🔮 Facebook Prophet Time-Series Forecasting
- **Additive Seasonality with Exogenous Regressors:** Incorporates daily Fourier seasonality, ambient temperature regressors, 3-tier occupancy schedules, and 24-hour autoregressive lag features (`lag_24`).
- **Continuous In-Sample Fit & 95% Uncertainty Bounds:** Generates continuous historical model fits alongside future predictions with full 95% confidence intervals (`yhat_lower`, `yhat_upper`).
- **Overnight Baseload Floor Calibration:** Calibrated baseload protection preventing negative or unrealistic overnight consumption drops.
- **Accuracy Benchmarking:** Evaluates against a Naive 24-hour Lag Seasonal Baseline across **MAE**, **RMSE**, and **MAPE** with automated improvement quantification.
- **On-Demand Retraining:** REST API trigger (`POST /api/retrain`) to retrain and persist updated model binaries.

### 3. 🇮🇳 Indian Rupee (₹) Progressive Slab Tariff Engine
- **Telescopic Slab Structure:**
  | Slab Bracket | Rate per Unit (kWh) |
  |---|---|
  | **0 – 50 Units** | **₹3.35** / kWh |
  | **51 – 100 Units** | **₹4.25** / kWh |
  | **101 – 150 Units** | **₹5.35** / kWh |
  | **151 – 200 Units** | **₹7.20** / kWh |
  | **201 – 250 Units** | **₹8.50** / kWh |
  | **> 250 Units** | **₹8.50** / kWh |
- **Actual vs. Predicted Referenced Savings:** Directly quantifies financial savings against both recorded historical consumption and Prophet forecasted demands.
- **Itemized Energy Reduction Strategies:**
  - Peak Load Shifting (18:00 – 22:00)
  - Occupancy Setback Automation
  - Standby & Vampire Load Elimination (01:00 – 05:00)
  - Weather-Adaptive Thermal Pre-Cooling

### 4. 🚨 Statistical Anomaly Detection
- Detects consumption spikes using **24-hour Rolling Z-Score** statistics ($Z = \frac{X - \mu}{\sigma}$).
- Configurable sensitivity thresholds ($Z \ge 1.0 - 5.0$).
- Categorizes anomalies by severity: **Low**, **Medium**, **High**, **Critical**.
- Generates automated contextual root causes (e.g., HVAC over-cycling, off-hours power leaks, weekend load stacking).

### 5. ⚡ Peak Usage Window & Load-Shifting Profiling
- Classifies diurnal operating windows: Morning Peak (07:00 – 10:00), Evening Peak (17:00 – 22:00), Overnight Baseline (00:00 – 05:00).
- Calculates load-shifting potential (kWh/day) from peak hours to off-peak tariff periods.

---

## 🖥️ Page Layout & UI Architecture

### 📊 1. Overview Dashboard (`/dashboard`)
- **Top Main Centerpiece:** Full-width **Prophet Predicted Energy Demand & 95% Confidence Bounds** chart with continuous historical cyan curve, purple Prophet model fit, 95% shaded confidence ribbon, and vertical *"Now / Forecast Start"* marker.
- **Bottom Section:** Compact **Historical Energy Consumption & Telemetry** (height: 260px) with interactive time-aggregation buttons (Daily 90d / Hourly Recent / Weekly) and optional live telemetry watcher.
- **Metric Cards:** Real-time summary cards for Total Consumption, Daily Average, Temperature, Occupancy, Anomalies Count, and Monthly Potential Savings in **₹ INR**.

### 📋 2. Hourly Predictions Stream (`/forecast`)
- **Top Summary Aggregations:** **Forecast Horizon Key Metrics & Summary** cards (Total Horizon Demand, Average Hourly Rate, Peak Forecast Demand, and Min Off-Peak Demand).
- **Interactive Horizon Switcher:** Next 24 Hours, Next 48 Hours, Next 72 Hours, Next 7 Days (168h).
- **Search & Filter:** Instant table query filtering by timestamp, hour, or occupancy tier.
- **CSV Data Export:** One-click CSV download of active stream predictions with exogenous regressors.
- **Model Evaluation:** Prophet vs. Naive Baseline MAPE, RMSE, and improvement percentage.

### 💰 3. Cost & Savings Optimization Engine (`/savings`)
- Displays active Indian Slab Tariff rate cards.
- Compares Actual Monthly Cost vs. Predicted Monthly Cost and variance in **₹ INR**.
- Itemized breakdown table for each optimization strategy.
- Target reduction scenario matrices (5% to 25% targets) calculated at progressive slab rates.

---

## 🏗️ Project Structure

```
smart-energy-optimizer/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI server, route handlers & hourly streaming worker
│   │   ├── core/                       # App configuration & dynamic path resolution
│   │   │   ├── __init__.py
│   │   │   └── config.py               # Centralized settings & environment variables
│   │   ├── services/                   # ML & Domain logic services
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py             # EDA aggregations & profiling
│   │   │   ├── anomaly_detection.py    # Rolling Z-score anomaly detector
│   │   │   ├── forecasting.py          # Prophet time-series engine with in-sample overlay
│   │   │   ├── peak_analysis.py        # Peak load & load shifting analysis
│   │   │   ├── preprocessing.py        # Data ingestion, cleaning & synthetic generation
│   │   │   ├── savings.py              # Indian slab tariff engine & itemized savings
│   │   │   └── streamer.py             # 1:1 real-world clock catch-up & streaming engine
│   │   ├── models/                     # Trained ML model binaries & metadata
│   │   │   ├── forecasting_meta.json
│   │   │   └── prophet_model.pkl
│   │   └── data/                       # Processed power consumption dataset
│   │       └── processed_power_consumption.csv
│   ├── tests/                          # Automated Pytest suite (13 tests)
│   │   ├── __init__.py
│   │   ├── conftest.py                 # Test fixtures & configuration
│   │   ├── test_fastapi_prophet.py     # FastAPI API integration tests
│   │   ├── test_preprocessing.py       # Data pipeline unit tests
│   │   ├── test_forecasting.py         # Forecasting engine unit tests
│   │   ├── test_anomaly.py             # Anomaly detection unit tests
│   │   └── test_savings.py             # Indian slab tariff & savings tests
│   ├── requirements.txt                # Python backend dependencies
│   └── Dockerfile                      # Backend container definition
│
├── frontend/
│   ├── templates/                      # Jinja2 HTML Templates
│   │   ├── base.html                   # Master layout with navigation sidebar
│   │   ├── dashboard.html              # Main dashboard with Prophet top plot & compact telemetry
│   │   ├── forecast.html               # Hourly Predictions Stream centerpiece & metrics
│   │   ├── anomalies.html              # Anomaly detection logs & threshold tuning
│   │   ├── peaks.html                  # Peak usage breakdown & load shifting
│   │   ├── recommendations.html        # Energy saving action cards
│   │   └── savings.html                # Indian Rupee slab cost & savings calculator
│   └── static/                         # Static UI assets
│       ├── css/
│       │   └── style.css               # Glassmorphism dark theme stylesheet
│       └── js/
│           ├── anomalies.js            # Anomaly charts & threshold handler
│           ├── dashboard.js            # Overview charts & metrics
│           ├── forecast.js             # Forecast charts & retrain handler
│           └── savings.js              # Tariff calculator logic
│
├── docker-compose.yml                  # Docker Compose orchestration
├── Dockerfile                          # Root production container
├── run.py                              # Root application launcher
└── README.md                           # Documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
# Option A: Root Launcher
python run.py

# Option B: Direct Uvicorn
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🧪 Automated Testing

Run the full pytest suite to verify all 13 unit and integration tests:
```bash
pytest backend/tests -v
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/summary` | GET | Returns key dataset metrics, total kWh, peak hour, anomaly count, and slab baseline cost. |
| `GET /api/consumption` | GET | Returns consumption time series (`period=daily\|hourly\|weekly`, `limit=90`). |
| `GET /api/forecast` | GET | Returns Prophet predictions, in-sample fitted history, 95% bounds (`horizon=24\|48\|72\|168`). |
| `GET /api/stream/status` | GET | Returns real-time streaming status, latest recorded timestamp, and total records. |
| `GET /api/stream/latest` | GET | Returns the most recent streaming record and current system time. |
| `GET /api/anomalies` | GET | Returns flagged high-consumption anomaly events (`threshold=2.0`). |
| `GET /api/peaks` | GET | Returns peak usage hours and load-shifting recommendations. |
| `GET /api/recommendations` | GET | Returns rule-based energy saving recommendations and estimated kWh impact. |
| `POST /api/calculate-savings` | POST | Computes Indian Rupee (₹) slab costs, actual vs. predicted baseline, and itemized strategies. |
| `POST /api/retrain` | POST | Triggers Prophet model retraining and updates model binaries. |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
