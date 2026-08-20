# Smart Energy Consumption Optimizer

An enterprise-grade, full-stack Data Science and Machine Learning Web Application built with **FastAPI**, **Facebook Prophet**, **Pandas**, **Scikit-learn**, and **Vanilla JavaScript / Chart.js**. The system analyzes building and household electricity usage patterns, forecasts future hourly energy consumption using time-series models with Weather and Occupancy feature engineering, identifies abnormal power spikes (anomalies), highlights peak usage hours, and provides rule-based energy-saving recommendations and financial tariff cost calculators.

---

## 🌟 Key Features

1. **Data Preprocessing & Fast Caching**
   - Ingests raw UCI minute-level household electricity consumption data.
   - Automatically handles missing values using time-aware interpolation.
   - Resamples data into hourly aggregates and caches preprocessed data in `backend/app/data/processed_power_consumption.csv` for sub-second API responses.
   - Auto-generates a realistic 1-year synthetic dataset if the raw dataset file is not present.

2. **Exploratory Data Analysis & Visual Profiling**
   - Total, daily, and hourly energy metrics.
   - Interactive consumption time-series charts (Daily 90d, Hourly 7d, Weekly 1y).
   - Hourly distribution profiles (00:00 – 23:00) and Day-of-Week radar charts with Weather and Occupancy correlation.

3. **Time-Series Forecasting (Facebook Prophet)**
   - Prophet model with additive daily/weekly seasonality, ambient temperature regressors, and occupancy ratio regressors.
   - User-selectable forecast horizons (24 hours, 48 hours, 72 hours, 7 days).
   - Evaluation metrics: **MAE**, **RMSE**, **MAPE**.
   - Naive baseline model comparison (24-hour lag predictor) and quantifiable accuracy improvements.
   - On-demand model retraining via REST API (`POST /api/retrain`).

4. **Statistical Anomaly Detection**
   - Detects abnormal electricity consumption spikes using 24-hour rolling Z-Score statistics ($Z = \frac{X - \mu}{\sigma}$).
   - Configurable sensitivity thresholds ($Z \ge 1.0 - 5.0$).
   - Categorizes anomaly events by severity: **Low**, **Medium**, **High**, **Critical**.
   - Generates contextual root-cause explanations (e.g., HVAC load during low occupancy, overnight power leaks, weekend demand stacks).

5. **Peak Usage Window Identification**
   - Identifies top peak consumption hours and delta percentages above household baseline.
   - Categorizes period averages: Morning Peak (07-10), Evening Peak (17-22), Overnight Baseline (00-05).
   - Quantifies load shifting potential (kWh/day) from peak hours to off-peak overnight hours.

6. **Energy Saving Recommendation Engine**
   - Rule-based optimization system offering advice on peak load shifting, occupancy automation, phantom standby reduction, and thermal pre-cooling.
   - Quantifies monthly kWh savings for each recommendation.

7. **Financial Cost & Tariff Calculator**
   - Customizable electricity tariff rate ($/kWh) and monthly consumption inputs.
   - Projects monthly and annual flat and Time-Of-Use (TOU) electricity costs.
   - Provides scenario projections for 5%, 10%, 15%, 20%, and 25% energy reduction goals.

---

## 🏗️ Project Architecture

The codebase follows a clean, modular, deployment-ready architecture with strict separation between backend services, frontend assets, tests, and deployment orchestration:

```
smart-energy-optimizer/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI Application Server & Route Endpoints
│   │   ├── core/                       # App Configuration & Dynamic Path Resolution
│   │   │   ├── __init__.py
│   │   │   └── config.py               # Centralized settings & environment variables
│   │   ├── services/                   # ML & Domain Logic Services
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py             # EDA aggregations & profiling
│   │   │   ├── anomaly_detection.py    # Rolling Z-score anomaly detector
│   │   │   ├── forecasting.py          # Prophet time-series engine
│   │   │   ├── peak_analysis.py        # Peak load & load shifting analysis
│   │   │   ├── preprocessing.py        # Data ingestion, cleaning & feature engineering
│   │   │   └── savings.py              # Recommendation engine & tariff calculator
│   │   ├── models/                     # Trained ML model binaries & metadata
│   │   │   ├── forecasting_meta.json
│   │   │   ├── ml_forecast_model.pkl
│   │   │   ├── prophet_model.pkl
│   │   │   └── sarima_model.pkl
│   │   └── data/                       # Raw & processed consumption datasets
│   │       ├── sample_power_consumption.txt
│   │       └── processed_power_consumption.csv
│   ├── tests/                          # Automated Pytest Suite
│   │   ├── __init__.py
│   │   ├── conftest.py                 # Test fixtures & configuration
│   │   ├── test_fastapi_prophet.py     # FastAPI API integration tests
│   │   ├── test_preprocessing.py       # Data pipeline unit tests
│   │   ├── test_forecasting.py         # Forecasting engine unit tests
│   │   ├── test_anomaly.py             # Anomaly detection unit tests
│   │   └── test_savings.py             # Tariff calculation tests
│   ├── requirements.txt                # Python backend dependencies
│   └── Dockerfile                      # Backend container definition
│
├── frontend/
│   ├── templates/                      # Jinja2 HTML Templates
│   │   ├── base.html                   # Master layout with sidebar
│   │   ├── dashboard.html              # Main overview dashboard
│   │   ├── forecast.html               # Time-series forecast & horizon selector
│   │   ├── anomalies.html              # Anomaly logs & severity breakdown
│   │   ├── peaks.html                  # Peak usage breakdown & load shifting
│   │   ├── recommendations.html        # Energy saving tips
│   │   └── savings.html                # Financial cost & savings calculator
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
├── Dockerfile                          # Root production multi-stage container
├── .env.example                        # Environment variable template
├── .gitignore                          # Git ignore specification
├── run.py                              # Root application launcher
└── README.md                           # Documentation
```

---

## ⚙️ Installation & Setup

### 1. Create Virtual Environment
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

---

## 🚀 Running the Application

### Option 1: Quick Root Launcher
```bash
python run.py
```

### Option 2: Using Uvicorn Directly
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🐳 Containerized Deployment (Docker & Docker Compose)

### Using Docker Compose
```bash
# Build and start the container
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Using Standard Docker
```bash
# Build image
docker build -t smart-energy-optimizer .

# Run container
docker run -d -p 8000:8000 --name energy-optimizer smart-energy-optimizer
```

---

## 🧪 Running Automated Tests

Run the full test suite with pytest:
```bash
pytest backend/tests/ -v
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/summary` | GET | Returns key dataset metrics, total kWh, peak hour, anomalies count, and baseline monthly cost. |
| `GET /api/consumption` | GET | Returns consumption time series. Query params: `period=daily\|hourly\|weekly`, `limit=90`. |
| `GET /api/hourly` | GET | Returns average consumption profiles by hour of day (0-23) and day of week. |
| `GET /api/forecast` | GET | Returns Prophet predictions & metrics. Query params: `horizon=24\|48\|72\|168`, `retrain=true\|false`. |
| `GET /api/anomalies` | GET | Returns flagged high-consumption anomaly events. Query param: `threshold=2.0`. |
| `GET /api/peaks` | GET | Returns peak usage hours and load-shifting recommendations. |
| `GET /api/recommendations` | GET | Returns rule-based energy saving recommendations and estimated kWh impact. |
| `POST /api/calculate-savings` | POST | Calculates flat & TOU tariff costs and savings scenarios. Body: `{"tariff_rate": 0.15, "monthly_kWh": 500}`. |
| `POST /api/retrain` | POST | Triggers Prophet model retraining and saves new model binaries into `backend/app/models/`. |
