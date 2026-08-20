# Smart Energy Consumption Optimizer

A complete end-to-end Data Science and Machine Learning Web Application built with **Flask**, **Pandas**, **Scikit-learn**, **Statsmodels**, and **Chart.js**. The system analyzes household electricity usage patterns, forecasts future hourly energy consumption using time-series models (SARIMA / ML Regressor), identifies abnormal power spikes (anomalies), highlights peak usage hours, and provides rule-based energy-saving recommendations and financial tariff cost calculators.

---

## 🌟 Key Features

1. **Data Preprocessing & Fast Caching**
   - Ingests raw UCI minute-level household electricity consumption data.
   - Automatically handles missing values (`?`) using time-aware interpolation.
   - Resamples data into hourly aggregates and caches preprocessed data in `data/processed_power_consumption.csv` for fast API responses.
   - Auto-generates a realistic 1-year sample dataset if the UCI raw dataset file is not present.

2. **Exploratory Data Analysis & Visual Profiling**
   - Total, daily, and hourly energy metrics.
   - Interactive consumption time-series charts (Daily 90d, Hourly 7d, Weekly 1y).
   - Hourly distribution profiles (00:00 – 23:00) and Day-of-Week radar charts.

3. **Time-Series Forecasting (SARIMA & ML)**
   - SARIMA model $(1,1,1) \times (1,0,1)_7$ with fast lag-feature ML regressor fallback.
   - User-selectable forecast horizons (24 hours, 48 hours, 72 hours, 7 days).
   - Evaluation metrics: **MAE**, **RMSE**, **MAPE**.
   - Naive baseline model comparison ($t-168$ lag predictor).
   - On-demand model retraining via web API.

4. **Statistical Anomaly Detection**
   - Detects abnormal electricity consumption spikes using 24-hour rolling Z-Score statistics ($Z = \frac{X - \mu}{\sigma}$).
   - Configurable sensitivity thresholds ($Z \ge 2.0, 2.5, 3.0$).
   - Categorizes anomaly events by severity: **Low**, **Medium**, **High**, **Critical**.
   - Generates contextual explanations (e.g., overnight power leaks, weekend heavy load stacks).

5. **Peak Usage Window Identification**
   - Identifies top 10 peak consumption hours and delta percentages above household baseline.
   - Categorizes period averages: Morning Peak (07-10), Evening Peak (17-22), Overnight Baseline (00-05).

6. **Energy Saving Recommendation Engine**
   - Rule-based optimization system offering advice on load shifting, phantom standby reduction, appliance checkups, and equipment upgrades.
   - Quantifies monthly kWh savings for each recommendation.

7. **Financial Cost & Tariff Calculator**
   - Customizable electricity tariff rate ($/kWh) and monthly consumption inputs.
   - Projects monthly and annual electricity costs.
   - Provides scenario projections for 5%, 10%, 15%, 20%, and 25% energy reduction goals.

---

## 🏗️ Project Architecture

```
smart-energy-optimizer/
│
├── app.py                      # Flask Application Server & API Routes
├── requirements.txt            # Python Dependencies
├── README.md                   # Technical Documentation
│
├── data/
│   ├── household_power_consumption.txt   # UCI raw dataset (user-placed)
│   ├── sample_power_consumption.txt      # Mock dataset (auto-generated if raw missing)
│   └── processed_power_consumption.csv   # Cached preprocessed hourly data
│
├── models/
│   ├── sarima_model.pkl                  # Saved SARIMA model binary
│   ├── ml_forecast_model.pkl             # Saved ML regressor binary
│   └── forecasting_meta.json             # Model metrics & metadata
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Ingestion, cleaning & resampling
│   ├── analysis.py                       # EDA aggregations & profiles
│   ├── forecasting.py                    # SARIMA & lag-feature forecasting engine
│   ├── anomaly_detection.py              # Rolling Z-score anomaly detector
│   ├── peak_analysis.py                  # Peak load analysis
│   └── savings.py                        # Recommendation engine & tariff calculator
│
├── templates/
│   ├── base.html                         # Master layout with sidebar
│   ├── dashboard.html                    # Main overview dashboard
│   ├── forecast.html                     # Time series forecast & horizon selector
│   ├── anomalies.html                    # Anomaly logs & severity view
│   ├── peaks.html                        # Peak usage breakdown
│   ├── recommendations.html              # Energy saving tips
│   └── savings.html                      # Cost & savings calculator
│
├── static/
│   ├── css/
│   │   └── style.css                     # Custom glassmorphism dark theme
│   └── js/
│       ├── dashboard.js                  # Overview charts & metrics
│       ├── forecast.js                   # Forecast charts & retraining
│       ├── anomalies.js                  # Anomaly spike charts
│       └── savings.js                    # Calculator interactivity
│
└── tests/
    ├── test_preprocessing.py             # Preprocessing unit tests
    ├── test_forecasting.py               # Forecasting unit tests
    ├── test_anomaly.py                   # Anomaly detection unit tests
    ├── test_savings.py                   # Savings formula unit tests
    └── test_api.py                       # Flask REST API integration tests
```

---

## 📊 Dataset Setup Instructions

### Option A: Using the Official UCI Dataset
1. Download the [UCI Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2. Extract the downloaded `household_power_consumption.zip` archive.
3. Place the file `household_power_consumption.txt` inside the `data/` folder:
   ```
   smart-energy-optimizer/data/household_power_consumption.txt
   ```
4. Start the application. The system will automatically detect the raw dataset and preprocess it into `data/processed_power_consumption.csv`.

### Option B: Automatic Mock Dataset (Zero-Setup Testing)
If `household_power_consumption.txt` is not present, the application automatically generates a synthetic 1-year sample dataset (`data/sample_power_consumption.txt`) matching UCI schema and column definitions so all features and tests can be evaluated instantly.

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

### Start Flask Server
```bash
python app.py
```
Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 🧪 Running Automated Tests

Run the complete pytest test suite:
```bash
pytest tests/ -v
```
Or run individual test modules:
```bash
python -m unittest tests/test_preprocessing.py
python -m unittest tests/test_forecasting.py
python -m unittest tests/test_anomaly.py
python -m unittest tests/test_savings.py
python -m unittest tests/test_api.py
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/summary` | GET | Returns key dataset metrics, total kWh, peak hour, anomalies count, and baseline monthly cost. |
| `GET /api/consumption` | GET | Returns consumption time series. Query params: `period=daily\|hourly\|weekly`, `limit=90`. |
| `GET /api/hourly` | GET | Returns average consumption profiles by hour of day (0-23) and day of week. |
| `GET /api/forecast` | GET | Returns time-series predictions & metrics. Query params: `horizon=24\|48\|72\|168`, `retrain=true\|false`. |
| `GET /api/anomalies` | GET | Returns flagged high-consumption anomaly events. Query param: `threshold=2.0`. |
| `GET /api/peaks` | GET | Returns top 10 peak usage hours and morning/evening/overnight window averages. |
| `GET /api/recommendations` | GET | Returns rule-based energy saving recommendations and estimated kWh impact. |
| `POST /api/calculate-savings` | POST | Calculates tariff costs and 5%-25% savings scenarios. Body: `{"tariff_rate": 0.15, "monthly_kWh": 500}`. |
| `POST /api/retrain` | POST | Triggers forecasting model retraining and saves new model binaries into `models/`. |

---

## 🔮 Future Enhancements

1. **Deep Learning Forecasting (LSTM / Prophet)**
   - Add optional PyTorch/TensorFlow LSTM neural network models for multi-step long-horizon predictions.
2. **Appliance-Level Sub-metering Disaggregation (NILM)**
   - Implement Non-Intrusive Load Monitoring (NILM) algorithms to disaggregate individual appliance power curves from global active power.
3. **Automated Load Shifting Integration**
   - Provide smart home IoT integrations (e.g. Home Assistant / MQTT) to trigger automated device scheduling during low-tariff off-peak windows.
