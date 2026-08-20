import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.app.core.config import STATIC_DIR, TEMPLATES_DIR, APP_HOST, APP_PORT, DEBUG
from backend.app.services.preprocessing import get_processed_data
from backend.app.services.analysis import (
    get_summary_metrics, get_hourly_profile, get_day_of_week_profile,
    get_monthly_profile, get_consumption_time_series
)
from backend.app.services.forecasting import get_forecast_results, EnergyForecaster
from backend.app.services.anomaly_detection import detect_anomalies
from backend.app.services.peak_analysis import analyze_peak_usage
from backend.app.services.savings import generate_recommendations, calculate_cost_and_savings

# Global dataset cache
_DATA_CACHE = None


def load_dataset(force_reload: bool = False):
    global _DATA_CACHE
    if _DATA_CACHE is None or force_reload:
        _DATA_CACHE = get_processed_data(force_reprocess=force_reload)
    return _DATA_CACHE


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Smart Energy Consumption Optimizer (FastAPI + Prophet)...")
    load_dataset()
    print("Dataset successfully loaded into memory cache.")
    yield


# Initialize FastAPI App
app = FastAPI(
    title="Smart Energy Consumption Optimizer API",
    description="Enterprise Energy Management API powered by Facebook Prophet time-series forecasting, FastAPI, and Weather/Occupancy feature engineering.",
    version="2.0.0",
    lifespan=lifespan
)

# Setup Static & Templates Directories
os.makedirs(str(STATIC_DIR), exist_ok=True)
os.makedirs(str(TEMPLATES_DIR), exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Pydantic Schemas
class SavingsRequest(BaseModel):
    monthly_kWh: Optional[float] = Field(None, description="Monthly energy consumption in kWh")
    tariff_rate: float = Field(0.15, description="Tariff rate per kWh")
    currency_symbol: str = Field("$", description="Currency symbol ($ or ₹ or €)")


class RetrainRequest(BaseModel):
    force_reload_data: bool = Field(True, description="Force reprocess dataset before retraining")


# --- HTML Page Routes ---

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"active_page": "dashboard"})


@app.get("/forecast", response_class=HTMLResponse)
async def forecast_page(request: Request):
    return templates.TemplateResponse(request=request, name="forecast.html", context={"active_page": "forecast"})


@app.get("/anomalies", response_class=HTMLResponse)
async def anomalies_page(request: Request):
    return templates.TemplateResponse(request=request, name="anomalies.html", context={"active_page": "anomalies"})


@app.get("/peaks", response_class=HTMLResponse)
async def peaks_page(request: Request):
    return templates.TemplateResponse(request=request, name="peaks.html", context={"active_page": "peaks"})


@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations_page(request: Request):
    return templates.TemplateResponse(request=request, name="recommendations.html", context={"active_page": "recommendations"})


@app.get("/savings", response_class=HTMLResponse)
async def savings_page(request: Request):
    return templates.TemplateResponse(request=request, name="savings.html", context={"active_page": "savings"})


# --- REST API Endpoints ---

@app.get("/api/summary")
async def api_summary():
    try:
        df = load_dataset()
        summary = get_summary_metrics(df)
        
        avg_monthly_kwh = summary['avg_daily_kWh'] * 30.0
        cost_info = calculate_cost_and_savings(avg_monthly_kwh, tariff_rate=0.15)
        summary['estimated_monthly_cost'] = cost_info['current_estimates']['flat_monthly_cost']
        summary['potential_monthly_savings'] = cost_info['savings_scenarios'][2]['saved_cost_monthly']  # 15% scenario
        
        anom_res = detect_anomalies(df)
        summary['anomaly_count'] = anom_res['summary']['total_anomalies']
        
        return {"status": "success", "data": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/consumption")
async def api_consumption(period: str = Query('daily', pattern="^(hourly|daily|weekly)$"), limit: int = Query(90, ge=1, le=1000)):
    try:
        df = load_dataset()
        series_data = get_consumption_time_series(df, period=period, limit=limit)
        return {"status": "success", "period": period, "data": series_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hourly")
async def api_hourly():
    try:
        df = load_dataset()
        hourly_prof = get_hourly_profile(df)
        dow_prof = get_day_of_week_profile(df)
        return {
            "status": "success",
            "hourly_profile": hourly_prof,
            "day_of_week_profile": dow_prof
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast")
async def api_forecast(horizon: int = Query(24, ge=1, le=168), retrain: bool = Query(False)):
    try:
        df = load_dataset()
        forecast_data = get_forecast_results(df, horizon_hours=horizon, force_retrain=retrain)
        return {"status": "success", "data": forecast_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomalies")
async def api_anomalies(threshold: float = Query(2.0, ge=1.0, le=5.0)):
    try:
        df = load_dataset()
        res = detect_anomalies(df, z_threshold=threshold)
        return {"status": "success", "data": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/peaks")
async def api_peaks():
    try:
        df = load_dataset()
        peak_res = analyze_peak_usage(df)
        return {"status": "success", "data": peak_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recommendations")
async def api_recommendations():
    try:
        df = load_dataset()
        anom_res = detect_anomalies(df)
        peak_res = analyze_peak_usage(df)
        recs = generate_recommendations(df, anomalies_summary=anom_res['summary'], peak_info=peak_res)
        return {"status": "success", "data": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calculate-savings")
async def api_calculate_savings(req: SavingsRequest):
    try:
        df = load_dataset()
        if req.monthly_kWh is not None and req.monthly_kWh > 0:
            monthly_kWh = req.monthly_kWh
        else:
            avg_daily = float(df['Energy_kWh'].resample('D').sum().mean())
            monthly_kWh = avg_daily * 30.0

        res = calculate_cost_and_savings(
            monthly_kWh=monthly_kWh,
            tariff_rate=req.tariff_rate,
            currency_symbol=req.currency_symbol
        )
        return {"status": "success", "data": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/retrain")
async def api_retrain(req: RetrainRequest = RetrainRequest()):
    try:
        df = load_dataset(force_reload=req.force_reload_data)
        forecaster = EnergyForecaster()
        meta = forecaster.train(df)
        return {"status": "success", "message": "Prophet model successfully retrained.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("backend.app.main:app", host=APP_HOST, port=APP_PORT, reload=DEBUG)
