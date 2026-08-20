"""
Core configuration module for the Smart Energy Consumption Optimizer.
Provides environment-aware path resolutions and runtime settings.
"""

import os
from pathlib import Path

# Base Directory Resolvers
CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
BACKEND_DIR = APP_DIR.parent
ROOT_DIR = BACKEND_DIR.parent

# Application Directories
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", str(ROOT_DIR / "frontend")))
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", str(FRONTEND_DIR / "templates")))
STATIC_DIR = Path(os.getenv("STATIC_DIR", str(FRONTEND_DIR / "static")))


def _resolve_data_dir() -> Path:
    env_override = os.getenv("DATA_DIR")
    if env_override:
        return Path(env_override)
    # Check backend/data, backend/app/data, or root data
    for candidate in [BACKEND_DIR / "data", APP_DIR / "data", ROOT_DIR / "data"]:
        if candidate.exists() and (candidate / "sample_power_consumption.txt").exists():
            return candidate
    return BACKEND_DIR / "data"


def _resolve_models_dir() -> Path:
    env_override = os.getenv("MODELS_DIR")
    if env_override:
        return Path(env_override)
    # Check backend/models, backend/app/models, or root models
    for candidate in [BACKEND_DIR / "models", APP_DIR / "models", ROOT_DIR / "models"]:
        if candidate.exists() and (candidate / "prophet_model.pkl").exists():
            return candidate
    return BACKEND_DIR / "models"


DATA_DIR = _resolve_data_dir()
MODELS_DIR = _resolve_models_dir()

# Data File Paths
RAW_DATA_PATH = str(DATA_DIR / "household_power_consumption.txt")
SAMPLE_DATA_PATH = str(DATA_DIR / "sample_power_consumption.txt")
PROCESSED_DATA_PATH = str(DATA_DIR / "processed_power_consumption.csv")

# Model File Paths
PROPHET_MODEL_PATH = str(MODELS_DIR / "prophet_model.pkl")
METADATA_PATH = str(MODELS_DIR / "forecasting_meta.json")
ML_FORECAST_MODEL_PATH = str(MODELS_DIR / "ml_forecast_model.pkl")
SARIMA_MODEL_PATH = str(MODELS_DIR / "sarima_model.pkl")

# Server Configurations
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
