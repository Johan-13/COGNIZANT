"""
Root application runner for Smart Energy Consumption Optimizer.
Usage:
    python run.py
"""
import sys
import os
from pathlib import Path

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import APP_HOST, APP_PORT, DEBUG

if __name__ == '__main__':
    import uvicorn
    print(f"Starting Smart Energy Consumption Optimizer on http://{APP_HOST}:{APP_PORT}")
    uvicorn.run("backend.app.main:app", host=APP_HOST, port=APP_PORT, reload=True)
