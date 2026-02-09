# NetPredict: 5G Network Anomaly Detection

An end-to-end MLOps project for detecting anomalies in 5G network telemetry data.

## Features
- Automated data pipeline for network KPI processing
- XGBoost model with MLflow experiment tracking
- FastAPI inference service with Docker containerization
- Prometheus/Grafana monitoring stack
- CI/CD with GitHub Actions

## Quick Start
```bash
# 1. Clone repo
git clone <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training pipeline
python src/models/train.py

# 4. Start API server
uvicorn src.api.main:app --reload