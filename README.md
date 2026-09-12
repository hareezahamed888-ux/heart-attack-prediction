# Heart Attack Prediction System

A machine learning web app that estimates heart attack risk from common clinical features.  
Includes a Random Forest model, a FastAPI backend, and a clean browser UI — ready to deploy on Vercel.

## Live features
- Interactive web form at the root URL
- REST API: `POST /api/predict`
- Health check: `GET /api/health`
- Original CLI scripts (`train.py`, `predict.py`) still work locally

## Quick start (local)

```bash
pip install -r requirements.txt
python train.py          # optional – API can train on first request
python -m uvicorn api.index:app --reload
```

Then open http://localhost:8000 (or serve `public/index.html`).

### CLI prediction example
```bash
python predict.py --age 55 --sex 1 --cp 2 --trestbps 130 --chol 250 --fbs 0 --restecg 1 --thalach 150 --exang 0 --oldpeak 1.5 --slope 2 --ca 0 --thal 2
```

## Deploy to Vercel

1. This repository is already linked to the Vercel project.
2. Push to `main` triggers a new deployment.
3. Framework preset: **Other** (or leave default).
4. No extra environment variables needed.

After deployment the form is available at your Vercel URL and calls `/api/predict` on the same domain.

## API

**POST /api/predict**

```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

Response:

```json
{
  "prediction": 1,
  "label": "Heart attack likely",
  "probability": 0.82,
  "risk_percent": 82.0
}
```

## Features used by the model
- age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

## Disclaimer
This project is for educational purposes only. It does **not** provide medical advice or diagnosis.
