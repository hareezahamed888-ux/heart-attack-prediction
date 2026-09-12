from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import sys

# Ensure project root is on path so we can import heart_attack_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heart_attack_model import FEATURE_COLUMNS, load_model, predict as model_predict

app = FastAPI(title="Heart Attack Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at cold start
_model = None

def get_model():
    global _model
    if _model is None:
        # Prefer the committed model path relative to project root
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model",
            "heart_attack_model.pkl",
        )
        _model = load_model(model_path)
    return _model


class PatientInput(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure")
    chol: int = Field(..., ge=50, le=600, description="Serum cholesterol mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=1, le=3, description="Thalassemia (1=normal, 2=fixed, 3=reversible)")


@app.get("/")
def root():
    return {"status": "ok", "message": "Heart Attack Prediction API", "docs": "/docs"}


@app.get("/api/health")
def health():
    try:
        m = get_model()
        return {"status": "ok", "model_loaded": m is not None}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/api/predict")
def predict(patient: PatientInput):
    try:
        model = get_model()
        patient_info = [
            patient.age,
            patient.sex,
            patient.cp,
            patient.trestbps,
            patient.chol,
            patient.fbs,
            patient.restecg,
            patient.thalach,
            patient.exang,
            patient.oldpeak,
            patient.slope,
            patient.ca,
            patient.thal,
        ]
        result = model_predict(patient_info, model=model)
        label = "Heart attack likely" if result["prediction"] == 1 else "Heart attack unlikely"
        return {
            "prediction": result["prediction"],
            "label": label,
            "probability": round(result["probability"], 4),
            "risk_percent": round(result["probability"] * 100, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
