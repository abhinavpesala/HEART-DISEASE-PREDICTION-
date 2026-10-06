from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

app = FastAPI(title="Cardiac Risk Stack")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path("backend/models")

try:
    tab_model = joblib.load(MODEL_DIR / "tabular_model.joblib")
    ecg_model = joblib.load(MODEL_DIR / "ecg_model.joblib")
    echo_model = joblib.load(MODEL_DIR / "echo_model.joblib")
    meta = joblib.load(MODEL_DIR / "meta_learner.joblib")
    
    tab_pre = joblib.load(MODEL_DIR / "tab_preprocessor.joblib")
    ecg_pre = joblib.load(MODEL_DIR / "ecg_preprocessor.joblib")
    echo_pre = joblib.load(MODEL_DIR / "echo_preprocessor.joblib")
    
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    
    print("✅ Models loaded")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure you ran notebook.py first to generate models")
    raise

class PatientInput(BaseModel):
    Age: float
    Sex: str
    DM: int
    HTN: int
    BP: int
    PR: int
    Typical_Chest_Pain: int = 0
    Atypical: int = 0
    Nonanginal: int = 0
    Dyspnea: int = 0
    EF_TTE: float = 50
    Region_RWMA: str = "N"
    VHD: str = "N"
    BBB: str = "N"
    Function_Class: str = "N"
    Q_Wave: int = 0
    St_Elevation: int = 0
    St_Depression: int = 0
    Tinversion: int = 0
    LVH: int = 0
    
    class Config:
        schema_extra = {
            "example": {
                "Age": 60,
                "Sex": "Male",
                "DM": 1,
                "HTN": 1,
                "BP": 130,
                "PR": 75,
                "Typical_Chest_Pain": 1,
                "EF_TTE": 50
            }
        }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": True,
        "metadata": metadata
    }

@app.post("/predict")
def predict(patient: PatientInput):
    """
    Predict cardiac risk from patient data
    
    Returns:
        - risk_score: probability (0-1)
        - risk_label: High/Moderate/Low
        - confidence: model confidence
        - branch_scores: individual branch probabilities
    """
    try:
        # Convert to dict and then DataFrame
        patient_dict = patient.dict()
        row = pd.DataFrame([patient_dict])
        
        # Map column names (underscores to spaces)
        col_map = {
            "Typical_Chest_Pain": "Typical Chest Pain",
            "Region_RWMA": "Region RWMA",
            "EF_TTE": "EF-TTE",
            "Q_Wave": "Q Wave",
            "St_Elevation": "St Elevation",
            "St_Depression": "St Depression",
            "Function_Class": "Function Class"
        }
        row.rename(columns=col_map, inplace=True)
        
        # Define branch columns
        ecg_cols = ["PR", "BP", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Region RWMA"]
        echo_cols = ["EF-TTE", "Region RWMA", "VHD"]
        tabular_cols = [c for c in row.columns if c not in ecg_cols + echo_cols]
        
        # Extract branch data
        X_tab = row[[c for c in tabular_cols if c in row.columns]].fillna(0)
        X_ecg = row[[c for c in ecg_cols if c in row.columns]].fillna(0)
        X_echo = row[[c for c in echo_cols if c in row.columns]].fillna(0)
        
        # Preprocess
        tab_proc = tab_pre.transform(X_tab)
        ecg_proc = ecg_pre.transform(X_ecg)
        echo_proc = echo_pre.transform(X_echo)
        
        # Get branch predictions
        tab_p = float(tab_model.predict_proba(tab_proc)[0, 1])
        ecg_p = float(ecg_model.predict_proba(ecg_proc)[0, 1])
        echo_p = float(echo_model.predict_proba(echo_proc)[0, 1])
        
        # Meta-learner final prediction
        final_p = float(meta.predict_proba(np.array([[tab_p, ecg_p, echo_p]]))[0, 1])
        
        # Risk categorization
        risk = "High" if final_p >= 0.7 else "Moderate" if final_p >= 0.4 else "Low"
        
        return {
            "risk_score": round(final_p, 4),
            "risk_label": risk,
            "confidence": round(1 - abs(final_p - 0.5) * 2, 4),
            "branch_scores": {
                "tabular": round(tab_p, 4),
                "ecg": round(ecg_p, 4),
                "echo": round(echo_p, 4)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)