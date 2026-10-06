"""
Complete Cardiac Risk Stacked Ensemble Pipeline
Run this entire file to train models and generate artifacts.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import logging

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: FOUNDATION MODEL MANAGER (inline)
# ============================================================================

class ECGFMEncoder:
    """ECG-FM Foundation Model Encoder"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.device = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            import torch
            from transformers import AutoModel
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading ECG-FM from {model_path}...")
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ ECG-FM loaded")
        except Exception as e:
            logger.warning(f"ECG-FM load failed: {e}. Using fallback.")
            self.model = None
    
    def encode(self, ecg_data):
        if self.model is not None:
            try:
                import torch
                if isinstance(ecg_data, np.ndarray):
                    ecg_tensor = torch.from_numpy(ecg_data).float().unsqueeze(0)
                else:
                    ecg_tensor = ecg_data.float().unsqueeze(0)
                
                ecg_tensor = ecg_tensor.to(self.device)
                
                with torch.no_grad():
                    embeddings = self.model(ecg_tensor)
                
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                
                return embeddings[0].cpu().numpy()
            except Exception as e:
                logger.warning(f"Encoding failed: {e}. Using fallback.")
                return self._fallback_encode(ecg_data)
        else:
            return self._fallback_encode(ecg_data)
    
    def _fallback_encode(self, ecg_data):
        if isinstance(ecg_data, np.ndarray):
            embedding = np.zeros(384)
            embedding[:min(len(ecg_data), 384)] = ecg_data[:384]
            return embedding
        return np.zeros(384)


class EchoPrimeEncoder:
    """EchoPrime Foundation Model Encoder"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.device = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            import torch
            from transformers import AutoModel
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading EchoPrime from {model_path}...")
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ EchoPrime loaded")
        except Exception as e:
            logger.warning(f"EchoPrime load failed: {e}. Using fallback.")
            self.model = None
    
    def encode(self, echo_data):
        if self.model is not None:
            try:
                import torch
                if isinstance(echo_data, np.ndarray):
                    echo_tensor = torch.from_numpy(echo_data).float().unsqueeze(0)
                else:
                    echo_tensor = echo_data.float().unsqueeze(0)
                
                echo_tensor = echo_tensor.to(self.device)
                
                with torch.no_grad():
                    embeddings = self.model(echo_tensor)
                
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                
                return embeddings[0].cpu().numpy()
            except Exception as e:
                logger.warning(f"Encoding failed: {e}. Using fallback.")
                return self._fallback_encode(echo_data)
        else:
            return self._fallback_encode(echo_data)
    
    def _fallback_encode(self, echo_data):
        if isinstance(echo_data, np.ndarray):
            embedding = np.zeros(256)
            embedding[:min(len(echo_data), 256)] = echo_data[:256]
            return embedding
        return np.zeros(256)


class FoundationModelManager:
    """Manager for ECG-FM and EchoPrime models"""
    
    def __init__(self, ecg_model_path=None, echo_model_path=None):
        self.ecg_fm = ECGFMEncoder(ecg_model_path)
        self.echoprime = EchoPrimeEncoder(echo_model_path)
        logger.info("✅ Foundation model manager initialized")
    
    def encode_batch_ecg(self, ecg_batch):
        embeddings = []
        for ecg_sample in ecg_batch:
            embedding = self.ecg_fm.encode(ecg_sample)
            embeddings.append(embedding)
        return np.vstack(embeddings)
    
    def encode_batch_echo(self, echo_batch):
        embeddings = []
        for echo_sample in echo_batch:
            embedding = self.echoprime.encode(echo_sample)
            embeddings.append(embedding)
        return np.vstack(embeddings)

# ============================================================================
# SECTION 2: CONFIG & INITIALIZATION
# ============================================================================

TARGET = "Cath"

ECG_LIKE_COLS = [
    "PR", "BP", "Q Wave", "St Elevation", "St Depression",
    "Tinversion", "LVH", "Region RWMA"
]

ECHO_LIKE_COLS = [
    "EF-TTE", "Region RWMA", "VHD", "LAD", "LCX", "RCA"
]

DROP_COLS = [
    "LAD", "LCX", "RCA",
    "Exertional CP",
    "CHF", "LowTH Ang", "CVA", "Weak Peripheral Pulse", "CRF", "Thyroid Disease"
]

# Initialize foundation models (set paths if available)
ecg_model_path = None   # e.g., "./models/ecg_fm" or "username/ecg-fm"
echo_model_path = None  # e.g., "./models/echo_prime" or "username/echoprime"

fm = FoundationModelManager(
    ecg_model_path=ecg_model_path,
    echo_model_path=echo_model_path
)

logger.info("✅ Configuration loaded")

# ============================================================================
# SECTION 3: LOAD & CLEAN DATA
# ============================================================================

logger.info("\n[LOADING DATA]")
df = pd.read_csv("dataset.csv")

# Fix label typo
if "Sex" in df.columns:
    df["Sex"] = df["Sex"].replace({"Fmale": "Female"})

# Convert target
if TARGET in df.columns:
    df[TARGET] = df[TARGET].map({"CAD": 1, "Normal": 0})

# Remove leakage / constant columns
for c in DROP_COLS:
    if c in df.columns:
        df = df.drop(columns=[c])

# Y/N to binary
yn_cols = [
    "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity",
    "DLP", "Airway disease", "Edema", "Lung rales",
    "Systolic Murmur", "Diastolic Murmur",
    "Typical Chest Pain", "Atypical", "Nonanginal",
    "Dyspnea", "Q Wave", "St Elevation", "St Depression",
    "Tinversion", "LVH", "Poor R Progression"
]

for c in yn_cols:
    if c in df.columns:
        df[c] = df[c].replace({"Y": 1, "N": 0, "Yes": 1, "No": 0})
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Ordinal VHD
if "VHD" in df.columns:
    df["VHD"] = df["VHD"].map({"N": 0, "mild": 1, "Moderate": 2, "Severe": 3})
    df["VHD"] = pd.to_numeric(df["VHD"], errors="coerce")

# Feature engineering
if {"BP", "PR"}.issubset(df.columns):
    df["pulse_pressure"] = df["BP"] - df["PR"]

if "EF-TTE" in df.columns:
    df["ef_low_50"] = (df["EF-TTE"] < 50).astype(int)

# Remove redundant columns
for c in ["Weight", "Length", "Obesity"]:
    if c in df.columns:
        df = df.drop(columns=[c])

logger.info(f"✅ Data cleaned. Shape: {df.shape}")
logger.info(f"Target distribution: {df[TARGET].value_counts().to_dict()}")

# ============================================================================
# SECTION 4: BUILD BRANCHES & PREPROCESSORS
# ============================================================================

logger.info("\n[BUILDING BRANCHES]")

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

ecg_cols = [c for c in ECG_LIKE_COLS if c in X.columns]
echo_cols = [c for c in ECHO_LIKE_COLS if c in X.columns]
tabular_cols = [c for c in X.columns if c not in ecg_cols + echo_cols]

X_tab = X[tabular_cols]
X_ecg = X[ecg_cols]
X_echo = X[echo_cols]

# Tabular preprocessor
num_cols_tab = X_tab.select_dtypes(include=["number"]).columns.tolist()
cat_cols_tab = [c for c in X_tab.columns if c not in num_cols_tab]

transformers_tab = []
if num_cols_tab:
    transformers_tab.append(
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols_tab)
    )
if cat_cols_tab:
    transformers_tab.append(
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols_tab)
    )

tab_pre = ColumnTransformer(transformers=transformers_tab, remainder="drop")

# ECG and Echo preprocessors
num_cols_ecg = X_ecg.select_dtypes(include=["number"]).columns.tolist()
ecg_pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols_ecg)
    ],
    remainder="drop"
)

num_cols_echo = X_echo.select_dtypes(include=["number"]).columns.tolist()
echo_pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols_echo)
    ],
    remainder="drop"
)

# Transform data
X_tab_proc = tab_pre.fit_transform(X_tab)
X_ecg_proc = ecg_pre.fit_transform(X_ecg)
X_echo_proc = echo_pre.fit_transform(X_echo)

# Foundation model encoding
if len(X_ecg_proc) > 0:
    ecg_embeddings = fm.encode_batch_ecg(np.asarray(X_ecg_proc))
else:
    ecg_embeddings = np.empty((len(X), 384))

if len(X_echo_proc) > 0:
    echo_embeddings = fm.encode_batch_echo(np.asarray(X_echo_proc))
else:
    echo_embeddings = np.empty((len(X), 256))

logger.info("✅ Branch features built")
logger.info(f"  Tabular: {X_tab_proc.shape}")
logger.info(f"  ECG embeddings: {ecg_embeddings.shape}")
logger.info(f"  Echo embeddings: {echo_embeddings.shape}")

# ============================================================================
# SECTION 5: OOF STACKING & META-LEARNER
# ============================================================================

logger.info("\n[TRAINING OOF & META-LEARNER]")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tab_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary:logistic"
)

ecg_model = LGBMClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

echo_model = CatBoostClassifier(
    iterations=500,
    depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
    verbose=False
)

# OOF predictions
logger.info("Generating OOF for Tabular...")
oof_tab = cross_val_predict(tab_model, X_tab_proc, y, cv=cv, method="predict_proba")[:, 1]

logger.info("Generating OOF for ECG...")
oof_ecg = cross_val_predict(ecg_model, ecg_embeddings, y, cv=cv, method="predict_proba")[:, 1]

logger.info("Generating OOF for Echo...")
oof_echo = cross_val_predict(echo_model, echo_embeddings, y, cv=cv, method="predict_proba")[:, 1]

X_meta = np.column_stack([oof_tab, oof_ecg, oof_echo])

meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
meta.fit(X_meta, y)

meta_oof = meta.predict_proba(X_meta)[:, 1]

logger.info("✅ Meta-learner trained")
logger.info(f"  ROC-AUC: {roc_auc_score(y, meta_oof):.4f}")
logger.info(f"  Brier: {brier_score_loss(y, meta_oof):.4f}")

# ============================================================================
# SECTION 6: FINAL TRAINING & SAVE ARTIFACTS
# ============================================================================

logger.info("\n[FINAL TRAINING & SAVE]")

tab_model.fit(X_tab_proc, y)
ecg_model.fit(ecg_embeddings, y)
echo_model.fit(echo_embeddings, y)

tab_pred = tab_model.predict_proba(X_tab_proc)[:, 1]
ecg_pred = ecg_model.predict_proba(ecg_embeddings)[:, 1]
echo_pred = echo_model.predict_proba(echo_embeddings)[:, 1]

X_meta_final = np.column_stack([tab_pred, ecg_pred, echo_pred])
meta.fit(X_meta_final, y)

artifact_dir = Path("backend/models")
artifact_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(tab_model, artifact_dir / "tabular_model.joblib")
joblib.dump(ecg_model, artifact_dir / "ecg_model.joblib")
joblib.dump(echo_model, artifact_dir / "echo_model.joblib")
joblib.dump(meta, artifact_dir / "meta_learner.joblib")

joblib.dump(tab_pre, artifact_dir / "tab_preprocessor.joblib")
joblib.dump(ecg_pre, artifact_dir / "ecg_preprocessor.joblib")
joblib.dump(echo_pre, artifact_dir / "echo_preprocessor.joblib")

with open(artifact_dir / "metadata.json", "w") as f:
    json.dump({
        "branch_order": ["tabular", "ecg", "echo"],
        "roc_auc": float(roc_auc_score(y, meta_oof)),
        "brier": float(brier_score_loss(y, meta_oof)),
        "foundation_models_used": {
            "ecg_fm": ecg_model_path is not None,
            "echo_prime": echo_model_path is not None
        }
    }, f, indent=2)

logger.info(f"✅ All artifacts saved to {artifact_dir}")

# ============================================================================
# SECTION 7: SMOKE TEST (Single Prediction)
# ============================================================================

logger.info("\n[SMOKE TEST]")

sample = pd.DataFrame([{
    "Age": 60,
    "Sex": "Male",
    "DM": 1,
    "HTN": 1,
    "BP": 130,
    "PR": 75,
    "Typical Chest Pain": 1,
    "Atypical": 0,
    "Nonanginal": 0,
    "Dyspnea": 0,
    "EF-TTE": 50,
    "Region RWMA": "N",
    "VHD": "N",
    "BBB": "N",
    "Function Class": "N",
    "Q Wave": 0,
    "St Elevation": 0,
    "St Depression": 0,
    "Tinversion": 0,
    "LVH": 0
}])

tab_proc = tab_pre.transform(sample[tabular_cols])
ecg_proc = ecg_pre.transform(sample[ecg_cols])
echo_proc = echo_pre.transform(sample[echo_cols])

ecg_emb = fm.encode_batch_ecg(np.asarray(ecg_proc))
echo_emb = fm.encode_batch_echo(np.asarray(echo_proc))

tab_p = tab_model.predict_proba(tab_proc)[0, 1]
ecg_p = ecg_model.predict_proba(ecg_emb)[0, 1]
echo_p = echo_model.predict_proba(echo_emb)[0, 1]

final_input = np.array([[tab_p, ecg_p, echo_p]])
final_p = meta.predict_proba(final_input)[0, 1]

logger.info("Sample prediction:")
logger.info(f"  Tabular: {tab_p:.4f}")
logger.info(f"  ECG: {ecg_p:.4f}")
logger.info(f"  Echo: {echo_p:.4f}")
logger.info(f"  Final CAD probability: {final_p:.4f}")

logger.info("\n✅ PIPELINE COMPLETE")
#pip install -r requirements.txt