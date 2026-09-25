"""Project-wide settings and feature definitions.

Category codes below were verified by matching every row of data/heart.csv
against the original UCI Cleveland dataset (see INTERVIEW_GUIDE.md).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_PATH = ROOT / "models" / "heart_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

RANDOM_STATE = 42
TARGET = "has_disease"  # 1 = heart disease, 0 = healthy (after label fix)

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Human-readable names shown in the app
NICE_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate (stress test)",
    "exang": "Chest pain during exercise",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST segment slope",
    "ca": "Blocked major vessels",
    "thal": "Thalassemia test",
}

# Code -> meaning, in THIS dataset's encoding (verified against UCI)
CODES = {
    "sex": {1: "Male", 0: "Female"},
    "cp": {0: "Asymptomatic (no pain)", 1: "Atypical angina",
           2: "Non-anginal pain", 3: "Typical angina"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {1: "Normal", 2: "ST-T wave abnormality", 0: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {2: "Upsloping", 1: "Flat", 0: "Downsloping"},
    "thal": {2: "Normal", 1: "Fixed defect", 3: "Reversible defect"},
}

# Allowed input ranges (a little wider than the training data)
RANGES = {
    "age": (18, 100, 50),
    "trestbps": (80, 220, 130),
    "chol": (100, 600, 240),
    "thalach": (60, 220, 150),
    "oldpeak": (0.0, 7.0, 1.0),
    "ca": (0, 3, 0),
}

UNITS = {"trestbps": "mm Hg", "chol": "mg/dl", "thalach": "bpm", "age": "years"}
