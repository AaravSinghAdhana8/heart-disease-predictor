"""Model pipelines + prediction and explanation helpers."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY, CATEGORICAL, CODES, FEATURES, MODEL_PATH, NICE_NAMES, NUMERIC, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", categorical, CATEGORICAL),
    ])


def candidate_models() -> dict:
    """Models we compare. Each is a full pipeline: preprocessing + classifier."""
    return {
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=RANDOM_STATE),
    }


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


# ---------- inference ----------

def load_bundle(path=MODEL_PATH, auto_train=True) -> dict:
    """Load the saved model bundle.

    Bundle = {'model', 'model_name', 'baseline', 'threshold', 'train_ranges', 'sklearn_version'}

    If the file is missing, can't be read, or was saved with a different
    scikit-learn version, the model is retrained automatically (~15 seconds).
    This keeps the project runnable on any computer or on Streamlit Cloud.
    """
    import warnings
    import sklearn

    if path.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bundle = joblib.load(path)
            if bundle.get("sklearn_version") == sklearn.__version__:
                return bundle
            print("Saved model was built with a different scikit-learn version; retraining...")
        except Exception as e:  # corrupted or incompatible file
            print(f"Could not load saved model ({e}); retraining...")
    if not auto_train:
        raise FileNotFoundError(f"No usable model at {path}. Run: python train.py")
    import train  # project root is on sys.path when running app/CLI/tests
    train.main()
    return joblib.load(path)


def to_frame(patient: dict) -> pd.DataFrame:
    row = {f: patient.get(f, np.nan) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES).astype(float)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a user-supplied table into clean model input.

    Accepts sex as 1/0 or Male/Female, converts everything to numbers,
    and marks the dataset's hidden missing codes (ca=4, thal=0) as NaN.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    X = df[FEATURES].copy()
    sex_map = {"male": 1, "m": 1, "1": 1, "1.0": 1, "female": 0, "f": 0, "0": 0, "0.0": 0}
    X["sex"] = X["sex"].astype(str).str.strip().str.lower().map(sex_map)
    X = X.apply(pd.to_numeric, errors="coerce")
    X.loc[X["ca"] == 4, "ca"] = np.nan
    X.loc[X["thal"] == 0, "thal"] = np.nan
    return X


def predict_proba(bundle, X: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(X[FEATURES].astype(float))[:, 1]


def risk_band(p: float) -> str:
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Moderate"
    return "High"


def explain(bundle, patient: dict, top_k: int = 5) -> pd.DataFrame:
    """What-if explanation.

    For each feature, swap the patient's value with a 'typical healthy'
    reference value and see how much the predicted risk changes.
    Positive impact = this value pushes risk UP.
    """
    base_df = to_frame(patient)
    p = predict_proba(bundle, base_df)[0]
    rows = []
    for f in FEATURES:
        ref = bundle["baseline"][f]
        if pd.isna(patient.get(f)) or float(patient[f]) == float(ref):
            continue
        alt = base_df.copy()
        alt[f] = ref
        impact = p - predict_proba(bundle, alt)[0]
        rows.append({
            "feature": NICE_NAMES[f],
            "your_value": describe_value(f, patient[f]),
            "healthy_reference": describe_value(f, ref),
            "impact": impact,
        })
    out = pd.DataFrame(rows, columns=["feature", "your_value", "healthy_reference", "impact"])
    if out.empty:
        return out
    return out.reindex(out["impact"].abs().sort_values(ascending=False).index).head(top_k)


def describe_value(f, v):
    if f in CODES:
        return CODES[f].get(int(v), str(v))
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)
