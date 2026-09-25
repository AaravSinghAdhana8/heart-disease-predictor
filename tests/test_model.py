"""Run:  python -m pytest -q"""
import pandas as pd
import pytest

from src.data import clean, load_raw
from src.model import load_bundle, predict_proba, prepare_input, to_frame

HEALTHY = dict(age=35, sex=0, cp=0, trestbps=115, chol=190, fbs=0, restecg=1,
               thalach=180, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)
SICK = dict(age=65, sex=1, cp=0, trestbps=150, chol=290, fbs=1, restecg=0,
            thalach=105, exang=1, oldpeak=3.0, slope=1, ca=3, thal=3)


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_label_is_fixed():
    df = clean(load_raw())
    # Exercise-induced chest pain must be linked to MORE disease, not less
    rates = df.groupby("exang")["has_disease"].mean()
    assert rates[1] > rates[0]


def test_healthy_patient_is_low_risk(bundle):
    assert predict_proba(bundle, to_frame(HEALTHY))[0] < 0.3


def test_sick_patient_is_high_risk(bundle):
    assert predict_proba(bundle, to_frame(SICK))[0] > 0.7


def test_not_everyone_at_risk(bundle):
    X = clean(load_raw()).drop(columns="has_disease")
    share = (predict_proba(bundle, X) >= 0.5).mean()
    assert 0.3 < share < 0.6  # real disease rate is ~46%


def test_more_blocked_vessels_raise_risk(bundle):
    risks = [predict_proba(bundle, to_frame({**HEALTHY, "ca": c}))[0] for c in range(4)]
    assert risks[3] > risks[0]


def test_text_sex_and_missing_codes_accepted(bundle):
    df = pd.DataFrame([{**HEALTHY, "sex": "Female", "ca": 4, "thal": 0}])
    X = prepare_input(df)
    assert X["ca"].isna().all() and X["thal"].isna().all()
    assert 0 <= predict_proba(bundle, X)[0] <= 1
