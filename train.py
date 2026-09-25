"""Train, compare and save the heart disease model.

Run:  python train.py
"""
import json

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split

from src.config import FEATURES, METRICS_PATH, MODEL_PATH, NICE_NAMES, RANDOM_STATE, TARGET, CATEGORICAL, BINARY
from src.data import clean, load_raw
from src.model import candidate_models, make_pipeline

THRESHOLD = 0.5


def main():
    raw = load_raw()
    print(f"Raw data: {raw.shape[0]} rows")
    df = clean(raw)
    X, y = df[FEATURES], df[TARGET]
    print(f"After cleaning: {len(df)} rows | disease rate: {y.mean():.1%}")
    print(f"Missing values imputed -> ca: {X['ca'].isna().sum()}, thal: {X['thal'].isna().sum()}")

    # 1) Hold out 20% as a test set the model never sees during selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # 2) Compare candidate models with repeated 5-fold cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scoring = ["roc_auc", "accuracy", "recall", "precision", "f1"]
    cv_rows = []
    print("\nCross-validation on training set (5-fold x 5 repeats):")
    for name, clf in candidate_models().items():
        res = cross_validate(make_pipeline(clf), X_train, y_train, cv=cv, scoring=scoring)
        row = {"model": name}
        for s in scoring:
            row[s] = float(res[f"test_{s}"].mean())
            row[f"{s}_std"] = float(res[f"test_{s}"].std())
        cv_rows.append(row)
        print(f"  {name:20s} ROC-AUC {row['roc_auc']:.3f} ± {row['roc_auc_std']:.3f} | "
              f"Acc {row['accuracy']:.3f} | Recall {row['recall']:.3f}")

    best = max(cv_rows, key=lambda r: r["roc_auc"])
    best_name = best["model"]
    print(f"\nSelected: {best_name} (highest mean ROC-AUC)")

    # 3) Final, honest check on the untouched test set
    pipe = make_pipeline(clone(candidate_models()[best_name])).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    test = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)
    print("Test set:", {k: round(v, 3) for k, v in test.items()})
    print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)

    # Which features matter? (permutation importance on the test set)
    imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                 n_repeats=20, random_state=RANDOM_STATE)
    importance = sorted(
        [{"feature": NICE_NAMES[f], "importance": float(m)} for f, m in zip(FEATURES, imp.importances_mean)],
        key=lambda d: -d["importance"])

    # 4) Retrain on ALL data for the deployed model (more data = better model)
    final = make_pipeline(clone(candidate_models()[best_name])).fit(X, y)

    # Healthy reference values for "what-if" explanations = typical healthy patient
    healthy = X[y == 0]
    baseline = {}
    for f in FEATURES:
        if f in CATEGORICAL or f in BINARY or f == "ca":
            baseline[f] = float(healthy[f].mode().iloc[0])
        else:
            baseline[f] = float(healthy[f].median())

    # 5) Sanity check: the old model failed exactly this test
    healthy_case = dict(age=35, sex=0, cp=0, trestbps=115, chol=190, fbs=0, restecg=1,
                        thalach=180, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)
    sick_case = dict(age=65, sex=1, cp=0, trestbps=150, chol=290, fbs=1, restecg=0,
                     thalach=105, exang=1, oldpeak=3.0, slope=1, ca=3, thal=3)
    for label, case in [("healthy", healthy_case), ("sick", sick_case)]:
        p = final.predict_proba(pd.DataFrame([case])[FEATURES].astype(float))[0, 1]
        print(f"Sanity check - {label} patient -> risk {p:.1%}")
    p_h = final.predict_proba(pd.DataFrame([healthy_case])[FEATURES].astype(float))[0, 1]
    p_s = final.predict_proba(pd.DataFrame([sick_case])[FEATURES].astype(float))[0, 1]
    assert p_h < 0.3 < 0.7 < p_s, "Sanity check failed: model is not behaving sensibly!"

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": final,
        "sklearn_version": sklearn.__version__,
        "model_name": best_name,
        "baseline": baseline,
        "threshold": THRESHOLD,
        "train_ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
    }, MODEL_PATH)

    metrics = {
        "model_name": best_name,
        "n_rows": int(len(df)),
        "disease_rate": float(y.mean()),
        "cv": cv_rows,
        "test": test,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "importance": importance,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model  -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
