"""Load and clean the heart disease dataset.

Three problems exist in the popular Kaggle heart.csv (303 rows):
1. The 'target' column is INVERTED: target=1 actually means NO disease.
   We create `has_disease = 1 - target` so that 1 really means disease.
2. Missing values are hidden as real-looking codes: ca=4 and thal=0.
   We turn them into NaN and let the model pipeline impute them.
3. One exact duplicate row, which we drop.
"""
import numpy as np
import pandas as pd

from .config import DATA_PATH, FEATURES, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    # utf-8-sig strips the invisible BOM some copies of this file start with
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix 1: correct the inverted label
    if "target" in df.columns:
        df[TARGET] = 1 - df["target"]
        df = df.drop(columns="target")

    # Fix 2: hidden missing values
    df.loc[df["ca"] == 4, "ca"] = np.nan
    df.loc[df["thal"] == 0, "thal"] = np.nan

    # Fix 3: duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_clean(path=DATA_PATH):
    df = clean(load_raw(path))
    return df[FEATURES], df[TARGET]
