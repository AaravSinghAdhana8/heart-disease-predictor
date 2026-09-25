# Data Issue Report: Inverted Target Label

## Symptom

The original version of this project predicted "AT RISK" for almost every
normal-looking patient.

| Test patient | Old model output |
|---|---|
| Healthy 30-year-old, max heart rate 190, no ST depression, normal thal | **75.8% — AT RISK** |
| 70-year-old, 3 blocked vessels, exercise angina, ST depression 3.5, reversible defect | **9.1% — NOT at risk** |

The model was confidently backwards, yet its test accuracy was **83.6%**.

## Step 1 — Check whether the data makes medical sense

Share of rows with `target = 1`, grouped by known risk factors (raw Kaggle file):

| Risk factor | Value | % with target = 1 |
|---|---|---|
| Exercise-induced angina (`exang`) | No | 70% |
| | **Yes** | **23%** |
| Blocked vessels (`ca`) | 0 | 74% |
| | **3** | **15%** |
| Thal test (`thal`) | Normal | 78% |
| | **Reversible defect** | **24%** |

Correlation with `target`: `oldpeak` −0.43, `age` −0.23, `thalach` (max heart
rate) +0.42.

Every well-known risk factor is associated with **less** `target = 1`. That only
makes sense if `target = 1` means *healthy*.

## Step 2 — Prove it against the original source

The original UCI Cleveland data spells out the diagnosis as Yes/No. All 303 rows
were matched on age, blood pressure, cholesterol, max heart rate and oldpeak:

| Kaggle `target` | UCI: No disease | UCI: Disease |
|---|---|---|
| 0 | 0 | **138** |
| 1 | **165** | 0 |

**303 of 303 rows are reversed.** `target = 1` means no heart disease.

The same comparison decoded every category code used by the app:

| Column | Code → meaning |
|---|---|
| cp | 0 asymptomatic, 1 atypical angina, 2 non-anginal, 3 typical angina |
| restecg | 0 LV hypertrophy, 1 normal, 2 ST-T abnormality |
| slope | 0 downsloping, 1 flat, 2 upsloping |
| thal | 1 fixed defect, 2 normal, 3 reversible defect, **0 = missing** |
| ca | 0–3 vessels, **4 = missing** |

## Why accuracy didn't reveal the bug

The model learned the inverted pattern perfectly and was tested against the same
inverted labels, so it scored well. Accuracy measures agreement with the labels,
not agreement with reality. Only a common-sense check of predictions exposed it.

## Additional issues found

1. Missing values disguised as real codes (`ca = 4`: 5 rows, `thal = 0`: 2 rows).
2. Category codes treated as numbers (no one-hot encoding).
3. One duplicate row.
4. The command-line script defaulted to `thal = 3` (reversible defect), itself a disease signal.
5. No tests or sanity checks.

## Fix

- `src/data.py`: `has_disease = 1 - target`, hidden missing codes → NaN, duplicates dropped.
- `src/model.py`: one-hot encoding + imputation inside a pipeline.
- `train.py`: refuses to save a model that fails the healthy/sick sanity check.
- `tests/test_model.py`: permanent regression tests.

## After the fix

| Test patient | New model |
|---|---|
| Healthy 30-year-old (same as above) | **11.6% — Low** |
| 70-year-old with 3 blocked vessels (same as above) | **93.4% — High** |

## Takeaway

Always sanity-check a model's predictions against domain knowledge, not just
its metrics. A high accuracy score can hide a model that is exactly wrong.
