"""Command-line predictions.

Examples:
  python predict_cli.py --age 63 --sex Male --cp 0 --thalach 110 --exang 1 --oldpeak 2.5 --ca 2 --thal 3
  python predict_cli.py --input-csv sample_data/patients.csv --output-csv predictions.csv
"""
import argparse

import pandas as pd

from src.config import CODES, FEATURES
from src.model import explain, load_bundle, predict_proba, prepare_input, risk_band


def main():
    ap = argparse.ArgumentParser(description="Predict heart disease risk.")
    ap.add_argument("--input-csv", help="CSV with one patient per row")
    ap.add_argument("--output-csv", default="predictions.csv")
    ap.add_argument("--age", type=int, default=50)
    ap.add_argument("--sex", default="Male", help="Male/Female or 1/0")
    ap.add_argument("--cp", type=int, choices=[0, 1, 2, 3], default=2, help=str(CODES["cp"]))
    ap.add_argument("--trestbps", type=int, default=130)
    ap.add_argument("--chol", type=int, default=240)
    ap.add_argument("--fbs", type=int, choices=[0, 1], default=0)
    ap.add_argument("--restecg", type=int, choices=[0, 1, 2], default=1, help=str(CODES["restecg"]))
    ap.add_argument("--thalach", type=int, default=150)
    ap.add_argument("--exang", type=int, choices=[0, 1], default=0)
    ap.add_argument("--oldpeak", type=float, default=1.0)
    ap.add_argument("--slope", type=int, choices=[0, 1, 2], default=2, help=str(CODES["slope"]))
    ap.add_argument("--ca", type=int, choices=[0, 1, 2, 3], default=0)
    ap.add_argument("--thal", type=int, choices=[1, 2, 3], default=2, help=str(CODES["thal"]))
    args = ap.parse_args()

    bundle = load_bundle()

    if args.input_csv:
        raw = pd.read_csv(args.input_csv, encoding="utf-8-sig")
        probs = predict_proba(bundle, prepare_input(raw))
        out = raw.copy()
        out["risk_probability"] = probs.round(3)
        out["risk_band"] = [risk_band(p) for p in probs]
        out.to_csv(args.output_csv, index=False)
        print(out[["risk_probability", "risk_band"]].to_string())
        print(f"\nSaved {len(out)} predictions to {args.output_csv}")
        return

    patient_df = prepare_input(pd.DataFrame([{f: getattr(args, f) for f in FEATURES}]))
    patient = patient_df.iloc[0].to_dict()
    p = float(predict_proba(bundle, patient_df)[0])
    print(f"\nRisk of heart disease: {p:.1%}  ({risk_band(p)} risk)")
    print("\nTop factors (+ raises risk, - lowers risk):")
    for _, r in explain(bundle, patient).iterrows():
        print(f"  {r['impact']*100:+6.1f} pts  {r['feature']}: {r['your_value']} "
              f"(healthy ref: {r['healthy_reference']})")


if __name__ == "__main__":
    main()
