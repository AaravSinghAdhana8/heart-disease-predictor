"""
Run:  streamlit run app.py
"""
import json
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__":
    try:
        from streamlit import runtime
        running_in_streamlit = runtime.exists()
    except Exception:
        running_in_streamlit = True
    if not running_in_streamlit:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())

from src.config import CODES, FEATURES, METRICS_PATH, NICE_NAMES, RANGES, UNITS  # noqa: E402
from src.data import clean, load_raw  # noqa: E402
from src.model import explain, load_bundle, predict_proba, prepare_input, risk_band, to_frame  # noqa: E402


def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


NEW_STREAMLIT = _st_version() >= (1, 50)
FULL_WIDTH = {"width": "stretch"} if NEW_STREAMLIT else {"use_container_width": True}

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀", layout="wide")

INK, MUTED, PAPER = "#13233A", "#5B6B7F", "#F5F8FA"
COLORS = {"Low": "#1F7A5C", "Moderate": "#B7791F", "High": "#C62839"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button {{ font-family: 'Public Sans', system-ui, sans-serif; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.result {{ background: {PAPER}; border-left: 6px solid var(--c); border-radius: 4px; padding: 1.2rem 1.4rem; }}
.result .pct {{ font-size: 3.2rem; font-weight: 800; color: var(--c); line-height: 1; }}
.result .band {{ font-size: 1.1rem; font-weight: 600; color: var(--c); margin-top: .3rem; }}
.result .note {{ color: {MUTED}; font-size: .9rem; margin-top: .6rem; }}
.small {{ color: {MUTED}; font-size: .85rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the model (first run can take ~15 seconds)...")
def get_bundle():
    try:
        return load_bundle()
    except Exception as e:
        st.error(f"Could not prepare the model: {e}")
        return None


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


def ecg_svg(color: str, p: float) -> str:
    """Heartbeat line whose spike height grows with risk."""
    h = 12 + 40 * p
    pts = []
    for k in range(4):
        x = k * 150
        pts += [f"{x},60", f"{x+50},60", f"{x+60},{60-h*0.3:.0f}", f"{x+70},60",
                f"{x+80},60", f"{x+88},{60+h*0.35:.0f}", f"{x+96},{60-h:.0f}",
                f"{x+104},{60+h*0.5:.0f}", f"{x+112},60", f"{x+150},60"]
    return (f'<svg viewBox="0 0 600 120" width="100%" height="70" preserveAspectRatio="none">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round"/></svg>')


bundle = get_bundle()
if bundle is None:
    st.info("Run `python train.py` in the project folder, then refresh this page.")
    st.stop()

st.title("Heart Disease Risk Predictor — by Aarav Singh Adhana")
st.markdown(f'<p class="small">Estimates the chance of heart disease from 13 clinical measurements. '
            f'Model: {bundle["model_name"]}, trained on the UCI Cleveland dataset. '
            f'For learning and demonstration only — not a medical diagnosis.</p>', unsafe_allow_html=True)

tab_one, tab_batch, tab_perf, tab_data = st.tabs(
    ["Check one patient", "Check many patients (CSV)", "Model performance", "Data & fixes"])

# ---------------------------------------------------------------- one patient
PRESETS = {
    "Healthy example": dict(age=35, sex=0, cp=0, trestbps=115, chol=190, fbs=0, restecg=1,
                            thalach=180, exang=0, oldpeak=0.0, slope=2, ca=0, thal=2),
    "High-risk example": dict(age=65, sex=1, cp=0, trestbps=150, chol=290, fbs=1, restecg=0,
                              thalach=105, exang=1, oldpeak=3.0, slope=1, ca=3, thal=3),
}
DEFAULT = dict(age=50, sex=1, cp=2, trestbps=130, chol=240, fbs=0, restecg=1,
               thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=2)

def load_preset(values):
    for k, v in values.items():
        st.session_state[f"in_{k}"] = v


with tab_one:
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    c1.button("Fill healthy example", on_click=load_preset, args=(PRESETS["Healthy example"],))
    c2.button("Fill high-risk example", on_click=load_preset, args=(PRESETS["High-risk example"],))
    c3.button("Reset", on_click=load_preset, args=(DEFAULT,))

    left, right = st.columns([1.15, 1], gap="large")
    p = {}
    with left:
        st.subheader("Patient details")
        a, b = st.columns(2)

        def num(col, f, step=1):
            lo, hi, d = RANGES[f]
            label = NICE_NAMES[f] + (f" ({UNITS[f]})" if f in UNITS else "")
            key = f"in_{f}"
            st.session_state.setdefault(key, DEFAULT[f])
            return col.number_input(label, min_value=lo, max_value=hi, step=step, key=key)

        def cat(col, f):
            key = f"in_{f}"
            st.session_state.setdefault(key, DEFAULT[f])
            opts = list(CODES[f].keys())
            return col.selectbox(NICE_NAMES[f], opts, format_func=lambda v: CODES[f][v], key=key)

        p["age"] = num(a, "age")
        p["sex"] = cat(b, "sex")
        p["cp"] = cat(a, "cp")
        p["trestbps"] = num(b, "trestbps")
        p["chol"] = num(a, "chol")
        p["fbs"] = cat(b, "fbs")
        p["restecg"] = cat(a, "restecg")
        p["thalach"] = num(b, "thalach")
        p["exang"] = cat(a, "exang")
        p["oldpeak"] = num(b, "oldpeak", step=0.1)
        p["slope"] = cat(a, "slope")
        p["ca"] = num(b, "ca")
        p["thal"] = cat(a, "thal")

        with st.expander("What do these fields mean?"):
            st.markdown("""
- **Chest pain type** – *Asymptomatic* sounds harmless, but in this data it is the group with the **most** heart disease (silent disease often shows no typical pain).
- **Max heart rate** – highest rate reached in a treadmill stress test. A heart that can't reach a high rate is a warning sign.
- **ST depression / slope** – ECG changes during exercise. More depression and a flat/downsloping slope point to poor blood flow.
- **Blocked major vessels** – number of vessels (0–3) coloured by fluoroscopy.
- **Thalassemia test** – blood-flow scan; a *reversible defect* is a strong disease signal.
""")

    prob = float(predict_proba(bundle, to_frame(p))[0])
    band = risk_band(prob)
    color = COLORS[band]

    with right:
        st.subheader("Result")
        verdict = "Likely heart disease" if prob >= bundle["threshold"] else "Unlikely heart disease"
        st.markdown(f"""
<div class="result" style="--c:{color}">
  <div class="pct">{prob:.0%}</div>
  <div class="band">{band} risk · {verdict}</div>
  {ecg_svg(color, prob)}
  <div class="note">Probability from the model. Low &lt; 30% ≤ Moderate &lt; 60% ≤ High.</div>
</div>""", unsafe_allow_html=True)

        warns = [NICE_NAMES[f] for f, (lo, hi) in bundle["train_ranges"].items()
                 if f in p and not pd.isna(p[f]) and not (lo <= float(p[f]) <= hi)]
        if warns:
            st.warning("Outside the range seen in training, so treat with care: " + ", ".join(warns))

        st.markdown("#### What drives this prediction")
        ex = explain(bundle, p)
        if ex.empty:
            st.info("All values match a typical healthy patient.")
        else:
            ex = ex.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=ex["impact"] * 100, y=ex["feature"], orientation="h",
                marker_color=[COLORS["High"] if v > 0 else COLORS["Low"] for v in ex["impact"]],
                customdata=ex[["your_value", "healthy_reference"]].values,
                hovertemplate="%{y}<br>Your value: %{customdata[0]}<br>"
                              "Healthy reference: %{customdata[1]}<br>Effect: %{x:+.1f} pts<extra></extra>"))
            fig.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=30),
                              xaxis_title="Change in risk (percentage points)",
                              plot_bgcolor="white", font=dict(family="Public Sans, Arial, sans-serif", color=INK))
            fig.add_vline(x=0, line_color=MUTED, line_width=1)
            st.plotly_chart(fig, **FULL_WIDTH)
            st.markdown('<p class="small">Red bars push risk up, green bars pull it down. '
                        'Each bar = how much the risk would change if that one value were '
                        'replaced by a typical healthy value.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Predict for many patients at once")
    st.markdown("Upload a CSV with these columns: `" + "`, `".join(FEATURES) + "`. "
                "`sex` can be 1/0 or Male/Female. Extra columns (like a name or `target`) are kept.")
    template = pd.DataFrame([PRESETS["Healthy example"], PRESETS["High-risk example"]])[FEATURES]
    st.download_button("Download a template CSV", template.to_csv(index=False), "patients_template.csv")

    up = st.file_uploader("Patient CSV", type="csv")
    if up is not None:
        try:
            raw = pd.read_csv(up, encoding="utf-8-sig")
            X = prepare_input(raw)
            n_blank = int(X.isna().sum().sum())
            probs = predict_proba(bundle, X)
            out = raw.copy()
            out["risk_probability"] = probs.round(3)
            out["risk_band"] = [risk_band(v) for v in probs]
            out["prediction"] = ["Likely disease" if v >= bundle["threshold"] else "Unlikely"
                                 for v in probs]
            m1, m2, m3 = st.columns(3)
            m1.metric("Patients", len(out))
            m2.metric("High risk", int((out["risk_band"] == "High").sum()))
            m3.metric("Low risk", int((out["risk_band"] == "Low").sum()))
            if n_blank:
                st.info(f"{n_blank} blank or unreadable values were filled with typical values from training data.")
            st.dataframe(out, **FULL_WIDTH, hide_index=True)
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:  # clear message instead of a crash
            st.error(f"Could not read this file: {e}")

# ---------------------------------------------------------------- performance
with tab_perf:
    m = get_metrics()
    if m is None:
        st.info("Run `python train.py` to generate metrics.")
    else:
        st.subheader(f"How good is the model? ({m['model_name']})")
        t = m["test"]
        k = st.columns(5)
        for col, (name, key) in zip(k, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                        ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]):
            col.metric(name, f"{t[key]:.2f}")
        st.markdown('<p class="small">Measured on a 20% test set the model never saw during training.</p>',
                    unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            cm = m["confusion_matrix"]
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Predicted healthy", "Predicted disease"], y=["Actually healthy", "Actually disease"],
                text=cm, texttemplate="%{text}", colorscale=[[0, "#EEF3F7"], [1, INK]], showscale=False))
            fig.update_layout(title="Confusion matrix", height=330, yaxis_autorange="reversed",
                              margin=dict(l=0, r=0, t=40, b=0), font=dict(family="Public Sans, Arial, sans-serif"))
            st.plotly_chart(fig, **FULL_WIDTH)
        with g2:
            r = m["roc_curve"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", line=dict(color=INK, width=3),
                                     name=f"Model (AUC {t['roc_auc']:.2f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=MUTED),
                                     name="Random guess"))
            fig.update_layout(title="ROC curve", height=330, xaxis_title="False positive rate",
                              yaxis_title="True positive rate", margin=dict(l=0, r=0, t=40, b=0),
                              legend=dict(x=0.45, y=0.1), font=dict(family="Public Sans, Arial, sans-serif"),
                              plot_bgcolor="white")
            st.plotly_chart(fig, **FULL_WIDTH)

        st.markdown("#### Models compared (5-fold cross-validation, repeated 5 times)")
        cv = pd.DataFrame(m["cv"])[["model", "roc_auc", "roc_auc_std", "accuracy", "recall", "precision", "f1"]]
        st.dataframe(cv.round(3), **FULL_WIDTH, hide_index=True)

        st.markdown("#### Which measurements matter most overall")
        imp = pd.DataFrame(m["importance"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=INK))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=30), plot_bgcolor="white",
                          xaxis_title="Drop in ROC-AUC when this feature is shuffled",
                          font=dict(family="Public Sans, Arial, sans-serif"))
        st.plotly_chart(fig, **FULL_WIDTH)

# ---------------------------------------------------------------- data
with tab_data:
    st.subheader("Why the old version said 'at risk' for everyone")
    st.markdown("""
The Kaggle copy of this dataset has its **`target` column flipped**: `target = 1` means the patient is
**healthy**, not sick. We proved this by matching all 303 rows with the original UCI Cleveland file,
where the label is spelled out (Yes/No). Every single row was reversed.

The old model learned "healthy looks like disease", so any normal-looking patient came out *at risk*,
and truly sick patients came out *safe*.

**Fixes in this version**
1. New label `has_disease = 1 - target`.
2. Hidden missing values (`ca = 4`, `thal = 0`) converted to missing and filled by the pipeline.
3. Duplicate row removed.
4. Categorical codes (chest pain, ECG, slope, thal) one-hot encoded instead of treated as numbers.
5. Three models compared with cross-validation; a sanity test blocks saving a model that behaves backwards.
""")
    df = clean(load_raw())
    s = df.groupby("exang")["has_disease"].mean().rename(index=CODES["exang"])
    st.markdown("**Quick sanity check after the fix** — disease rate by *chest pain during exercise*:")
    st.dataframe(s.map("{:.0%}".format).to_frame("disease rate"))
