"""
CodeAlpha - Task 4: Diabetes Prediction - Streamlit Demo
Interactive web app to predict diabetes risk from clinical data.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
# Random Forest importances from the notebook
TOP_FEATURES = [
    ("Glucose level", 0.268), ("Body mass index, BMI", 0.158),
    ("Family history of diabetes, DPF", 0.124), ("Age", 0.120), ("Insulin", 0.092),
]
# key, label, unit
COMPARE = [
    ("Glucose", "Glucose", "mg/dL"),
    ("BMI", "Body mass index", "kg/m²"),
    ("Age", "Age", "years"),
    ("BloodPressure", "Blood pressure", "mm Hg"),
    ("DiabetesPedigreeFunction", "Family history of diabetes", ""),
    ("Pregnancies", "Pregnancies", ""),
]
CHARTS = [
    ("model_comparison.png", "Model comparison", "Accuracy, precision, recall, F1 and ROC-AUC for each model."),
    ("roc_curves.png", "ROC curves", "How well each model separates diabetic from healthy patients."),
    ("confusion_matrices.png", "Confusion matrices", "Correct and wrong predictions on the 154 test patients."),
    ("feature_importance.png", "Feature importance", "Which measurements the Random Forest relies on most."),
]

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    @keyframes rise { from { opacity: 0; transform: translateY(12px); } }

    .hero {
        background: linear-gradient(135deg, #065f46 0%, #059669 50%, #10b981 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 40px rgba(6, 95, 70, 0.3);
        position: relative;
        overflow: hidden;
        animation: rise .6s ease-out both;
    }
    .hero::after {
        content: "";
        position: absolute;
        top: -60%;
        right: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 65%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.22);
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 0.12em;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.6rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
        position: relative;
        z-index: 1;
        color: white;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
        max-width: 720px;
        position: relative;
        z-index: 1;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
        animation: rise .5s ease-out both;
    }
    .result-card.high {
        border-left: 6px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
    }
    .result-card.mid {
        border-left: 6px solid #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
    }
    .result-card.low {
        border-left: 6px solid #10b981;
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    }
    .result-card.neutral {
        background: #f8fafc;
        border-style: dashed;
        text-align: center;
    }

    .gauge-wrapper {
        display: flex;
        justify-content: center;
        margin: 1rem 0 1.5rem 0;
        animation: rise .5s ease-out both;
    }
    .gauge {
        width: 170px;
        height: 170px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
    }
    .gauge-inner {
        width: 138px;
        height: 138px;
        background: white;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .gauge-inner small { font-size: .7rem; font-weight: 600; color: #64748b; }

    /* patient vs population strips */
    .strip { margin-bottom: 1.1rem; }
    .strip-head { display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: .5rem; color: #475569; }
    .strip-head b { color: #065f46; }
    .track { position: relative; height: 9px; border-radius: 6px;
        background: linear-gradient(90deg, #a7f3d0, #fde68a 55%, #fca5a5); }
    .track i { position: absolute; transform: translateX(-50%); }
    .track .tick { top: -4px; width: 4px; height: 17px; border-radius: 3px; border: 1px solid white; }
    .track .you { top: -6px; width: 21px; height: 21px; border-radius: 50%; background: #065f46;
        border: 3px solid white; box-shadow: 0 0 0 1px #065f46, 0 3px 8px rgba(6,95,70,.4);
        transition: left .5s ease; }
    .legend { display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: .78rem; color: #64748b; margin-top: .4rem; }
    .legend b { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .35rem; vertical-align: -1px; }

    /* metric tiles in the performance tab */
    .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; margin-bottom: .5rem; }
    .tile { border-radius: 16px; padding: 1.1rem 1.2rem; color: white; animation: rise .5s ease-out both;
        box-shadow: 0 8px 22px -10px rgba(6, 95, 70, .6); }
    .tile span { font-size: .78rem; font-weight: 600; opacity: .9; text-transform: uppercase; letter-spacing: .06em; }
    .tile div { font-size: 1.9rem; font-weight: 800; line-height: 1.2; }

    /* One solid colour for the whole sidebar height (a gradient left a visible seam) */
    [data-testid="stSidebar"] {
        background: #064e3b !important;
    }
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        background: transparent !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown {
        color: #d1fae5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #6ee7b7 !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #6ee7b7 !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #047857 !important;
    }

    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #059669, #047857) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(5, 150, 105, 0.5) !important;
    }

    .stSlider label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.88rem !important;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        padding: 2.5rem 0 1rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        line-height: 1.7;
    }
    .footer strong {
        color: #1e293b;
    }
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ============ LOAD ARTIFACTS ============
@st.cache_resource
def load_artifacts():
    return (joblib.load(BASE / "model.pkl"), joblib.load(BASE / "scaler.pkl"),
            joblib.load(BASE / "features.pkl"))


@st.cache_data
def load_results():
    path = BASE / "results.csv"
    return pd.read_csv(path).sort_values("ROC-AUC", ascending=False) if path.exists() else None


@st.cache_data
def load_reference():
    """Typical values of healthy and diabetic patients, for the comparison chart."""
    path = BASE / "data" / "diabetes.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None, names=COLUMNS)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
    return {
        "healthy": df[df.Outcome == 0].median().to_dict(),
        "diabetic": df[df.Outcome == 1].median().to_dict(),
        "p5": df.quantile(0.05).to_dict(),
        "p95": df.quantile(0.95).to_dict(),
    }


try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Run the notebook's save cell to generate "
        "`model.pkl`, `scaler.pkl`, and `features.pkl`."
    )
    st.stop()

results = load_results()
ref = load_reference()
best = results.iloc[0] if results is not None else None

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk AI")
    st.caption("Clinical decision support demo")
    st.divider()
    st.markdown("### 📊 Model Performance")
    if best is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", f"{best['Accuracy']:.1%}")
            st.metric("F1-Score", f"{best['F1-Score']:.1%}")
        with c2:
            st.metric("ROC-AUC", f"{best['ROC-AUC']:.3f}")
            st.metric("Precision", f"{best['Precision']:.1%}")
    st.divider()
    st.markdown("### 🧠 Algorithm")
    st.caption("XGBoost (gradient boosting, 200 trees)")
    st.divider()
    st.markdown("### 📁 Dataset")
    st.caption("""
**Pima Indians Diabetes**
768 patients · 8 features
UCI ML Repository
    """)
    st.divider()
    st.markdown("### 🔬 Top Features")
    st.caption("""
1. Glucose (26.8%)
2. Body mass index (15.8%)
3. Family history of diabetes (12.4%)
    """)
    st.divider()
    st.caption("⚠️ For educational purposes only. Not a substitute for professional medical advice.")

# ============ HERO ============
st.markdown("""
<div class="hero">
    <div class="hero-badge">MACHINE LEARNING · CODEALPHA INTERNSHIP</div>
    <h1>🩺 Diabetes Risk Predictor</h1>
    <p>An AI-powered clinical decision support tool trained on 768 patient records from the Pima Indians Diabetes dataset.</p>
</div>
""", unsafe_allow_html=True)


def compare_html(values: dict) -> str:
    """Strips placing the patient between typical healthy and diabetic patients."""
    rows = ""
    for key, label, unit in COMPARE:
        lo, hi = ref["p5"][key], ref["p95"][key]

        def pos(v):
            return float(np.clip((v - lo) / (hi - lo), 0, 1) * 100)

        shown = f"{values[key]:.2f}" if key == "DiabetesPedigreeFunction" else f"{values[key]:g}"
        rows += (
            f'<div class="strip"><div class="strip-head"><span>{label}</span><b>{shown} {unit}</b></div>'
            f'<div class="track">'
            f'<i class="tick" style="left:{pos(ref["healthy"][key]):.1f}%;background:#10b981"></i>'
            f'<i class="tick" style="left:{pos(ref["diabetic"][key]):.1f}%;background:#ef4444"></i>'
            f'<i class="you" style="left:{pos(values[key]):.1f}%"></i></div></div>'
        )
    legend = (
        '<div class="legend"><span><b style="background:#065f46"></b>This patient</span>'
        '<span><b style="background:#10b981"></b>Typical without diabetes</span>'
        '<span><b style="background:#ef4444"></b>Typical with diabetes</span></div>'
    )
    return f'<div class="result-card">{rows}{legend}</div>'


tab_predict, tab_perf, tab_about = st.tabs(["🩺 Prediction", "📊 Model performance", "ℹ️ About"])

# ============ TAB 1: PREDICTION ============
with tab_predict:
    left, right = st.columns([1.6, 1], gap="large")

    with left:
        with st.form("prediction_form"):
            st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age (years)", 21, 90, 35, 1)
            with c2:
                pregnancies = st.slider("Number of pregnancies", 0, 17, 2, 1)

            st.markdown('<div class="section-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                glucose = st.slider("Glucose (mg/dL)", 40, 250, 120, 5)
                blood_pressure = st.slider("Blood pressure (mm Hg)", 30, 130, 70, 2)
                skin_thickness = st.slider("Triceps skinfold (mm)", 7, 100, 25, 1)
            with c2:
                insulin = st.slider("Insulin (μU/mL)", 14, 850, 100, 2)
                bmi = st.slider("Body mass index, BMI (kg/m²)", 15.0, 70.0, 28.0, 0.5)
                dpf = st.slider("Family history of diabetes (DPF)", 0.05, 2.5, 0.5, 0.01)

            st.markdown("")
            submitted = st.form_submit_button(
                "🔍 Predict Diabetes Risk",
                type="primary",
                width="stretch",
                key="predict",
            )

    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    with right:
        st.markdown('<div class="section-title">📋 Risk Assessment</div>', unsafe_allow_html=True)

        if not submitted:
            st.markdown("""
            <div class="result-card neutral">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🩺</div>
                <div style="color: #64748b; font-weight: 500; line-height: 1.6;">
                    Fill in the patient data on the left and click
                    <strong style="color:#1e293b;">Predict</strong> to see the risk assessment.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            raw = pd.DataFrame([values])[feature_names]
            proba = float(model.predict_proba(scaler.transform(raw))[0, 1])

            # One set of thresholds drives the colour, the title and the advice.
            if proba < 0.35:
                level, color, icon, title, text = (
                    "low", "#10b981", "✅", "Low Diabetes Risk",
                    "The model indicates a low probability of diabetes. "
                    "<strong>Maintain a healthy lifestyle and routine checkups.</strong>")
            elif proba < 0.60:
                level, color, icon, title, text = (
                    "mid", "#f59e0b", "🟠", "Moderate Diabetes Risk",
                    "Mixed signals. <strong>Worth monitoring and discussing at the next checkup.</strong>")
            else:
                level, color, icon, title, text = (
                    "high", "#ef4444", "⚠️", "Elevated Diabetes Risk",
                    "The model indicates an elevated probability of diabetes. "
                    "<strong>Medical consultation and further tests are recommended.</strong>")

            st.markdown(f"""
            <div class="gauge-wrapper">
                <div class="gauge" style="background: conic-gradient({color} {proba*100}%, #e2e8f0 {proba*100}% 100%);">
                    <div class="gauge-inner" style="color: {color};">{proba:.0%}<small>chance of diabetes</small></div>
                </div>
            </div>
            <div class="result-card {level}">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">{title}</div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

            items = "".join(f"<li>{name} ({share:.1%})</li>" for name, share in TOP_FEATURES)
            st.markdown(f"""
            <div class="result-card" style="background: #f0fdf4;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #065f46; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem;">
                    🔬 Top Predictive Features
                </div>
                <ul style="margin: 0; padding-left: 1.2rem; color: #334155; font-size: 0.88rem; line-height: 1.9;">{items}</ul>
                <div style="font-size: .75rem; color: #64748b; margin-top: .6rem;">Importances from the Random Forest model.</div>
            </div>
            """, unsafe_allow_html=True)

            st.caption("⚠️ This is a machine learning estimate, not a medical diagnosis.")

    if submitted and ref is not None:
        st.markdown('<div class="section-title">📍 How this patient compares</div>', unsafe_allow_html=True)
        st.markdown(compare_html(values), unsafe_allow_html=True)

# ============ TAB 2: MODEL PERFORMANCE ============
with tab_perf:
    if best is None:
        st.info("Run the notebook to generate results.csv and see the model comparison.")
    else:
        st.markdown(f'<div class="section-title">🏆 Best model: {best["Model"]}</div>', unsafe_allow_html=True)
        tiles = [
            ("Accuracy", f"{best['Accuracy']:.1%}", "#047857", "#059669"),
            ("Precision", f"{best['Precision']:.1%}", "#0f766e", "#14b8a6"),
            ("Recall", f"{best['Recall']:.1%}", "#065f46", "#10b981"),
            ("ROC-AUC", f"{best['ROC-AUC']:.3f}", "#064e3b", "#047857"),
        ]
        html = "".join(
            f'<div class="tile" style="background:linear-gradient(135deg,{c1},{c2});animation-delay:{i * 80}ms">'
            f'<span>{name}</span><div>{val}</div></div>'
            for i, (name, val, c1, c2) in enumerate(tiles)
        )
        st.markdown(f'<div class="tiles">{html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">📋 All models</div>', unsafe_allow_html=True)
        metrics = [c for c in results.columns if c != "Model"]
        st.dataframe(
            results,
            hide_index=True,
            column_config={
                c: st.column_config.ProgressColumn(c, format="%.3f", min_value=0.0, max_value=1.0)
                for c in metrics
            },
            alt="Accuracy, precision, recall, F1-score and ROC-AUC of the four models",
        )

    charts = [(BASE / "images" / f, t, d) for f, t, d in CHARTS if (BASE / "images" / f).exists()]
    if charts:
        st.markdown('<div class="section-title">📈 Charts from the notebook</div>', unsafe_allow_html=True)
        cols = st.columns(2, gap="medium")
        for i, (path, title, desc) in enumerate(charts):
            with cols[i % 2].container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
                st.image(str(path), width="stretch", alt=title)

# ============ TAB 3: ABOUT ============
with tab_about:
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1.container(border=True, height="stretch"):
        st.markdown("**🧪 How it works**")
        st.markdown(
            "- Trained on the Pima Indians Diabetes dataset (768 patients, 8 features).\n"
            "- Logistic Regression, SVM, Random Forest and XGBoost were compared.\n"
            "- XGBoost has the best ROC-AUC and powers this app."
        )
    with c2.container(border=True, height="stretch"):
        st.markdown("**🧹 Data handling**")
        st.markdown(
            "- Zeros in glucose, blood pressure, skinfold, insulin and BMI mean *missing*.\n"
            "- They were replaced by the median.\n"
            "- Features are standardised with a scaler fitted on the training set."
        )
    with c3.container(border=True, height="stretch"):
        st.markdown("**⚠️ Limitations**")
        st.markdown(
            "- Small dataset from one population (Pima women aged 21+).\n"
            "- About 4 in 10 diabetic patients are missed at the default threshold.\n"
            "- The probability is an estimate, not a diagnosis."
        )

# ============ FOOTER ============
st.markdown("""
<div class="footer">
    <strong>CodeAlpha Machine Learning Internship</strong> · Built by <strong>Sawadogo Noé</strong><br>
    Pima Indians Diabetes dataset · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)
