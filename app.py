"""
CodeAlpha - Task 4: Diabetes Prediction - Streamlit Demo
Interactive web app to predict diabetes risk from clinical data.
"""
import streamlit as st
import pandas as pd
import joblib

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

    html, body, [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .hero {
        background: linear-gradient(135deg, #065f46 0%, #059669 50%, #10b981 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(6, 95, 70, 0.3);
        position: relative;
        overflow: hidden;
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
    }
    .result-card.high {
        border-left: 6px solid #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #ffffff 100%);
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
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #064e3b 0%, #065f46 100%);
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

    .stButton > button, .stFormSubmitButton > button {
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
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 28px rgba(5, 150, 105, 0.5) !important;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
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
</style>
""", unsafe_allow_html=True)

# ============ LOAD ARTIFACTS ============
@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl"), joblib.load("scaler.pkl"), joblib.load("features.pkl")

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ Model files not found. Run the notebook's save cell to generate "
        "`model.pkl`, `scaler.pkl`, and `features.pkl`."
    )
    st.stop()

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk AI")
    st.caption("Clinical decision support demo")
    st.divider()
    st.markdown("### 📊 Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", "76.0%")
        st.metric("F1-Score", "0.641")
    with c2:
        st.metric("ROC-AUC", "0.827")
        st.metric("Precision", "0.674")
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
2. BMI (15.8%)
3. Diabetes Pedigree (12.4%)
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

# ============ LAYOUT ============
left, right = st.columns([1.6, 1], gap="large")

with left:
    with st.form("prediction_form"):
        st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age (years)", 18, 90, 35, 1)
        with c2:
            pregnancies = st.slider("Number of pregnancies", 0, 17, 2, 1)

        st.markdown('<div class="section-title">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            glucose = st.slider("Glucose (mg/dL)", 40, 250, 120, 5)
            blood_pressure = st.slider("Blood pressure (mm Hg)", 30, 130, 70, 2)
            skin_thickness = st.slider("Skin thickness (mm)", 0, 100, 25, 1)
        with c2:
            insulin = st.slider("Insulin (μU/mL)", 0, 900, 100, 10)
            bmi = st.slider("BMI", 15.0, 70.0, 28.0, 0.5)
            dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)

        st.markdown("")
        submitted = st.form_submit_button(
            "🔍 Predict Diabetes Risk",
            type="primary",
            use_container_width=True
        )

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
        raw = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }])

        raw = raw.reindex(columns=feature_names, fill_value=0)
        raw_scaled = scaler.transform(raw)

        proba = float(model.predict_proba(raw_scaled)[0, 1])
        prediction = int(model.predict(raw_scaled)[0])

        if proba < 0.33:
            color = "#10b981"
        elif proba < 0.66:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        st.markdown(f"""
        <div class="gauge-wrapper">
            <div class="gauge" style="background: conic-gradient({color} {proba*100}%, #e2e8f0 {proba*100}% 100%);">
                <div class="gauge-inner" style="color: {color};">{proba:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
            <div class="result-card high">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⚠️</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #dc2626; margin-bottom: 0.5rem;">
                    Elevated Diabetes Risk
                </div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    The model indicates an elevated probability of diabetes.
                    <strong>Medical consultation and further tests are recommended.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card low">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">✅</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #059669; margin-bottom: 0.5rem;">
                    Low Diabetes Risk
                </div>
                <div style="color: #475569; font-size: 0.9rem; line-height: 1.6;">
                    The model indicates a low probability of diabetes.
                    <strong>Maintain a healthy lifestyle and routine checkups.</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="result-card" style="background: #f0fdf4;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #065f46; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.75rem;">
                🔬 Top Predictive Features
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; color: #334155; font-size: 0.88rem; line-height: 1.9;">
                <li>Glucose level (26.8%)</li>
                <li>Body Mass Index — BMI (15.8%)</li>
                <li>Diabetes Pedigree Function (12.4%)</li>
                <li>Age (12.0%)</li>
                <li>Insulin (9.2%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚠️ This is a machine learning estimate, not a medical diagnosis.")

# ============ FOOTER ============
st.markdown("""
<div class="footer">
    <strong>CodeAlpha Machine Learning Internship</strong> · Built by <strong>Sawadogo Noé</strong><br>
    Pima Indians Diabetes dataset · XGBoost · Streamlit
</div>
""", unsafe_allow_html=True)