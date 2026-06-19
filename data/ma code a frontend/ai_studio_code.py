import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISPAIS — Mzuzu University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model loader ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "app_model"

@st.cache_resource(show_spinner="Loading AI Models...")
def load_models():
    try:
        # Check if directory exists
        if not MODEL_DIR.exists():
            return {"loaded": False, "error": f"Folder 'app_model' not found at {MODEL_DIR}"}
        
        return {
            "rf_full":        joblib.load(MODEL_DIR / "rf_full_model.pkl"),
            "rf_behav":       joblib.load(MODEL_DIR / "rf_behav_model.pkl"),
            "scaler_full":    joblib.load(MODEL_DIR / "scaler_full.pkl"),
            "scaler_behav":   joblib.load(MODEL_DIR / "scaler_behav.pkl"),
            "all_features":   joblib.load(MODEL_DIR / "all_feature_names.pkl"),
            "behav_features": joblib.load(MODEL_DIR / "behav_feature_names.pkl"),
            "band_order":     joblib.load(MODEL_DIR / "band_order.pkl"),
            "band_to_int":    joblib.load(MODEL_DIR / "band_to_int.pkl"),
            "int_to_band":    joblib.load(MODEL_DIR / "int_to_band.pkl"),
            "feat_maxes":     joblib.load(MODEL_DIR / "feat_maxes.pkl") if (MODEL_DIR / "feat_maxes.pkl").exists() else {"study_hours_per_day": 12.0, "internet_usage_hours": 12.0},
            "loaded": True,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}

# Load models once
models = load_models()

# ── SHAP Explainer Cache (CRITICAL FIX) ──────────────────────────────────────
@st.cache_resource
def get_explainer(mode="full"):
    if not models.get("loaded"): return None
    model = models["rf_full"] if mode == "full" else models["rf_behav"]
    # Using TreeExplainer for Random Forest
    return shap.TreeExplainer(model)

# ── Prediction helpers ───────────────────────────────────────────────────────
def assign_mzuni_band(score: float) -> str:
    if   score <= 34: return "Repeat"
    elif score <= 44: return "Supplementary"
    elif score <= 64: return "Good"
    else:             return "Excellent"

def build_features(raw: dict, feat_maxes: dict) -> dict:
    d = raw.copy()
    d["study_x_attendance"] = d["study_hours_per_day"] * d["attendance_percentage"] / 100
    d["distraction_ratio"]  = min(d["internet_usage_hours"] / (d["study_hours_per_day"] + 0.1), 10)
    sn = d["study_hours_per_day"]   / feat_maxes["study_hours_per_day"]
    an = d["attendance_percentage"] / 100
    in_ = d["internet_usage_hours"] / feat_maxes["internet_usage_hours"]
    d["risk_score"] = (1 - sn)*0.4 + (1 - an)*0.4 + in_*0.2
    return d

RECOMMENDATIONS = {
    "Repeat": {
        "student": ["Seek academic counselling immediately.", "Increase study hours to 4+ daily.", "Attend every class.", "Reduce internet usage to <1hr.", "Form a study group."],
        "lecturer": ["Flag for urgent review.", "Escalate to HOD.", "Provide foundational practice.", "Schedule mid-semester check-in."]
    },
    "Supplementary": {
        "student": ["Target specific CA gaps.", "Attend all remaining tutorials.", "Reduce distraction internet hours.", "Use office hours for feedback."],
        "lecturer": ["Provide past papers.", "Schedule office hours check-in.", "Monitor attendance closely."]
    },
    "Good": {
        "student": ["Maintain consistency.", "Add 30 mins study daily.", "Attempt past exam papers.", "Aim for Excellent band."],
        "lecturer": ["Encourage with extension tasks.", "Suggest peer tutoring roles."]
    },
    "Excellent": {
        "student": ["Maintain current habits.", "Mentor peers in lower bands.", "Explore advanced coursework."],
        "lecturer": ["Recognise performance.", "Assign peer mentorship role."]
    }
}

def predict_student(raw_data: dict, mode: str = "full") -> dict:
    m = models
    data = build_features(raw_data, m["feat_maxes"])
    
    if mode == "full":
        band = assign_mzuni_band(data.get("exam_score", 0))
        model = m["rf_full"]
        feats = m["all_features"]
    else:
        band = None
        model = m["rf_behav"]
        feats = m["behav_features"]

    input_df = pd.DataFrame([data])[feats]
    proba = model.predict_proba(input_df)[0]
    pred_int = model.predict(input_df)[0]
    pred_band = m["int_to_band"][pred_int]
    
    final_band = band if mode == "full" else pred_band
    class_idx = m["band_to_int"][final_band]
    confidence = proba[class_idx] * 100

    # Optimized SHAP call
    explainer = get_explainer(mode)
    # check_additivity=False prevents hangs in some scikit-learn versions
    sv = explainer.shap_values(input_df, check_additivity=False)
    
    # Handle list vs array output in SHAP
    if isinstance(sv, list):
        sv_cls = sv[class_idx][0]
    else:
        sv_cls = sv[0, :, class_idx]

    drivers = pd.Series(np.abs(sv_cls), index=feats).sort_values(ascending=False).head(3)
    
    return {
        "mode": mode,
        "band": final_band,
        "ml_prediction": pred_band,
        "confidence": round(confidence, 1),
        "probabilities": {b: round(proba[i]*100, 1) for i, b in enumerate(m["band_order"])},
        "top_drivers": drivers.index.tolist(),
        "driver_values": drivers.to_dict(),
        "student_actions": RECOMMENDATIONS[final_band]["student"],
        "lecturer_actions": RECOMMENDATIONS[final_band]["lecturer"],
    }

# ── UI/CSS Logic (Simplified for brevity, matches your design) ───────────────
st.markdown("""<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #1a7a3e; color: white; }
    .report-card { background: #f9f9f9; padding: 20px; border-radius: 15px; border-left: 5px solid #1a7a3e; }
</style>""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "mode" not in st.session_state: st.session_state.mode = None
if "submitted" not in st.session_state: st.session_state.submitted = False

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🎓 Mzuzu University — ISPAIS")
st.caption("Intelligent Student Performance Assessment & Intervention System")

if not models["loaded"]:
    st.error(f"⚠️ **Error Loading Models:** {models['error']}")
    st.info(f"Make sure your .pkl files are in: `{MODEL_DIR}`")
    st.stop()

# ── Role Selection ───────────────────────────────────────────────────────────
if st.session_state.mode is None:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("I am a Student"):
            st.session_state.mode = "student"
            st.rerun()
    with c2:
        if st.button("I am a Lecturer"):
            st.session_state.mode = "lecturer"
            st.rerun()
else:
    if st.button("← Switch Role"):
        st.session_state.mode = None
        st.session_state.submitted = False
        st.rerun()

# ── Main Panels ──────────────────────────────────────────────────────────────
if st.session_state.mode == "student":
    st.subheader("Student Performance Analysis")
    
    with st.expander("Enter Study Habits", expanded=True):
        c1, c2 = st.columns(2)
        study = c1.number_input("Study Hours/Day", 0.0, 15.0, 3.0)
        attend = c2.number_input("Attendance %", 0.0, 100.0, 80.0)
        sleep = c1.number_input("Sleep Hours", 0.0, 12.0, 7.0)
        internet = c2.number_input("Non-study Internet Hours", 0.0, 15.0, 2.0)
        
        has_score = st.checkbox("I have my CA/Exam Score")
        score = None
        if has_score:
            score = st.number_input("Exam Score %", 0.0, 100.0, 50.0)

    if st.button("Run Analysis"):
        raw = {
            "study_hours_per_day": study,
            "attendance_percentage": attend,
            "sleep_hours": sleep,
            "internet_usage_hours": internet
        }
        if has_score: raw["exam_score"] = score
        
        try:
            with st.spinner("AI is analyzing your patterns..."):
                res = predict_student(raw, mode="full" if has_score else "early")
                st.session_state.result = res
                st.session_state.submitted = True
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    if st.session_state.submitted:
        res = st.session_state.result
        st.success(f"### Predicted Band: {res['band']}")
        st.progress(res['confidence']/100)
        st.write(f"Confidence: {res['confidence']}%")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Personal Action Plan:**")
            for a in res['student_actions']: st.write(f"- {a}")
        with col_b:
            st.write("**Top Drivers (SHAP):**")
            for d, v in res['driver_values'].items():
                st.write(f"- {d.replace('_',' ')}: {round(v, 3)}")

elif st.session_state.mode == "lecturer":
    st.subheader("Lecturer Cohort Dashboard")
    uploaded_file = st.file_uploader("Upload Class CSV", type="csv")
    
    if uploaded_file and st.button("Process Class"):
        df = pd.read_csv(uploaded_file)
        results = []
        
        # Batch processing
        with st.spinner(f"Analyzing {len(df)} students..."):
            for _, row in df.iterrows():
                # Note: We skip individual SHAP in lecturer batch mode to prevent browser hang
                # Only calculate band and probability
                raw = {
                    "study_hours_per_day": row["study_hours_per_day"],
                    "attendance_percentage": row["attendance_percentage"],
                    "sleep_hours": row["sleep_hours"],
                    "internet_usage_hours": row["internet_usage_hours"],
                    "exam_score": row["exam_score"]
                }
                data = build_features(raw, models["feat_maxes"])
                input_df = pd.DataFrame([data])[models["all_features"]]
                pred = models["rf_full"].predict(input_df)[0]
                results.append(models["int_to_band"][pred])
        
        df["Predicted_Band"] = results
        st.write("### Cohort Overview")
        st.bar_chart(df["Predicted_Band"].value_counts())
        st.dataframe(df)

st.divider()
st.caption("Mzuzu University ICT Dept | Nathan Bvumbwe | 2025")