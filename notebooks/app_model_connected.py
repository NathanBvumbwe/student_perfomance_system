"""
BSDS0322 — Intelligent Student Performance Assessment & Intervention System
Mzuzu University, ICT Department — Nathan Bvumbwe
Model-connected version: number inputs, SHAP explanations, both modes live.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISPAIS — Mzuzu University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model loader ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "app_model")

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        return {
            "rf_full":        joblib.load(f"{MODEL_DIR}/rf_full_model.pkl"),
            "rf_behav":       joblib.load(f"{MODEL_DIR}/rf_behav_model.pkl"),
            "scaler_full":    joblib.load(f"{MODEL_DIR}/scaler_full.pkl"),
            "scaler_behav":   joblib.load(f"{MODEL_DIR}/scaler_behav.pkl"),
            "all_features":   joblib.load(f"{MODEL_DIR}/all_feature_names.pkl"),
            "behav_features": joblib.load(f"{MODEL_DIR}/behav_feature_names.pkl"),
            "band_order":     joblib.load(f"{MODEL_DIR}/band_order.pkl"),
            "band_to_int":    joblib.load(f"{MODEL_DIR}/band_to_int.pkl"),
            "int_to_band":    joblib.load(f"{MODEL_DIR}/int_to_band.pkl"),
            "feat_maxes":     joblib.load(f"{MODEL_DIR}/feat_maxes.pkl")
                              if os.path.exists(f"{MODEL_DIR}/feat_maxes.pkl")
                              else {"study_hours_per_day": 12.0, "internet_usage_hours": 12.0},
            "loaded": True,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}

models = load_models()

# ── Prediction helpers ───────────────────────────────────────────────────────
def assign_mzuni_band(score: float) -> str:
    if   score <= 34: return "Repeat"
    elif score <= 44: return "Supplementary"
    elif score <= 64: return "Good"
    else:             return "Excellent"

def build_features(raw: dict, feat_maxes: dict) -> dict:
    d = raw.copy()
    d["study_x_attendance"] = d["study_hours_per_day"] * d["attendance_percentage"] / 100
    d["distraction_ratio"]  = min(
        d["internet_usage_hours"] / (d["study_hours_per_day"] + 0.1), 10
    )
    sn = d["study_hours_per_day"]   / feat_maxes["study_hours_per_day"]
    an = d["attendance_percentage"] / 100
    in_ = d["internet_usage_hours"] / feat_maxes["internet_usage_hours"]
    d["risk_score"] = (1 - sn)*0.4 + (1 - an)*0.4 + in_*0.2
    return d

def normalise_shap(sv, n):
    if sv.ndim == 3 and sv.shape[0] == n: return sv
    if sv.ndim == 3 and sv.shape[0] == 4: return sv.transpose(1, 2, 0)
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

RECOMMENDATIONS = {
    "Repeat": {
        "student": [
            "Seek academic counselling immediately — do not wait until exams.",
            "Increase study hours to at least 4 hours per day.",
            "Attend every class — missing even one session compounds the risk.",
            "Reduce recreational internet usage to under 1 hour on study days.",
            "Form a study group with Good or Excellent band peers.",
        ],
        "lecturer": [
            "Flag this student for urgent one-on-one academic review.",
            "Escalate to the academic advisor or HOD immediately.",
            "Provide additional practice questions on foundational concepts.",
            "Schedule a mid-semester progress check-in.",
        ],
    },
    "Supplementary": {
        "student": [
            "You are close to the pass boundary — focused effort now can shift your outcome.",
            "Target the specific CA components where marks were lost.",
            "Attend all remaining classes and tutorials without exception.",
            "Reduce internet usage on study days.",
            "Use office hours to get feedback on weak areas.",
        ],
        "lecturer": [
            "Provide supplementary practice materials and past questions.",
            "Schedule an office hours check-in for this student.",
            "Monitor attendance closely over the next two weeks.",
        ],
    },
    "Good": {
        "student": [
            "You are on track — maintain your current study routine.",
            "Push for consistency: aim to add 30 minutes of study per day.",
            "Attempt past exam papers under timed conditions.",
            "Set a stretch target of the Excellent band for the final exam.",
        ],
        "lecturer": [
            "Encourage with extension tasks or challenge questions.",
            "Suggest peer tutoring to reinforce and deepen knowledge.",
        ],
    },
    "Excellent": {
        "student": [
            "Outstanding — maintain your current habits.",
            "Consider mentoring peers in lower bands.",
            "Explore supplementary reading to prepare for advanced coursework.",
        ],
        "lecturer": [
            "Recognise this performance publicly where appropriate.",
            "Assign a peer mentorship role to extend impact across the cohort.",
        ],
    },
}

def predict_student(raw_data: dict, mode: str = "full") -> dict:
    m    = models
    data = build_features(raw_data, m["feat_maxes"])
    if mode == "full":
        band  = assign_mzuni_band(data["exam_score"])
        model = m["rf_full"]
        feats = m["all_features"]
    else:
        band  = None
        model = m["rf_behav"]
        feats = m["behav_features"]

    input_df   = pd.DataFrame([data])[feats]
    proba      = model.predict_proba(input_df)[0]
    pred_int   = model.predict(input_df)[0]
    pred_band  = m["int_to_band"][pred_int]
    final_band = band if mode == "full" else pred_band
    confidence = proba[m["band_to_int"][final_band]] * 100

    exp     = shap.TreeExplainer(model)
    sv      = normalise_shap(np.array(exp.shap_values(input_df)), 1)
    sv_cls  = sv[0, :, m["band_to_int"][final_band]]
    drivers = (
        pd.Series(np.abs(sv_cls), index=feats)
        .sort_values(ascending=False)
        .head(3)
    )
    return {
        "mode":             mode,
        "band":             final_band,
        "ml_prediction":    pred_band,
        "confidence":       round(confidence, 1),
        "probabilities":    {b: round(proba[i]*100, 1) for i, b in enumerate(m["band_order"])},
        "top_drivers":      drivers.index.tolist(),
        "driver_values":    drivers.round(4).to_dict(),
        "student_actions":  RECOMMENDATIONS[final_band]["student"],
        "lecturer_actions": RECOMMENDATIONS[final_band]["lecturer"],
    }

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{
    --g900:#0a2e1a;--g800:#0f4526;--g700:#166534;--g600:#1a7a3e;
    --g500:#22a050;--g400:#3dbd6b;--g300:#6dd496;--g200:#a8ecc2;
    --g100:#d4f7e4;--g050:#edfaf3;--white:#ffffff;--off:#f7fdf9;
    --text:#0a2e1a;--muted:#4a7a5e;--border:#c8edda;
    --shadow:0 2px 20px rgba(10,46,26,0.08);
    --shadow-lg:0 8px 40px rgba(10,46,26,0.14);
}
*{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{
    background:var(--white)!important;
    font-family:'DM Sans',sans-serif;color:var(--text);}
[data-testid="stSidebar"]{display:none;}
[data-testid="stHeader"]{background:transparent!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]>div{padding:0!important;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3{font-family:'DM Serif Display',serif;}

.nav-bar{background:var(--g800);padding:0 48px;height:68px;display:flex;
    align-items:center;justify-content:space-between;position:sticky;
    top:0;z-index:100;border-bottom:3px solid var(--g600);}
.nav-logo{display:flex;align-items:center;gap:14px;}
.nav-logo-badge{background:var(--g500);color:var(--white);width:40px;height:40px;
    border-radius:10px;display:flex;align-items:center;justify-content:center;
    font-family:'DM Serif Display',serif;font-size:18px;font-weight:bold;flex-shrink:0;}
.nav-title{color:var(--white);font-family:'DM Serif Display',serif;font-size:17px;line-height:1.2;}
.nav-subtitle{color:var(--g200);font-size:11px;font-weight:300;letter-spacing:0.5px;text-transform:uppercase;}
.nav-pill{background:rgba(255,255,255,0.12);color:var(--g100);
    border:1px solid rgba(255,255,255,0.2);border-radius:20px;
    padding:6px 14px;font-size:12px;font-weight:500;}

.hero{background:linear-gradient(135deg,var(--g800) 0%,var(--g700) 50%,var(--g600) 100%);
    padding:72px 48px 64px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:340px;height:340px;
    border-radius:50%;background:rgba(255,255,255,0.04);pointer-events:none;}
.hero::after{content:'';position:absolute;bottom:-40px;left:30%;width:200px;height:200px;
    border-radius:50%;background:rgba(61,189,107,0.12);pointer-events:none;}
.hero-eyebrow{color:var(--g300);font-size:12px;font-weight:600;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:16px;}
.hero-title{color:var(--white);font-family:'DM Serif Display',serif;
    font-size:clamp(32px,4vw,52px);line-height:1.15;margin-bottom:20px;max-width:640px;}
.hero-title em{color:var(--g300);font-style:italic;}
.hero-body{color:var(--g100);font-size:16px;line-height:1.7;max-width:520px;font-weight:300;}
.hero-stats{display:flex;gap:32px;margin-top:40px;flex-wrap:wrap;}
.hero-stat{border-left:2px solid var(--g400);padding-left:16px;}
.hero-stat-num{color:var(--white);font-family:'DM Serif Display',serif;font-size:28px;line-height:1;}
.hero-stat-lbl{color:var(--g200);font-size:12px;font-weight:400;margin-top:4px;}

.mode-section{background:var(--off);padding:56px 48px;}
.section-label{font-size:11px;font-weight:600;letter-spacing:2.5px;
    text-transform:uppercase;color:var(--g500);margin-bottom:10px;}
.section-title{font-family:'DM Serif Display',serif;font-size:28px;
    color:var(--g900);margin-bottom:8px;}
.section-body{color:var(--muted);font-size:15px;margin-bottom:36px;font-weight:300;}

.mode-card{background:var(--white);border:2px solid var(--border);border-radius:16px;
    padding:32px;transition:all 0.2s ease;position:relative;overflow:hidden;}
.mode-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
    background:var(--g400);transform:scaleX(0);transition:transform 0.25s;transform-origin:left;}
.mode-card:hover::before,.mode-card.active::before{transform:scaleX(1);}
.mode-card:hover{border-color:var(--g400);box-shadow:var(--shadow-lg);transform:translateY(-2px);}
.mode-card.active{border-color:var(--g500);box-shadow:var(--shadow-lg);background:var(--g050);}
.mode-icon{width:52px;height:52px;border-radius:14px;display:flex;
    align-items:center;justify-content:center;font-size:24px;margin-bottom:20px;}
.mode-icon.student{background:var(--g100);}
.mode-icon.lecturer{background:#fef3c7;}
.mode-card-title{font-family:'DM Serif Display',serif;font-size:20px;
    color:var(--g900);margin-bottom:8px;}
.mode-card-body{color:var(--muted);font-size:14px;line-height:1.6;margin-bottom:20px;}
.mode-card-tags{display:flex;gap:8px;flex-wrap:wrap;}
.tag{background:var(--g100);color:var(--g700);border-radius:20px;
    padding:4px 12px;font-size:12px;font-weight:500;}
.tag.amber{background:#fef3c7;color:#92400e;}

.panel{background:var(--white);padding:56px 48px;border-top:1px solid var(--border);
    animation:slideIn 0.35s ease;}
@keyframes slideIn{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.panel-header{display:flex;align-items:flex-start;gap:20px;margin-bottom:40px;
    padding-bottom:32px;border-bottom:1px solid var(--border);}
.panel-icon-wrap{width:56px;height:56px;border-radius:16px;display:flex;
    align-items:center;justify-content:center;font-size:26px;flex-shrink:0;}
.panel-icon-wrap.student{background:var(--g100);}
.panel-icon-wrap.lecturer{background:#fef3c7;}
.panel-title{font-family:'DM Serif Display',serif;font-size:26px;color:var(--g900);margin-bottom:6px;}
.panel-desc{color:var(--muted);font-size:14px;line-height:1.6;}

.field-group{background:var(--off);border:1px solid var(--border);
    border-radius:12px;padding:24px;margin-bottom:20px;}
.field-group-title{font-size:12px;font-weight:600;letter-spacing:1.5px;
    text-transform:uppercase;color:var(--g600);margin-bottom:16px;
    padding-bottom:12px;border-bottom:1px solid var(--border);}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select{
    border:1.5px solid var(--border)!important;border-radius:8px!important;
    font-family:'DM Sans',sans-serif!important;background:var(--white)!important;
    color:var(--text)!important;font-size:14px!important;}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus{
    border-color:var(--g500)!important;
    box-shadow:0 0 0 3px rgba(34,160,80,0.15)!important;}
label[data-testid="stWidgetLabel"] p{
    font-size:13px!important;font-weight:500!important;
    color:var(--g800)!important;font-family:'DM Sans',sans-serif!important;}
.stButton>button{
    background:var(--g600)!important;color:var(--white)!important;
    border:none!important;border-radius:10px!important;padding:14px 36px!important;
    font-family:'DM Sans',sans-serif!important;font-size:15px!important;
    font-weight:600!important;letter-spacing:0.3px!important;
    cursor:pointer!important;transition:all 0.2s!important;width:100%!important;}
.stButton>button:hover{
    background:var(--g700)!important;transform:translateY(-1px)!important;
    box-shadow:0 4px 16px rgba(10,46,26,0.25)!important;}

.info-box{background:var(--g050);border:1px solid var(--g200);
    border-left:4px solid var(--g500);border-radius:0 10px 10px 0;
    padding:16px 20px;margin-bottom:20px;}
.info-box-title{font-weight:600;font-size:13px;color:var(--g700);margin-bottom:4px;}
.info-box-body{font-size:13px;color:var(--muted);line-height:1.5;}

.result-band{border-radius:14px;padding:28px 32px;margin-bottom:20px;}
.result-band.repeat   {background:#fef2f2;border:1.5px solid #fca5a5;}
.result-band.supp     {background:#fffbeb;border:1.5px solid #fcd34d;}
.result-band.good     {background:var(--g050);border:1.5px solid var(--g200);}
.result-band.excellent{background:#f0fdf4;border:1.5px solid var(--g300);}
.result-band-label{font-size:11px;font-weight:700;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:6px;}
.result-band.repeat    .result-band-label{color:#b91c1c;}
.result-band.supp      .result-band-label{color:#b45309;}
.result-band.good      .result-band-label{color:var(--g600);}
.result-band.excellent .result-band-label{color:var(--g700);}
.result-band-name{font-family:'DM Serif Display',serif;font-size:30px;}
.result-band.repeat    .result-band-name{color:#7f1d1d;}
.result-band.supp      .result-band-name{color:#78350f;}
.result-band.good      .result-band-name{color:var(--g800);}
.result-band.excellent .result-band-name{color:var(--g900);}

.action-list{list-style:none;padding:0;margin:0;}
.action-item{display:flex;gap:12px;align-items:flex-start;padding:12px 0;
    border-bottom:1px solid var(--border);font-size:14px;color:var(--text);line-height:1.5;}
.action-item:last-child{border-bottom:none;}
.action-dot{width:8px;height:8px;border-radius:50%;background:var(--g400);
    flex-shrink:0;margin-top:6px;}

.driver-bar-wrap{margin-bottom:10px;}
.driver-label{font-size:12px;font-weight:500;color:var(--g700);margin-bottom:4px;}
.driver-bar-bg{background:var(--g100);border-radius:4px;height:8px;overflow:hidden;}
.driver-bar-fill{background:var(--g500);height:8px;border-radius:4px;}

.upload-zone{border:2px dashed var(--g300);border-radius:14px;padding:48px 32px;
    text-align:center;background:var(--g050);margin-bottom:24px;}
.upload-icon{font-size:40px;margin-bottom:12px;}
.upload-title{font-family:'DM Serif Display',serif;font-size:18px;
    color:var(--g800);margin-bottom:6px;}
.upload-body{color:var(--muted);font-size:13px;margin-bottom:16px;}
.upload-badge{display:inline-block;background:var(--g700);color:var(--white);
    border-radius:6px;padding:4px 12px;font-size:12px;font-weight:600;letter-spacing:1px;}

.footer{background:var(--g900);padding:40px 48px;display:flex;
    justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;}
.footer-left{color:var(--g200);font-size:13px;line-height:1.8;}
.footer-right{display:flex;gap:8px;flex-wrap:wrap;}
.footer-badge{background:rgba(255,255,255,0.08);color:var(--g200);
    border:1px solid rgba(255,255,255,0.1);border-radius:6px;
    padding:6px 12px;font-size:11px;font-weight:500;letter-spacing:0.5px;}
.divider{height:1px;background:var(--border);margin:32px 0;}

@media(max-width:700px){
    .nav-bar,.hero,.mode-section,.panel{padding-left:20px;padding-right:20px;}
    .footer{padding:32px 20px;flex-direction:column;}
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("mode", None), ("student_submitted", False),
             ("student_result", None), ("lecturer_submitted", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">
    <div class="nav-logo-badge">M</div>
    <div>
      <div class="nav-title">ISPAIS</div>
      <div class="nav-subtitle">Mzuzu University · ICT Department</div>
    </div>
  </div>
  <div class="nav-pill">Academic Year 2024 / 25</div>
</div>
""", unsafe_allow_html=True)

if not models.get("loaded"):
    st.error(
        f"⚠️ Model files not found in `app_model/`. "
        f"Make sure `app_model/` sits next to `app.py` and contains the .pkl files. "
        f"Error: {models.get('error','unknown')}"
    )

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Proactive Academic Intervention</div>
  <div class="hero-title">
    Know where you stand —<br><em>before</em> the final exam.
  </div>
  <div class="hero-body">
    An intelligent early-warning system that maps your continuous assessment
    performance to Mzuzu University's grading bands and delivers personalised,
    AI-explained action plans — for students and lecturers alike.
  </div>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-num">4</div>
      <div class="hero-stat-lbl">Performance bands</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">2</div>
      <div class="hero-stat-lbl">Prediction modes</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">XAI</div>
      <div class="hero-stat-lbl">SHAP explanations</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">ICT</div>
      <div class="hero-stat-lbl">Dept. pilot</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# MODE SELECTOR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="mode-section">
  <div class="section-label">Get started</div>
  <div class="section-title">Who are you?</div>
  <div class="section-body">
    Choose your role to access the right tools. Each mode is tailored to your
    data access and goals.
  </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col_pad = st.columns([1, 1, 0.6])

with col1:
    s_act = "active" if st.session_state.mode == "student" else ""
    st.markdown(f"""
    <div class="mode-card {s_act}" style="margin:0 0 0 48px;">
      <div class="mode-icon student">🎓</div>
      <div class="mode-card-title">I am a Student</div>
      <div class="mode-card-body">Enter your study habits and CA results to
        discover your performance band and get a personalised improvement
        plan powered by SHAP explanations.</div>
      <div class="mode-card-tags">
        <span class="tag">Early warning</span>
        <span class="tag">Post-CA</span>
        <span class="tag">Action plan</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)
    if st.button("Continue as Student →", key="btn_student"):
        st.session_state.mode = "student"
        st.session_state.student_submitted = False
        st.session_state.student_result = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    l_act = "active" if st.session_state.mode == "lecturer" else ""
    st.markdown(f"""
    <div class="mode-card {l_act}">
      <div class="mode-icon lecturer">📋</div>
      <div class="mode-card-title">I am a Lecturer</div>
      <div class="mode-card-body">Upload your class grades file to get a full
        cohort analysis, at-risk student flags, SHAP driver charts, and
        targeted delivery recommendations.</div>
      <div class="mode-card-tags">
        <span class="tag">Cohort overview</span>
        <span class="tag amber">Bulk upload</span>
        <span class="tag">Delivery tips</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Continue as Lecturer →", key="btn_lecturer"):
        st.session_state.mode = "lecturer"
        st.session_state.lecturer_submitted = False
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# STUDENT PANEL
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "student":

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-header">
      <div class="panel-icon-wrap student">🎓</div>
      <div>
        <div class="panel-title">Student Assessment</div>
        <div class="panel-desc">
          Type your values in the fields below. <strong>Exam score is optional</strong> —
          leave it unticked for an early warning based on study habits alone,
          or tick it for a confirmed post-CA band assignment.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">🔒 Your data is private</div>
      <div class="info-box-body">
        This system does not store or share any data you enter.
        All results are advisory tools only — not deterministic judgements.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Personal info
    st.markdown('<div class="field-group"><div class="field-group-title">Personal Information</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        student_id = st.text_input("Student ID", placeholder="e.g. BSDS0322", key="s_id")
    with p2:
        age = st.number_input("Age", min_value=16, max_value=60, value=20, step=1, key="s_age")
    with p3:
        gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Prefer not to say"], key="s_gender")
    st.markdown('</div>', unsafe_allow_html=True)

    # Study habits — NUMBER INPUTS
    st.markdown('<div class="field-group"><div class="field-group-title">Study Habits *</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0, max_value=24.0, value=3.0, step=0.5,
            help="Average hours spent studying each day this semester.",
            key="s_study"
        )
        sleep_hours = st.number_input(
            "Sleep hours per night",
            min_value=0.0, max_value=24.0, value=7.0, step=0.5,
            help="Average hours of sleep per night.",
            key="s_sleep"
        )
    with h2:
        attendance = st.number_input(
            "Attendance percentage (%)",
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            help="Percentage of classes attended this semester.",
            key="s_attend"
        )
        internet = st.number_input(
            "Daily non-study internet usage (hours)",
            min_value=0.0, max_value=24.0, value=2.0, step=0.5,
            help="Hours per day on social media, streaming, gaming, etc.",
            key="s_internet"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Exam score (optional)
    st.markdown('<div class="field-group"><div class="field-group-title">Assessment Score — Optional (for confirmed prediction)</div>', unsafe_allow_html=True)
    has_score = st.checkbox("I have my CA / exam score", value=False, key="s_has_score")
    if has_score:
        exam_score = st.number_input(
            "CA / Exam score (%)",
            min_value=0.0, max_value=100.0, value=55.0, step=0.5,
            key="s_score"
        )
        if   exam_score <= 34: preview = "🔴 Repeat zone (0–34%)"
        elif exam_score <= 44: preview = "🟡 Supplementary zone (35–44%)"
        elif exam_score <= 64: preview = "🟢 Good zone (45–64%)"
        else:                  preview = "✅ Excellent zone (65–100%)"
        st.markdown(
            f"<p style='font-size:13px;color:var(--muted);margin-top:2px;'>"
            f"Band preview: <strong>{preview}</strong></p>",
            unsafe_allow_html=True
        )
    else:
        exam_score = None
    st.markdown('</div>', unsafe_allow_html=True)

    # Submit button
    if st.button("Analyse My Performance →", key="btn_student_submit"):
        if gender == "Select...":
            st.warning("Please select your gender to continue.")
        elif not models.get("loaded"):
            st.error("Model files not found. Check that app_model/ is next to app.py.")
        else:
            with st.spinner("Running two-layer analysis…"):
                raw = {
                    "study_hours_per_day":   study_hours,
                    "attendance_percentage": attendance,
                    "sleep_hours":           sleep_hours,
                    "internet_usage_hours":  internet,
                }
                if has_score:
                    raw["exam_score"] = exam_score
                try:
                    result = predict_student(raw, mode="full" if has_score else "early")
                    st.session_state.student_result    = result
                    st.session_state.student_submitted = True
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # Results
    if st.session_state.student_submitted and st.session_state.student_result:
        res      = st.session_state.student_result
        band     = res["band"]
        mode_lbl = "Post-CA Confirmed" if res["mode"] == "full" else "Early Warning — Behavioural"
        band_cls = {"Repeat":"repeat","Supplementary":"supp",
                    "Good":"good","Excellent":"excellent"}.get(band, "good")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-label" style="margin-top:8px;">Results — {mode_lbl}</div>
        <div class="section-title" style="margin-bottom:24px;">Your Performance Report</div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1.4])

        with r1:
            # Band card
            st.markdown(f"""
            <div class="result-band {band_cls}">
              <div class="result-band-label">Predicted Band</div>
              <div class="result-band-name">{band}</div>
              <p style="font-size:13px;margin-top:10px;color:var(--muted);">
                Model confidence: <strong>{res['confidence']}%</strong>
              </p>
              <p style="font-size:12px;color:var(--muted);margin-top:4px;">
                {"Score mapped to Mzuni threshold. Layer 1 rule engine applied."
                 if res['mode']=='full'
                 else "Based on study habits only. Add exam score for confirmed result."}
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="field-group" style="margin-top:0;"><div class="field-group-title">Band Probabilities</div>', unsafe_allow_html=True)
            prob_colors = {"Repeat":"#E24B4A","Supplementary":"#EF9F27",
                           "Good":"#1D9E75","Excellent":"#534AB7"}
            for b, p in res["probabilities"].items():
                clr = prob_colors[b]
                st.markdown(f"""
                <div class="driver-bar-wrap">
                  <div style="display:flex;justify-content:space-between;">
                    <span class="driver-label">{b}</span>
                    <span style="font-size:12px;font-weight:600;color:{clr};">{p}%</span>
                  </div>
                  <div class="driver-bar-bg">
                    <div class="driver-bar-fill" style="width:{p}%;background:{clr};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # SHAP drivers
            st.markdown('<div class="field-group"><div class="field-group-title">Top SHAP Drivers</div>', unsafe_allow_html=True)
            max_v = max(res["driver_values"].values()) if res["driver_values"] else 1
            for feat, val in res["driver_values"].items():
                pct = (val / max_v * 100) if max_v > 0 else 0
                st.markdown(f"""
                <div class="driver-bar-wrap">
                  <div style="display:flex;justify-content:space-between;">
                    <span class="driver-label">{feat.replace('_',' ').title()}</span>
                    <span style="font-size:11px;color:var(--muted);">{val:.4f}</span>
                  </div>
                  <div class="driver-bar-bg">
                    <div class="driver-bar-fill" style="width:{pct:.0f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<p style="font-size:11px;color:var(--muted);margin-top:6px;">These features influenced your prediction most strongly.</p></div>', unsafe_allow_html=True)

            # Input summary
            st.markdown('<div class="field-group"><div class="field-group-title">Input Summary</div>', unsafe_allow_html=True)
            for k, v in {
                "Study hrs/day": f"{study_hours} hrs",
                "Attendance":    f"{attendance}%",
                "Sleep":         f"{sleep_hours} hrs",
                "Internet":      f"{internet} hrs",
                **({"Exam score": f"{exam_score}%"} if has_score else {}),
            }.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:13px;padding:6px 0;border-bottom:1px solid var(--border);'>"
                    f"<span style='color:var(--muted);'>{k}</span><strong>{v}</strong></div>",
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            # Student recommendations
            st.markdown('<div style="margin-bottom:16px;"><div class="field-group-title" style="font-size:11px;letter-spacing:1.5px;">WHAT YOU SHOULD DO</div></div>', unsafe_allow_html=True)
            st.markdown(
                '<ul class="action-list">' +
                "".join(f'<li class="action-item"><div class="action-dot"></div>{a}</li>'
                        for a in res["student_actions"]) +
                '</ul>',
                unsafe_allow_html=True
            )

            # Lecturer recommendations
            st.markdown('<div style="margin-top:28px;margin-bottom:16px;"><div class="field-group-title" style="font-size:11px;letter-spacing:1.5px;">WHAT YOUR LECTURER SHOULD DO</div></div>', unsafe_allow_html=True)
            st.markdown(
                '<ul class="action-list">' +
                "".join(f'<li class="action-item"><div class="action-dot" style="background:#EF9F27;"></div>{a}</li>'
                        for a in res["lecturer_actions"]) +
                '</ul>',
                unsafe_allow_html=True
            )

            if res["mode"] == "early":
                st.markdown("""
                <div class="info-box" style="margin-top:24px;">
                  <div class="info-box-title">💡 Early warning mode active</div>
                  <div class="info-box-body">
                    No exam score was provided so the early warning model was used.
                    Tick "I have my CA / exam score" and re-submit for a confirmed band.
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# LECTURER PANEL
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "lecturer":

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-header">
      <div class="panel-icon-wrap lecturer">📋</div>
      <div>
        <div class="panel-title">Lecturer Dashboard</div>
        <div class="panel-desc">
          Upload your class grades CSV to receive a full cohort band distribution,
          at-risk student flags with confidence scores, a SHAP driver chart, and
          personalised course delivery recommendations.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">📄 Required CSV columns</div>
      <div class="info-box-body">
        <strong>student_id, study_hours_per_day, attendance_percentage,
        sleep_hours, internet_usage_hours, exam_score</strong>.
        Optional: age, gender. One row per student.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Lecturer details
    st.markdown('<div class="field-group"><div class="field-group-title">Lecturer Details</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        lec_name   = st.text_input("Full name", placeholder="e.g. Dr. E. Ngalande", key="l_name")
    with lc2:
        lec_course = st.text_input("Course / Unit", placeholder="e.g. Data Structures", key="l_course")
    with lc3:
        lec_year   = st.selectbox("Year of study", ["Select...", "Year 1", "Year 2", "Year 3", "Year 4"], key="l_year")
    st.markdown('</div>', unsafe_allow_html=True)

    # Upload zone
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon">📊</div>
      <div class="upload-title">Upload Class Grades</div>
      <div class="upload-body">Drag and drop your CSV here, or click to browse</div>
      <span class="upload-badge">CSV ONLY</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose CSV", type=["csv"],
        key="l_file", label_visibility="collapsed"
    )

    # Options
    st.markdown('<div class="field-group" style="margin-top:20px;"><div class="field-group-title">Analysis Options</div>', unsafe_allow_html=True)
    ao1, ao2, ao3 = st.columns(3)
    with ao1:
        show_atrisk   = st.checkbox("Flag at-risk students", value=True, key="l_atrisk")
    with ao2:
        show_shap     = st.checkbox("SHAP driver chart", value=True, key="l_shap")
    with ao3:
        show_delivery = st.checkbox("Delivery recommendations", value=True, key="l_delivery")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Analyse Class →", key="btn_lecturer_submit"):
        if lec_year == "Select...":
            st.warning("Please select the year of study.")
        elif uploaded_file is None:
            st.warning("Please upload a CSV file before continuing.")
        elif not models.get("loaded"):
            st.error("Model not loaded. Check app_model/ folder.")
        else:
            st.session_state.lecturer_submitted = True

    # Results
    if st.session_state.lecturer_submitted and uploaded_file is not None:

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label" style="margin-top:8px;">Cohort Analysis</div>
        <div class="section-title" style="margin-bottom:24px;">Class Performance Report</div>
        """, unsafe_allow_html=True)

        with st.spinner("Running model on all students…"):
            try:
                df_up = pd.read_csv(uploaded_file)
            except Exception:
                st.error("Could not read CSV. Check file format.")
                st.stop()

            required_cols = ["study_hours_per_day", "attendance_percentage",
                             "sleep_hours", "internet_usage_hours", "exam_score"]
            missing = [c for c in required_cols if c not in df_up.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            records = []
            for _, row in df_up.iterrows():
                raw = {c: float(row[c]) for c in required_cols}
                res = predict_student(raw, mode="full")
                records.append({
                    "student_id":  str(row.get("student_id", "—")),
                    "exam_score":  raw["exam_score"],
                    "band":        res["band"],
                    "confidence":  res["confidence"],
                    "top_driver":  res["top_drivers"][0].replace("_", " ").title()
                                   if res["top_drivers"] else "—",
                })
            result_df = pd.DataFrame(records)

        n = len(result_df)
        st.success(f"✓ Analysed {n:,} students successfully.")

        # Band distribution cards
        st.markdown('<div class="section-label" style="margin-top:24px;margin-bottom:12px;">Band Distribution</div>', unsafe_allow_html=True)
        band_dist   = result_df["band"].value_counts().reindex(
            ["Repeat","Supplementary","Good","Excellent"], fill_value=0
        )
        band_styles = {
            "Repeat":        ("🔴","#fef2f2","#b91c1c"),
            "Supplementary": ("🟡","#fffbeb","#b45309"),
            "Good":          ("🟢","#edfaf3","#166534"),
            "Excellent":     ("✅","#f0fdf4","#14532d"),
        }
        bc1, bc2, bc3, bc4 = st.columns(4)
        for col, (band, count) in zip([bc1,bc2,bc3,bc4], band_dist.items()):
            icon, bg, clr = band_styles[band]
            pct = count / n * 100
            col.markdown(f"""
            <div style="background:{bg};border-radius:12px;padding:20px;text-align:center;">
              <div style="font-size:24px;margin-bottom:6px;">{icon}</div>
              <div style="font-family:'DM Serif Display',serif;font-size:28px;color:{clr};">{count}</div>
              <div style="font-size:12px;font-weight:600;color:{clr};margin:4px 0;">{band}</div>
              <div style="font-size:11px;color:#6b7280;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP driver chart
        if show_shap:
            st.markdown('<div class="section-label" style="margin-top:32px;margin-bottom:8px;">Most Common Risk Driver per Student (SHAP)</div>', unsafe_allow_html=True)
            driver_counts = result_df["top_driver"].value_counts()
            fig, ax = plt.subplots(figsize=(8, max(3, len(driver_counts)*0.55)))
            colors = ["#22a050" if i == 0 else "#6dd496" for i in range(len(driver_counts))]
            driver_counts.plot.barh(ax=ax, color=colors, edgecolor="white")
            ax.set_xlabel("Number of students", fontsize=11)
            ax.set_title("Top SHAP driver — cohort view", fontsize=12)
            ax.invert_yaxis()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # At-risk table
        if show_atrisk:
            st.markdown('<div class="section-label" style="margin-top:32px;margin-bottom:8px;">At-Risk Students — Repeat Band</div>', unsafe_allow_html=True)
            at_risk = result_df[result_df["band"] == "Repeat"].reset_index(drop=True)
            if len(at_risk) == 0:
                st.success("No students are currently in the Repeat band.")
            else:
                st.dataframe(at_risk, use_container_width=True, hide_index=True)

        # Full table
        with st.expander("View full class results"):
            st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Delivery recommendations
        if show_delivery:
            repeat_pct = band_dist.get("Repeat", 0) / n * 100
            risk_pct   = (band_dist.get("Repeat",0) + band_dist.get("Supplementary",0)) / n * 100

            st.markdown('<div class="section-label" style="margin-top:32px;margin-bottom:12px;">Course Delivery Recommendations</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)

            with d1:
                items = []
                if repeat_pct > 15:
                    items.append(f"⚠️ {repeat_pct:.1f}% of students are in the Repeat band — above the 15% warning threshold. Immediate cohort-level action is recommended.")
                items += [
                    "Schedule one-on-one check-ins with all Repeat band students before Week 8.",
                    "Provide foundational recap materials for high-failure topics.",
                    "Consider peer mentoring: pair Excellent ↔ Repeat students.",
                    "Escalate persistent at-risk cases to the academic advisor or HOD.",
                ]
                st.markdown(
                    '<div class="field-group"><div class="field-group-title">For At-Risk Students</div><ul class="action-list">' +
                    "".join(f'<li class="action-item"><div class="action-dot"></div>{a}</li>' for a in items) +
                    '</ul></div>',
                    unsafe_allow_html=True
                )

            with d2:
                items2 = [
                    f"{'⚠️ ' if risk_pct>35 else ''}{risk_pct:.1f}% of students are in Repeat or Supplementary — {'review pacing urgently.' if risk_pct>35 else 'monitor closely.'}",
                    "Increase formative assessments mid-semester to catch gaps early.",
                    "Share anonymised cohort data with the HOD to benchmark across courses.",
                    "The SHAP driver chart above shows which behaviours most predict poor outcomes in your class — use it to focus your guidance.",
                ]
                st.markdown(
                    '<div class="field-group"><div class="field-group-title">For Course Delivery</div><ul class="action-list">' +
                    "".join(f'<li class="action-item"><div class="action-dot" style="background:#EF9F27;"></div>{a}</li>' for a in items2) +
                    '</ul></div>',
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <div class="footer-left">
    <strong style="color:white;">ISPAIS — Intelligent Student Performance Assessment &amp; Intervention System</strong><br>
    Mzuzu University · ICT Department · BSDS0322 Nathan Bvumbwe<br>
    Supervisor: Emmanuel Ngalande · Final Year Project 2025
  </div>
  <div class="footer-right">
    <span class="footer-badge">RANDOM FOREST</span>
    <span class="footer-badge">TWO-LAYER PIPELINE</span>
    <span class="footer-badge">SHAP XAI</span>
    <span class="footer-badge">STREAMLIT</span>
  </div>
</div>
""", unsafe_allow_html=True)
