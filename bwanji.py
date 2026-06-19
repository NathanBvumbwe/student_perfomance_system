import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISPAIS — Mzuzu University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "app_model_colab (1)"   # ← retrained with updated Mzuni bands

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    if not MODEL_DIR.exists():
        return {"loaded": False,
                "error": f"Folder 'app_model_v2' not found at {MODEL_DIR}"}
    try:
        feat_maxes_path = MODEL_DIR / "feat_maxes.pkl"
        feat_maxes = (
            joblib.load(feat_maxes_path)
            if feat_maxes_path.exists()
            else {"study_hours_per_day": 12.0, "internet_usage_hours": 12.0}
        )
        return {
            "loaded":        True,
            "rf_full":       joblib.load(MODEL_DIR / "rf_full_model.pkl"),
            "rf_behav":      joblib.load(MODEL_DIR / "rf_behav_model.pkl"),
            "scaler_full":   joblib.load(MODEL_DIR / "scaler_full.pkl"),
            "scaler_behav":  joblib.load(MODEL_DIR / "scaler_behav.pkl"),
            "all_features":  joblib.load(MODEL_DIR / "all_feature_names.pkl"),
            "behav_features":joblib.load(MODEL_DIR / "behav_feature_names.pkl"),
            "band_order":    joblib.load(MODEL_DIR / "band_order.pkl"),
            "band_to_int":   joblib.load(MODEL_DIR / "band_to_int.pkl"),
            "int_to_band":   joblib.load(MODEL_DIR / "int_to_band.pkl"),
            "feat_maxes":    feat_maxes,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}

models = load_models()

# ── SHAP explainer cache ──────────────────────────────────────────────────────
@st.cache_resource
def get_explainer(mode="full"):
    if not models.get("loaded"):
        return None
    model = models["rf_full"] if mode == "full" else models["rf_behav"]
    return shap.TreeExplainer(model)

# ── Updated Layer 1 rule engine ───────────────────────────────────────────────
# Updated Mzuni bands (per SRS revision):
#   Fail          :  0 – 34%
#   Supplementable: 35 – 49%   (upper raised from 44%)
#   Pass          : 50 – 64%   (lower raised from 45%)
#   Excellent     : 65 – 100%
def assign_mzuni_band(score: float) -> str:
    if   score <= 34: return "Fail"
    elif score <= 49: return "Supplementable"
    elif score <= 64: return "Pass"
    else:             return "Excellent"

def band_preview(score: float) -> str:
    if   score <= 34: return "🔴  Fail zone  (0 – 34%)"
    elif score <= 49: return "🟡  Supplementable zone  (35 – 49%)"
    elif score <= 64: return "🟢  Pass zone  (50 – 64%)"
    else:             return "✅  Excellent zone  (65 – 100%)"

# ── Feature engineering — must match retraining notebook exactly ──────────────
def build_features(raw: dict, feat_maxes: dict) -> dict:
    d = raw.copy()
    d["study_x_attendance"] = (
        d["study_hours_per_day"] * d["attendance_percentage"] / 100
    )
    d["distraction_ratio"] = min(
        d["internet_usage_hours"] / (d["study_hours_per_day"] + 0.1), 10
    )
    sn  = d["study_hours_per_day"]   / feat_maxes["study_hours_per_day"]
    an  = d["attendance_percentage"] / 100
    in_ = d["internet_usage_hours"]  / feat_maxes["internet_usage_hours"]
    d["risk_score"] = (1 - sn)*0.4 + (1 - an)*0.4 + in_*0.2
    return d

# ── Updated recommendations ───────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Fail": {
        "student": [
            "Seek academic counselling immediately — do not wait until exams.",
            "Increase study hours to at least 4 hours per day.",
            "Attend every class — missing even one session compounds the risk.",
            "Reduce recreational internet usage to under 1 hour on study days.",
            "Form a study group with Pass or Excellent band peers.",
        ],
        "lecturer": [
            "Flag this student for urgent one-on-one academic review.",
            "Escalate to the academic advisor or HOD immediately.",
            "Provide additional practice questions on foundational concepts.",
            "Schedule a mid-semester progress check-in.",
        ],
    },
    "Supplementable": {
        "student": [
            "You are in the supplementable range — focused effort now can shift your outcome.",
            "Target the specific CA components where marks were lost.",
            "Attend all remaining classes and tutorials without exception.",
            "Reduce internet usage on study days.",
            "Use office hours to get targeted feedback on weak areas.",
        ],
        "lecturer": [
            "Provide supplementary practice materials and past exam questions.",
            "Schedule an office hours check-in for this student.",
            "Monitor attendance closely over the next two weeks.",
        ],
    },
    "Pass": {
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

# ── Band styling ──────────────────────────────────────────────────────────────
BAND_STYLE = {
    "Fail":          {"bg":"#fef2f2","border":"#fca5a5","text":"#7f1d1d","icon":"🔴"},
    "Supplementable":{"bg":"#fffbeb","border":"#fcd34d","text":"#78350f","icon":"🟡"},
    "Pass":          {"bg":"#edfaf3","border":"#a8ecc2","text":"#0f4526","icon":"🟢"},
    "Excellent":     {"bg":"#f0fdf4","border":"#6dd496","text":"#0a2e1a","icon":"✅"},
}
PROB_COLORS = {
    "Fail":          "#E24B4A",
    "Supplementable":"#EF9F27",
    "Pass":          "#1D9E75",
    "Excellent":     "#534AB7",
}

# ── Core prediction function ──────────────────────────────────────────────────
def predict_student(raw_data: dict, mode: str = "full") -> dict:
    m    = models
    data = build_features(raw_data, m["feat_maxes"])

    if mode == "full":
        band  = assign_mzuni_band(data.get("exam_score", 0))
        model = m["rf_full"]
        feats = m["all_features"]
    else:
        band  = None
        model = m["rf_behav"]
        feats = m["behav_features"]

    input_df  = pd.DataFrame([data])[feats]
    proba     = model.predict_proba(input_df)[0]
    pred_int  = int(model.predict(input_df)[0])
    pred_band = m["int_to_band"][pred_int]

    final_band = band if mode == "full" else pred_band
    class_idx  = m["band_to_int"][final_band]
    confidence = float(proba[class_idx]) * 100

    # SHAP
    explainer = get_explainer(mode)
    sv = explainer.shap_values(input_df, check_additivity=False)

    if isinstance(sv, list):
        sv_cls = sv[class_idx][0]
    else:
        try:
            sv_cls = sv[0, :, class_idx]
        except Exception:
            sv_cls = np.zeros(len(feats))

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
        "probabilities":    {
            b: round(float(proba[i]) * 100, 1)
            for i, b in enumerate(m["band_order"])
        },
        "top_drivers":      drivers.index.tolist(),
        "driver_values":    {k: round(float(v), 4) for k, v in drivers.items()},
        "student_actions":  RECOMMENDATIONS[final_band]["student"],
        "lecturer_actions": RECOMMENDATIONS[final_band]["lecturer"],
    }

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --g800:#0F4526; --g700:#166534; --g600:#1a7a3e;
    --g500:#22A050; --g400:#3DBD6B; --g200:#A8ECC2;
    --g100:#D4F7E4; --g050:#EDF9F3;
    --white:#ffffff; --off:#F7FDF9;
    --border:#C8EDDA; --muted:#4a7a5e; --text:#0A2E1A;
}
*{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{
    background:var(--white)!important;
    font-family:'DM Sans',sans-serif; color:var(--text);}
[data-testid="stSidebar"]{display:none;}
[data-testid="stHeader"]{background:transparent!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stMain"]>div{padding:0!important;}
#MainMenu,footer,header{visibility:hidden;}

/* Nav */
.nav{background:var(--g800);padding:0 48px;height:68px;display:flex;
    align-items:center;justify-content:space-between;
    position:sticky;top:0;z-index:100;border-bottom:3px solid var(--g600);}
.nav-brand{display:flex;align-items:center;gap:14px;}
.nav-badge{background:var(--g500);color:white;width:40px;height:40px;
    border-radius:10px;display:flex;align-items:center;justify-content:center;
    font-weight:bold;font-size:18px;flex-shrink:0;}
.nav-title{color:white;font-size:17px;font-weight:600;line-height:1.2;}
.nav-sub{color:var(--g200);font-size:11px;text-transform:uppercase;letter-spacing:0.5px;}
.nav-pill{background:rgba(255,255,255,0.12);color:var(--g100);
    border:1px solid rgba(255,255,255,0.2);border-radius:20px;
    padding:6px 14px;font-size:12px;}

/* Hero */
.hero{background:linear-gradient(135deg,var(--g800) 0%,var(--g700) 50%,var(--g600) 100%);
    padding:64px 48px 56px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;
    border-radius:50%;background:rgba(255,255,255,0.04);pointer-events:none;}
.hero-eyebrow{color:var(--g200);font-size:12px;font-weight:600;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:14px;}
.hero-title{color:white;font-family:'DM Serif Display',serif;
    font-size:clamp(28px,4vw,48px);line-height:1.15;margin-bottom:16px;max-width:600px;}
.hero-title em{color:var(--g200);font-style:italic;}
.hero-body{color:var(--g100);font-size:15px;line-height:1.7;
    max-width:500px;font-weight:300;margin-bottom:36px;}
.hero-stats{display:flex;gap:28px;flex-wrap:wrap;}
.hero-stat{border-left:2px solid var(--g400);padding-left:14px;}
.hero-stat-num{color:white;font-family:'DM Serif Display',serif;font-size:26px;line-height:1;}
.hero-stat-lbl{color:var(--g200);font-size:12px;margin-top:3px;}

/* Mode section */
.mode-section{background:var(--off);padding:52px 48px;}
.section-label{font-size:11px;font-weight:600;letter-spacing:2.5px;
    text-transform:uppercase;color:var(--g500);margin-bottom:8px;}
.section-title{font-family:'DM Serif Display',serif;font-size:26px;
    color:var(--g800);margin-bottom:6px;}
.section-body{color:var(--muted);font-size:14px;margin-bottom:32px;}

/* Mode cards */
.mode-card{background:white;border:2px solid var(--border);border-radius:16px;
    padding:28px;transition:all 0.2s;position:relative;overflow:hidden;}
.mode-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
    background:var(--g400);transform:scaleX(0);transition:transform 0.25s;
    transform-origin:left;}
.mode-card:hover::before,.mode-card.active::before{transform:scaleX(1);}
.mode-card:hover{border-color:var(--g400);
    box-shadow:0 8px 32px rgba(10,46,26,0.12);transform:translateY(-2px);}
.mode-card.active{border-color:var(--g500);background:var(--g050);}
.mode-icon{width:48px;height:48px;border-radius:13px;display:flex;
    align-items:center;justify-content:center;font-size:22px;margin-bottom:18px;}
.mode-icon.s{background:var(--g100);}
.mode-icon.l{background:#fef3c7;}
.mode-card-title{font-family:'DM Serif Display',serif;font-size:19px;
    color:var(--g800);margin-bottom:7px;}
.mode-card-body{color:var(--muted);font-size:13.5px;line-height:1.6;margin-bottom:18px;}
.mode-card-tags{display:flex;gap:7px;flex-wrap:wrap;}
.tag{background:var(--g100);color:var(--g700);border-radius:18px;
    padding:3px 11px;font-size:11.5px;font-weight:500;}
.tag.a{background:#fef3c7;color:#92400e;}

/* Panel */
.panel{background:white;padding:52px 48px;border-top:1px solid var(--border);
    animation:slideIn 0.3s ease;}
@keyframes slideIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
.panel-hdr{display:flex;align-items:flex-start;gap:18px;margin-bottom:36px;
    padding-bottom:28px;border-bottom:1px solid var(--border);}
.panel-icon-wrap{width:52px;height:52px;border-radius:14px;display:flex;
    align-items:center;justify-content:center;font-size:24px;flex-shrink:0;}
.panel-icon-wrap.s{background:var(--g100);}
.panel-icon-wrap.l{background:#fef3c7;}
.panel-title{font-family:'DM Serif Display',serif;font-size:24px;
    color:var(--g800);margin-bottom:5px;}
.panel-desc{color:var(--muted);font-size:13.5px;line-height:1.6;}

/* Field groups */
.fg{background:var(--off);border:1px solid var(--border);
    border-radius:12px;padding:22px;margin-bottom:18px;}
.fg-title{font-size:11px;font-weight:700;letter-spacing:1.5px;
    text-transform:uppercase;color:var(--g600);margin-bottom:14px;
    padding-bottom:10px;border-bottom:1px solid var(--border);}

/* Widget overrides */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input{
    border:1.5px solid var(--border)!important;border-radius:8px!important;
    font-family:'DM Sans',sans-serif!important;font-size:14px!important;}
[data-testid="stNumberInput"] input:focus{
    border-color:var(--g500)!important;
    box-shadow:0 0 0 3px rgba(34,160,80,0.15)!important;}
label[data-testid="stWidgetLabel"] p{
    font-size:13px!important;font-weight:500!important;color:var(--g800)!important;}
.stButton>button{
    background:var(--g600)!important;color:white!important;
    border:none!important;border-radius:10px!important;
    padding:14px 32px!important;font-size:14.5px!important;
    font-weight:600!important;width:100%!important;transition:all 0.2s!important;}
.stButton>button:hover{
    background:var(--g700)!important;transform:translateY(-1px)!important;
    box-shadow:0 4px 14px rgba(10,46,26,0.25)!important;}

/* Info box */
.info-box{background:var(--g050);border:1px solid var(--g200);
    border-left:4px solid var(--g500);border-radius:0 10px 10px 0;
    padding:14px 18px;margin-bottom:18px;}
.info-box-title{font-weight:600;font-size:13px;color:var(--g700);margin-bottom:3px;}
.info-box-body{font-size:13px;color:var(--muted);line-height:1.5;}

/* Band card */
.band-card{border-radius:14px;padding:24px 28px;margin-bottom:18px;}
.band-label-text{font-size:11px;font-weight:700;letter-spacing:2px;
    text-transform:uppercase;margin-bottom:6px;}
.band-name{font-size:32px;font-weight:700;margin-bottom:6px;}
.band-meta{font-size:12px;color:#555;}

/* Bars */
.bar-row{margin-bottom:9px;}
.bar-label{display:flex;justify-content:space-between;
    font-size:12px;font-weight:500;color:var(--g700);margin-bottom:3px;}
.bar-bg{background:var(--g100);border-radius:4px;height:8px;overflow:hidden;}
.bar-fill{height:8px;border-radius:4px;}

/* Action list */
.al{list-style:none;padding:0;margin:0;}
.ai{display:flex;gap:10px;align-items:flex-start;padding:10px 0;
    border-bottom:1px solid var(--border);font-size:13.5px;line-height:1.5;}
.ai:last-child{border-bottom:none;}
.ad{width:7px;height:7px;border-radius:50%;background:var(--g400);
    flex-shrink:0;margin-top:6px;}

/* Upload zone */
.upload-zone{border:2px dashed var(--g200);border-radius:12px;
    padding:40px;text-align:center;background:var(--g050);margin-bottom:20px;}
.upload-icon{font-size:36px;margin-bottom:10px;}
.upload-title{font-family:'DM Serif Display',serif;font-size:18px;
    color:var(--g800);margin-bottom:5px;}
.upload-body{color:var(--muted);font-size:13px;margin-bottom:14px;}
.upload-badge{display:inline-block;background:var(--g700);color:white;
    border-radius:6px;padding:3px 10px;font-size:11px;font-weight:600;}

/* Band dist card */
.bdc{border-radius:12px;padding:18px;text-align:center;}
.bdc-num{font-size:30px;font-weight:700;line-height:1;}
.bdc-name{font-size:12px;font-weight:600;margin:5px 0;}
.bdc-pct{font-size:12px;color:#666;}

/* Divider */
.div{height:1px;background:var(--border);margin:28px 0;}

/* Footer */
.footer{background:var(--g800);padding:36px 48px;display:flex;
    justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;}
.footer-left{color:var(--g200);font-size:12.5px;line-height:1.8;}
.footer-left strong{color:white;}
.footer-right{display:flex;gap:7px;flex-wrap:wrap;}
.fbadge{background:rgba(255,255,255,0.08);color:var(--g200);
    border:1px solid rgba(255,255,255,0.12);border-radius:5px;
    padding:5px 10px;font-size:10.5px;font-weight:500;letter-spacing:0.5px;}

@media(max-width:700px){
    .nav,.hero,.mode-section,.panel,.footer{padding-left:20px;padding-right:20px;}
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("mode",        None),
    ("submitted",   False),
    ("result",      None),
    ("lec_results", None),
    ("lec_df",      None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nav">
  <div class="nav-brand">
    <div class="nav-badge">M</div>
    <div>
      <div class="nav-title">ISPAIS</div>
      <div class="nav-sub">Mzuzu University · ICT Department</div>
    </div>
  </div>
  <div class="nav-pill">Academic Year 2025 / 26</div>
</div>
""", unsafe_allow_html=True)

# Model error banner
if not models.get("loaded"):
    st.error(
        f"⚠️ Models not loaded — {models.get('error','unknown')}. "
        f"Make sure `app_model_v2/` sits next to `app.py` and contains "
        f"all .pkl files from the retraining notebook."
    )
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Proactive Academic Intervention</div>
  <div class="hero-title">Know where you stand —<br><em>before</em> the final exam.</div>
  <div class="hero-body">An intelligent early-warning system that maps your
    continuous assessment performance to Mzuzu University's grading bands and
    delivers personalised, AI-explained action plans.</div>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-num">4</div>
      <div class="hero-stat-lbl">Mzuni bands</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">2</div>
      <div class="hero-stat-lbl">Prediction modes</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-num">XAI</div>
      <div class="hero-stat-lbl">SHAP driven</div>
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
  <div class="section-body">Choose your role to access the right tools.</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col_pad = st.columns([1, 1, 0.6])

with col1:
    s_act = "active" if st.session_state.mode == "student" else ""
    st.markdown(f"""
    <div class="mode-card {s_act}" style="margin:0 0 0 48px;">
      <div class="mode-icon s">🎓</div>
      <div class="mode-card-title">I am a Student</div>
      <div class="mode-card-body">Enter your study habits and CA results to
        discover your performance band and get a personalised improvement plan
        powered by SHAP explanations.</div>
      <div class="mode-card-tags">
        <span class="tag">Early warning</span>
        <span class="tag">Post-CA</span>
        <span class="tag">Action plan</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='padding:0 48px;'>", unsafe_allow_html=True)
    if st.button("Continue as Student →", key="btn_student"):
        st.session_state.mode      = "student"
        st.session_state.submitted = False
        st.session_state.result    = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    l_act = "active" if st.session_state.mode == "lecturer" else ""
    st.markdown(f"""
    <div class="mode-card {l_act}">
      <div class="mode-icon l">📋</div>
      <div class="mode-card-title">I am a Lecturer</div>
      <div class="mode-card-body">Upload your class grades CSV to receive a
        full cohort analysis, Fail band flags, SHAP driver chart, and
        targeted delivery recommendations.</div>
      <div class="mode-card-tags">
        <span class="tag">Cohort view</span>
        <span class="tag a">Bulk upload</span>
        <span class="tag">Delivery tips</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Continue as Lecturer →", key="btn_lecturer"):
        st.session_state.mode        = "lecturer"
        st.session_state.lec_results = None
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# STUDENT PANEL
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "student":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-hdr">
      <div class="panel-icon-wrap s">🎓</div>
      <div>
        <div class="panel-title">Student Assessment</div>
        <div class="panel-desc">Type your values below. Exam score is optional —
          leave it unticked for an early warning from study habits alone, or
          tick it for a confirmed post-CA band assignment.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">🔒 Your data is private</div>
      <div class="info-box-body">Nothing you enter is stored or shared.
        Results are advisory tools only — not deterministic judgements.</div>
    </div>
    """, unsafe_allow_html=True)

    # Study habits
    st.markdown('<div class="fg"><div class="fg-title">Study Habits *</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        study    = st.number_input("Study hours per day",
                                   min_value=0.0, max_value=24.0,
                                   value=3.0, step=0.5, key="s_study",
                                   help="Average daily study hours this semester.")
        sleep    = st.number_input("Sleep hours per night",
                                   min_value=0.0, max_value=24.0,
                                   value=7.0, step=0.5, key="s_sleep",
                                   help="Average nightly sleep.")
    with h2:
        attend   = st.number_input("Attendance percentage (%)",
                                   min_value=0.0, max_value=100.0,
                                   value=75.0, step=1.0, key="s_attend",
                                   help="Percentage of classes attended.")
        internet = st.number_input("Daily non-study internet (hours)",
                                   min_value=0.0, max_value=24.0,
                                   value=2.0, step=0.5, key="s_internet",
                                   help="Social media, streaming, gaming, etc.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Exam score
    st.markdown('<div class="fg"><div class="fg-title">CA / Exam Score — Optional</div>', unsafe_allow_html=True)
    has_score = st.checkbox("I have my CA / exam score", value=False, key="s_has_score")
    score = None
    if has_score:
        score = st.number_input("Exam score (%)",
                                min_value=0.0, max_value=100.0,
                                value=55.0, step=0.5, key="s_score")
        st.markdown(
            f"<p style='font-size:13px;color:var(--muted);margin-top:4px;'>"
            f"Band preview: <strong>{band_preview(score)}</strong></p>",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Submit
    if st.button("Analyse My Performance →", key="btn_analyse"):
        raw = {
            "study_hours_per_day":   study,
            "attendance_percentage": attend,
            "sleep_hours":           sleep,
            "internet_usage_hours":  internet,
        }
        if has_score and score is not None:
            raw["exam_score"] = score

        with st.spinner("Running two-layer analysis…"):
            try:
                res = predict_student(raw, mode="full" if (has_score and score is not None) else "early")
                st.session_state.result    = res
                st.session_state.submitted = True
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # Results
    if st.session_state.submitted and st.session_state.result:
        res  = st.session_state.result
        band = res["band"]
        sty  = BAND_STYLE.get(band, BAND_STYLE["Pass"])

        st.markdown('<div class="div"></div>', unsafe_allow_html=True)
        mode_lbl = "Post-CA Confirmed" if res["mode"] == "full" else "Early Warning — Behavioural"
        st.markdown(
            f"<p style='font-size:11px;font-weight:700;letter-spacing:2px;"
            f"text-transform:uppercase;color:var(--g500);margin-bottom:16px;'>"
            f"Results — {mode_lbl}</p>",
            unsafe_allow_html=True
        )

        r1, r2 = st.columns([1, 1.4])

        with r1:
            # Band card
            st.markdown(f"""
            <div class="band-card" style="background:{sty['bg']};border:1.5px solid {sty['border']};">
              <div class="band-label-text" style="color:{sty['text']};">{sty['icon']}  Predicted Band</div>
              <div class="band-name" style="color:{sty['text']};">{band}</div>
              <div class="band-meta">
                Model confidence: <strong>{res['confidence']}%</strong><br>
                {"Layer 1 rule engine applied — policy-correct."
                 if res['mode']=='full'
                 else "Study habits only. Add exam score for confirmed result."}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="fg" style="margin-top:0;"><div class="fg-title">Band Probabilities</div>', unsafe_allow_html=True)
            for b, p in res["probabilities"].items():
                clr = PROB_COLORS.get(b, "#22A050")
                st.markdown(f"""
                <div class="bar-row">
                  <div class="bar-label">
                    <span>{b}</span>
                    <span style="color:{clr};font-weight:600;">{p}%</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{p}%;background:{clr};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # SHAP drivers
            st.markdown('<div class="fg"><div class="fg-title">Top SHAP Drivers</div>', unsafe_allow_html=True)
            max_v = max(res["driver_values"].values()) if res["driver_values"] else 1
            for feat, val in res["driver_values"].items():
                pct = (val / max_v * 100) if max_v > 0 else 0
                st.markdown(f"""
                <div class="bar-row">
                  <div style="display:flex;justify-content:space-between;">
                    <span class="bar-label">{feat.replace('_',' ').title()}</span>
                    <span style="font-size:11px;color:var(--muted);">{val:.4f}</span>
                  </div>
                  <div class="bar-bg">
                    <div class="bar-fill" style="width:{pct:.0f}%;background:var(--g500);"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<p style="font-size:11px;color:var(--muted);margin-top:6px;">These features influenced your prediction most strongly.</p></div>', unsafe_allow_html=True)

            # Input summary
            st.markdown('<div class="fg"><div class="fg-title">Input Summary</div>', unsafe_allow_html=True)
            summary = {"Study hrs/day":f"{study} hrs","Attendance":f"{attend}%",
                       "Sleep":f"{sleep} hrs","Internet":f"{internet} hrs"}
            if has_score and score is not None:
                summary["Exam score"] = f"{score}%"
            for k, v in summary.items():
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:13px;padding:6px 0;border-bottom:1px solid var(--border);'>"
                    f"<span style='color:var(--muted);'>{k}</span><strong>{v}</strong></div>",
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div style="margin-bottom:14px;"><div class="fg-title" style="font-size:11px;letter-spacing:1.5px;">WHAT YOU SHOULD DO</div></div>', unsafe_allow_html=True)
            st.markdown(
                '<ul class="al">' +
                "".join(f'<li class="ai"><div class="ad"></div>{a}</li>' for a in res["student_actions"]) +
                '</ul>',
                unsafe_allow_html=True
            )
            st.markdown('<div style="margin-top:24px;margin-bottom:14px;"><div class="fg-title" style="font-size:11px;letter-spacing:1.5px;">WHAT YOUR LECTURER SHOULD DO</div></div>', unsafe_allow_html=True)
            st.markdown(
                '<ul class="al">' +
                "".join(f'<li class="ai"><div class="ad" style="background:#EF9F27;"></div>{a}</li>' for a in res["lecturer_actions"]) +
                '</ul>',
                unsafe_allow_html=True
            )
            if res["mode"] == "early":
                st.markdown("""
                <div class="info-box" style="margin-top:22px;">
                  <div class="info-box-title">💡 Early warning mode active</div>
                  <div class="info-box-body">No exam score was provided — the
                    behavioural early warning model was used. Tick the exam score
                    checkbox and re-submit for a confirmed band.</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close panel

# ════════════════════════════════════════════════════════════════════════════
# LECTURER PANEL
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "lecturer":
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel-hdr">
      <div class="panel-icon-wrap l">📋</div>
      <div>
        <div class="panel-title">Lecturer Dashboard</div>
        <div class="panel-desc">Upload your class CSV to receive a full cohort
          band distribution, Fail band flags, SHAP driver analysis, and
          personalised course delivery recommendations.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">📄 Required CSV columns</div>
      <div class="info-box-body">
        <strong>student_id, study_hours_per_day, attendance_percentage,
        sleep_hours, internet_usage_hours, exam_score</strong> — one row per student.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload zone
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon">📊</div>
      <div class="upload-title">Upload Class Grades</div>
      <div class="upload-body">Drag and drop your CSV here, or click to browse</div>
      <span class="upload-badge">CSV ONLY</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                label_visibility="collapsed", key="l_file")

    ao1, ao2 = st.columns(2)
    show_atrisk   = ao1.checkbox("Flag Fail band students", value=True, key="l_at")
    show_delivery = ao2.checkbox("Delivery recommendations", value=True, key="l_del")

    if st.button("Analyse Class →", key="btn_lec"):
        if uploaded is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                df_up = pd.read_csv(uploaded)
            except Exception:
                st.error("Could not read CSV. Check file format.")
                st.stop()

            required = ["study_hours_per_day", "attendance_percentage",
                        "sleep_hours", "internet_usage_hours", "exam_score"]
            missing  = [c for c in required if c not in df_up.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            with st.spinner(f"Running model on {len(df_up):,} students…"):
                records = []
                for _, row in df_up.iterrows():
                    raw  = {c: float(row[c]) for c in required}
                    data = build_features(raw, models["feat_maxes"])
                    inp  = pd.DataFrame([data])[models["all_features"]]
                    pred = int(models["rf_full"].predict(inp)[0])
                    band = models["int_to_band"][pred]
                    prob = models["rf_full"].predict_proba(inp)[0]
                    conf = float(prob[pred]) * 100
                    records.append({
                        "student_id": str(row.get("student_id", "—")),
                        "exam_score": raw["exam_score"],
                        "band":       band,
                        "confidence": round(conf, 1),
                    })
                result_df = pd.DataFrame(records)

            st.session_state.lec_results = result_df
            st.session_state.lec_df      = df_up

    if st.session_state.lec_results is not None:
        result_df = st.session_state.lec_results
        df_up     = st.session_state.lec_df
        n         = len(result_df)
        required  = ["study_hours_per_day","attendance_percentage",
                     "sleep_hours","internet_usage_hours","exam_score"]

        st.success(f"✓ Analysed {n:,} students successfully.")

        # Band distribution cards
        st.markdown('<div style="margin:24px 0 12px;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--g500);">Band Distribution</div>', unsafe_allow_html=True)
        band_dist = result_df["band"].value_counts().reindex(
            models["band_order"], fill_value=0
        )
        bd_cols = st.columns(len(models["band_order"]))
        for col, (band, count) in zip(bd_cols, band_dist.items()):
            sty = BAND_STYLE.get(band, BAND_STYLE["Pass"])
            pct = count / n * 100
            col.markdown(f"""
            <div class="bdc" style="background:{sty['bg']};border:1px solid {sty['border']};">
              <div style="font-size:26px;margin-bottom:6px;">{sty['icon']}</div>
              <div class="bdc-num" style="color:{sty['text']};">{count}</div>
              <div class="bdc-name" style="color:{sty['text']};">{band}</div>
              <div class="bdc-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Fail band table
        if show_atrisk:
            st.markdown('<div style="margin:24px 0 8px;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--g500);">Fail Band — At-Risk Students</div>', unsafe_allow_html=True)
            fail_df  = result_df[result_df["band"] == "Fail"].reset_index(drop=True)
            fail_pct = len(fail_df) / n * 100
            if len(fail_df) == 0:
                st.success("No students are currently in the Fail band.")
            else:
                if fail_pct > 15:
                    st.warning(f"⚠️ {fail_pct:.1f}% of the class is in the Fail band — above the 15% warning threshold.")
                st.dataframe(fail_df, use_container_width=True, hide_index=True)

        # SHAP cohort driver chart
        st.markdown('<div style="margin:24px 0 8px;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--g500);">Most Common Risk Driver — SHAP Cohort View</div>', unsafe_allow_html=True)
        sample_ids = result_df.sample(min(150, n), random_state=42)["student_id"].tolist()
        top_drivers = []
        explainer   = get_explainer("full")

        for sid in sample_ids:
            rows = df_up[df_up["student_id"].astype(str) == str(sid)]
            if rows.empty:
                continue
            r    = rows.iloc[0]
            raw  = {c: float(r[c]) for c in required}
            data = build_features(raw, models["feat_maxes"])
            inp  = pd.DataFrame([data])[models["all_features"]]
            band_val = result_df[result_df["student_id"] == str(sid)]["band"].values
            if len(band_val) == 0:
                continue
            cidx = models["band_to_int"].get(band_val[0], 0)
            sv   = explainer.shap_values(inp, check_additivity=False)
            if isinstance(sv, list):
                sv_cls = sv[cidx][0]
            else:
                try:
                    sv_cls = sv[0, :, cidx]
                except Exception:
                    continue
            feat_imp = pd.Series(np.abs(sv_cls), index=models["all_features"])
            top_drivers.append(feat_imp.idxmax().replace("_", " ").title())

        if top_drivers:
            driver_counts = pd.Series(Counter(top_drivers)).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, max(2.5, len(driver_counts)*0.55)))
            colors = ["#22a050" if i == len(driver_counts)-1 else "#6dd496"
                      for i in range(len(driver_counts))]
            driver_counts.plot.barh(ax=ax, color=colors, edgecolor="white")
            ax.set_xlabel("Number of students", fontsize=11)
            ax.set_title("Top SHAP driver per student — cohort sample", fontsize=12)
            for spine in ["top","right"]:
                ax.spines[spine].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Full results table
        with st.expander("View full class results table"):
            st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Delivery recommendations
        if show_delivery:
            fail_pct = band_dist.get("Fail", 0) / n * 100
            supp_pct = band_dist.get("Supplementable", 0) / n * 100
            risk_pct = fail_pct + supp_pct

            st.markdown('<div style="margin:24px 0 12px;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--g500);">Course Delivery Recommendations</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)

            with d1:
                items = []
                if fail_pct > 15:
                    items.append(f"⚠️ {fail_pct:.1f}% of students are in the Fail band — immediate cohort-level action recommended.")
                items += [
                    "Schedule one-on-one check-ins with all Fail band students before Week 8.",
                    "Provide foundational recap materials for high-failure topics.",
                    "Consider peer mentoring: pair Excellent ↔ Fail students.",
                    "Escalate persistent Fail cases to the academic advisor or HOD.",
                ]
                st.markdown(
                    '<div class="fg"><div class="fg-title">For At-Risk Students</div><ul class="al">' +
                    "".join(f'<li class="ai"><div class="ad"></div>{a}</li>' for a in items) +
                    '</ul></div>', unsafe_allow_html=True
                )

            with d2:
                items2 = [
                    f"{'⚠️ ' if risk_pct>35 else ''}{risk_pct:.1f}% of students are in Fail or Supplementable — "
                    f"{'review pacing urgently.' if risk_pct>35 else 'monitor closely.'}",
                    "Increase formative assessments mid-semester to catch gaps early.",
                    "Share anonymised cohort data with the HOD for departmental benchmarking.",
                    "Use the SHAP driver chart above to target your guidance where it matters most.",
                ]
                st.markdown(
                    '<div class="fg"><div class="fg-title">For Course Delivery</div><ul class="al">' +
                    "".join(f'<li class="ai"><div class="ad" style="background:#EF9F27;"></div>{a}</li>' for a in items2) +
                    '</ul></div>', unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)  # close panel

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <div class="footer-left">
    <strong>ISPAIS — Intelligent Student Performance Assessment &amp; Intervention System</strong><br>
    Mzuzu University · ICT Department · Nathan Bvumbwe (BSDS0322)<br>
    Supervisor: Emmanuel Ngalande · Final Year Project 2026
  </div>
  <div class="footer-right">
    <span class="fbadge">RANDOM FOREST</span>
    <span class="fbadge">TWO-LAYER PIPELINE</span>
    <span class="fbadge">SHAP XAI</span>
    <span class="fbadge">UPDATED MZUNI BANDS</span>
    <span class="fbadge">STREAMLIT</span>
  </div>
</div>
""", unsafe_allow_html=True)
