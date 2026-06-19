"""
app.py — ISPAIS entry point.
Run with:  streamlit run app.py

Folder structure required:
  student_performance_system/
  ├── app.py
  ├── predictor.py
  ├── student_view.py
  ├── lecturer_view.py
  └── app_model/
      ├── rf_full_model.pkl
      ├── rf_behav_model.pkl
      ├── scaler_full.pkl
      ├── scaler_behav.pkl
      ├── all_feature_names.pkl
      ├── behav_feature_names.pkl
      ├── band_order.pkl
      ├── band_to_int.pkl
      ├── int_to_band.pkl
      └── feat_maxes.pkl
"""

import streamlit as st
from predictor import load_models
import student_view
import lecturer_view

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISPAIS — Mzuzu University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Warm up model on first load (cached — only runs once per session)
models = load_models()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --g900:#0a2e1a; --g800:#0f4526; --g700:#166534; --g600:#1a7a3e;
    --g500:#22a050; --g400:#3dbd6b; --g300:#6dd496; --g200:#a8ecc2;
    --g100:#d4f7e4; --g050:#edfaf3; --white:#ffffff; --off:#f7fdf9;
    --text:#0a2e1a; --muted:#4a7a5e; --border:#c8edda;
    --shadow:0 2px 20px rgba(10,46,26,0.08);
    --shadow-lg:0 8px 40px rgba(10,46,26,0.14);
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
.nav-pill{background:rgba(255,255,255,0.12);color:var(--g100);border:1px solid rgba(255,255,255,0.2);
    border-radius:20px;padding:6px 14px;font-size:12px;font-weight:500;}
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
.section-title{font-family:'DM Serif Display',serif;font-size:28px;color:var(--g900);margin-bottom:8px;}
.section-body{color:var(--muted);font-size:15px;margin-bottom:36px;font-weight:300;}
.mode-card{background:var(--white);border:2px solid var(--border);border-radius:16px;
    padding:32px;transition:all 0.2s ease;position:relative;overflow:hidden;}
.mode-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;
    background:var(--g400);transform:scaleX(0);transition:transform 0.25s;transform-origin:left;}
.mode-card:hover::before,.mode-card.active::before{transform:scaleX(1);}
.mode-card:hover{border-color:var(--g400);box-shadow:var(--shadow-lg);transform:translateY(-2px);}
.mode-card.active{border-color:var(--g500);box-shadow:var(--shadow-lg);background:var(--g050);}
.mode-icon{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;
    justify-content:center;font-size:24px;margin-bottom:20px;}
.mode-icon.student{background:var(--g100);}
.mode-icon.lecturer{background:#fef3c7;}
.mode-card-title{font-family:'DM Serif Display',serif;font-size:20px;color:var(--g900);margin-bottom:8px;}
.mode-card-body{color:var(--muted);font-size:14px;line-height:1.6;margin-bottom:20px;}
.mode-card-tags{display:flex;gap:8px;flex-wrap:wrap;}
.tag{background:var(--g100);color:var(--g700);border-radius:20px;padding:4px 12px;font-size:12px;font-weight:500;}
.tag.amber{background:#fef3c7;color:#92400e;}
.panel{background:var(--white);padding:56px 48px;border-top:1px solid var(--border);animation:slideIn 0.35s ease;}
@keyframes slideIn{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.panel-header{display:flex;align-items:flex-start;gap:20px;margin-bottom:40px;
    padding-bottom:32px;border-bottom:1px solid var(--border);}
.panel-icon-wrap{width:56px;height:56px;border-radius:16px;display:flex;align-items:center;
    justify-content:center;font-size:26px;flex-shrink:0;}
.panel-icon-wrap.student{background:var(--g100);}
.panel-icon-wrap.lecturer{background:#fef3c7;}
.panel-title{font-family:'DM Serif Display',serif;font-size:26px;color:var(--g900);margin-bottom:6px;}
.panel-desc{color:var(--muted);font-size:14px;line-height:1.6;}
.field-group{background:var(--off);border:1px solid var(--border);border-radius:12px;
    padding:24px;margin-bottom:20px;}
.field-group-title{font-size:12px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;
    color:var(--g600);margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--border);}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select{
    border:1.5px solid var(--border)!important;border-radius:8px!important;
    font-family:'DM Sans',sans-serif!important;background:var(--white)!important;
    color:var(--text)!important;font-size:14px!important;}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus{
    border-color:var(--g500)!important;box-shadow:0 0 0 3px rgba(34,160,80,0.15)!important;}
label[data-testid="stWidgetLabel"] p{
    font-size:13px!important;font-weight:500!important;
    color:var(--g800)!important;font-family:'DM Sans',sans-serif!important;}
.stButton>button{
    background:var(--g600)!important;color:var(--white)!important;border:none!important;
    border-radius:10px!important;padding:14px 36px!important;font-family:'DM Sans',sans-serif!important;
    font-size:15px!important;font-weight:600!important;letter-spacing:0.3px!important;
    cursor:pointer!important;transition:all 0.2s!important;width:100%!important;}
.stButton>button:hover{
    background:var(--g700)!important;transform:translateY(-1px)!important;
    box-shadow:0 4px 16px rgba(10,46,26,0.25)!important;}
.info-box{background:var(--g050);border:1px solid var(--g200);border-left:4px solid var(--g500);
    border-radius:0 10px 10px 0;padding:16px 20px;margin-bottom:20px;}
.info-box-title{font-weight:600;font-size:13px;color:var(--g700);margin-bottom:4px;}
.info-box-body{font-size:13px;color:var(--muted);line-height:1.5;}
.result-band{border-radius:14px;padding:28px 32px;margin-bottom:20px;}
.result-band.repeat   {background:#fef2f2;border:1.5px solid #fca5a5;}
.result-band.supp     {background:#fffbeb;border:1.5px solid #fcd34d;}
.result-band.good     {background:var(--g050);border:1.5px solid var(--g200);}
.result-band.excellent{background:#f0fdf4;border:1.5px solid var(--g300);}
.result-band-label{font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;}
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
.action-dot{width:8px;height:8px;border-radius:50%;background:var(--g400);flex-shrink:0;margin-top:6px;}
.driver-bar-wrap{margin-bottom:10px;}
.driver-label{font-size:12px;font-weight:500;color:var(--g700);margin-bottom:4px;}
.driver-bar-bg{background:var(--g100);border-radius:4px;height:8px;overflow:hidden;}
.driver-bar-fill{background:var(--g500);height:8px;border-radius:4px;}
.upload-zone{border:2px dashed var(--g300);border-radius:14px;padding:48px 32px;
    text-align:center;background:var(--g050);margin-bottom:24px;}
.upload-icon{font-size:40px;margin-bottom:12px;}
.upload-title{font-family:'DM Serif Display',serif;font-size:18px;color:var(--g800);margin-bottom:6px;}
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
    .footer{padding:32px 20px;flex-direction:column;}}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("mode", None),
    ("student_result", None),
    ("student_inputs", {}),
    ("lecturer_result_df", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# NAV
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
        f"Make sure the folder is next to app.py. "
        f"Error: {models.get('error','unknown')}"
    )

# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Proactive Academic Intervention</div>
  <div class="hero-title">Know where you stand —<br><em>before</em> the final exam.</div>
  <div class="hero-body">
    An intelligent early-warning system that maps your continuous assessment
    performance to Mzuzu University's grading bands and delivers personalised,
    AI-explained action plans — for students and lecturers alike.
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="hero-stat-num">4</div>
      <div class="hero-stat-lbl">Performance bands</div></div>
    <div class="hero-stat"><div class="hero-stat-num">2</div>
      <div class="hero-stat-lbl">Prediction modes</div></div>
    <div class="hero-stat"><div class="hero-stat-num">XAI</div>
      <div class="hero-stat-lbl">SHAP explanations</div></div>
    <div class="hero-stat"><div class="hero-stat-num">ICT</div>
      <div class="hero-stat-lbl">Dept. pilot</div></div>
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
      <div class="mode-icon student">🎓</div>
      <div class="mode-card-title">I am a Student</div>
      <div class="mode-card-body">Enter your study habits and CA results to discover
        your performance band and get a personalised improvement plan powered by
        SHAP explanations.</div>
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
        st.session_state.student_result = None
        st.session_state.student_inputs = {}
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    l_act = "active" if st.session_state.mode == "lecturer" else ""
    st.markdown(f"""
    <div class="mode-card {l_act}">
      <div class="mode-icon lecturer">📋</div>
      <div class="mode-card-title">I am a Lecturer</div>
      <div class="mode-card-body">Upload your class grades file to get a full cohort
        analysis, at-risk student flags, SHAP driver charts, and targeted delivery
        recommendations.</div>
      <div class="mode-card-tags">
        <span class="tag">Cohort overview</span>
        <span class="tag amber">Bulk upload</span>
        <span class="tag">Delivery tips</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Continue as Lecturer →", key="btn_lecturer"):
        st.session_state.mode = "lecturer"
        st.session_state.lecturer_result_df = None
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# PANEL ROUTER — delegates to view modules
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "student":
    student_view.render()
elif st.session_state.mode == "lecturer":
    lecturer_view.render()

# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <div class="footer-left">
    <strong style="color:white;">ISPAIS — Intelligent Student Performance
    Assessment &amp; Intervention System</strong><br>
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