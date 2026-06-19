"""
BSDS0322 — Intelligent Student Performance Assessment & Intervention System
Mzuzu University, ICT Department
Frontend only — model integration comes next.
"""

import streamlit as st
import pandas as pd
import time

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISPAIS — Mzuzu University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS — Mzuni green & white theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Tokens ── */
:root {
    --g900: #0a2e1a;
    --g800: #0f4526;
    --g700: #166534;
    --g600: #1a7a3e;
    --g500: #22a050;
    --g400: #3dbd6b;
    --g300: #6dd496;
    --g200: #a8ecc2;
    --g100: #d4f7e4;
    --g050: #edfaf3;
    --white: #ffffff;
    --off:   #f7fdf9;
    --text:  #0a2e1a;
    --muted: #4a7a5e;
    --border:#c8edda;
    --shadow: 0 2px 20px rgba(10,46,26,0.08);
    --shadow-lg: 0 8px 40px rgba(10,46,26,0.14);
}

/* ── Reset ── */
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: var(--white) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] { display: none; }
[data-testid="stHeader"]  { background: transparent !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* ── Nav bar ── */
.nav-bar {
    background: var(--g800);
    padding: 0 48px;
    height: 68px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 3px solid var(--g600);
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 14px;
}
.nav-logo-badge {
    background: var(--g500);
    color: var(--white);
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Serif Display', serif;
    font-size: 18px; font-weight: bold;
    flex-shrink: 0;
}
.nav-title {
    color: var(--white);
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
    line-height: 1.2;
}
.nav-subtitle {
    color: var(--g200);
    font-size: 11px;
    font-weight: 300;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.nav-pill {
    background: rgba(255,255,255,0.12);
    color: var(--g100);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, var(--g800) 0%, var(--g700) 50%, var(--g600) 100%);
    padding: 72px 48px 64px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 340px; height: 340px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(61,189,107,0.12);
    pointer-events: none;
}
.hero-eyebrow {
    color: var(--g300);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-title {
    color: var(--white);
    font-family: 'DM Serif Display', serif;
    font-size: clamp(32px, 4vw, 52px);
    line-height: 1.15;
    margin-bottom: 20px;
    max-width: 640px;
}
.hero-title em {
    color: var(--g300);
    font-style: italic;
}
.hero-body {
    color: var(--g100);
    font-size: 16px;
    line-height: 1.7;
    max-width: 520px;
    font-weight: 300;
    margin-bottom: 0;
}
.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 40px;
    flex-wrap: wrap;
}
.hero-stat {
    border-left: 2px solid var(--g400);
    padding-left: 16px;
}
.hero-stat-num {
    color: var(--white);
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    line-height: 1;
}
.hero-stat-lbl {
    color: var(--g200);
    font-size: 12px;
    font-weight: 400;
    margin-top: 4px;
}

/* ── Mode selector ── */
.mode-section {
    background: var(--off);
    padding: 56px 48px;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--g500);
    margin-bottom: 10px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: var(--g900);
    margin-bottom: 8px;
}
.section-body {
    color: var(--muted);
    font-size: 15px;
    margin-bottom: 36px;
    font-weight: 300;
}

/* ── Mode cards ── */
.mode-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    max-width: 900px;
}
.mode-card {
    background: var(--white);
    border: 2px solid var(--border);
    border-radius: 16px;
    padding: 32px;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.mode-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--g400);
    transform: scaleX(0);
    transition: transform 0.25s ease;
    transform-origin: left;
}
.mode-card:hover::before,
.mode-card.active::before { transform: scaleX(1); }
.mode-card:hover {
    border-color: var(--g400);
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}
.mode-card.active {
    border-color: var(--g500);
    box-shadow: var(--shadow-lg);
    background: var(--g050);
}
.mode-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    margin-bottom: 20px;
}
.mode-icon.student { background: var(--g100); }
.mode-icon.lecturer { background: #fef3c7; }
.mode-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: var(--g900);
    margin-bottom: 8px;
}
.mode-card-body {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
    margin-bottom: 20px;
}
.mode-card-tags {
    display: flex; gap: 8px; flex-wrap: wrap;
}
.tag {
    background: var(--g100);
    color: var(--g700);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
}
.tag.amber { background: #fef3c7; color: #92400e; }

/* ── Form panels ── */
.panel {
    background: var(--white);
    padding: 56px 48px;
    border-top: 1px solid var(--border);
    animation: slideIn 0.35s ease;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.panel-header {
    display: flex;
    align-items: flex-start;
    gap: 20px;
    margin-bottom: 40px;
    padding-bottom: 32px;
    border-bottom: 1px solid var(--border);
}
.panel-icon-wrap {
    width: 56px; height: 56px;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    flex-shrink: 0;
}
.panel-icon-wrap.student { background: var(--g100); }
.panel-icon-wrap.lecturer { background: #fef3c7; }
.panel-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: var(--g900);
    margin-bottom: 6px;
}
.panel-desc {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
}

/* ── Field groups ── */
.field-group {
    background: var(--off);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}
.field-group-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--g600);
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select {
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    background: var(--white) !important;
    color: var(--text) !important;
    font-size: 14px !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"]   input:focus {
    border-color: var(--g500) !important;
    box-shadow: 0 0 0 3px rgba(34,160,80,0.15) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: var(--g500) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--g800) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button {
    background: var(--g600) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 36px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--g700) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(10,46,26,0.25) !important;
}

/* ── Mode pills (top of form panels) ── */
.mode-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--g100);
    color: var(--g700);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 28px;
    cursor: pointer;
    border: none;
    transition: background 0.15s;
}
.mode-pill:hover { background: var(--g200); }

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed var(--g300);
    border-radius: 14px;
    padding: 48px 32px;
    text-align: center;
    background: var(--g050);
    margin-bottom: 24px;
    transition: all 0.2s;
}
.upload-zone:hover {
    border-color: var(--g500);
    background: var(--g100);
}
.upload-icon { font-size: 40px; margin-bottom: 12px; }
.upload-title {
    font-family: 'DM Serif Display', serif;
    font-size: 18px; color: var(--g800); margin-bottom: 6px;
}
.upload-body {
    color: var(--muted); font-size: 13px; margin-bottom: 16px;
}
.upload-badge {
    display: inline-block;
    background: var(--g700); color: var(--white);
    border-radius: 6px; padding: 4px 12px;
    font-size: 12px; font-weight: 600;
    letter-spacing: 1px;
}

/* ── Info box ── */
.info-box {
    background: var(--g050);
    border: 1px solid var(--g200);
    border-left: 4px solid var(--g500);
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-bottom: 20px;
}
.info-box-title {
    font-weight: 600; font-size: 13px;
    color: var(--g700); margin-bottom: 4px;
}
.info-box-body {
    font-size: 13px; color: var(--muted); line-height: 1.5;
}

/* ── Result cards (placeholder) ── */
.result-band {
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.result-band.repeat   { background: #fef2f2; border: 1.5px solid #fca5a5; }
.result-band.supp     { background: #fffbeb; border: 1.5px solid #fcd34d; }
.result-band.good     { background: var(--g050); border: 1.5px solid var(--g200); }
.result-band.excellent{ background: #f0fdf4; border: 1.5px solid var(--g300); }
.result-band-label {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 6px;
}
.result-band.repeat    .result-band-label { color: #b91c1c; }
.result-band.supp      .result-band-label { color: #b45309; }
.result-band.good      .result-band-label { color: var(--g600); }
.result-band.excellent .result-band-label { color: var(--g700); }
.result-band-name {
    font-family: 'DM Serif Display', serif;
    font-size: 30px;
}
.result-band.repeat    .result-band-name { color: #7f1d1d; }
.result-band.supp      .result-band-name { color: #78350f; }
.result-band.good      .result-band-name { color: var(--g800); }
.result-band.excellent .result-band-name { color: var(--g900); }

/* ── Action items ── */
.action-list { list-style: none; padding: 0; margin: 0; }
.action-item {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
    font-size: 14px; color: var(--text); line-height: 1.5;
}
.action-item:last-child { border-bottom: none; }
.action-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--g400); flex-shrink: 0; margin-top: 6px;
}

/* ── Footer ── */
.footer {
    background: var(--g900);
    padding: 40px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}
.footer-left { color: var(--g200); font-size: 13px; line-height: 1.8; }
.footer-right {
    display: flex; gap: 8px; flex-wrap: wrap;
}
.footer-badge {
    background: rgba(255,255,255,0.08);
    color: var(--g200);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── Divider ── */
.divider {
    height: 1px; background: var(--border);
    margin: 32px 0;
}

/* ── Responsive ── */
@media (max-width: 700px) {
    .nav-bar { padding: 0 20px; }
    .hero    { padding: 48px 20px; }
    .mode-section, .panel { padding: 40px 20px; }
    .mode-grid { grid-template-columns: 1fr; }
    .footer    { padding: 32px 20px; flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = None         # None | 'student' | 'lecturer'
if "student_submitted" not in st.session_state:
    st.session_state.student_submitted = False
if "lecturer_submitted" not in st.session_state:
    st.session_state.lecturer_submitted = False


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


# ════════════════════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">Proactive Academic Intervention</div>
  <div class="hero-title">
    Know where you stand —<br>
    <em>before</em> the final exam.
  </div>
  <div class="hero-body">
    An intelligent early-warning system that maps your continuous assessment (CA)
    performance to Mzuzu University's grading bands and delivers personalised
    action plans — for students and lecturers alike.
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

# Cards as Streamlit columns (so buttons work)
col1, col2, col_pad = st.columns([1, 1, 0.6])

with col1:
    student_active = "active" if st.session_state.mode == "student" else ""
    st.markdown(f"""
    <div class="mode-card {student_active}" style="margin: 0 0 0 48px;">
      <div class="mode-icon student">🎓</div>
      <div class="mode-card-title">I am a Student</div>
      <div class="mode-card-body">
        Enter your study habits and CA results to discover your performance band
        and get a personalised improvement plan.
      </div>
      <div class="mode-card-tags">
        <span class="tag">Early warning</span>
        <span class="tag">Post-CA</span>
        <span class="tag">Action plan</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='padding: 0 48px;'>", unsafe_allow_html=True)
    if st.button("Continue as Student →", key="btn_student"):
        st.session_state.mode = "student"
        st.session_state.student_submitted = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    lecturer_active = "active" if st.session_state.mode == "lecturer" else ""
    st.markdown(f"""
    <div class="mode-card {lecturer_active}">
      <div class="mode-icon lecturer">📋</div>
      <div class="mode-card-title">I am a Lecturer</div>
      <div class="mode-card-body">
        Upload your class grades file to get a full cohort analysis, at-risk
        student flags, and targeted delivery recommendations.
      </div>
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

    # Header
    st.markdown("""
    <div class="panel-header">
      <div class="panel-icon-wrap student">🎓</div>
      <div>
        <div class="panel-title">Student Assessment</div>
        <div class="panel-desc">
          Fill in your details below. Fields marked with an asterisk are required
          for all prediction modes. <strong>Exam score</strong> is only needed for
          the confirmed post-CA prediction — leave it blank for an early warning
          based on your study habits alone.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Info box
    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">🔒 Your data is private</div>
      <div class="info-box-body">
        This system uses a synthetic proxy dataset for development and does not
        store or share any data you enter. Results are for advisory purposes only.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Personal info ──
    st.markdown('<div class="field-group"><div class="field-group-title">Personal Information</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        student_id = st.text_input("Student ID", placeholder="e.g. BSDS0322", key="s_id")
    with p2:
        age = st.number_input("Age", min_value=16, max_value=60, value=20, key="s_age")
    with p3:
        gender = st.selectbox("Gender", ["Select...", "Male", "Female", "Prefer not to say"], key="s_gender")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Study habits ──
    st.markdown('<div class="field-group"><div class="field-group-title">Study Habits *</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    with h1:
        study_hours = st.slider(
            "Study hours per day",
            min_value=0.0, max_value=12.0, value=3.0, step=0.5,
            format="%.1f hrs", key="s_study"
        )
        sleep_hours = st.slider(
            "Sleep hours per night",
            min_value=3.0, max_value=12.0, value=7.0, step=0.5,
            format="%.1f hrs", key="s_sleep"
        )
    with h2:
        attendance = st.slider(
            "Attendance percentage",
            min_value=0, max_value=100, value=75, step=1,
            format="%d%%", key="s_attend"
        )
        internet = st.slider(
            "Daily internet usage (non-study)",
            min_value=0.0, max_value=12.0, value=2.0, step=0.5,
            format="%.1f hrs", key="s_internet"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CA / Exam score ──
    st.markdown('<div class="field-group"><div class="field-group-title">Assessment Score (optional — for confirmed prediction)</div>', unsafe_allow_html=True)
    e1, e2 = st.columns([1, 2])
    with e1:
        has_score = st.checkbox("I have my CA / exam score", value=False, key="s_has_score")
    if has_score:
        with e2:
            exam_score = st.number_input(
                "CA / Exam score (%)",
                min_value=0.0, max_value=100.0, value=55.0, step=0.5,
                key="s_score"
            )
        # Show which band this falls in live
        if exam_score <= 34:
            band_preview = "🔴 Repeat zone (0–34%)"
        elif exam_score <= 44:
            band_preview = "🟡 Supplementary zone (35–44%)"
        elif exam_score <= 64:
            band_preview = "🟢 Good zone (45–64%)"
        else:
            band_preview = "✅ Excellent zone (65–100%)"
        st.markdown(f"<p style='font-size:13px;color:var(--muted);margin-top:4px;'>Band preview: <strong>{band_preview}</strong></p>", unsafe_allow_html=True)
    else:
        exam_score = None

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ──
    st.markdown("<div style='margin-top: 8px;'>", unsafe_allow_html=True)
    if st.button("Analyse My Performance →", key="btn_student_submit"):
        if gender == "Select...":
            st.warning("Please select your gender to continue.")
        else:
            st.session_state.student_submitted = True

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Results placeholder ──
    if st.session_state.student_submitted:
        mode_label = "Post-CA Confirmed Prediction" if has_score else "Early Warning (Behavioural)"

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-label" style="margin-top:8px;">Results — {mode_label}</div>
        <div class="section-title" style="margin-bottom:24px;">Your Performance Report</div>
        """, unsafe_allow_html=True)

        with st.spinner("Analysing your data..."):
            time.sleep(1.2)

        # Band result placeholder (will be driven by model later)
        if has_score and exam_score <= 34:
            band_cls, band_name = "repeat",    "Repeat"
        elif has_score and exam_score <= 44:
            band_cls, band_name = "supp",      "Supplementary"
        elif has_score and exam_score <= 64:
            band_cls, band_name = "good",      "Good"
        elif has_score:
            band_cls, band_name = "excellent", "Excellent"
        else:
            # Early warning placeholder
            if study_hours < 2 or attendance < 50:
                band_cls, band_name = "repeat", "At Risk (Early Warning)"
            elif study_hours < 3.5:
                band_cls, band_name = "supp",   "Supplementary Risk"
            else:
                band_cls, band_name = "good",   "On Track"

        r1, r2 = st.columns([1, 1.4])

        with r1:
            st.markdown(f"""
            <div class="result-band {band_cls}">
              <div class="result-band-label">Predicted Band</div>
              <div class="result-band-name">{band_name}</div>
              <p style="font-size:13px;margin-top:12px;color:var(--muted);">
                {"Based on your exam score mapped to Mzuni grading thresholds." if has_score
                 else "Based on your study habits and attendance. Add your exam score for a confirmed result."}
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="field-group" style="margin-top:0;">
              <div class="field-group-title">Input Summary</div>
            """, unsafe_allow_html=True)
            summary_data = {
                "Study hours/day": f"{study_hours} hrs",
                "Attendance":      f"{attendance}%",
                "Sleep":           f"{sleep_hours} hrs",
                "Internet (non-study)": f"{internet} hrs",
            }
            if has_score:
                summary_data["Exam score"] = f"{exam_score}%"
            for k, v in summary_data.items():
                st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:6px 0;border-bottom:1px solid var(--border);'><span style='color:var(--muted);'>{k}</span><strong>{v}</strong></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div style="margin-bottom:12px;">
              <div class="field-group-title" style="font-size:11px;letter-spacing:1.5px;">
                RECOMMENDED ACTIONS — STUDENT
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Placeholder recommendations — will come from model
            placeholder_actions = {
                "repeat": [
                    "Seek academic counselling immediately — do not wait until exams.",
                    "Increase study hours to at least 4 hours per day.",
                    "Attend every class — missing even one session compounds the risk.",
                    "Reduce recreational internet usage to under 1 hour on study days.",
                    "Form a study group with Good or Excellent band peers.",
                ],
                "supp": [
                    "You are close to the pass boundary — focused effort now can shift your outcome.",
                    "Target the specific CA components where marks were lost.",
                    "Attend all remaining classes and tutorials without exception.",
                    "Use office hours to get feedback on weak areas.",
                ],
                "good": [
                    "You are on track — maintain your current study routine.",
                    "Push for consistency: aim to add 30 minutes of study per day.",
                    "Attempt past exam papers under timed conditions.",
                    "Set a stretch target of the Excellent band.",
                ],
                "excellent": [
                    "Outstanding — maintain your current habits.",
                    "Consider mentoring peers in lower bands.",
                    "Explore supplementary reading for advanced coursework.",
                ],
            }
            actions = placeholder_actions.get(band_cls, placeholder_actions["good"])
            items_html = "".join(
                f'<li class="action-item"><div class="action-dot"></div>{a}</li>'
                for a in actions
            )
            st.markdown(f'<ul class="action-list">{items_html}</ul>', unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-top:20px;">
              <div class="info-box-title">💡 Model note</div>
              <div class="info-box-body">
                These recommendations are currently based on rule thresholds.
                Full ML + SHAP explanations will be connected in the next version.
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close panel


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
          Upload your class grades file to receive a full cohort band distribution,
          a list of at-risk students flagged for intervention, and personalised
          course delivery recommendations.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <div class="info-box-title">📄 Accepted file format</div>
      <div class="info-box-body">
        Upload a CSV file with one row per student. Required columns:
        <strong>student_id, study_hours_per_day, attendance_percentage,
        sleep_hours, internet_usage_hours, exam_score</strong>.
        Optional: gender, age.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Lecturer details ──
    st.markdown('<div class="field-group"><div class="field-group-title">Lecturer Details</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        lec_name   = st.text_input("Full name", placeholder="e.g. Dr. E. Ngalande", key="l_name")
    with lc2:
        lec_course = st.text_input("Course / Unit", placeholder="e.g. Data Structures", key="l_course")
    with lc3:
        lec_year   = st.selectbox("Year of study", ["Select...", "Year 1", "Year 2", "Year 3", "Year 4"], key="l_year")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── File upload ──
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-icon">📊</div>
      <div class="upload-title">Upload Class Grades</div>
      <div class="upload-body">Drag and drop your CSV file here, or click to browse</div>
      <span class="upload-badge">CSV ONLY</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="l_file",
        label_visibility="collapsed"
    )

    # ── Analysis options ──
    st.markdown('<div class="field-group" style="margin-top:20px;"><div class="field-group-title">Analysis Options</div>', unsafe_allow_html=True)
    ao1, ao2, ao3 = st.columns(3)
    with ao1:
        show_atRisk    = st.checkbox("Flag at-risk students (Repeat band)", value=True, key="l_atrisk")
    with ao2:
        show_shap      = st.checkbox("Include SHAP driver analysis", value=True, key="l_shap")
    with ao3:
        show_delivery  = st.checkbox("Course delivery recommendations", value=True, key="l_delivery")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ──
    if st.button("Analyse Class →", key="btn_lecturer_submit"):
        if lec_year == "Select...":
            st.warning("Please select the year of study.")
        elif uploaded_file is None:
            st.warning("Please upload a grades CSV file before continuing.")
        else:
            st.session_state.lecturer_submitted = True

    # ── Results placeholder ──
    if st.session_state.lecturer_submitted and uploaded_file is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label" style="margin-top:8px;">Cohort Analysis</div>
        <div class="section-title" style="margin-bottom:24px;">Class Performance Report</div>
        """, unsafe_allow_html=True)

        with st.spinner("Processing your class data..."):
            time.sleep(1.4)
            try:
                df_upload = pd.read_csv(uploaded_file)
                n_students = len(df_upload)
                has_score_col = 'exam_score' in df_upload.columns
            except Exception:
                st.error("Could not read the CSV file. Please check the format and try again.")
                st.stop()

        st.success(f"✓ File loaded — {n_students:,} student records found.")

        # ── Band distribution preview (placeholder) ──
        st.markdown("""
        <div class="section-label" style="margin-top:24px;margin-bottom:8px;">Band Distribution</div>
        """, unsafe_allow_html=True)

        if has_score_col:
            df_upload['band'] = df_upload['exam_score'].apply(
                lambda s: 'Repeat' if s<=34 else 'Supplementary' if s<=44 else 'Good' if s<=64 else 'Excellent'
            )
            band_dist = df_upload['band'].value_counts().reindex(
                ['Repeat','Supplementary','Good','Excellent'], fill_value=0
            )

            bc1, bc2, bc3, bc4 = st.columns(4)
            band_styles = {
                'Repeat':        ('🔴', '#fef2f2', '#b91c1c'),
                'Supplementary': ('🟡', '#fffbeb', '#b45309'),
                'Good':          ('🟢', '#edfaf3', '#166534'),
                'Excellent':     ('✅', '#f0fdf4', '#14532d'),
            }
            for col, (band, count) in zip([bc1,bc2,bc3,bc4], band_dist.items()):
                icon, bg, clr = band_styles[band]
                pct = count/n_students*100 if n_students > 0 else 0
                col.markdown(f"""
                <div style="background:{bg};border-radius:12px;padding:20px;text-align:center;">
                  <div style="font-size:24px;margin-bottom:6px;">{icon}</div>
                  <div style="font-family:'DM Serif Display',serif;font-size:26px;color:{clr};">{count}</div>
                  <div style="font-size:12px;font-weight:600;color:{clr};margin:4px 0;">{band}</div>
                  <div style="font-size:11px;color:#6b7280;">{pct:.1f}% of class</div>
                </div>
                """, unsafe_allow_html=True)

            # At-risk table
            if show_atRisk:
                st.markdown("""
                <div class="section-label" style="margin-top:32px;margin-bottom:8px;">
                  At-Risk Students — Repeat Band
                </div>
                """, unsafe_allow_html=True)
                at_risk = df_upload[df_upload['band'] == 'Repeat']
                if len(at_risk) == 0:
                    st.success("No students are currently in the Repeat band.")
                else:
                    st.dataframe(
                        at_risk.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True
                    )

        else:
            st.info("Add an `exam_score` column to your CSV to see band distribution charts.")

        # ── Delivery recommendations placeholder ──
        if show_delivery:
            st.markdown("""
            <div class="section-label" style="margin-top:32px;margin-bottom:8px;">
              Course Delivery Recommendations
            </div>
            """, unsafe_allow_html=True)

            d1, d2 = st.columns(2)
            with d1:
                st.markdown("""
                <div class="field-group">
                  <div class="field-group-title">For At-Risk Students</div>
                  <ul class="action-list">
                    <li class="action-item"><div class="action-dot"></div>
                      Schedule one-on-one check-ins with Repeat band students before Week 8.
                    </li>
                    <li class="action-item"><div class="action-dot"></div>
                      Provide foundational recap materials for topics with highest failure rates.
                    </li>
                    <li class="action-item"><div class="action-dot"></div>
                      Consider peer mentoring pairs: Excellent ↔ Repeat students.
                    </li>
                    <li class="action-item"><div class="action-dot"></div>
                      Escalate persistent at-risk cases to the academic advisor.
                    </li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)
            with d2:
                st.markdown("""
                <div class="field-group">
                  <div class="field-group-title">For Course Delivery</div>
                  <ul class="action-list">
                    <li class="action-item"><div class="action-dot"></div>
                      If &gt;20% of the class is in Repeat or Supplementary, review
                      pacing and delivery approach for the most tested topics.
                    </li>
                    <li class="action-item"><div class="action-dot"></div>
                      Increase formative assessments mid-semester to catch gaps early.
                    </li>
                    <li class="action-item"><div class="action-dot"></div>
                      Share anonymised cohort data with the HOD to benchmark against
                      other courses in the department.
                    </li>
                    <li class="action-item"><div class="action-dot"></div>
                      Use SHAP driver analysis (available in the full model) to identify
                      which student behaviours most predict poor outcomes in your class.
                    </li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:8px;">
          <div class="info-box-title">💡 Full ML analysis coming next</div>
          <div class="info-box-body">
            Band distribution is currently computed from score thresholds only.
            The next version will run the full two-layer model and SHAP explanations
            for each student in the uploaded file.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close panel


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
    <span class="footer-badge">K-MEANS CLUSTERING</span>
    <span class="footer-badge">RANDOM FOREST</span>
    <span class="footer-badge">SHAP XAI</span>
    <span class="footer-badge">STREAMLIT</span>
  </div>
</div>
""", unsafe_allow_html=True)
