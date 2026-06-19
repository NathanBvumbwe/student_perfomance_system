"""
student_view.py — Student panel for ISPAIS.
Called by app.py when mode == 'student'.
"""

import streamlit as st
from predictor import load_models, predict_one, BAND_ORDER


# ── Colour maps ───────────────────────────────────────────────────────────────
BAND_CSS = {
    "Repeat":        "repeat",
    "Supplementary": "supp",
    "Good":          "good",
    "Excellent":     "excellent",
}
PROB_COLORS = {
    "Repeat":        "#E24B4A",
    "Supplementary": "#EF9F27",
    "Good":          "#1D9E75",
    "Excellent":     "#534AB7",
}


def _action_list(actions: list, dot_color: str = "var(--g400)") -> str:
    items = "".join(
        f'<li class="action-item">'
        f'<div class="action-dot" style="background:{dot_color};"></div>{a}'
        f'</li>'
        for a in actions
    )
    return f'<ul class="action-list">{items}</ul>'


def render_results(res: dict, inputs: dict) -> None:
    """Render the prediction result cards."""
    band     = res["band"]
    band_cls = BAND_CSS.get(band, "good")
    mode_lbl = "Post-CA Confirmed" if res["mode"] == "full" else "Early Warning — Behavioural"

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-label" style="margin-top:8px;">Results — {mode_lbl}</div>'
        f'<div class="section-title" style="margin-bottom:24px;">Your Performance Report</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.4])

    # ── LEFT: band card, probability bars, SHAP drivers, input summary ────────
    with left:
        st.markdown(f"""
        <div class="result-band {band_cls}">
          <div class="result-band-label">Predicted Band</div>
          <div class="result-band-name">{band}</div>
          <p style="font-size:13px;margin-top:10px;color:var(--muted);">
            Model confidence: <strong>{res['confidence']}%</strong>
          </p>
          <p style="font-size:12px;color:var(--muted);margin-top:4px;">
            {"Score mapped to Mzuni threshold via Layer 1 rule engine."
             if res['mode'] == 'full'
             else "Based on study habits only. Add exam score for a confirmed band."}
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown(
            '<div class="field-group" style="margin-top:0;">'
            '<div class="field-group-title">Band Probabilities</div>',
            unsafe_allow_html=True,
        )
        for b, p in res["probabilities"].items():
            clr = PROB_COLORS[b]
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

        # SHAP top drivers
        st.markdown(
            '<div class="field-group">'
            '<div class="field-group-title">Top SHAP Drivers</div>',
            unsafe_allow_html=True,
        )
        max_v = max(res["driver_values"].values()) if res["driver_values"] else 1
        for feat, val in res["driver_values"].items():
            pct = (val / max_v * 100) if max_v > 0 else 0
            label = feat.replace("_", " ").title()
            st.markdown(f"""
            <div class="driver-bar-wrap">
              <div style="display:flex;justify-content:space-between;">
                <span class="driver-label">{label}</span>
                <span style="font-size:11px;color:var(--muted);">{val:.4f}</span>
              </div>
              <div class="driver-bar-bg">
                <div class="driver-bar-fill" style="width:{pct:.0f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;color:var(--muted);margin-top:6px;">'
            'These features influenced your prediction most strongly.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Input summary
        st.markdown(
            '<div class="field-group">'
            '<div class="field-group-title">Input Summary</div>',
            unsafe_allow_html=True,
        )
        for k, v in inputs.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:13px;padding:6px 0;border-bottom:1px solid var(--border);'>"
                f"<span style='color:var(--muted);'>{k}</span>"
                f"<strong>{v}</strong></div>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: recommendations ─────────────────────────────────────────────
    with right:
        st.markdown(
            '<div style="margin-bottom:16px;">'
            '<div class="field-group-title" style="font-size:11px;letter-spacing:1.5px;">'
            'WHAT YOU SHOULD DO</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(_action_list(res["student_actions"]), unsafe_allow_html=True)

        st.markdown(
            '<div style="margin-top:28px;margin-bottom:16px;">'
            '<div class="field-group-title" style="font-size:11px;letter-spacing:1.5px;">'
            'WHAT YOUR LECTURER SHOULD DO</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _action_list(res["lecturer_actions"], dot_color="#EF9F27"),
            unsafe_allow_html=True,
        )

        if res["mode"] == "early":
            st.markdown("""
            <div class="info-box" style="margin-top:24px;">
              <div class="info-box-title">💡 Early warning mode active</div>
              <div class="info-box-body">
                No exam score was provided — the behavioural early warning model was used.
                Tick "I have my CA / exam score" and re-submit for a confirmed band.
              </div>
            </div>
            """, unsafe_allow_html=True)


def render() -> None:
    """Render the full student panel."""

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="panel-header">
      <div class="panel-icon-wrap student">🎓</div>
      <div>
        <div class="panel-title">Student Assessment</div>
        <div class="panel-desc">
          Type your values in the fields below.
          <strong>Exam score is optional</strong> — leave it unticked for an
          early warning based on study habits alone, or tick it for a confirmed
          post-CA band assignment.
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

    # ── Personal info ────────────────────────────────────────────────────────
    st.markdown('<div class="field-group"><div class="field-group-title">Personal Information</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.text_input("Student ID", placeholder="e.g. BSDS0322", key="s_id")
    with p2:
        st.number_input("Age", min_value=16, max_value=60, value=20, step=1, key="s_age")
    with p3:
        gender = st.selectbox(
            "Gender",
            ["Select...", "Male", "Female", "Prefer not to say"],
            key="s_gender",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Study habits ─────────────────────────────────────────────────────────
    st.markdown('<div class="field-group"><div class="field-group-title">Study Habits *</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0, max_value=24.0, value=3.0, step=0.5,
            help="Average hours spent studying each day this semester.",
            key="s_study",
        )
        sleep_hours = st.number_input(
            "Sleep hours per night",
            min_value=0.0, max_value=24.0, value=7.0, step=0.5,
            help="Average hours of sleep per night.",
            key="s_sleep",
        )
    with h2:
        attendance = st.number_input(
            "Attendance percentage (%)",
            min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            help="Percentage of classes attended this semester.",
            key="s_attend",
        )
        internet = st.number_input(
            "Daily non-study internet usage (hours)",
            min_value=0.0, max_value=24.0, value=2.0, step=0.5,
            help="Hours per day on social media, streaming, gaming, etc.",
            key="s_internet",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Exam score ───────────────────────────────────────────────────────────
    st.markdown('<div class="field-group"><div class="field-group-title">Assessment Score — Optional</div>', unsafe_allow_html=True)
    has_score = st.checkbox("I have my CA / exam score", value=False, key="s_has_score")
    exam_score = None
    if has_score:
        exam_score = st.number_input(
            "CA / Exam score (%)",
            min_value=0.0, max_value=100.0, value=55.0, step=0.5,
            key="s_score",
        )
        if   exam_score <= 34: preview = "🔴 Repeat zone (0–34%)"
        elif exam_score <= 44: preview = "🟡 Supplementary zone (35–44%)"
        elif exam_score <= 64: preview = "🟢 Good zone (45–64%)"
        else:                  preview = "✅ Excellent zone (65–100%)"
        st.markdown(
            f"<p style='font-size:13px;color:var(--muted);margin-top:2px;'>"
            f"Band preview: <strong>{preview}</strong></p>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ───────────────────────────────────────────────────────────────
    submit = st.button("Analyse My Performance →", key="btn_student_submit")

    if submit:
        if gender == "Select...":
            st.warning("Please select your gender to continue.")
        elif not load_models().get("loaded"):
            st.error("Model files not found. Check that app_model/ is next to app.py.")
        else:
            raw = {
                "study_hours_per_day":   study_hours,
                "attendance_percentage": attendance,
                "sleep_hours":           sleep_hours,
                "internet_usage_hours":  internet,
            }
            if has_score and exam_score is not None:
                raw["exam_score"] = exam_score

            mode = "full" if has_score and exam_score is not None else "early"

            with st.spinner("Running two-layer analysis…"):
                try:
                    result = predict_one(raw, mode=mode)
                    # Store result and inputs in session state
                    st.session_state["student_result"] = result
                    st.session_state["student_inputs"] = {
                        "Study hrs/day":  f"{study_hours} hrs",
                        "Attendance":     f"{attendance}%",
                        "Sleep":          f"{sleep_hours} hrs",
                        "Internet":       f"{internet} hrs",
                        **({"Exam score": f"{exam_score}%"} if has_score else {}),
                    }
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

    # ── Show results if they exist in session state ───────────────────────────
    if st.session_state.get("student_result"):
        render_results(
            st.session_state["student_result"],
            st.session_state.get("student_inputs", {}),
        )

    st.markdown('</div>', unsafe_allow_html=True)