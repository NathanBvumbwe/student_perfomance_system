"""
lecturer_view.py — Lecturer panel for ISPAIS.
Called by app.py when mode == 'lecturer'.
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictor import load_models, predict_batch


BAND_STYLES = {
    "Repeat":        ("🔴", "#fef2f2", "#b91c1c"),
    "Supplementary": ("🟡", "#fffbeb", "#b45309"),
    "Good":          ("🟢", "#edfaf3", "#166534"),
    "Excellent":     ("✅", "#f0fdf4", "#14532d"),
}

REQUIRED_COLS = [
    "study_hours_per_day", "attendance_percentage",
    "sleep_hours", "internet_usage_hours", "exam_score",
]


def _action_list(actions: list, dot_color: str = "var(--g400)") -> str:
    items = "".join(
        f'<li class="action-item">'
        f'<div class="action-dot" style="background:{dot_color};"></div>{a}'
        f'</li>'
        for a in actions
    )
    return f'<ul class="action-list">{items}</ul>'


def render_results(result_df: pd.DataFrame, show_atrisk: bool,
                   show_shap: bool, show_delivery: bool) -> None:
    n = len(result_df)
    st.success(f"✓ Analysed {n:,} students successfully.")

    # ── Band distribution cards ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-label" style="margin-top:24px;margin-bottom:12px;">'
        'Band Distribution</div>',
        unsafe_allow_html=True,
    )
    band_dist = (
        result_df["band"]
        .value_counts()
        .reindex(["Repeat", "Supplementary", "Good", "Excellent"], fill_value=0)
    )
    bc1, bc2, bc3, bc4 = st.columns(4)
    for col, (band, count) in zip([bc1, bc2, bc3, bc4], band_dist.items()):
        icon, bg, clr = BAND_STYLES[band]
        pct = count / n * 100 if n > 0 else 0
        col.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:20px;text-align:center;
                    border:1px solid {clr}22;">
          <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
          <div style="font-family:'DM Serif Display',serif;font-size:32px;
                      color:{clr};line-height:1;">{count}</div>
          <div style="font-size:12px;font-weight:600;color:{clr};margin:6px 0;">{band}</div>
          <div style="font-size:12px;color:#6b7280;">{pct:.1f}% of class</div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP driver chart ────────────────────────────────────────────────────
    if show_shap and "top_driver" in result_df.columns:
        st.markdown(
            '<div class="section-label" style="margin-top:32px;margin-bottom:8px;">'
            'Most Common Risk Driver per Student (SHAP)</div>',
            unsafe_allow_html=True,
        )
        driver_counts = result_df["top_driver"].value_counts()
        fig, ax = plt.subplots(figsize=(8, max(2.5, len(driver_counts) * 0.55)))
        colors = ["#22a050" if i == 0 else "#6dd496" for i in range(len(driver_counts))]
        driver_counts.plot.barh(ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Number of students", fontsize=11)
        ax.set_title("Top SHAP driver — cohort view", fontsize=12, pad=10)
        ax.invert_yaxis()
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── At-risk table ────────────────────────────────────────────────────────
    if show_atrisk:
        st.markdown(
            '<div class="section-label" style="margin-top:32px;margin-bottom:8px;">'
            'At-Risk Students — Repeat Band</div>',
            unsafe_allow_html=True,
        )
        at_risk = result_df[result_df["band"] == "Repeat"].reset_index(drop=True)
        if len(at_risk) == 0:
            st.success("No students are currently in the Repeat band.")
        else:
            st.dataframe(at_risk, use_container_width=True, hide_index=True)

    # Full results
    with st.expander("View full class results table"):
        st.dataframe(result_df, use_container_width=True, hide_index=True)

    # ── Delivery recommendations ─────────────────────────────────────────────
    if show_delivery:
        repeat_n   = band_dist.get("Repeat", 0)
        supp_n     = band_dist.get("Supplementary", 0)
        repeat_pct = repeat_n / n * 100
        risk_pct   = (repeat_n + supp_n) / n * 100

        st.markdown(
            '<div class="section-label" style="margin-top:32px;margin-bottom:12px;">'
            'Course Delivery Recommendations</div>',
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns(2)

        with d1:
            items = []
            if repeat_pct > 15:
                items.append(
                    f"⚠️ {repeat_pct:.1f}% of students are in the Repeat band "
                    f"— above the 15% warning threshold. Immediate cohort-level "
                    f"action is recommended."
                )
            items += [
                "Schedule one-on-one check-ins with all Repeat band students before Week 8.",
                "Provide foundational recap materials for high-failure topics.",
                "Consider peer mentoring: pair Excellent ↔ Repeat students.",
                "Escalate persistent at-risk cases to the academic advisor or HOD.",
            ]
            st.markdown(
                '<div class="field-group">'
                '<div class="field-group-title">For At-Risk Students</div>' +
                _action_list(items) +
                '</div>',
                unsafe_allow_html=True,
            )

        with d2:
            items2 = [
                (
                    f"{'⚠️ ' if risk_pct > 35 else ''}{risk_pct:.1f}% of students "
                    f"are in Repeat or Supplementary — "
                    f"{'review pacing and foundational coverage urgently.' if risk_pct > 35 else 'monitor closely.'}"
                ),
                "Increase formative assessments mid-semester to catch gaps early.",
                "Share anonymised cohort data with the HOD to benchmark across courses.",
                "The SHAP driver chart shows which behaviours most predict poor outcomes "
                "in your class — use it to focus your guidance.",
            ]
            st.markdown(
                '<div class="field-group">'
                '<div class="field-group-title">For Course Delivery</div>' +
                _action_list(items2, dot_color="#EF9F27") +
                '</div>',
                unsafe_allow_html=True,
            )


def render() -> None:
    """Render the full lecturer panel."""

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

    # ── Lecturer details ─────────────────────────────────────────────────────
    st.markdown('<div class="field-group"><div class="field-group-title">Lecturer Details</div>', unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.text_input("Full name", placeholder="e.g. Dr. E. Ngalande", key="l_name")
    with lc2:
        st.text_input("Course / Unit", placeholder="e.g. Data Structures", key="l_course")
    with lc3:
        lec_year = st.selectbox(
            "Year of study",
            ["Select...", "Year 1", "Year 2", "Year 3", "Year 4"],
            key="l_year",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────────
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
        key="l_file", label_visibility="collapsed",
    )

    # ── Options ──────────────────────────────────────────────────────────────
    st.markdown('<div class="field-group" style="margin-top:20px;"><div class="field-group-title">Analysis Options</div>', unsafe_allow_html=True)
    ao1, ao2, ao3 = st.columns(3)
    with ao1:
        show_atrisk   = st.checkbox("Flag at-risk students", value=True, key="l_atrisk")
    with ao2:
        show_shap     = st.checkbox("SHAP driver chart", value=True, key="l_shap")
    with ao3:
        show_delivery = st.checkbox("Delivery recommendations", value=True, key="l_delivery")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Submit ───────────────────────────────────────────────────────────────
    submit = st.button("Analyse Class →", key="btn_lecturer_submit")

    if submit:
        if lec_year == "Select...":
            st.warning("Please select the year of study.")
        elif uploaded_file is None:
            st.warning("Please upload a CSV file before continuing.")
        elif not load_models().get("loaded"):
            st.error("Model not loaded. Check that app_model/ is next to app.py.")
        else:
            try:
                df_up = pd.read_csv(uploaded_file)
            except Exception:
                st.error("Could not read CSV. Check file format.")
                st.stop()

            missing = [c for c in REQUIRED_COLS if c not in df_up.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                st.stop()

            with st.spinner(f"Running model on {len(df_up):,} students…"):
                try:
                    result_df = predict_batch(df_up)
                    st.session_state["lecturer_result_df"]   = result_df
                    st.session_state["lecturer_show_atrisk"] = show_atrisk
                    st.session_state["lecturer_show_shap"]   = show_shap
                    st.session_state["lecturer_show_delivery"] = show_delivery
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")
                    st.stop()

    # ── Show results if they exist ────────────────────────────────────────────
    if st.session_state.get("lecturer_result_df") is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-label" style="margin-top:8px;">Cohort Analysis</div>
        <div class="section-title" style="margin-bottom:24px;">Class Performance Report</div>
        """, unsafe_allow_html=True)

        render_results(
            st.session_state["lecturer_result_df"],
            st.session_state.get("lecturer_show_atrisk", True),
            st.session_state.get("lecturer_show_shap", True),
            st.session_state.get("lecturer_show_delivery", True),
        )

    st.markdown('</div>', unsafe_allow_html=True)