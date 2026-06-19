"""
predictor.py — Model loading and prediction logic for ISPAIS.
Imported by student_view.py and lecturer_view.py.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "app_model")

# ── Band definitions ─────────────────────────────────────────────────────────
BAND_ORDER  = ["Repeat", "Supplementary", "Good", "Excellent"]
BAND_TO_INT = {b: i for i, b in enumerate(BAND_ORDER)}
INT_TO_BAND = {i: b for i, b in enumerate(BAND_ORDER)}

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


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load and cache all model artefacts. Called once at startup."""
    required = [
        "rf_full_model.pkl", "rf_behav_model.pkl",
        "scaler_full.pkl",   "scaler_behav.pkl",
        "all_feature_names.pkl", "behav_feature_names.pkl",
        "band_order.pkl",    "band_to_int.pkl",  "int_to_band.pkl",
    ]
    missing = [f for f in required if not os.path.exists(f"{MODEL_DIR}/{f}")]
    if missing:
        return {"loaded": False, "error": f"Missing files: {missing}"}

    try:
        feat_maxes_path = f"{MODEL_DIR}/feat_maxes.pkl"
        feat_maxes = (
            joblib.load(feat_maxes_path)
            if os.path.exists(feat_maxes_path)
            else {"study_hours_per_day": 12.0, "internet_usage_hours": 12.0}
        )
        return {
            "loaded":        True,
            "rf_full":       joblib.load(f"{MODEL_DIR}/rf_full_model.pkl"),
            "rf_behav":      joblib.load(f"{MODEL_DIR}/rf_behav_model.pkl"),
            "all_features":  joblib.load(f"{MODEL_DIR}/all_feature_names.pkl"),
            "behav_features":joblib.load(f"{MODEL_DIR}/behav_feature_names.pkl"),
            "feat_maxes":    feat_maxes,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}


def assign_mzuni_band(score: float) -> str:
    """Layer 1 rule engine — always matches Mzuni policy."""
    if   score <= 34: return "Repeat"
    elif score <= 44: return "Supplementary"
    elif score <= 64: return "Good"
    else:             return "Excellent"


def build_features(raw: dict, feat_maxes: dict) -> dict:
    """Add engineered features — must match notebook exactly."""
    d = raw.copy()
    d["study_x_attendance"] = (
        d["study_hours_per_day"] * d["attendance_percentage"] / 100
    )
    d["distraction_ratio"] = min(
        d["internet_usage_hours"] / (d["study_hours_per_day"] + 0.1), 10
    )
    sn  = d["study_hours_per_day"]   / feat_maxes["study_hours_per_day"]
    an  = d["attendance_percentage"] / 100
    inp = d["internet_usage_hours"]  / feat_maxes["internet_usage_hours"]
    d["risk_score"] = (1 - sn)*0.4 + (1 - an)*0.4 + inp*0.2
    return d


def _normalise_shap(sv, n: int) -> np.ndarray:
    if sv.ndim == 3 and sv.shape[0] == n:
        return sv
    if sv.ndim == 3 and sv.shape[0] == 4:
        return sv.transpose(1, 2, 0)
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")


def predict_one(raw_data: dict, mode: str = "full") -> dict:
    """
    Run two-layer prediction for one student.

    Parameters
    ----------
    raw_data : dict
        Keys: study_hours_per_day, attendance_percentage, sleep_hours,
              internet_usage_hours. Plus exam_score if mode='full'.
    mode : 'full' | 'early'

    Returns
    -------
    dict with band, confidence, probabilities, top_drivers, recommendations.
    """
    m    = load_models()
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
    pred_int   = int(model.predict(input_df)[0])
    pred_band  = INT_TO_BAND[pred_int]
    final_band = band if mode == "full" else pred_band
    confidence = float(proba[BAND_TO_INT[final_band]]) * 100

    # SHAP
    exp    = shap.TreeExplainer(model)
    sv     = _normalise_shap(np.array(exp.shap_values(input_df)), 1)
    sv_cls = sv[0, :, BAND_TO_INT[final_band]]
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
            for i, b in enumerate(BAND_ORDER)
        },
        "top_drivers":      drivers.index.tolist(),
        "driver_values":    {k: round(float(v), 4) for k, v in drivers.items()},
        "student_actions":  RECOMMENDATIONS[final_band]["student"],
        "lecturer_actions": RECOMMENDATIONS[final_band]["lecturer"],
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run full-mode prediction on every row of a DataFrame.
    Returns a results DataFrame with band, confidence, top_driver columns.
    """
    records = []
    for _, row in df.iterrows():
        raw = {
            "study_hours_per_day":   float(row["study_hours_per_day"]),
            "attendance_percentage": float(row["attendance_percentage"]),
            "sleep_hours":           float(row["sleep_hours"]),
            "internet_usage_hours":  float(row["internet_usage_hours"]),
            "exam_score":            float(row["exam_score"]),
        }
        res = predict_one(raw, mode="full")
        records.append({
            "student_id":  str(row.get("student_id", "—")),
            "exam_score":  raw["exam_score"],
            "band":        res["band"],
            "confidence":  res["confidence"],
            "top_driver":  (
                res["top_drivers"][0].replace("_", " ").title()
                if res["top_drivers"] else "—"
            ),
        })
    return pd.DataFrame(records)