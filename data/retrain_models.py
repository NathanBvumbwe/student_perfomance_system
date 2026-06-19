"""
BSDS0322 — ISPAIS Model Retraining Script
==========================================
Updated Mzuni grading bands (per SRS revision):
  Fail          :  0 – 34%
  Supplementable: 35 – 49%
  Pass          : 50 – 64%
  Excellent     : 65 – 100%

Run this script from the project root with your venv active:
    python retrain_models.py

All artefacts are saved to  app_model_v2/
The old app_model/ folder is NOT touched.
After verifying the new models, rename the folders as needed.
"""

import os, time, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score,
    accuracy_score, mean_squared_error, roc_curve, auc
)
import shap

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH  = "data/student_performance_dataset_bigger.csv"
MODEL_DIR  = "app_model_v2/"
REPORT_DIR = "report_assets_v2/"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

print("=" * 65)
print("ISPAIS — Model Retraining with Updated Mzuni Bands")
print("=" * 65)

# ── Updated band definitions ──────────────────────────────────────────────────
# Single source of truth — change thresholds here only.
BAND_ORDER  = ["Fail", "Supplementable", "Pass", "Excellent"]
BAND_TO_INT = {b: i for i, b in enumerate(BAND_ORDER)}
INT_TO_BAND = {i: b for i, b in enumerate(BAND_ORDER)}

BAND_THRESHOLDS = {
    "Fail":          (0,  34),
    "Supplementable":(35, 49),
    "Pass":          (50, 64),
    "Excellent":     (65, 100),
}

PALETTE = {
    "Fail":          "#E24B4A",
    "Supplementable":"#EF9F27",
    "Pass":          "#1D9E75",
    "Excellent":     "#534AB7",
}

def assign_mzuni_band(score: float) -> str:
    """
    Layer 1 rule engine — updated Mzuni grading bands.
    Deterministic: always policy-correct.
    Note: The Dean's List condition (avg ≥ 70, no course < 65)
    cannot be computed from a single exam_score; the 65% threshold
    is used as the Excellent boundary for single-course prediction.
    """
    if   score <= 34: return "Fail"
    elif score <= 49: return "Supplementable"
    elif score <= 64: return "Pass"
    else:             return "Excellent"

# ── Stage 1: Load data ─────────────────────────────────────────────────────────
print("\n[1/8] Loading dataset...")
t0 = time.time()
df = pd.read_csv(DATA_PATH).reset_index(drop=True)
print(f"      Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"      Missing: {df.isnull().sum().sum()} values")
print(f"      Loaded in {time.time()-t0:.1f}s")

# ── Stage 2: Layer 1 — assign updated bands ───────────────────────────────────
print("\n[2/8] Applying Layer 1 — updated band assignment...")
df["mzuni_band"]     = df["exam_score"].apply(assign_mzuni_band)
df["mzuni_band_int"] = df["mzuni_band"].map(BAND_TO_INT)

print("      Band distribution (Layer 1):")
band_counts = df["mzuni_band"].value_counts().reindex(BAND_ORDER)
for band, count in band_counts.items():
    pct        = count / len(df) * 100
    mean_score = df[df["mzuni_band"] == band]["exam_score"].mean()
    print(f"        {band:<15}: {count:>8,}  ({pct:5.1f}%)  "
          f"mean score = {mean_score:.1f}%")

# Band distribution plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(BAND_ORDER, band_counts.values,
            color=[PALETTE[b] for b in BAND_ORDER], edgecolor="white")
axes[0].set_title("Band Distribution (Updated Thresholds)")
axes[0].set_ylabel("Student Count")
for i, v in enumerate(band_counts.values):
    axes[0].text(i, v + 1000, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=9)

for band in BAND_ORDER:
    subset = df[df["mzuni_band"] == band]["exam_score"]
    axes[1].hist(subset, bins=50, alpha=0.75, label=band,
                 color=PALETTE[band], edgecolor="none")
for thresh, lbl in [(34, "34%"), (49, "49%"), (64, "64%")]:
    axes[1].axvline(thresh, color="black", linestyle="--", linewidth=1)
    axes[1].text(thresh + 0.5, axes[1].get_ylim()[1] * 0.9,
                 lbl, fontsize=8)
axes[1].set_title("Exam Score by Updated Mzuni Band")
axes[1].set_xlabel("Exam Score (%)")
axes[1].legend()
plt.suptitle("Stage 2 — Updated Band Validation", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}01_band_distribution.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}01_band_distribution.png")

# ── Stage 3: Feature engineering ─────────────────────────────────────────────
print("\n[3/8] Engineering features...")

RAW_BEHAV = [
    "study_hours_per_day",
    "attendance_percentage",
    "sleep_hours",
    "internet_usage_hours",
]

# Engineered features
df["study_x_attendance"] = (
    df["study_hours_per_day"] * df["attendance_percentage"] / 100
)
df["distraction_ratio"] = (
    df["internet_usage_hours"] / (df["study_hours_per_day"] + 0.1)
).clip(upper=10)

# Store training-set max values for inference-time normalisation
FEAT_MAXES = {
    "study_hours_per_day":  float(df["study_hours_per_day"].max()),
    "internet_usage_hours": float(df["internet_usage_hours"].max()),
}
study_norm    = df["study_hours_per_day"]   / FEAT_MAXES["study_hours_per_day"]
attend_norm   = df["attendance_percentage"] / 100
internet_norm = df["internet_usage_hours"]  / FEAT_MAXES["internet_usage_hours"]
df["risk_score"] = (
    (1 - study_norm) * 0.4 +
    (1 - attend_norm) * 0.4 +
    internet_norm * 0.2
)

ENGINEERED     = ["study_x_attendance", "distraction_ratio", "risk_score"]
BEHAV_FEATURES = RAW_BEHAV + ENGINEERED
ALL_FEATURES   = BEHAV_FEATURES + ["exam_score"]

print(f"      Behavioural features ({len(BEHAV_FEATURES)}): {BEHAV_FEATURES}")
print(f"      Full features        ({len(ALL_FEATURES)}): {ALL_FEATURES}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(df[ALL_FEATURES].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax, linewidths=0.5, square=True)
ax.set_title("Stage 3 — Feature Correlation Heatmap (Updated)")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}02_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}02_correlation_heatmap.png")

# ── Stage 4: Save scalers and artefacts ──────────────────────────────────────
print("\n[4/8] Fitting scalers and saving base artefacts...")

scaler_full  = StandardScaler().fit(df[ALL_FEATURES])
scaler_behav = StandardScaler().fit(df[BEHAV_FEATURES])

artefacts = {
    "scaler_full.pkl":          scaler_full,
    "scaler_behav.pkl":         scaler_behav,
    "all_feature_names.pkl":    ALL_FEATURES,
    "behav_feature_names.pkl":  BEHAV_FEATURES,
    "band_order.pkl":           BAND_ORDER,
    "band_to_int.pkl":          BAND_TO_INT,
    "int_to_band.pkl":          INT_TO_BAND,
    "feat_maxes.pkl":           FEAT_MAXES,
    "rule_engine.pkl":          assign_mzuni_band,
}
for fname, obj in artefacts.items():
    joblib.dump(obj, f"{MODEL_DIR}{fname}")
    print(f"      Saved: {MODEL_DIR}{fname}")

# ── Stage 5: Train / test split ───────────────────────────────────────────────
print("\n[5/8] Splitting data (80/20 stratified)...")
y = df["mzuni_band_int"]

X_all_tr, X_all_te, y_train, y_test = train_test_split(
    df[ALL_FEATURES], y,
    test_size=0.2, random_state=42, stratify=y
)
X_beh_tr = df[BEHAV_FEATURES].loc[X_all_tr.index]
X_beh_te = df[BEHAV_FEATURES].loc[X_all_te.index]

print(f"      Train: {len(X_all_tr):,}  |  Test: {len(X_all_te):,}")
print("      Train class distribution:")
for i, b in enumerate(BAND_ORDER):
    n = (y_train == i).sum()
    print(f"        {b:<15}: {n:>8,}  ({n/len(y_train)*100:.1f}%)")

# ── Stage 6: Full model ───────────────────────────────────────────────────────
print("\n[6/8] Training full model (all features including exam_score)...")
t0 = time.time()

RF_PARAMS_FULL = dict(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
rf_full = RandomForestClassifier(**RF_PARAMS_FULL)
rf_full.fit(X_all_tr, y_train)

cv_full = cross_val_score(
    rf_full, X_all_tr, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1_macro", n_jobs=-1
)
y_pred_full = rf_full.predict(X_all_te)

print(f"      5-Fold CV F1 (macro): {cv_full.mean():.4f} ± {cv_full.std():.4f}")
print(f"      Test set report:")
print(classification_report(y_test, y_pred_full, target_names=BAND_ORDER))

# Confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred_full),
    display_labels=BAND_ORDER
).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Stage 6 — Full Model Confusion Matrix (Updated Bands)")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}03a_confusion_full.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}03a_confusion_full.png")

joblib.dump(rf_full, f"{MODEL_DIR}rf_full_model.pkl")
size_kb = os.path.getsize(f"{MODEL_DIR}rf_full_model.pkl") / 1024
print(f"      Saved: {MODEL_DIR}rf_full_model.pkl  ({size_kb:.0f} KB)")
print(f"      Training time: {time.time()-t0:.1f}s")

# ── Stage 7: Behavioural model — GridSearchCV + compress ──────────────────────
print("\n[7/8] Training behavioural model (early warning — no exam score)...")

# 7a — Baseline
print("      7a. Baseline (no tuning)...")
rf_base = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=10,
    max_features="sqrt", random_state=42, n_jobs=-1,
    class_weight="balanced"
)
rf_base.fit(X_beh_tr, y_train)
y_pred_base = rf_base.predict(X_beh_te)
f1_base = f1_score(y_test, y_pred_base, average="macro")
rep_base = classification_report(
    y_test, y_pred_base, target_names=BAND_ORDER, output_dict=True
)
recall_fail_base = rep_base["Fail"]["recall"]
print(f"      Baseline F1 (macro): {f1_base:.4f}  "
      f"| Fail recall: {recall_fail_base:.4f}")

# 7b — GridSearchCV on 20k subsample
print("      7b. GridSearchCV (~5 mins)...")
t0 = time.time()
gs_idx = X_beh_tr.sample(20_000, random_state=42).index
X_gs   = X_beh_tr.loc[gs_idx]
y_gs   = y_train.loc[gs_idx]

param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [10, 15, 20],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     ["sqrt", "log2"],
}
gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1,
                           class_weight="balanced"),
    param_grid,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
)
gs.fit(X_gs, y_gs)
best_params = gs.best_params_
print(f"      Best params: {best_params}")
print(f"      Best CV F1 : {gs.best_score_:.4f}")
print(f"      Search time: {time.time()-t0:.1f}s")

# 7c — Retrain on full training set with best params
print("      7c. Retraining on full training set...")
rf_tuned = RandomForestClassifier(
    **best_params, random_state=42, n_jobs=-1,
    class_weight="balanced"
)
rf_tuned.fit(X_beh_tr, y_train)
y_pred_tuned = rf_tuned.predict(X_beh_te)
f1_tuned = f1_score(y_test, y_pred_tuned, average="macro")
rep_tuned = classification_report(
    y_test, y_pred_tuned, target_names=BAND_ORDER, output_dict=True
)
recall_fail_tuned = rep_tuned["Fail"]["recall"]
print(f"      Tuned F1 (macro): {f1_tuned:.4f}  "
      f"| Fail recall: {recall_fail_tuned:.4f}")

# 7d — Compression: find smallest model keeping ≥ 95% of tuned F1
print("      7d. Compressing model...")
target_f1    = f1_tuned * 0.95
best_n_trees = best_params["n_estimators"]  # fallback
best_kb      = None

for n_trees in [25, 50, 75, 100]:
    rf_s = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf_s.fit(X_beh_tr, y_train)
    f1_s = f1_score(y_test, rf_s.predict(X_beh_te), average="macro")
    tmp  = f"{MODEL_DIR}_tmp.pkl"
    joblib.dump(rf_s, tmp)
    kb   = os.path.getsize(tmp) / 1024
    os.remove(tmp)
    ok   = f1_s >= target_f1
    print(f"        {n_trees:>3} trees  F1={f1_s:.4f}  {kb:>8.0f} KB  "
          f"{'✓ meets target' if ok else '✗'}")
    if ok and (best_kb is None or kb < best_kb):
        best_n_trees = n_trees
        best_kb      = kb

print(f"      Selected: {best_n_trees} trees ({best_kb:.0f} KB)")

# 7e — Final compressed behavioural model
print("      7e. Training final compressed model...")
rf_behav_final = RandomForestClassifier(
    n_estimators=best_n_trees,
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    random_state=42, n_jobs=-1, class_weight="balanced"
)
rf_behav_final.fit(X_beh_tr, y_train)
y_pred_final = rf_behav_final.predict(X_beh_te)
y_prob_final = rf_behav_final.predict_proba(X_beh_te) # Added for ROC

f1_final     = f1_score(y_test, y_pred_final, average="macro")
rep_final    = classification_report(
    y_test, y_pred_final, target_names=BAND_ORDER, output_dict=True
)
recall_fail_final = rep_final["Fail"]["recall"]

joblib.dump(rf_behav_final, f"{MODEL_DIR}rf_behav_model.pkl")
size_final = os.path.getsize(f"{MODEL_DIR}rf_behav_model.pkl") / 1024
print(f"      Final F1 (macro): {f1_final:.4f}  "
      f"| Fail recall: {recall_fail_final:.4f}")
print(f"      Model size: {size_final:.0f} KB  ({size_final/1024:.1f} MB)")
print()
print("      Full per-class report (final behavioural model):")
print(classification_report(y_test, y_pred_final, target_names=BAND_ORDER))

# ── NEW BLOCK: Comprehensive Evaluation & ROC Curve ───────────────────────────
print("\n      --- Comprehensive Model Evaluation Metrics ---")
acc_final = accuracy_score(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score (Macro)", "RMSE (Ordinal)"],
    "Score": [round(acc_final, 4), round(f1_final, 4), round(rmse_final, 4)]
})
print(metrics_df.to_string(index=False))

# Calculate AUC for each class using OvR (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_final[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

print("\n      --- AUC Scores per Band ---")
auc_df = pd.DataFrame({
    "Band": BAND_ORDER,
    "AUC": [round(roc_auc[i], 4) for i in range(n_classes)]
})
print(auc_df.to_string(index=False))

# Plot ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
colors = [PALETTE[b] for b in BAND_ORDER]
for i, color, band in zip(range(n_classes), colors, BAND_ORDER):
    ax.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {band} (AUC = {roc_auc[i]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Stage 7 — Multiclass ROC Curve (Behavioural Model)')
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}03c_roc_curve_behav.png", bbox_inches="tight")
plt.close()
print(f"\n      Saved: {REPORT_DIR}03c_roc_curve_behav.png")
# ──────────────────────────────────────────────────────────────────────────────

# Confusion matrices: baseline vs final
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, preds, title in [
    (axes[0], y_pred_base,  "Baseline (imbalanced, no tuning)"),
    (axes[1], y_pred_final, f"Final ({best_n_trees} trees, compressed)"),
]:
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds),
        display_labels=BAND_ORDER
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
plt.suptitle("Stage 7 — Behavioural Model: Baseline vs Final (Updated Bands)",
             fontsize=13)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}03b_confusion_behav.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}03b_confusion_behav.png")

# ── Stage 8: SHAP explainability ─────────────────────────────────────────────
print("\n[8/8] Running SHAP explainability...")

def normalise_shap(sv, n):
    if sv.ndim == 3 and sv.shape[0] == n:  return sv
    if sv.ndim == 3 and sv.shape[0] == 4:  return sv.transpose(1, 2, 0)
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

shap_sample_all  = X_all_te.sample(3000, random_state=42)
shap_sample_beh  = X_beh_te.loc[shap_sample_all.index]

exp_full  = shap.TreeExplainer(rf_full)
exp_behav = shap.TreeExplainer(rf_behav_final)

sv_full  = normalise_shap(
    np.array(exp_full.shap_values(shap_sample_all)), 3000
)
sv_behav = normalise_shap(
    np.array(exp_behav.shap_values(shap_sample_beh)), 3000
)

# Global importance
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plt.sca(axes[0])
shap.summary_plot(sv_full, shap_sample_all,
                  class_names=BAND_ORDER, plot_type="bar", show=False)
axes[0].set_title("Full model (post-CA) — Updated Bands")
plt.sca(axes[1])
shap.summary_plot(sv_behav, shap_sample_beh,
                  class_names=BAND_ORDER, plot_type="bar", show=False)
axes[1].set_title("Behavioural model (early warning) — Updated Bands")
plt.suptitle("Stage 8 — SHAP Global Feature Importance", fontsize=13)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}04a_shap_global.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}04a_shap_global.png")

# Beeswarm for Fail band (most critical)
for class_idx, band_name, fname in [
    (0, "Fail",          "04b_shap_fail.png"),
    (1, "Supplementable","04c_shap_supplementable.png"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plt.sca(axes[0])
    shap.summary_plot(sv_full[:, :, class_idx],  shap_sample_all,  show=False)
    axes[0].set_title(f"Full model — {band_name}")
    plt.sca(axes[1])
    shap.summary_plot(sv_behav[:, :, class_idx], shap_sample_beh, show=False)
    axes[1].set_title(f"Behavioural — {band_name}")
    plt.suptitle(f"Stage 8 — SHAP Beeswarm: {band_name} Band", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}{fname}", bbox_inches="tight")
    plt.close()
    print(f"      Saved: {REPORT_DIR}{fname}")

# Feature importances
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, model, feats, title in [
    (axes[0], rf_full,        ALL_FEATURES,   "Full model"),
    (axes[1], rf_behav_final, BEHAV_FEATURES, "Behavioural model"),
]:
    imp = pd.Series(model.feature_importances_,
                    index=feats).sort_values()
    imp.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Gini importance")
plt.suptitle("Stage 8 — Feature Importances (Updated Bands)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}04d_rf_importances.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}04d_rf_importances.png")

# ── Progression summary table ─────────────────────────────────────────────────
progression = pd.DataFrame([
    {"Model": "Baseline (raw features, no tuning)",
     "F1 macro": round(f1_base,  4),
     "Fail recall": round(recall_fail_base,  4),
     "Notes": "Starting point"},
    {"Model": "Tuned (GridSearchCV)",
     "F1 macro": round(f1_tuned, 4),
     "Fail recall": round(recall_fail_tuned, 4),
     "Notes": str(best_params)},
    {"Model": f"Final compressed ({best_n_trees} trees)",
     "F1 macro": round(f1_final, 4),
     "Fail recall": round(recall_fail_final, 4),
     "Notes": f"{size_final:.0f} KB — app-ready"},
])
progression.to_csv(f"{REPORT_DIR}05_model_progression.csv", index=False)
print(f"\n      Saved: {REPORT_DIR}05_model_progression.csv")

# ── Band profile heatmap ──────────────────────────────────────────────────────
profile = (
    df.groupby("mzuni_band")[ALL_FEATURES]
    .mean().round(2).reindex(BAND_ORDER)
)
profile["count"]      = df["mzuni_band"].value_counts().reindex(BAND_ORDER)
profile["% of total"] = (profile["count"] / len(df) * 100).round(1)
profile_norm = (
    (profile[ALL_FEATURES] - profile[ALL_FEATURES].min()) /
    (profile[ALL_FEATURES].max() - profile[ALL_FEATURES].min())
)
fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(profile_norm, annot=profile[ALL_FEATURES],
            fmt=".2f", cmap="RdYlGn", linewidths=0.5,
            ax=ax, vmin=0, vmax=1)
ax.set_title("Band Profile Heatmap — Updated Mzuni Bands")
plt.tight_layout()
plt.savefig(f"{REPORT_DIR}06_band_heatmap.png", bbox_inches="tight")
plt.close()
print(f"      Saved: {REPORT_DIR}06_band_heatmap.png")

# ── Final model inventory ─────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RETRAINING COMPLETE — app_model_v2/ contents:")
print("=" * 65)
for fname in sorted(os.listdir(MODEL_DIR)):
    if fname.startswith("_"): continue
    kb = os.path.getsize(f"{MODEL_DIR}{fname}") / 1024
    print(f"  {fname:<40} {kb:>8.0f} KB")

print("\nModel performance summary:")
print(f"  Full model CV F1 (macro)            : {cv_full.mean():.4f}")
print(f"  Behav baseline F1 (macro)           : {f1_base:.4f}")
print(f"  Behav final F1 (macro, compressed)  : {f1_final:.4f}")
print(f"  Behav Fail recall                   : {recall_fail_final:.4f}")

print("\nUpdated Mzuni band thresholds:")
for band, (lo, hi) in BAND_THRESHOLDS.items():
    n   = band_counts[band]
    pct = n / len(df) * 100
    print(f"  {band:<15}: {lo:>3}–{hi:>3}%  →  {n:>8,} students ({pct:.1f}%)")

print("\nNext steps:")
print("  1. Review report_assets_v2/ plots for sanity checks")
print("  2. Rename app_model_v2/ → app_model/ when satisfied")
print("  3. Update RECOMMENDATIONS dict in predictor.py")
print("     (change 'Repeat'→'Fail', 'Supplementary'→'Supplementable',")
print("      'Good'→'Pass', 'Excellent' stays)")
print("  4. Run: streamlit run app.py  and test both modes")
print("=" * 65)