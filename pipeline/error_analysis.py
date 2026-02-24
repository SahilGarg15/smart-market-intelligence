"""
pipeline/error_analysis.py

Explainability & Error Analysis — v2
Generates:
  1. Feature importance bar chart  (top 10 features)
  2. Residual distribution histogram
  3. Residual summary written back into metadata.json

Why bother?
  A trained model is a black box until you explain it. Feature importance
  answers the question "what does the model actually use?" — essential for
  stakeholder trust and for catching data leakage.
  Residual analysis reveals *where* the model fails (e.g., systematic
  under-forecasting on Mondays) which guides the next iteration.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

ANALYSIS_DIR = "analysis"
METADATA_PATH = "models/metadata.json"


# ---------------------------------------------------------------------------
# 1. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, top_n: int = 10) -> str:
    """
    Generate and save a horizontal bar chart of the top-N feature importances.

    Only works for tree-based models (RandomForest, GradientBoosting).
    For linear models we fall back to coefficient magnitudes.

    Returns the saved file path.
    """
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Feature Importance (Gini)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        label = "Coefficient Magnitude (|coef|)"
    else:
        print("  [Warning] Model has no feature_importances_ or coef_. Skipping plot.")
        return ""

    # Pair and sort
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:top_n]
    names  = [p[0] for p in top_pairs]
    values = [p[1] for p in top_pairs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names[::-1], values[::-1], color="#2196F3", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_xlabel(label)
    ax.set_title(f"Top {top_n} Features — Smart Market Intelligence v2")
    ax.set_xlim(0, max(values) * 1.2)
    fig.tight_layout()

    out_path = os.path.join(ANALYSIS_DIR, "feature_importance.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Feature importance chart saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. Residual analysis
# ---------------------------------------------------------------------------

def analyse_residuals(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute residual statistics and generate a histogram.

    Residual = actual − predicted.
      • Mean ≈ 0  → model is unbiased on average
      • Large std → model struggles on volatile days (e.g., December surge)
      • % > 20%   → proportion of days with >20% miss (operational risk)

    Returns a dict suitable for embedding in metadata.json.
    """
    residuals = y_test - y_pred
    abs_pct_errors = np.abs(residuals / np.where(y_test != 0, y_test, np.nan)) * 100

    summary = {
        "mean_residual":        round(float(np.mean(residuals)),     2),
        "std_residual":         round(float(np.std(residuals)),      2),
        "max_residual":         round(float(np.max(residuals)),      2),
        "min_residual":         round(float(np.min(residuals)),      2),
        "pct_over_20pct_error": round(float(np.nanmean(abs_pct_errors > 20) * 100), 2),
    }

    print(f"\n  Residual summary:")
    for k, v in summary.items():
        print(f"    {k:<30}: {v}")

    # Histogram
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=30, color="#FF5722", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero line")
    ax.axvline(np.mean(residuals), color="gold", linestyle="-",
               linewidth=1.5, label=f"Mean={np.mean(residuals):,.0f}")
    ax.set_xlabel("Residual  (Actual − Predicted)  £")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution — Sales Forecast")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(ANALYSIS_DIR, "residual_distribution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Residual histogram saved → {out_path}")

    return summary


# ---------------------------------------------------------------------------
# 3. Master function — run both and patch metadata.json
# ---------------------------------------------------------------------------

def run_error_analysis(
    model,
    y_test:       np.ndarray,
    y_pred:       np.ndarray,
    feature_names: list,
) -> None:
    """
    Run feature-importance + residual analysis and patch metadata.json.
    Called from ml_engine.py after model selection is complete.
    """
    print("=" * 60)
    print("PIPELINE STEP 6 — Explainability & Error Analysis")
    print("=" * 60)

    plot_feature_importance(model, feature_names)
    residual_summary = analyse_residuals(y_test, y_pred)

    # Patch the existing metadata.json with residual results
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as fh:
            meta = json.load(fh)
        meta["residual_analysis"] = residual_summary
        with open(METADATA_PATH, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"\n  ✓ Residual summary added to {METADATA_PATH}")

    print("\n[Error Analysis] Complete ✓\n")
