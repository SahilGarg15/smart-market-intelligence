"""
pipeline/model_selection.py

Model Selection — v2
Pick the best model from benchmark results, train its final version on the
full dataset, persist the artefacts, and write a rich metadata.json.

Decision rule:  best_model = argmin(MAPE)

Why MAPE as the selection criterion?
  • RMSE is sensitive to the scale of Sales values, which vary by country
    and promotion period — making cross-run comparisons misleading.
  • R² can be inflated when the mean of y_test is far from 0.
  • MAPE is scale-independent and translates directly to a business statement:
    "Forecasts are off by X % on average." That's what stakeholders care about.
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_by_name(name: str, best_params: dict | None):
    """
    Instantiate a model from its string name.
    If name == 'RandomForest' and best_params is provided, use tuned params.
    """
    if name == "Linear":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "RandomForest":
        if best_params:
            return RandomForestRegressor(
                **best_params, random_state=42, n_jobs=-1
            )
        return RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_split=5, random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Unknown model name: {name}")


def _compute_holdout_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate the final model on the last TimeSeriesSplit fold (the 'holdout').

    Why last fold only (not average)?
      We generate the feature-importance and residual plots on a single
      contiguous test window — averaging across folds would prevent us from
      plotting a coherent residual time-series.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    _, test_idx = splits[-1]

    X_test, y_test = X[test_idx], y[test_idx]
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2":   round(r2,   4),
    }, y_test, y_pred


# ---------------------------------------------------------------------------
# Main selection + persistence function
# ---------------------------------------------------------------------------

def select_and_save(
    benchmark_results: dict,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    best_params: dict | None = None,
    model_dir: str = "models",
) -> dict:
    """
    1. Pick the best model by MAPE.
    2. Retrain it on the FULL dataset with the best hyperparameters.
    3. Save model + scaler artefacts.
    4. Write metadata.json with version, metrics, params, residual summary.

    Returns
    -------
    dict with keys:  model_name, model, y_test, y_pred, feature_names, metrics
    (used by error_analysis.py to generate plots without retraining)
    """
    print("=" * 60)
    print("PIPELINE STEP 5 — Model Selection")
    print("=" * 60)

    # Pick by MAPE
    best_name = min(benchmark_results, key=lambda k: benchmark_results[k]["MAPE"])
    print(f"\nBenchmark summary:")
    for name, m in benchmark_results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<15} RMSE={m['RMSE']:>8,.2f}  "
              f"MAPE={m['MAPE']:>6.2f}%  R²={m['R2']:.4f}{marker}")

    # Build and train final model
    params = best_params if best_name == "RandomForest" else None
    model  = _build_model_by_name(best_name, params)
    model.fit(X, y)
    print(f"\n✓ Best model '{best_name}' retrained on full dataset ({len(X)} samples)")

    # Holdout evaluation (last fold)
    metrics, y_test, y_pred = _compute_holdout_metrics(model, X, y)
    print(f"\nHoldout metrics (last fold):")
    print(f"  RMSE: £{metrics['RMSE']:,.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²:   {metrics['R2']:.4f}")

    # Persist artefacts
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved → {model_path}")

    # Build metadata
    metadata = {
        "version":       "v2",
        "model_name":    best_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": feature_names,
        "n_features":    len(feature_names),
        "n_training_samples": len(X),
        "metrics": metrics,
        "hyperparameters": (
            params if params else (
                {"alpha": 1.0} if best_name == "Ridge" else {}
            )
        ),
        "benchmark_comparison": benchmark_results,
    }

    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"✓ Metadata saved → {meta_path}\n")

    return {
        "model_name":    best_name,
        "model":         model,
        "y_test":        y_test,
        "y_pred":        y_pred,
        "feature_names": feature_names,
        "metrics":       metrics,
    }
