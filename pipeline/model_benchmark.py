"""
pipeline/model_benchmark.py

Model Benchmarking Engine — v2
Train and evaluate multiple regression models under identical TimeSeriesSplit
cross-validation, then return a uniform metrics dictionary.

Why benchmark instead of just using Random Forest?
  "Best" is empirical, not assumed. For the Online Retail dataset, a Ridge
  regression with good features might outperform a Complex RF. Benchmarking
  lets the data decide — an important engineering discipline.

Models evaluated:
  • LinearRegression   — baseline; fastest; interpretable
  • Ridge              — L2-regularised linear; handles correlated features
  • RandomForestRegressor — non-linear; robust to outliers
"""

import json
import os
import time

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Guard against division-by-zero on rare zero-sales days.
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def _get_models() -> dict:
    """
    Returns fresh model instances.

    Keeping parameters modest for a 305-row dataset:
      RF: n_estimators=100, max_depth=10 avoids heavy overfitting on small data.
    """
    return {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def benchmark_models(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Evaluate every model in the catalogue using TimeSeriesSplit.

    Why TimeSeriesSplit (not random KFold)?
      Random splitting would allow future data to 'leak' into training folds —
      the model would learn patterns it shouldn't know at prediction time.
      TimeSeriesSplit always trains on the past and tests on the future.

    Metrics averaged across all folds:
      RMSE  — error in the same unit as Sales (£); penalises large misses
      MAPE  — percentage error; scale-invariant; board-level KPI
      R²    — proportion of variance explained; 1.0 = perfect fit

    Returns
    -------
    dict  {model_name: {"RMSE": float, "MAPE": float, "R2": float,
                        "train_time_s": float}}
    """
    print("=" * 60)
    print("PIPELINE STEP 3 — Model Benchmarking")
    print("=" * 60)
    print(f"CV strategy: TimeSeriesSplit(n_splits={n_splits})\n")

    tscv    = TimeSeriesSplit(n_splits=n_splits)
    models  = _get_models()
    results = {}

    for name, model in models.items():
        fold_rmse, fold_mape, fold_r2 = [], [], []
        t0 = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            fold_rmse.append(_rmse(y_te, y_pred))
            fold_mape.append(_mape(y_te, y_pred))
            fold_r2.append(_r2(y_te, y_pred))

        elapsed = time.time() - t0

        results[name] = {
            "RMSE":         round(float(np.mean(fold_rmse)), 2),
            "MAPE":         round(float(np.mean(fold_mape)), 2),
            "R2":           round(float(np.mean(fold_r2)),   4),
            "train_time_s": round(elapsed, 3),
        }

        print(f"  {name:<15} RMSE={results[name]['RMSE']:>8,.2f}  "
              f"MAPE={results[name]['MAPE']:>6.2f}%  "
              f"R²={results[name]['R2']:>6.4f}  "
              f"({elapsed:.2f}s)")

    # Save artefact for reference / dashboard display
    os.makedirs("models", exist_ok=True)
    out_path = "models/benchmark_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nBenchmark results saved → {out_path}\n")

    return results


if __name__ == "__main__":
    import pandas as pd
    from pipeline.feature_engineering import build_forecasting_features, get_X_y

    df = pd.read_csv("data/cleaned_data.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    daily  = build_forecasting_features(df)
    X, y, _ = get_X_y(daily)

    results = benchmark_models(X, y)
    print(json.dumps(results, indent=2))
