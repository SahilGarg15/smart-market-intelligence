"""
run_pipeline.py

Master training pipeline — Smart Market Intelligence Platform v2

Run this script once to go from raw data to a production-ready model:

  python run_pipeline.py

Pipeline steps:
  1. Load + validate raw data
  2. Build forecasting features (lag, rolling, calendar)
  3. Benchmark baseline models (Linear, Ridge, RandomForest)
  4. Tune RandomForest hyperparameters via GridSearchCV
  5. Select best model by MAPE; retrain on full data
  6. Explainability & error analysis (feature importance, residuals)

Outputs written to disk:
  data/cleaned_data.csv          — cleaned transaction data
  data/daily_sales.csv           — engineered daily feature set
  models/benchmark_results.json  — CV metrics for all models
  models/best_model.pkl          — final trained model (joblib)
  models/metadata.json           — version, metrics, hyperparams, residuals
  analysis/feature_importance.png
  analysis/residual_distribution.png
"""

import json
import sys
import time
import traceback

import pandas as pd

from pipeline.data_validation  import load_and_validate
from pipeline.feature_engineering import build_forecasting_features, get_X_y
from pipeline.model_benchmark  import benchmark_models
from pipeline.hyperparameter_tuning import tune_random_forest
from pipeline.model_selection  import select_and_save
from pipeline.error_analysis   import run_error_analysis


def main():
    t_pipeline_start = time.time()

    print("=" * 60)
    print("  Smart Market Intelligence Platform v2 — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # STEP 1: Data Validation
    # ------------------------------------------------------------------
    try:
        df = load_and_validate()
    except Exception as exc:
        print(f"\n[FATAL] Data validation failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 2: Feature Engineering
    # ------------------------------------------------------------------
    try:
        daily = build_forecasting_features(df)
        X, y, feature_names = get_X_y(daily)
        print(f"\n[Pipeline] Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"[Pipeline] Target range:  £{y.min():,.0f}  –  £{y.max():,.0f}")
    except Exception as exc:
        print(f"\n[FATAL] Feature engineering failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3: Benchmark
    # ------------------------------------------------------------------
    try:
        benchmark_results = benchmark_models(X, y)
    except Exception as exc:
        print(f"\n[FATAL] Benchmarking failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 4: Hyperparameter Tuning (RandomForest only)
    # ------------------------------------------------------------------
    best_params = None
    try:
        tuning_result = tune_random_forest(X, y)
        best_params   = tuning_result["best_params"]
    except Exception as exc:
        print(f"\n[Warning] Tuning failed — using default RF params. Error: {exc}")

    # ------------------------------------------------------------------
    # STEP 5: Model Selection + Persistence
    # ------------------------------------------------------------------
    try:
        selection = select_and_save(
            benchmark_results = benchmark_results,
            X                 = X,
            y                 = y,
            feature_names     = feature_names,
            best_params       = best_params,
        )
    except Exception as exc:
        print(f"\n[FATAL] Model selection failed: {exc}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 6: Error Analysis + Explainability
    # ------------------------------------------------------------------
    try:
        run_error_analysis(
            model         = selection["model"],
            y_test        = selection["y_test"],
            y_pred        = selection["y_pred"],
            feature_names = selection["feature_names"],
        )
    except Exception as exc:
        # Non-fatal: plots are nice-to-have; model is already saved
        print(f"\n[Warning] Error analysis failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_pipeline_start
    m = selection["metrics"]
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Best model  : {selection['model_name']}")
    print(f"  RMSE        : £{m['RMSE']:,.2f}")
    print(f"  MAPE        :  {m['MAPE']:.2f}%")
    print(f"  R²          :  {m['R2']:.4f}")
    print(f"  Total time  :  {elapsed:.1f}s")
    print("=" * 60)
    print("\nNext step: start the API")
    print("  uvicorn api.app:app --reload\n")


if __name__ == "__main__":
    main()
