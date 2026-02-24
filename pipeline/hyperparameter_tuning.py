"""
pipeline/hyperparameter_tuning.py

Hyperparameter Tuning — v2
Run GridSearchCV on RandomForestRegressor using TimeSeriesSplit.

Why only tune Random Forest?
  LinearRegression has no meaningful hyperparameters to tune.
  Ridge alpha can be tuned with RidgeCV in one line — done separately.
  RF has the most knobs that meaningfully affect bias-variance trade-off on
  this dataset and is the model most likely to be selected as 'best'.

GridSearch strategy:
  We keep the grid deliberately small because:
    (a) The dataset is ~270 training rows — large grids overfit to CV noise.
    (b) Runtime on a laptop must stay under 2 minutes.
    (c) The gain from fine-tuning on 300 rows is marginal vs. feature quality.
"""

import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

RF_PARAM_GRID = {
    "n_estimators":    [100, 200],
    "max_depth":       [5, 10, None],
    "min_samples_split": [2, 5, 10],
}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _neg_rmse(y_true, y_pred):
    """
    GridSearchCV maximises the scorer, so we negate RMSE.
    Using RMSE (not MSE) keeps the scale interpretable in error messages.
    """
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Tuning function
# ---------------------------------------------------------------------------

def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """
    Run grid search and return the best hyperparameter dictionary.

    Why TimeSeriesSplit inside GridSearchCV?
      GridSearchCV by default uses stratified K-Fold. For time series that
      introduces look-ahead bias (future folds train on data after test fold).
      Passing cv=TimeSeriesSplit explicitly preserves temporal ordering inside
      each candidate evaluation.

    Returns
    -------
    best_params : dict  e.g. {"n_estimators": 200, "max_depth": 10, ...}
    """
    print("=" * 60)
    print("PIPELINE STEP 4 — Hyperparameter Tuning (RandomForest)")
    print("=" * 60)
    print(f"Parameter grid: {RF_PARAM_GRID}")
    n_candidates = (
        len(RF_PARAM_GRID["n_estimators"])
        * len(RF_PARAM_GRID["max_depth"])
        * len(RF_PARAM_GRID["min_samples_split"])
    )
    print(f"Candidates: {n_candidates} × {n_splits} folds = "
          f"{n_candidates * n_splits} fits\n")

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(_neg_rmse)

    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    t0 = time.time()
    gs = GridSearchCV(
        estimator  = base_rf,
        param_grid = RF_PARAM_GRID,
        cv         = tscv,
        scoring    = scorer,
        refit      = True,       # refit best estimator on full data
        verbose    = 1,
        n_jobs     = -1,
    )
    gs.fit(X, y)
    elapsed = time.time() - t0

    best_params = gs.best_params_
    best_rmse   = -gs.best_score_   # un-negate

    print(f"\nBest params : {best_params}")
    print(f"Best CV RMSE: £{best_rmse:,.2f}")
    print(f"Tuning time : {elapsed:.1f}s\n")

    return {
        "best_params":   best_params,
        "best_cv_rmse":  round(best_rmse, 2),
        "best_estimator": gs.best_estimator_,   # already fitted on full data
    }


if __name__ == "__main__":
    import pandas as pd
    from pipeline.feature_engineering import build_forecasting_features, get_X_y

    df = pd.read_csv("data/cleaned_data.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    daily = build_forecasting_features(df)
    X, y, _ = get_X_y(daily)

    result = tune_random_forest(X, y)
    print("Best params:", json.dumps(result["best_params"], indent=2))
