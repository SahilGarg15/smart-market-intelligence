"""
pipeline/feature_engineering.py

Feature Engineering — v2
Transforms the cleaned retail dataframe into two separate feature sets:
  A) RFM features  → used by the K-Means customer segmentation model
  B) Daily-sales features with lag/rolling windows → used by the forecasting models

Why a dedicated module?
  Feature engineering is the biggest driver of model performance.
  Isolating it makes every transformation auditable, testable, and reusable
  without re-running the full training script.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# A. Customer-segmentation features (RFM)
# ---------------------------------------------------------------------------

def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency / Frequency / Monetary value per customer.

        Recency  = days since the customer's last invoice (lower = more recent)
        Frequency= number of *unique* invoices placed
        Monetary = total revenue attributed to that customer

    We use the latest InvoiceDate in the dataset as the reference date
    (the 'snapshot date') so the numbers are internally consistent.
    """
    print("[Feature Engineering] Building RFM features …")

    snapshot_date = df["InvoiceDate"].max()

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
            Frequency = ("InvoiceNo",    "nunique"),
            Monetary  = ("TotalAmount",  "sum"),
        )
        .reset_index()
    )

    print(f"  → {len(rfm):,} customers | snapshot date: {snapshot_date.date()}")
    print(f"  Recency  — mean: {rfm['Recency'].mean():.0f} d, "
          f"max: {rfm['Recency'].max()} d")
    print(f"  Frequency — mean: {rfm['Frequency'].mean():.1f}, "
          f"max: {rfm['Frequency'].max()}")
    print(f"  Monetary — mean: £{rfm['Monetary'].mean():,.0f}, "
          f"max: £{rfm['Monetary'].max():,.0f}\n")

    return rfm


# ---------------------------------------------------------------------------
# B. Sales-forecasting features
# ---------------------------------------------------------------------------

def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate from transaction-level to daily-level.

    Also computes context features that are available at the *day* granularity
    without lookahead bias:
      - NumTransactions : volume proxy
      - NumCustomers    : breadth proxy
      - TotalQuantity   : unit volume proxy
    """
    daily = (
        df.groupby(df["InvoiceDate"].dt.date)
        .agg(
            Sales           = ("TotalAmount",  "sum"),
            NumTransactions = ("InvoiceNo",    "nunique"),
            NumCustomers    = ("CustomerID",   "nunique"),
            TotalQuantity   = ("Quantity",     "sum"),
        )
        .reset_index()
        .rename(columns={"InvoiceDate": "Date"})
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)
    return daily


def add_temporal_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the Date column.

    Why calendar features?
      Decision-tree models cannot infer cyclical patterns from raw dates.
      Supplying Month, DayOfWeek, etc. explicitly lets the tree partition
      on weekend-vs-weekday effects, seasonal patterns, etc.
    """
    d = daily.copy()
    d["Year"]       = d["Date"].dt.year
    d["Month"]      = d["Date"].dt.month
    d["Day"]        = d["Date"].dt.day
    d["DayOfWeek"]  = d["Date"].dt.dayofweek   # 0 = Monday
    d["Quarter"]    = d["Date"].dt.quarter
    d["DayOfYear"]  = d["Date"].dt.dayofyear
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["IsWeekend"]  = (d["DayOfWeek"] >= 5).astype(int)
    return d


def add_lag_features(daily: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Create lagged Sales columns (shift by N days).

    Lag features give the model explicit 'memory':
      Sales_Lag1  → yesterday's revenue
      Sales_Lag7  → same weekday last week (captures weekly seasonality)
      Sales_Lag14 → two weeks ago

    ⚠ We fill NaN lags with the column median to avoid dropping early rows
      that would reduce our already-small (305-day) dataset.
    """
    if lags is None:
        lags = [1, 7, 14]

    d = daily.copy()
    for lag in lags:
        col = f"Sales_Lag{lag}"
        d[col] = d["Sales"].shift(lag)
        d[col] = d[col].fillna(d["Sales"].median())
    return d


def add_rolling_features(daily: pd.DataFrame,
                         windows: list = None) -> pd.DataFrame:
    """
    Compute rolling mean over multiple windows.

    A 7-day moving average smooths intra-week noise; a 30-day average
    captures monthly trend. min_periods=1 prevents NaN in the first window.
    """
    if windows is None:
        windows = [7, 30]

    d = daily.copy()
    for w in windows:
        col = f"Sales_MA{w}"
        d[col] = d["Sales"].rolling(window=w, min_periods=1).mean()
    return d


def build_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — chains all time-series feature steps and saves the result.

    Returns the full daily DataFrame with all features appended.
    """
    print("[Feature Engineering] Building sales-forecasting features …")

    daily = build_daily_sales(df)
    daily = add_temporal_features(daily)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)

    # Persist for downstream use (API, dashboard)
    os.makedirs("data", exist_ok=True)
    daily.to_csv("data/daily_sales.csv", index=False)

    feature_cols = [c for c in daily.columns if c not in ("Date", "Sales")]
    print(f"  → {len(daily)} days | {len(feature_cols)} features: {feature_cols}\n")

    return daily


# ---------------------------------------------------------------------------
# Shared helper — extract X / y arrays from the daily DataFrame
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "Year", "Month", "Day", "DayOfWeek", "Quarter",
    "DayOfYear", "WeekOfYear", "IsWeekend",
    "NumTransactions", "NumCustomers", "TotalQuantity",
    "Sales_Lag1", "Sales_Lag7", "Sales_Lag14",
    "Sales_MA7", "Sales_MA30",
]

TARGET_COLUMN = "Sales"


def get_X_y(daily: pd.DataFrame):
    """
    Return feature matrix X and target vector y, keeping only the columns
    that were present in every run.

    Using a fixed column list ensures that the model trained in ml_engine
    and the features constructed in the API request are identical.
    """
    available = [c for c in FEATURE_COLUMNS if c in daily.columns]
    X = daily[available].values
    y = daily[TARGET_COLUMN].values
    return X, y, available


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    rfm   = build_rfm_features(df)
    daily = build_forecasting_features(df)
    X, y, cols = get_X_y(daily)

    print(f"X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"Feature columns: {cols}")
