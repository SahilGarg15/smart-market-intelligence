"""
pipeline/business_metrics.py

Business Metrics Computation Layer — Smart Market Intelligence Platform

Computes actionable business KPIs from daily_sales.csv and customer_segments.csv:
  • Revenue metrics  — daily/weekly growth, moving averages
  • Customer metrics — retention rate, repeat purchase rate, active customers
  • Sales metrics    — AOV, total orders, sales per customer
  • Segmentation     — revenue by segment, top-20% customer contribution

Usage:
    from pipeline.business_metrics import compute_all_metrics
    metrics = compute_all_metrics()
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DAILY_SALES_PATH = "data/daily_sales.csv"
CUSTOMER_SEGMENTS_PATH = "data/customer_segments.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_daily_sales() -> pd.DataFrame:
    """Load and parse the daily sales CSV."""
    if not os.path.exists(DAILY_SALES_PATH):
        raise FileNotFoundError(
            f"Daily sales data not found at '{DAILY_SALES_PATH}'. "
            "Run the training pipeline first: python run_pipeline.py"
        )
    df = pd.read_csv(DAILY_SALES_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _load_customer_segments() -> pd.DataFrame:
    """Load customer segmentation data (RFM + cluster assignments)."""
    if not os.path.exists(CUSTOMER_SEGMENTS_PATH):
        raise FileNotFoundError(
            f"Customer segments not found at '{CUSTOMER_SEGMENTS_PATH}'. "
            "Run the training pipeline first: python run_pipeline.py"
        )
    return pd.read_csv(CUSTOMER_SEGMENTS_PATH)


# ---------------------------------------------------------------------------
# 1. Revenue Metrics
# ---------------------------------------------------------------------------

def compute_revenue_metrics(daily: pd.DataFrame) -> dict:
    """
    Calculate revenue growth and trend indicators.

    Returns:
        revenue_growth_daily  — % change from second-to-last to last day
        revenue_growth_weekly — % change of last 7 days vs previous 7 days
        revenue_ma7           — latest 7-day moving average
        revenue_ma30          — latest 30-day moving average
        total_revenue         — lifetime total revenue
    """
    metrics = {}

    sales = daily["Sales"].values

    # Daily growth: last day vs day before
    if len(sales) >= 2:
        prev, curr = sales[-2], sales[-1]
        metrics["revenue_growth_daily"] = round(
            float((curr - prev) / prev) if prev != 0 else 0.0, 4
        )
    else:
        metrics["revenue_growth_daily"] = 0.0

    # Weekly growth: last 7 days vs previous 7 days
    if len(sales) >= 14:
        current_week = sales[-7:].sum()
        previous_week = sales[-14:-7].sum()
        metrics["revenue_growth_weekly"] = round(
            float((current_week - previous_week) / previous_week)
            if previous_week != 0 else 0.0, 4
        )
    else:
        metrics["revenue_growth_weekly"] = 0.0

    # Moving averages (latest value)
    if len(sales) >= 7:
        metrics["revenue_ma7"] = round(float(sales[-7:].mean()), 2)
    else:
        metrics["revenue_ma7"] = round(float(sales.mean()), 2)

    if len(sales) >= 30:
        metrics["revenue_ma30"] = round(float(sales[-30:].mean()), 2)
    else:
        metrics["revenue_ma30"] = round(float(sales.mean()), 2)

    metrics["total_revenue"] = round(float(sales.sum()), 2)

    return metrics


# ---------------------------------------------------------------------------
# 2. Customer Metrics
# ---------------------------------------------------------------------------

def compute_customer_metrics(segments: pd.DataFrame) -> dict:
    """
    Calculate customer health indicators from RFM segmentation data.

    Returns:
        retention_rate       — fraction of customers with Frequency >= 2
        repeat_purchase_rate — same concept (customers who came back at least once)
        active_customers     — customers with Recency <= 30 days
        total_customers      — total unique customers in the dataset
        active_customer_ratio — active / total
    """
    metrics = {}
    total = len(segments)

    if total == 0:
        return {
            "retention_rate": 0.0,
            "repeat_purchase_rate": 0.0,
            "active_customers": 0,
            "total_customers": 0,
            "active_customer_ratio": 0.0,
        }

    # Retention = customers who purchased more than once
    repeat_customers = segments[segments["Frequency"] >= 2]
    metrics["retention_rate"] = round(float(len(repeat_customers) / total), 4)

    # Repeat purchase rate — same as retention in transactional context
    metrics["repeat_purchase_rate"] = round(
        float(len(repeat_customers) / total), 4
    )

    # Active customers = purchased within last 30 days
    active = segments[segments["Recency"] <= 30]
    metrics["active_customers"] = int(len(active))
    metrics["total_customers"] = int(total)
    metrics["active_customer_ratio"] = round(float(len(active) / total), 4)

    return metrics


# ---------------------------------------------------------------------------
# 3. Sales Metrics
# ---------------------------------------------------------------------------

def compute_sales_metrics(daily: pd.DataFrame, segments: pd.DataFrame) -> dict:
    """
    Calculate operational sales KPIs.

    Returns:
        avg_order_value    — total revenue / total transactions
        total_orders       — sum of all daily transactions
        sales_per_customer — total revenue / unique customers
    """
    metrics = {}

    total_revenue = daily["Sales"].sum()
    total_orders = int(daily["NumTransactions"].sum())
    total_customers = len(segments)

    metrics["avg_order_value"] = (
        round(float(total_revenue / total_orders), 2) if total_orders > 0 else 0.0
    )
    metrics["total_orders"] = total_orders
    metrics["sales_per_customer"] = (
        round(float(total_revenue / total_customers), 2)
        if total_customers > 0 else 0.0
    )

    return metrics


# ---------------------------------------------------------------------------
# 4. Segmentation Metrics (RFM-connected)
# ---------------------------------------------------------------------------

def compute_segmentation_metrics(segments: pd.DataFrame) -> dict:
    """
    Analyse revenue distribution across customer segments.

    Returns:
        revenue_by_segment         — dict mapping segment name → total Monetary
        top_20_pct_contribution    — fraction of total revenue from top 20% spenders
    """
    metrics = {}

    # Revenue contribution by segment
    if "Segment" in segments.columns:
        seg_revenue = (
            segments.groupby("Segment")["Monetary"]
            .sum()
            .sort_values(ascending=False)
        )
        metrics["revenue_by_segment"] = {
            str(k): round(float(v), 2) for k, v in seg_revenue.items()
        }
    else:
        metrics["revenue_by_segment"] = {}

    # Top 20% customers contribution
    total_monetary = segments["Monetary"].sum()
    if total_monetary > 0 and len(segments) > 0:
        # Sort by Monetary descending, take top 20%
        sorted_monetary = segments["Monetary"].sort_values(ascending=False)
        top_n = max(1, int(len(sorted_monetary) * 0.20))
        top_revenue = sorted_monetary.iloc[:top_n].sum()
        metrics["top_20_pct_contribution"] = round(
            float(top_revenue / total_monetary), 4
        )
    else:
        metrics["top_20_pct_contribution"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_all_metrics() -> dict:
    """
    Compute all business metrics and return as a flat dictionary.

    This is the single entry point used by:
      • The /metrics API endpoint
      • The /insights engine
      • The Streamlit dashboard

    Returns a dict with keys:
        revenue_growth_daily, revenue_growth_weekly, revenue_ma7, revenue_ma30,
        total_revenue, retention_rate, repeat_purchase_rate, active_customers,
        total_customers, active_customer_ratio, avg_order_value, total_orders,
        sales_per_customer, revenue_by_segment, top_20_pct_contribution
    """
    daily = _load_daily_sales()
    segments = _load_customer_segments()

    metrics = {}
    metrics.update(compute_revenue_metrics(daily))
    metrics.update(compute_customer_metrics(segments))
    metrics.update(compute_sales_metrics(daily, segments))
    metrics.update(compute_segmentation_metrics(segments))

    return metrics


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    m = compute_all_metrics()
    print(json.dumps(m, indent=2))
