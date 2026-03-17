"""
pipeline/insight_engine.py

Rule-Based Insights Engine — Smart Market Intelligence Platform

Transforms computed business metrics into actionable, human-readable insights.
This is a pure rule-based engine (no LLM) — fast, deterministic, and auditable.

Data flow:
    daily_sales.csv + customer_segments.csv
        → business_metrics.compute_all_metrics()
            → insight_engine.generate_insights()
                → list of insight strings + severity levels

Usage:
    from pipeline.insight_engine import generate_insights
    from pipeline.business_metrics import compute_all_metrics

    metrics  = compute_all_metrics()
    insights = generate_insights(metrics)
"""

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Insight rules
# ---------------------------------------------------------------------------
# Each rule is a tuple: (condition_fn, severity, message)
# Severity levels: "positive", "warning", "critical", "info"

def _build_rules():
    """
    Define business insight rules.

    Each rule evaluates a condition on the metrics dict and produces
    a human-readable insight with a severity level. Rules are ordered
    from most impactful to least.
    """
    rules = []

    # --- Revenue trend rules ---
    rules.append({
        "key": "revenue_trend_up",
        "condition": lambda m: m.get("revenue_growth_weekly", 0) > 0.10,
        "severity": "positive",
        "message": (
            "Strong upward revenue trend — weekly growth exceeds 10% "
            "({growth:.1%})"
        ),
        "format": lambda m: {"growth": m["revenue_growth_weekly"]},
    })

    rules.append({
        "key": "revenue_trend_moderate_up",
        "condition": lambda m: 0.03 < m.get("revenue_growth_weekly", 0) <= 0.10,
        "severity": "positive",
        "message": (
            "Revenue growing steadily — {growth:.1%} weekly increase"
        ),
        "format": lambda m: {"growth": m["revenue_growth_weekly"]},
    })

    rules.append({
        "key": "revenue_trend_down",
        "condition": lambda m: m.get("revenue_growth_weekly", 0) < -0.10,
        "severity": "critical",
        "message": (
            "Revenue declining sharply — weekly drop of {growth:.1%}. "
            "Investigate root cause and review marketing spend"
        ),
        "format": lambda m: {"growth": abs(m["revenue_growth_weekly"])},
    })

    rules.append({
        "key": "revenue_trend_slight_down",
        "condition": lambda m: -0.10 <= m.get("revenue_growth_weekly", 0) < -0.03,
        "severity": "warning",
        "message": (
            "Revenue slightly declining — {growth:.1%} weekly decrease"
        ),
        "format": lambda m: {"growth": abs(m["revenue_growth_weekly"])},
    })

    # --- Customer retention rules ---
    rules.append({
        "key": "retention_critical",
        "condition": lambda m: m.get("retention_rate", 1) < 0.40,
        "severity": "critical",
        "message": (
            "Customer retention is critically low at {rate:.1%} — "
            "high churn risk. Implement urgent retention campaigns"
        ),
        "format": lambda m: {"rate": m["retention_rate"]},
    })

    rules.append({
        "key": "retention_low",
        "condition": lambda m: 0.40 <= m.get("retention_rate", 1) < 0.55,
        "severity": "warning",
        "message": (
            "Customer retention is below benchmark at {rate:.1%} — "
            "consider loyalty programs or re-engagement emails"
        ),
        "format": lambda m: {"rate": m["retention_rate"]},
    })

    rules.append({
        "key": "retention_excellent",
        "condition": lambda m: m.get("retention_rate", 0) > 0.75,
        "severity": "positive",
        "message": (
            "Excellent customer retention at {rate:.1%} — strong loyalty base"
        ),
        "format": lambda m: {"rate": m["retention_rate"]},
    })

    # --- Top customer concentration ---
    rules.append({
        "key": "top_customer_dependency",
        "condition": lambda m: m.get("top_20_pct_contribution", 0) > 0.60,
        "severity": "warning",
        "message": (
            "Revenue heavily dependent on top 20% of customers "
            "({contrib:.1%} of total) — diversification needed"
        ),
        "format": lambda m: {"contrib": m["top_20_pct_contribution"]},
    })

    rules.append({
        "key": "top_customer_extreme",
        "condition": lambda m: m.get("top_20_pct_contribution", 0) > 0.80,
        "severity": "critical",
        "message": (
            "Extreme revenue concentration — top 20% of customers "
            "generate {contrib:.1%} of revenue. Losing key accounts "
            "could be catastrophic"
        ),
        "format": lambda m: {"contrib": m["top_20_pct_contribution"]},
    })

    # --- Active customer ratio ---
    rules.append({
        "key": "low_active_ratio",
        "condition": lambda m: m.get("active_customer_ratio", 1) < 0.30,
        "severity": "warning",
        "message": (
            "Only {ratio:.1%} of customers are active (purchased in last 30 days) "
            "— re-engagement campaign recommended"
        ),
        "format": lambda m: {"ratio": m["active_customer_ratio"]},
    })

    rules.append({
        "key": "high_active_ratio",
        "condition": lambda m: m.get("active_customer_ratio", 0) > 0.60,
        "severity": "positive",
        "message": (
            "Healthy active customer base — {ratio:.1%} purchased within 30 days"
        ),
        "format": lambda m: {"ratio": m["active_customer_ratio"]},
    })

    # --- Average order value ---
    rules.append({
        "key": "high_aov",
        "condition": lambda m: m.get("avg_order_value", 0) > 500,
        "severity": "positive",
        "message": (
            "Average order value is strong at £{aov:,.0f} — "
            "indicates healthy customer spending"
        ),
        "format": lambda m: {"aov": m["avg_order_value"]},
    })

    rules.append({
        "key": "low_aov",
        "condition": lambda m: 0 < m.get("avg_order_value", 0) < 100,
        "severity": "info",
        "message": (
            "Average order value is £{aov:,.0f} — consider cross-sell "
            "or bundle strategies to increase basket size"
        ),
        "format": lambda m: {"aov": m["avg_order_value"]},
    })

    # --- Revenue MA trend ---
    rules.append({
        "key": "ma_divergence",
        "condition": lambda m: (
            m.get("revenue_ma7", 0) > 0
            and m.get("revenue_ma30", 0) > 0
            and m["revenue_ma7"] > m["revenue_ma30"] * 1.15
        ),
        "severity": "positive",
        "message": (
            "Short-term revenue trend is accelerating — 7-day MA "
            "(£{ma7:,.0f}) is {pct:.0%} above 30-day MA (£{ma30:,.0f})"
        ),
        "format": lambda m: {
            "ma7": m["revenue_ma7"],
            "ma30": m["revenue_ma30"],
            "pct": (m["revenue_ma7"] / m["revenue_ma30"]) - 1,
        },
    })

    rules.append({
        "key": "ma_bearish",
        "condition": lambda m: (
            m.get("revenue_ma7", 0) > 0
            and m.get("revenue_ma30", 0) > 0
            and m["revenue_ma7"] < m["revenue_ma30"] * 0.85
        ),
        "severity": "warning",
        "message": (
            "Short-term revenue is slowing — 7-day MA (£{ma7:,.0f}) "
            "is below 30-day MA (£{ma30:,.0f})"
        ),
        "format": lambda m: {
            "ma7": m["revenue_ma7"],
            "ma30": m["revenue_ma30"],
        },
    })

    return rules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insights(metrics: dict) -> dict:
    """
    Evaluate all rules against the provided metrics and return matching insights.

    Args:
        metrics: dict from compute_all_metrics()

    Returns:
        {
            "insights": [
                {"message": "...", "severity": "positive|warning|critical|info", "key": "..."},
                ...
            ],
            "total_insights": int,
            "generated_at": "ISO 8601 timestamp"
        }
    """
    rules = _build_rules()
    insights = []

    for rule in rules:
        try:
            if rule["condition"](metrics):
                fmt_args = rule["format"](metrics) if "format" in rule else {}
                insights.append({
                    "key": rule["key"],
                    "severity": rule["severity"],
                    "message": rule["message"].format(**fmt_args),
                })
        except (KeyError, ZeroDivisionError, TypeError):
            # Skip rules that fail due to missing data — graceful degradation
            continue

    # Sort: critical first, then warning, then info, then positive
    severity_order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 99))

    return {
        "insights": insights,
        "total_insights": len(insights),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pipeline.business_metrics import compute_all_metrics

    metrics = compute_all_metrics()
    result = generate_insights(metrics)
    print(json.dumps(result, indent=2))
