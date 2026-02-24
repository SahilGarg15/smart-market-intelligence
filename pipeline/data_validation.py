"""
pipeline/data_validation.py

Data Validation & Cleaning Pipeline — v2
Loads the Online Retail CSV, validates quality, cleans, and saves output files.

Why a separate module?
  Separating validation from feature engineering follows the "single responsibility"
  principle. If data quality breaks, the failure is isolated here, not buried inside
  a 400-line training script.
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
RAW_DATA_PATH      = "data/Online Retail.csv"
CLEANED_DATA_PATH  = "data/cleaned_data.csv"


# ---------------------------------------------------------------------------
# 1. Schema validation helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice",
    "CustomerID", "Country",
]


def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that every required column is present.
    Fails fast rather than producing mysterious KeyErrors downstream.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Schema mismatch — missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    print(f"  ✓ Schema OK — {len(df.columns)} columns present")


def validate_not_empty(df: pd.DataFrame) -> None:
    """An empty dataframe cannot produce a trained model."""
    if df.empty:
        raise ValueError(
            "Dataset is empty. Verify that the CSV file is not corrupted "
            "and was placed at: " + RAW_DATA_PATH
        )
    print(f"  ✓ Row count OK — {len(df):,} rows loaded")


def validate_price_integrity(df: pd.DataFrame, fatal_threshold: float = 0.01) -> None:
    """
    Negative UnitPrice indicates data-entry errors or return credits.

    Policy:
      • < 1% of rows negative → print a warning; clean_data() will remove them.
      • ≥ 1% of rows negative → raise, because something systemic is wrong
        (wrong sign convention, corrupted file, etc.) that warrants investigation.
    """
    if "UnitPrice" not in df.columns:
        return
    neg = df[df["UnitPrice"] < 0]
    if len(neg) == 0:
        print(f"  ✓ UnitPrice integrity OK — no negative prices")
        return
    pct = len(neg) / len(df)
    if pct >= fatal_threshold:
        raise ValueError(
            f"Found {len(neg):,} rows ({pct:.2%}) with negative UnitPrice — "
            f"this suggests systemic data corruption.\n"
            f"Sample values: {neg['UnitPrice'].head(5).tolist()}"
        )
    print(f"  ⚠  {len(neg):,} rows with negative UnitPrice ({pct:.4%}) — "
          f"will be removed in cleaning step")


def validate_quantity_range(df: pd.DataFrame, threshold: int = 100_000) -> None:
    """
    Quantities above 'threshold' in a single transaction are physiologically
    impossible for a standard retail operation and most likely data errors.
    """
    if "Quantity" not in df.columns:
        return
    extreme = df[df["Quantity"] > threshold]
    if len(extreme) > 0:
        raise ValueError(
            f"Found {len(extreme):,} rows with Quantity > {threshold:,}.\n"
            f"Max value seen: {df['Quantity'].max():,.0f}\n"
            f"This suggests data-entry errors or wrong units."
        )
    print(f"  ✓ Quantity range OK — no values above {threshold:,}")


def run_validation(df: pd.DataFrame) -> None:
    """
    Convenience wrapper — runs all validators in sequence.
    Called before cleaning so we never silently process bad data.
    """
    print("\n[Validation] Running guardrails …")
    validate_not_empty(df)
    validate_schema(df)
    validate_price_integrity(df)
    validate_quantity_range(df)
    print("[Validation] All checks passed ✓\n")


# ---------------------------------------------------------------------------
# 2. Cleaning pipeline
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic cleaning rules to the raw retail dataframe.

    Rules (in order):
      1. Drop rows with missing CustomerID     — un-attributable transactions
      2. Drop rows with Quantity  <= 0         — returns / credits
      3. Drop rows with UnitPrice <= 0         — free / error lines
      4. Compute TotalAmount                   — revenue per line item
      5. Parse InvoiceDate to datetime         — enables time-series features
    """
    initial = len(df)
    print(f"[Cleaning] Starting rows: {initial:,}")

    # Rule 1 — CustomerID is the key for segmentation; cannot be imputed
    df = df.dropna(subset=["CustomerID"])
    print(f"  → Dropped {initial - len(df):,} rows with missing CustomerID  "
          f"({len(df):,} remain)")

    # Rule 2 — negative / zero Quantity = returns or placeholders
    before = len(df)
    df = df[df["Quantity"] > 0]
    print(f"  → Dropped {before - len(df):,} rows with Quantity ≤ 0  "
          f"({len(df):,} remain)")

    # Rule 3 — zero/negative price lines carry no revenue signal
    before = len(df)
    df = df[df["UnitPrice"] > 0]
    print(f"  → Dropped {before - len(df):,} rows with UnitPrice ≤ 0  "
          f"({len(df):,} remain)")

    # Rule 4 — revenue per line
    df = df.copy()
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Rule 5 — datetime conversion enables .dt accessor in feature engineering
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    print(f"[Cleaning] Final rows: {len(df):,}  "
          f"({initial - len(df):,} total removed)\n")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Entry point
# ---------------------------------------------------------------------------

def load_and_validate(
    input_path: str = RAW_DATA_PATH,
    output_path: str = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """
    Full pipeline: load → validate → clean → persist.

    Returns the cleaned DataFrame so downstream modules can import it
    without re-reading from disk.
    """
    print("=" * 60)
    print("PIPELINE STEP 1 — Data Validation & Cleaning")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{input_path}'.\n"
            "Download 'Online Retail.xlsx' from the UCI ML Repository, "
            "convert to CSV, and save it to the data/ folder."
        )

    # encoding='latin-1' handles special characters in product descriptions
    df_raw = pd.read_csv(input_path, encoding="latin-1")
    print(f"Loaded {len(df_raw):,} rows from {input_path}")

    run_validation(df_raw)

    df_clean = clean_data(df_raw)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved cleaned data → {output_path}  ({size_mb:.2f} MB)\n")

    return df_clean


if __name__ == "__main__":
    load_and_validate()
