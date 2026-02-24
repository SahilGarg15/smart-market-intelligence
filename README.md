# Smart Market Intelligence Platform

**Tech Stack**: Python 3.14 · FastAPI · Streamlit · Scikit-Learn · PostgreSQL (Neon) · Plotly

An end-to-end retail ML platform covering data validation, RFM customer segmentation, multi-model sales forecasting, a REST API with cloud persistence, and an interactive dashboard — all built as a clean, modular v2 pipeline.

---

## Architecture

```
Raw CSV
  │
  ▼
pipeline/data_validation.py       ← fail-fast guardrails
pipeline/feature_engineering.py   ← RFM + lag/rolling features
pipeline/model_benchmark.py        ← Linear · Ridge · Random Forest × TimeSeriesSplit
pipeline/hyperparameter_tuning.py  ← GridSearchCV on best candidate
pipeline/model_selection.py        ← select by MAPE, retrain on full data, export
pipeline/error_analysis.py         ← feature importance + residual charts
  │
  ├─► models/best_model.pkl
  ├─► models/metadata.json
  └─► analysis/*.png
         │
         ▼
   api/app.py  (FastAPI)              dashboard.py  (Streamlit)
         │
         ▼
   db/  (Neon PostgreSQL)
   prediction_logs table
```

---

## Project Structure

```
market-intel/
│
├── pipeline/                          # Training pipeline (6 modules)
│   ├── data_validation.py             # Load CSV + 4 fail-fast guardrails
│   ├── feature_engineering.py         # RFM, daily aggregation, lag/rolling features
│   ├── model_benchmark.py             # Benchmark 3 models with TimeSeriesSplit
│   ├── hyperparameter_tuning.py       # GridSearchCV on Random Forest
│   ├── model_selection.py             # Pick best by MAPE, serialize model
│   ├── error_analysis.py              # Feature importance + residual analysis
│   └── __init__.py
│
├── api/
│   ├── app.py                         # FastAPI — /health /predict /drift /benchmark /metadata
│   └── __init__.py
│
├── db/
│   ├── db_config.py                   # SQLAlchemy engine + Neon connection pool
│   ├── db_models.py                   # PredictionLog ORM table
│   ├── db_logger.py                   # log_prediction() + get_recent_predictions()
│   └── __init__.py
│
├── utils/
│   └── data_loader.py                 # Legacy v1 helper (kept for reference)
│
├── data/
│   ├── Online Retail.csv              # Source dataset — place here before running
│   ├── cleaned_data.csv               # Auto-generated after pipeline
│   ├── daily_sales.csv                # Auto-generated after pipeline
│   ├── customer_segments.csv          # Auto-generated after pipeline
│   └── README.md
│
├── models/
│   ├── best_model.pkl                 # Primary model loaded by the API
│   ├── sales_model_v1.pkl             # Versioned backup for rollback
│   ├── kmeans_model.pkl               # Customer segmentation model
│   ├── scaler.pkl                     # RFM StandardScaler
│   ├── benchmark_results.json         # Per-model CV metrics
│   ├── metadata.json                  # Full training audit record
│   └── README.md
│
├── analysis/
│   ├── feature_importance.png         # Top-10 feature importances
│   └── residual_distribution.png      # Residual histogram
│
├── pipeline_notebook.ipynb            # Interactive step-by-step notebook
├── run_pipeline.py                    # One-shot pipeline runner script
├── dashboard.py                       # Streamlit dashboard
├── requirements.txt                   # Pinned Python dependencies
├── .env                               # DATABASE_URL — never committed
├── .env.example                       # Template for .env
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Environment Setup

```powershell
cd market-intel
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt --no-cache-dir
```

> **Python 3.14 note**: If you see ABI mismatch errors on import, add `--force-reinstall` to the pip command. This affects pre-compiled wheels (numpy, pandas, pyarrow, pydantic-core) when newer Python releases are ahead of wheel availability.

### 2. Configure Database

Copy `.env.example` to `.env` and fill in your PostgreSQL connection string:

```
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

A free Neon serverless PostgreSQL instance works out of the box: https://neon.tech

### 3. Download Dataset

Save the UCI Online Retail dataset as `data/Online Retail.csv`:

- Direct download: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
- Open in Excel → File → Save As → CSV UTF-8

Expected columns: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

---

## Running the Pipeline

### Option A — One-shot script

```powershell
.\venv\Scripts\python.exe run_pipeline.py
```

Runs all 6 pipeline steps sequentially, prints progress to console, and writes all artefacts to `models/` and `analysis/`.

**Typical output:**
```
[1/6] Data Validation ...   541,909 rows loaded, 4 guardrails passed
[2/6] Feature Engineering ...   406,829 clean rows, 374 daily rows x 16 features
[3/6] Model Benchmark ...
        Linear       MAPE=14.08%  RMSE=6,274  R2=0.791
        Ridge        MAPE=13.08%  RMSE=6,293  R2=0.788   <-- best
        RandomForest MAPE=14.74%  RMSE=8,528  R2=0.599
[4/6] Hyperparameter Tuning ...   best RF params found
[5/6] Model Selection ...   Ridge selected, retrained on full data
[6/6] Error Analysis ...   charts saved to analysis/
Pipeline complete in ~17 s
```

### Option B — Interactive Notebook

Open `pipeline_notebook.ipynb` in VS Code and run cells one by one. Each cell produces inline Plotly charts and printed summaries.

| Cell | Content |
|------|---------|
| 1 | Imports + environment check |
| 2 | Load & validate raw data |
| 3 | Clean & preprocess |
| 4 | RFM feature engineering |
| 5 | K-Means customer segmentation (pie chart) |
| 6 | Daily sales aggregation + lag/rolling features (trend chart) |
| 7 | Benchmark 3 models with TimeSeriesSplit (table) |
| 8 | Evaluation charts — actual vs predicted, residuals, feature importance |
| 9 | Model versioning + metadata export |
| 10 | Live API testing — requires server running (see below) |

---

## Running the Full Stack

Open three terminals simultaneously:

**Terminal 1 — API server**
```powershell
.\venv\Scripts\python.exe -m uvicorn api.app:app --reload
```
API base: http://127.0.0.1:8000  
Swagger UI: http://127.0.0.1:8000/docs

**Terminal 2 — Dashboard**
```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```
Dashboard: http://localhost:8501

**Terminal 3 — Notebook** *(optional, for notebook cell 10)*
> The API must be running before executing notebook cell 10.

---

## API Reference

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Model version, DB status, API version |
| `POST` | `/predict` | Sales forecast for N days |
| `GET` | `/benchmark` | Per-model CV results (RMSE, MAPE, R²) |
| `GET` | `/metadata` | Full training audit record |
| `GET` | `/drift` | Rolling mean vs training baseline |

### POST /predict

```json
// Request
{ "days": 7 }

// Response
{
  "model_version": "v2-Ridge-2026-02-24",
  "forecast_days": 7,
  "avg_prediction": 32377.13,
  "request_latency_ms": 7.3,
  "forecast_values": [31200.0, 33450.0, 34100.0, 29800.0, 28600.0, 35100.0, 33900.0]
}
```

### GET /drift

```json
{
  "training_mean": 29368.57,
  "rolling_mean_50": 31200.40,
  "drift_pct": 6.24,
  "drift_flag": false
}
```

`drift_flag` is `true` when `|drift_pct| > 20%` — indicating the live distribution has shifted significantly from training data.

### GET /benchmark

```json
{
  "Linear":       { "RMSE": 6274.73, "MAPE": 14.08, "R2": 0.7912 },
  "Ridge":        { "RMSE": 6292.69, "MAPE": 13.08, "R2": 0.7878 },
  "RandomForest": { "RMSE": 8528.42, "MAPE": 14.74, "R2": 0.5989 }
}
```

---

## Database Schema

Every `/predict` call is persisted to Neon PostgreSQL (`prediction_logs`):

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PK | Auto-increment |
| `timestamp` | TIMESTAMPTZ | UTC request time |
| `forecast_days` | INT | Number of days requested |
| `model_version` | TEXT | e.g. `v2-Ridge-2026-02-24` |
| `avg_prediction` | FLOAT | Mean of forecast_values |
| `request_latency_ms` | FLOAT | End-to-end request latency |
| `forecast_json` | TEXT | Full JSON array of daily values |

DB writes are non-fatal — a connection failure logs a warning but does not fail the `/predict` response.

---

## Model Performance

Results from the UCI Online Retail dataset run (February 2026):

**Cross-validation benchmark** (TimeSeriesSplit, 5 folds):

| Model | RMSE (GBP) | MAPE (%) | R² |
|-------|-----------|----------|-----|
| Linear Regression | 6,274 | 14.08 | 0.791 |
| **Ridge Regression** | **6,293** | **13.08** | **0.788** |
| Random Forest | 8,528 | 14.74 | 0.599 |

**Holdout evaluation (last fold only):**

| Metric | Value |
|--------|-------|
| RMSE | GBP 6,090 |
| MAPE | 9.55% |
| R² | 0.9355 |

Ridge was selected as best model by MAPE. The higher R² on the holdout fold (vs CV average) indicates stable generalisation on the most recent time period.

---

## Pipeline Design Decisions

### No data leakage — TimeSeriesSplit
Random `train_test_split` allows future data to leak into training (the model sees tomorrow to predict yesterday). `TimeSeriesSplit(n_splits=5)` always trains on the past and validates on the future, matching real deployment conditions.

### MAPE as the primary selection metric
MAPE is scale-independent: 10% error means the same whether daily sales are £1,000 or £100,000. It translates directly to business language and is easier to explain to non-technical stakeholders than RMSE.

### Soft price-validation guardrail
Negative `UnitPrice` rows (< 1% of data) are dropped with a warning rather than a hard fail. Hard-failing on 2 rows out of 541,909 would block the entire pipeline for a known data-entry artefact in the UCI dataset that does not affect model integrity.

### Log1p + StandardScaler for RFM clustering
RFM distributions are heavily right-skewed due to whale customers. `log1p` compresses the long tail so K-Means Euclidean distance is not dominated by outlier monetary values; `StandardScaler` then equalises feature scales across Recency, Frequency, and Monetary.

### Ridge over bare Linear Regression
Ridge (L2 regularisation) was selected by MAPE even though its RMSE was marginally higher. Regularisation reduces sensitivity to collinear lag features (Sales_Lag1, Sales_MA7, Sales_MA30), making the model more robust to slight distributional shifts without sacrificing predictive accuracy.

---

## Dataset

**Online Retail Dataset** — UCI Machine Learning Repository  
Daqing Chen et al., 2012. DOI: 10.24432/C5BW33

- ~541,909 transactions from a UK-based online retailer (2010–2011)
- ~4,300 unique customers across 38 countries
- 8 columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: data/Online Retail.csv` | Dataset not placed | Download and save to `data/` (see Quickstart §3) |
| `RuntimeError: Model not found at models/best_model.pkl` | Pipeline not run | Run `run_pipeline.py` first |
| `ModuleNotFoundError: No module named 'X'` | Wrong venv active | Use `.\venv\Scripts\python.exe` explicitly |
| API cell 10 — `ConnectionRefusedError` | Server not running | Start uvicorn in a separate terminal |
| `TypeError: string indices must be integers` in notebook cell 10 | API returned error dict | Check server logs; add `r.raise_for_status()` |
| Streamlit warning: `pandas<3 required` | Metadata mismatch | Upgrade: `pip install streamlit>=1.45 --upgrade` |

---

## References

1. Chen, D. et al. (2012). *Data Mining for the Online Retail Industry*. Journal of Database Marketing & Customer Strategy Management. DOI: 10.24432/C5BW33
2. Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
4. Hoerl, A. E. & Kennard, R. W. (1970). *Ridge Regression*. Technometrics, 12(1).
5. FastAPI documentation — https://fastapi.tiangolo.com
6. Streamlit documentation — https://docs.streamlit.io
7. Neon serverless PostgreSQL — https://neon.tech

---

**Version**: 2.0.0 · **Last updated**: February 2026 · **Status**: Production-ready
