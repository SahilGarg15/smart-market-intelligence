# 📊 Smart Market Intelligence Platform

**Tech Stack**: Python | Pandas | Scikit-Learn | FastAPI | Streamlit

A complete machine learning platform for retail analytics featuring customer segmentation, sales forecasting, REST API, and interactive dashboard.

---

## 🎯 Project Overview

This platform demonstrates end-to-end ML pipeline development:
- **Phase 1**: Data loading and preprocessing with validation
- **Phase 2**: ML model training (K-Means + Random Forest) with proper time-series splitting
- **Phase 3**: RESTful API deployment (FastAPI) with robust error handling
- **Phase 4**: Interactive web dashboard (Streamlit)

### **Key Features:**
- ✅ **Customer Segmentation**: RFM-based K-Means clustering
- ✅ **Sales Forecasting**: Random Forest regression with lag features
- ✅ **REST API**: FastAPI endpoints for real-time predictions
- ✅ **Dashboard**: Streamlit interface with Plotly visualizations
- ✅ **Data Validation**: Fail-fast guardrails for corrupted data
- ✅ **No Data Leakage**: TimeSeriesSplit for proper time-series validation
- ✅ **Model Versioning**: Timestamped models with metadata tracking
- ✅ **Production-Ready**: Input validation, error handling, startup checks
- ✅ **Well-Documented**: Extensive comments explaining each step

---

## 📁 Project Structure

```
market-intel/
├── data/                          # Dataset folder
│   ├── Online Retail.csv          # Original dataset (place here)
│   ├── cleaned_data.csv           # Cleaned data (auto-generated)
│   ├── daily_sales.csv            # Aggregated sales (auto-generated)
│   └── customer_segments.csv      # RFM segments (auto-generated)
├── models/                        # Trained ML models (auto-generated)
│   ├── kmeans_model.pkl           # Customer segmentation
│   ├── scaler.pkl                 # Feature scaler
│   ├── sales_forecast_model.pkl   # Sales predictor
│   ├── sales_model_v1.pkl         # Versioned backup (NEW)
│   └── metadata.json              # Training metadata (NEW)
├── utils/                         # Utility modules
│   └── data_loader.py             # Phase 1: Data loading & cleaning with validation
├── ml_engine.py                   # Phase 2: ML training pipeline (TimeSeriesSplit)
├── app.py                         # Phase 3: FastAPI backend (robust error handling)
├── dashboard.py                   # Phase 4: Streamlit frontend
├── requirements.txt               # Python dependencies
├── MODEL_LIMITATIONS.md           # Known issues & improvements
├── LIMITATIONS_FOR_REPORT.md      
└── README.md                      # This file
```

---

## 🚀 Quick Start Guide

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 500MB free disk space

### **Step 1: Clone & Setup**

```bash
# Navigate to project directory
cd market-intel

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Download Dataset**

Download the **Online Retail Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail):
- **Direct link**: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
- Convert to CSV format (Excel → Save As → CSV)
- Save as `data/Online Retail.csv`

**Expected columns:**
- InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### **Step 3: Run Phase 1 - Data Cleaning**

```bash
python utils/data_loader.py
```

**Output:**
- `data/cleaned_data.csv` - Cleaned dataset
- Console summary of data quality

**What it does:**
- **Validates data** (fails fast on empty/corrupted data)
- Removes rows with missing CustomerID
- Filters out negative quantities (returns)
- Creates TotalAmount column
- Converts dates to datetime format

### **Step 4: Run Phase 2 - Train ML Models**

```bash
python ml_engine.py
```

**Output:**
- `models/kmeans_model.pkl`, `models/scaler.pkl`, `models/sales_forecast_model.pkl`
- `models/sales_model_v1.pkl` (versioned backup)
- `models/metadata.json` (training date, metrics, hyperparameters)
- `data/daily_sales.csv`, `data/customer_segments.csv`
- Training metrics in console

**What it does:**
- Creates RFM features (Recency, Frequency, Monetary)
- Trains K-Means clustering (3 customer segments)
- Calculates Silhouette Score for cluster quality
- Aggregates sales by day
- **Uses TimeSeriesSplit** (5 folds, no data leakage!)
- Trains Random Forest for sales prediction
- Reports **RMSE** (dollar error) and **MAPE** (percentage error)
- **Saves versioned models** with metadata for rollback capability

### **Step 5: Run Phase 3 - Start API Server**

```bash
python app.py
```

Or with uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Endpoints:**
- 🌐 API Docs: http://localhost:8000/docs
- ✅ Health Check: `GET /health`
- 📊 Customer Segment: `POST /segment`
- 📈 Sales Forecast: `GET /forecast/{days}`

**New Production Features:**
- 🔒 **Startup Validation**: Fails if models missing (clear error message)
- ✅ **Input Validation**: Rejects negative RFM values, limits forecast to 30 days
- 🛡️ **Error Handling**: Try-catch blocks with detailed error logging
- 📊 **Health Endpoint**: Check model loading status

**Test with curl:**
```bash
# Segment a customer
curl -X POST "http://localhost:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{"Recency": 30, "Frequency": 5, "Monetary": 250.50}'

# Forecast next 7 days
curl http://localhost:8000/forecast/7
```

### **Step 6: Run Phase 4 - Launch Dashboard**

```bash
streamlit run dashboard.py
```

**Dashboard URL**: http://localhost:8501

**Features:**
- 📊 **Business Overview Tab**:
  - Key metrics (Total Revenue, Avg Daily Sales, Customer Count)
  - Daily sales trend chart
  - Customer segment distribution pie chart
  - Segment statistics table

- 🔮 **Prediction Lab Tab**:
  - Interactive customer segmentation with RFM sliders
  - Sales forecasting with customizable days
  - Real-time API integration
  - Visual predictions with Plotly charts

---

## 🏗️ MLOps Best Practices (Production Enhancements)

### **1. Data Validation Layer**
**File**: `utils/data_loader.py`

```python
def validate_data(df):
    # Fail fast on:
    # - Empty dataframes
    # - Negative UnitPrice (data corruption)
    # - Unrealistic Quantity >100,000 (data errors)
```

**Why?** Prevents training models on corrupted data, provides clear error messages early in the pipeline.

### **2. No Data Leakage - TimeSeriesSplit**
**File**: `ml_engine.py`

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# Always train on past, test on future ✓
```

**Before**: Random `train_test_split` could use future data to predict past (LEAKAGE!)  
**After**: TimeSeriesSplit ensures chronological split (train on 90%, test on 10%)

### **3. MAPE Metrics**
**File**: `ml_engine.py`

```python
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# Industry-standard percentage error metric
```

**Why?** Scale-independent (10% error is 10% whether sales are $100 or $10,000). Business-friendly metric.

### **4. Model Versioning**
**File**: `ml_engine.py`

```python
# Saves:
# - sales_forecast_model.pkl (primary)
# - sales_model_v1.pkl (versioned backup)
# - metadata.json (training date, RMSE, MAPE, hyperparameters)
```

**Benefits:**
- Track model performance over time
- Rollback to previous version if new model performs worse
- A/B test different model versions
- Audit trail for production systems

### **5. Robust API Error Handling**
**File**: `app.py`

```python
# Startup validation
if missing_files:
    raise RuntimeError("Models not found! Run ml_engine.py first.")

# Input validation
if days > 30:
    raise HTTPException(400, "Max 30 days (error compounds!)")

# Error logging
except Exception as e:
    print(f"Error: {error_details}")
    raise HTTPException(500, "Forecast failed. Check logs.")
```

**Why?** Production APIs must fail gracefully with clear messages, not cryptic stack traces.

---

## 🧠 Technical Details

### **Customer Segmentation**

**Algorithm**: K-Means Clustering  
**Features**: RFM (Recency, Frequency, Monetary)  
**Preprocessing**: Log transformation + StandardScaler

**Segments:**
- **Loyal**: Low recency, high frequency/monetary
- **At-Risk**: High recency, low frequency
- **Whale**: Very high monetary value

**Why K-Means?**
- Industry standard for RFM segmentation
- Fast and scalable
- Easy to interpret for business stakeholders

**Evaluation Metric**: Silhouette Score (0-1, higher is better)

### **Sales Forecasting**

**Algorithm**: Random Forest Regressor  
**Features**:
- Temporal: Year, Month, Day, DayOfWeek, Quarter, DayOfYear, WeekOfYear
- Context: NumTransactions, NumCustomers, TotalQuantity
- Lag: Previous day sales, same day last week
- Trend: 7-day moving average

**Why Random Forest?**
- Handles non-linear patterns (sales curves)
- Robust to outliers (holiday spikes)
- Provides feature importance
- No extensive feature engineering needed

**Evaluation Metrics**:
- **RMSE** (Root Mean Squared Error): Average error in dollars
- **MAE** (Mean Absolute Error): Average absolute error
- **MAPE** (Mean Absolute Percentage Error): Industry-standard percentage error
- **R²** (R-squared): Variance explained (0-1, higher is better)

**Training Method**: TimeSeriesSplit with 5 folds (no data leakage)

**Prediction Method**: Sliding window (uses previous predictions as inputs for future days)

---

## 📊 Dataset Information

**Source**: [UCI Machine Learning Repository - Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)

**Description**: Transactional data from UK-based online retail company (2010-2011)

**Size**: ~540,000 transactions, ~4,300 customers

**Attributes**:
- InvoiceNo: Invoice number (unique per transaction)
- StockCode: Product code
- Description: Product name
- Quantity: Quantity purchased
- InvoiceDate: Date and time of transaction
- UnitPrice: Price per unit
- CustomerID: Unique customer identifier
- Country: Customer's country

---


## 🎓 Learning Outcomes

By completing this project, you will demonstrate:

1. **Data Engineering**: Loading, cleaning, and transforming real-world messy data
2. **Feature Engineering**: Creating RFM features, temporal features, lag features
3. **Unsupervised Learning**: K-Means clustering for customer segmentation
4. **Supervised Learning**: Random Forest regression for time-series forecasting
5. **Model Evaluation**: Understanding MAE, RMSE, R², Silhouette Score
6. **API Development**: Building production-ready REST APIs with FastAPI
7. **Web Development**: Creating interactive dashboards with Streamlit
8. **Software Engineering**: Project structure, version control, documentation

---

## 🎤 Interview Preparation

### **Common Questions & Answers:**

**Q: Why did you remove rows with missing CustomerIDs?**  
A: CustomerID is essential for RFM analysis and segmentation. We can't impute it because it's a unique identifier, not a feature we can predict.

**Q: Why use StandardScaler for K-Means?**  
A: K-Means uses Euclidean distance. Without scaling, high-value features (Monetary in $1000s) would dominate low-value features (Recency in days), creating biased clusters.

**Q: How did you prevent data leakage in time-series forecasting?**  
A: I replaced random `train_test_split` with `TimeSeriesSplit` (5 folds). This ensures we ALWAYS train on past data and validate on future data, preventing the model from "seeing the future" during training.

**Q: What is MAPE and why is it important?**  
A: MAPE (Mean Absolute Percentage Error) is a scale-independent metric: `mean(|actual - pred| / |actual|) × 100`. A 10% error means the same thing whether sales are $100 or $10,000. It's an industry-standard metric that business stakeholders understand better than RMSE.

**Q: How do you handle model versioning?**  
A: I save both `sales_forecast_model.pkl` (current production model) and `sales_model_v1.pkl` (timestamped backup). I also create `metadata.json` with training date, RMSE, MAPE, and hyperparameters. This allows rollback if a new model performs worse and provides an audit trail.

**Q: What production safeguards did you implement in the API?**  
A: Three layers: (1) **Startup validation** - API refuses to start if model files are missing, (2) **Input validation** - Reject negative RFM values and limit forecast to 30 days, (3) **Error handling** - Try-catch blocks with detailed logging for debugging production issues.

---

## 📝 Usage Examples

### **API Examples**

```bash
# Customer Segmentation
curl -X POST "http://localhost:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "Recency": 10,
    "Frequency": 15,
    "Monetary": 2500.50
  }'

# Response:
{
  "cluster_id": 0,
  "segment_name": "Loyal",
  "rfm_input": {"Recency": 10, "Frequency": 15, "Monetary": 2500.50}
}

# Sales Forecast
curl http://localhost:8000/forecast/7

# Response:
{
  "forecast_days": 7,
  "predictions": [
    {"date": "2011-12-10", "predicted_sales": 15234.56},
    {"date": "2011-12-11", "predicted_sales": 14987.32},
    ...
  ]
}
```

### **Python Examples**

```python
import requests

# Segment a customer
response = requests.post(
    "http://localhost:8000/segment",
    json={"Recency": 30, "Frequency": 5, "Monetary": 250.50}
)
print(response.json())

# Get 14-day forecast
forecast = requests.get("http://localhost:8000/forecast/14")
print(forecast.json())
```

---

## 🛠️ Troubleshooting

### **Issue: Dataset not found**
```
FileNotFoundError: data/Online Retail.csv
```
**Solution**: Download the dataset and place it in `data/` folder (see Step 2)

### **Issue: API returns 503 Service Unavailable**
```
{"detail": "Models not loaded"}
```
**Solution**: Run `python ml_engine.py` first to train and save models

### **Issue: Dashboard shows "Data not available"**
```
Warning: Data not available. Please run ml_engine.py first.
```
**Solution**: Run Phase 2 (`python ml_engine.py`) to generate required CSV files

### **Issue: ModuleNotFoundError**
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### **Issue: Port already in use**
```
ERROR: [Errno 48] Address already in use
```
**Solution**: Change port or kill existing process:
```bash
# Change port
uvicorn app:app --port 8001

# Or kill existing process (Linux/Mac)
lsof -ti:8000 | xargs kill -9
```

---*Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total amount spent

**Why K-Means?**
- Simple and interpretable
- Works well with numerical features
- Industry standard for RFM segmentation

**Preprocessing:**
- StandardScaler to normalize features (mean=0, std=1)
- Prevents high-value monetary amounts from dominating

### Sales Prediction

**Algorithm**: Random Forest Regressor  
**Features**:
- Temporal: Year, Month, Day, DayOfWeek, Quarter, etc.
- Lag features: Previous day sales, same day last week
- Rolling averages: 7-day moving average
- Context: Number of transactions, customers, quantity

**Why Random Forest?**
- Handles non-linear patterns (sales don't follow straight lines)
- Robust to outliers (holiday spikes, slow days)
- Provides feature importance (what drives sales?)
- No extensive feature engineering needed

**Evaluation Metrics:**
- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (R-squared): How much variance is explained (1.0 = perfect)

## 📊 Key Insights for Interview

### Why this tech stack?

- **Pandas**: Standard for data manipulation in Python
- **Scikit-learn**: Industry-standard ML library, simple API
- **FastAPI**: Modern, fast API framework with auto-documentation
- **Streamlit**: Quick prototyping of data apps without HTML/CSS/JS

### Design Decisions

1. **RFM for Segmentation**: Proven marketing framework, easy to explain
2. **Random Forest over Linear Models**: Captures complex patterns in retail data
3. **Lag Features**: Yesterday's sales help predict today's (temporal dependency)
4. **Data Cleaning**: Removed cancelled orders, negative prices (data quality)
5. **Train/Test Split**: 80/20 to evaluate generalization

### Common Interview Questions

**Q: Why remove missing CustomerIDs?**  
A: We need CustomerID for segmentation. Cannot impute because they're unique identifiers.

**Q: Why standardize features for K-Means?**  
A: K-Means uses Euclidean distance. Without scaling, large-scale features (e.g., Monetary in $1000s) dominate small-scale features (e.g., Recency in days).

**Q: How do you handle overfitting?**  
A: For Random Forest, I limit max_depth and min_samples_split. Ideally, I would implement k-fold cross-validation and hyperparameter tuning with GridSearchCV.

**Q: Why Random Forest over Linear Regression?**  
A: Sales data has non-linear patterns (weekend spikes, seasonal trends). Random Forest captures these complex relationships without manual feature engineering, and it's robust to outliers.

---

## 📝 Usage Examples

### **API Examples**

```bash
# Customer Segmentation
curl -X POST "http://localhost:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "Recency": 10,
    "Frequency": 15,
    "Monetary": 2500.50
  }'

# Response:
{
  "cluster_id": 0,
  "segment_name": "Loyal",
  "rfm_input": {"Recency": 10, "Frequency": 15, "Monetary": 2500.50}
}

# Sales Forecast
curl http://localhost:8000/forecast/7

# Response:
{
  "forecast_days": 7,
  "predictions": [
    {"date": "2011-12-10", "predicted_sales": 15234.56},
    {"date": "2011-12-11", "predicted_sales": 14987.32},
    ...
  ]
}
```

### **Python Examples**

```python
import requests

# Segment a customer
response = requests.post(
    "http://localhost:8000/segment",
    json={"Recency": 30, "Frequency": 5, "Monetary": 250.50}
)
print(response.json())

# Get 14-day forecast
forecast = requests.get("http://localhost:8000/forecast/14")
print(forecast.json())
```

---

## 🛠️ Troubleshooting

### **Issue: Dataset not found**
```
FileNotFoundError: data/Online Retail.csv
```
**Solution**: Download the dataset and place it in `data/` folder (see Step 2)

### **Issue: API returns 503 Service Unavailable**
```
{"detail": "Models not loaded"}
```
**Solution**: Run `python ml_engine.py` first to train and save models

### **Issue: Dashboard shows "Data not available"**
```
Warning: Data not available. Please run ml_engine.py first.
```
**Solution**: Run Phase 2 (`python ml_engine.py`) to generate required CSV files

### **Issue: ModuleNotFoundError**
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### **Issue: Port already in use**
```
ERROR: [Errno 48] Address already in use
```
**Solution**: Change port or kill existing process:
```bash
# Change port
uvicorn app:app --port 8001

# Or kill existing process (Linux/Mac)
lsof -ti:8000 | xargs kill -9
```

---


## 🤝 Contributing

This is an academic project. Suggestions and feedback are welcome!

**Forking Guidelines:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Document your changes
4. Submit a pull request

---

## 🎯 Project Metrics

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~800 Python |
| **Development Time** | 4 weeks (part-time) |
| **API Endpoints** | 3 |
| **ML Models** | 2 (K-Means + Random Forest) |
| **Dashboard Tabs** | 2 |
| **Test Coverage** | 0% (future improvement) |

---


## 📚 References

1. **Dataset**: Daqing Chen, Sai Liang Sain, and Kun Guo, "Online Retail Dataset", UCI Machine Learning Repository, 2015. DOI: 10.24432/C5BW33

2. **RFM Analysis**: Hughes, A. M. (1994). Strategic database marketing. Chicago: Probus.

3. **K-Means**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations". Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.

4. **Random Forest**: Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5–32.

5. **FastAPI Documentation**: https://fastapi.tiangolo.com/

6. **Streamlit Documentation**: https://docs.streamlit.io/

---

## ⭐ Acknowledgments

- **UCI Machine Learning Repository** for providing the Online Retail Dataset
- **FastAPI** and **Streamlit** communities for excellent documentation
- **Academic Supervisor** for guidance and feedback
- **Open-source community** for Python ML libraries

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: ✅ Complete & Production-Ready
5. **Model Evaluation**: MAE, RMSE, R² metrics
6. **API Development**: Building REST APIs with FastAPI
7. **Data Visualization**: Interactive dashboards with Streamlit and Plotly
8. **Project Structure**: Organizing ML projects professionally

## 📚 Resources

- [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 👨‍💻 Author

[Sahil Garg](https://github.com/SahilGarg15)
