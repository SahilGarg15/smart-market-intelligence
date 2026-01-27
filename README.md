# 📊 Smart Market Intelligence Platform
**Tech Stack**: Python | Pandas | Scikit-Learn | FastAPI | Streamlit

A complete machine learning platform for retail analytics featuring customer segmentation, sales forecasting, REST API, and interactive dashboard.

---

## 🎯 Project Overview

This platform demonstrates end-to-end ML pipeline development:
- **Phase 1**: Data loading and preprocessing
- **Phase 2**: ML model training (K-Means + Random Forest)
- **Phase 3**: RESTful API deployment (FastAPI)
- **Phase 4**: Interactive web dashboard (Streamlit)

### **Key Features:**
- ✅ **Customer Segmentation**: RFM-based K-Means clustering
- ✅ **Sales Forecasting**: Random Forest regression with lag features
- ✅ **REST API**: FastAPI endpoints for real-time predictions
- ✅ **Dashboard**: Streamlit interface with Plotly visualizations
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
│   └── sales_forecast_model.pkl   # Sales predictor
├── utils/                         # Utility modules
│   └── data_loader.py             # Phase 1: Data loading & cleaning
├── ml_engine.py                   # Phase 2: ML training pipeline
├── app.py                         # Phase 3: FastAPI backend
├── dashboard.py                   # Phase 4: Streamlit frontend
├── requirements.txt               # Python dependencies
├── MODEL_LIMITATIONS.md           # Known issues & improvements
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
- `data/daily_sales.csv`, `data/customer_segments.csv`
- Training metrics in console

**What it does:**
- Creates RFM features (Recency, Frequency, Monetary)
- Trains K-Means clustering (3 customer segments)
- Calculates Silhouette Score for cluster quality
- Aggregates sales by day
- Trains Random Forest for sales prediction
- Reports RMSE and model performance

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
- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (R-squared): Variance explained (0-1, higher is better)

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

## ⚠️ Known Limitations

**Important**: This project has known limitations and areas for improvement. Please review [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md) for:
- Customer segmentation issues (fixed K, outlier sensitivity)
- Sales forecasting limitations (error compounding, no confidence intervals)
- Data processing concerns
- Recommended improvements
- Interview talking points

Understanding these limitations demonstrates critical thinking and engineering maturity in your project defense.

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
9. **Critical Thinking**: Identifying and documenting model limitations

---

## 🎤 Interview Preparation

### **Common Questions & Answers:**

**Q: Why did you remove rows with missing CustomerIDs?**  
A: CustomerID is essential for RFM analysis and segmentation. We can't impute it because it's a unique identifier, not a feature we can predict.

**Q: Why use StandardScaler for K-Means?**  
A: K-Means uses Euclidean distance. Without scaling, high-value features (Monetary in $1000s) would dominate low-value features (Recency in days), creating biased clusters.

**Q: How would you improve the sales forecasting?**  
A: Three main improvements: (1) Use ARIMA or LSTM models designed for time-series, (2) Add external features like holidays and promotions, (3) Implement proper time-series cross-validation instead of random train/test split.

**Q: What's the biggest limitation of your current approach?**  
A: The sliding window forecasting compounds prediction errors. Each day's forecast uses previous predictions as inputs, so errors accumulate exponentially. See [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md) for full details.

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

**Q: What's the biggest limitation of your current approach?**  
A: The sliding window forecasting compounds prediction errors. Each day's forecast uses previous predictions as inputs, so errors accumulate exponentially. See [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md) for full details.

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

## 🚀 Future Improvements

**Short-term (1-2 weeks):**
- [ ] Implement GridSearchCV for hyperparameter tuning
- [ ] Add k-fold cross-validation for robust evaluation
- [ ] Include confidence intervals in forecasts
- [ ] Add data validation checks for edge cases

**Medium-term (1-2 months):**
- [ ] Switch to ARIMA/LSTM for better time-series forecasting
- [ ] Add exogenous variables (holidays, promotions)
- [ ] Implement DBSCAN or Gaussian Mixture for dynamic clustering
- [ ] Add authentication to API endpoints (OAuth2/JWT)

**Long-term (3-6 months):**
- [ ] Real-time model retraining with streaming data
- [ ] Deploy to cloud platform (AWS/Azure/GCP)
- [ ] A/B testing framework for model variants
- [ ] Advanced analytics (cohort analysis, churn prediction)
- [ ] Multi-country support with currency normalization

See [`MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md) for detailed improvement roadmap.

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

---

**Good luck with your project! 🚀**
