# 🔍 Model Limitations & Known Issues

This document outlines the drawbacks, limitations, and potential errors in the Smart Market Intelligence Platform models. Understanding these limitations is crucial for accurate interpretation of results and future improvements.

---

## 📊 **Customer Segmentation Model (K-Means Clustering)**

### **Limitations:**

#### 1. **Fixed Number of Clusters**
- **Issue**: Currently hardcoded to 3 segments (Loyal, At-Risk, Whale)
- **Impact**: May not capture the true diversity of customer behavior
- **Solution**: Use Elbow Method or Silhouette Analysis to determine optimal K
- **Code Location**: `ml_engine.py`, line ~45

#### 2. **Sensitivity to Outliers**
- **Issue**: K-Means is highly sensitive to outliers (e.g., one-time big spenders)
- **Impact**: Cluster centers can be skewed by extreme values
- **Example**: A single customer spending $100,000 can distort the "Whale" segment
- **Solution**: Consider using DBSCAN or removing outliers (>99th percentile)

#### 3. **Assumes Spherical Clusters**
- **Issue**: K-Means assumes clusters are spherical and equally sized
- **Impact**: Cannot capture elongated or irregularly shaped customer groups
- **Reality**: Customer behavior often forms non-spherical patterns
- **Solution**: Try Gaussian Mixture Models (GMM) for more flexibility

#### 4. **Log Transformation Loses Interpretability**
- **Issue**: We apply `np.log1p()` to RFM values before clustering
- **Impact**: Cluster centers are in log space, harder to explain to business stakeholders
- **Code**: `ml_engine.py`, line ~50
- **Trade-off**: Necessary for better distribution, but reduces clarity

#### 5. **No Temporal Dynamics**
- **Issue**: Segments are static snapshots, not tracking customer journey over time
- **Impact**: Cannot detect customers transitioning between segments (e.g., from Loyal to At-Risk)
- **Solution**: Implement time-series clustering or cohort analysis

#### 6. **Segment Labels Are Arbitrary**
- **Issue**: "Loyal", "At-Risk", "Whale" labels are manually assigned based on inspection
- **Impact**: May not accurately reflect the true characteristics of each cluster
- **Example**: Cluster 0 might actually be "Low-Value" instead of "Loyal"
- **Solution**: Analyze cluster characteristics after each training and update labels

---

## 📈 **Sales Forecasting Model (Random Forest Regressor)**

### **Limitations:**

#### 1. **Sliding Window Prediction Compounds Errors**
- **Issue**: To predict Day 7, we use predictions from Days 1-6 as inputs
- **Impact**: Errors accumulate exponentially with each future day
- **Code Location**: `app.py`, line ~230-250
- **Example**: If Day 1 is 10% off, Day 7 could be 70% off
- **Solution**: Use proper time-series models (ARIMA, LSTM)

#### 2. **Only Uses Last 30 Days (Limited Historical Context)**
- **Issue**: Model only trained on 30-day windows
- **Impact**: Cannot capture long-term trends or yearly seasonality
- **Missing Patterns**: Cannot detect Christmas spikes if trained in June
- **Solution**: Use longer windows (90-365 days) or seasonal decomposition

#### 3. **No Exogenous Variables (External Factors)**
- **Issue**: Model doesn't account for:
  - Promotions/discounts
  - Holidays
  - Marketing campaigns
  - Competitor actions
  - Economic indicators
- **Impact**: Predictions will be inaccurate during special events
- **Solution**: Add feature columns for holidays, promotions, weather, etc.

#### 4. **Lag Features Create Data Leakage Risk**
- **Issue**: Using `Sales_Lag1` (yesterday's sales) makes the model "too good"
- **Impact**: High R² score doesn't reflect real-world prediction difficulty
- **Reality**: In production, you don't have tomorrow's lag features
- **Code**: `ml_engine.py`, line ~150-160

#### 5. **No Confidence Intervals**
- **Issue**: Model gives point predictions without uncertainty estimates
- **Impact**: Business cannot assess risk of decisions based on predictions
- **Example**: Predicted $10,000 could actually be $5,000-$15,000
- **Solution**: Use Quantile Regression or Conformal Prediction

#### 6. **Hyperparameters Not Tuned**
- **Issue**: Using default Random Forest parameters
- **Current Settings**: `n_estimators=100`, `max_depth=None`, `min_samples_split=2`
- **Impact**: Sub-optimal performance, possible overfitting
- **Solution**: Use GridSearchCV or RandomizedSearchCV to find best parameters

#### 7. **No Cross-Validation**
- **Issue**: Single train/test split (80/20)
- **Impact**: Performance metrics may be optimistic or pessimistic depending on the split
- **Solution**: Implement k-fold cross-validation or time-series cross-validation

#### 8. **Ignores Trend and Seasonality**
- **Issue**: Random Forest doesn't explicitly model trends or seasonal patterns
- **Impact**: May fail to predict gradual growth or seasonal spikes
- **Solution**: Use SARIMA, Prophet, or add trend/seasonal features

---

## 🔧 **Data Processing Issues**

### **1. Missing Data Handling**
- **Current Approach**: Drop all rows with missing CustomerID
- **Data Loss**: Could lose 20-30% of transactions
- **Impact**: Reduced training data, potential bias
- **Alternative**: Impute missing values or use transactions without CustomerID separately

### **2. Cancelled Transactions**
- **Current Approach**: Remove all invoices starting with 'C'
- **Missed Opportunity**: Could analyze return patterns
- **Impact**: Cannot model customer churn or product quality issues

### **3. Feature Engineering Limitations**
- **Missing Features**:
  - Customer lifetime value (CLV)
  - Product categories/types
  - Geographic patterns
  - Customer acquisition channel
- **Impact**: Model cannot leverage important business insights

### **4. No Data Validation**
- **Issue**: No checks for:
  - Negative prices (after filtering)
  - Duplicates
  - Unrealistic quantities (e.g., 10,000 units in one order)
- **Impact**: Garbage in, garbage out

---

## ⚠️ **Known Errors & Edge Cases**

### **Error 1: Empty Dataset After Cleaning**
- **Scenario**: If dataset has >90% missing CustomerIDs
- **Result**: `ValueError: Cannot train model on empty dataset`
- **Fix**: Add validation check in `data_loader.py`

### **Error 2: Insufficient Historical Data**
- **Scenario**: Less than 30 days of sales data
- **Result**: Sales model cannot create lag features
- **Fix**: Add check and fallback to simpler model

### **Error 3: Single Customer Segment**
- **Scenario**: All customers have similar RFM values
- **Result**: K-Means creates overlapping clusters
- **Fix**: Add variance check before clustering

### **Error 4: Negative Sales Predictions**
- **Scenario**: Random Forest can predict negative sales
- **Result**: API returns nonsensical forecasts
- **Fix**: Apply `max(0, prediction)` constraint (add in `app.py`)

### **Error 5: API Timeout on Large Forecasts**
- **Scenario**: Requesting forecast for 30 days with slow model
- **Result**: Browser timeout, incomplete response
- **Fix**: Implement async prediction or reduce max forecast days

---

## 🎯 **Metrics Interpretation Warnings**

### **1. R² Score Can Be Misleading**
- **Current R²**: ~0.85-0.95 (appears excellent)
- **Reality**: High R² due to using lag features (yesterday's sales)
- **Caution**: Does NOT mean model predicts well 7 days ahead
- **Better Metric**: MAPE (Mean Absolute Percentage Error)

### **2. Silhouette Score Ambiguity**
- **Current Score**: ~0.4-0.6 (printed in training)
- **Interpretation**: 
  - < 0.3 = Poor clustering
  - 0.3-0.5 = Weak structure
  - 0.5-0.7 = Reasonable structure
  - > 0.7 = Strong structure
- **Caution**: High score doesn't mean segments are business-relevant

### **3. RMSE in Dollars Is Absolute**
- **Example**: RMSE = $5,000
- **Problem**: Is this good or bad?
  - Good if average sales = $50,000/day (10% error)
  - Bad if average sales = $10,000/day (50% error)
- **Solution**: Always report percentage error (MAPE) alongside RMSE

---

## 🚀 **Recommended Improvements**

### **Short-Term (Easy Fixes)**
1. ✅ Add data validation checks
2. ✅ Implement negative prediction constraints
3. ✅ Add try-except error handling in API
4. ✅ Display MAPE alongside RMSE
5. ✅ Add loading indicators in dashboard

### **Medium-Term (Moderate Effort)**
1. 🔄 Implement Elbow Method for optimal K selection
2. 🔄 Add hyperparameter tuning (GridSearchCV)
3. 🔄 Implement cross-validation
4. 🔄 Add confidence intervals to forecasts
5. 🔄 Include holiday/promotion features

### **Long-Term (Major Enhancements)**
1. 🔮 Replace Random Forest with LSTM/Prophet for time series
2. 🔮 Implement customer journey tracking
3. 🔮 Add A/B testing framework
4. 🔮 Build model monitoring dashboard
5. 🔮 Implement online learning (model updates with new data)

---

## 📝 **Interview Talking Points**

When discussing these limitations in your interview:

### **Be Honest About Limitations:**
> "The K-Means clustering assumes spherical clusters, which may not capture all customer behavior patterns. In a production system, I would compare this with DBSCAN or hierarchical clustering."

### **Show Understanding of Trade-offs:**
> "I used log transformation to handle skewed RFM distributions, but this reduces interpretability. The trade-off is necessary for better clustering quality."

### **Demonstrate Problem-Solving:**
> "The sliding window forecasting compounds errors. A better approach would be ARIMA or LSTM, which are designed for time-series prediction."

### **Acknowledge Scope Constraints:**
> "Given the academic project timeframe, I focused on demonstrating end-to-end ML pipeline. In production, I would invest more in feature engineering and model selection."

### **Highlight Learning:**
> "Through this project, I learned that high R² doesn't always mean good predictions, especially with time-series data and lag features."

---
