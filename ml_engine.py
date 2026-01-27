"""
Machine Learning Engine for Market Intelligence Platform
Implements customer segmentation and sales forecasting
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os


def create_rfm_features(df):
    """
    Create RFM (Recency, Frequency, Monetary) features for customer segmentation
    
    RFM Analysis:
    - Recency: How recently did the customer make a purchase?
    - Frequency: How often do they purchase?
    - Monetary: How much money do they spend?
    
    Args:
        df: Cleaned dataframe with CustomerID, InvoiceDate, TotalAmount
        
    Returns:
        pd.DataFrame: RFM features per customer
    """
    print("\n=== Creating RFM Features ===")
    
    # Get the most recent date in the dataset as the reference point
    # Reason: We calculate "recency" relative to the last date in our data
    max_date = df['InvoiceDate'].max()
    print(f"Reference date (most recent): {max_date}")
    
    # Group by CustomerID and calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency: days since last purchase
        'InvoiceNo': 'nunique',  # Frequency: number of unique invoices (transactions)
        'TotalAmount': 'sum'  # Monetary: total amount spent
    }).reset_index()
    
    # Rename columns for clarity
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    print(f"\nRFM features created for {len(rfm)} customers")
    print("\nRFM Summary Statistics:")
    print(rfm.describe())
    
    return rfm


def customer_segmentation(rfm, n_clusters=3):
    """
    Perform customer segmentation using K-Means clustering on RFM features
    
    Args:
        rfm: DataFrame with RFM features
        n_clusters: Number of customer segments (default=3: Low, Mid, High value)
        
    Returns:
        tuple: (rfm_with_segments, kmeans_model, scaler, silhouette)
    """
    print("\n=== Customer Segmentation (K-Means Clustering) ===")
    
    # Extract RFM features for clustering
    # We exclude CustomerID as it's just an identifier, not a feature
    X = rfm[['Recency', 'Frequency', 'Monetary']].values
    
    # Standardize features before clustering
    # Reason: K-Means uses Euclidean distance. Without scaling, features with
    # larger ranges (e.g., Monetary in thousands) would dominate smaller ones
    # (e.g., Recency in days). StandardScaler transforms to mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nTraining K-Means with {n_clusters} clusters...")
    
    # Train K-Means clustering
    # random_state=42 ensures reproducible results
    # n_init=10 runs the algorithm 10 times with different initializations
    # and picks the best result (reduces impact of random initialization)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the RFM dataframe
    rfm_segmented = rfm.copy()
    rfm_segmented['Segment'] = clusters
    
    # Calculate Silhouette Score
    # Silhouette Score measures how well-separated the clusters are
    # Range: -1 to 1 (higher is better, >0.5 is good, >0.7 is excellent)
    # Reason: We can't use accuracy here (unsupervised learning has no labels)
    silhouette = silhouette_score(X_scaled, clusters)
    
    print(f"\nSilhouette Score: {silhouette:.4f}")
    if silhouette > 0.5:
        print("✓ Good cluster separation!")
    elif silhouette > 0.3:
        print("✓ Acceptable cluster separation")
    else:
        print("⚠ Clusters may overlap - consider adjusting number of clusters")
    
    # Analyze each segment to understand customer groups
    print("\n=== Segment Analysis ===")
    for segment in range(n_clusters):
        segment_data = rfm_segmented[rfm_segmented['Segment'] == segment]
        print(f"\nSegment {segment} ({len(segment_data)} customers):")
        print(f"  Avg Recency: {segment_data['Recency'].mean():.1f} days")
        print(f"  Avg Frequency: {segment_data['Frequency'].mean():.1f} purchases")
        print(f"  Avg Monetary: ${segment_data['Monetary'].mean():.2f}")
        
        # Interpret the segment based on RFM values
        avg_recency = segment_data['Recency'].mean()
        avg_monetary = segment_data['Monetary'].mean()
        
        if avg_recency < rfm['Recency'].median() and avg_monetary > rfm['Monetary'].median():
            segment_type = "HIGH VALUE (Recent + High Spending)"
        elif avg_monetary > rfm['Monetary'].median():
            segment_type = "MID VALUE (High Spending)"
        else:
            segment_type = "LOW VALUE (Low Engagement)"
        
        print(f"  Interpretation: {segment_type}")
    
    return rfm_segmented, kmeans, scaler, silhouette


def prepare_sales_forecasting_data(df, window_size=30, forecast_days=7):
    """
    Aggregate sales by day and prepare data for time series forecasting
    
    Args:
        df: Cleaned dataframe with InvoiceDate and TotalAmount
        window_size: Number of past days to use as features (default=30)
        forecast_days: Number of future days to predict (default=7)
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, daily_sales)
    """
    print("\n=== Preparing Sales Forecasting Data ===")
    
    # Aggregate sales by day
    # Reason: We predict daily total sales, not individual transactions
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
    daily_sales.columns = ['Date', 'Sales']
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)
    
    print(f"Daily sales data: {len(daily_sales)} days")
    print(f"Date range: {daily_sales['Date'].min()} to {daily_sales['Date'].max()}")
    print(f"Total sales: ${daily_sales['Sales'].sum():,.2f}")
    
    # Create sliding window features
    # For each day, we use the previous 'window_size' days as features
    # to predict the next 'forecast_days' days
    # Example: Use days 1-30 to predict day 31, days 2-31 to predict day 32, etc.
    
    X = []  # Features (past sales)
    y = []  # Target (future sales)
    
    # We need at least window_size + forecast_days data points
    min_data_points = window_size + forecast_days
    
    if len(daily_sales) < min_data_points:
        raise ValueError(f"Not enough data. Need at least {min_data_points} days, got {len(daily_sales)}")
    
    # Create training samples using sliding window
    for i in range(len(daily_sales) - window_size - forecast_days + 1):
        # Features: sales from the past 'window_size' days
        X.append(daily_sales['Sales'].iloc[i:i + window_size].values)
        
        # Target: average sales over the next 'forecast_days' days
        # Reason: We predict average daily sales for the forecast period
        # (could also predict exact values for each day separately)
        y.append(daily_sales['Sales'].iloc[i + window_size:i + window_size + forecast_days].mean())
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    # Use last 20% of data for testing (time series: don't shuffle!)
    # Reason: In time series, we test on future data, not random samples
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features per sample: {X_train.shape[1]} (last {window_size} days)")
    
    return X_train, y_train, X_test, y_test, daily_sales


def sales_forecasting(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model to forecast sales
    
    Args:
        X_train: Training features (past sales windows)
        y_train: Training targets (future average sales)
        X_test: Test features
        y_test: Test targets
        
    Returns:
        tuple: (model, rmse)
    """
    print("\n=== Sales Forecasting (Random Forest Regressor) ===")
    
    # Train Random Forest Regressor
    # Why Random Forest?
    # 1. Handles non-linear patterns (sales don't follow straight lines)
    # 2. Robust to outliers (holiday spikes, unusual days)
    # 3. No need for extensive feature engineering
    # 4. Provides feature importance (which past days matter most?)
    
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees (more = better but slower)
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=5,  # Require at least 5 samples to split
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE (Root Mean Squared Error)
    # RMSE measures the average prediction error in the same units as sales ($)
    # Lower RMSE = better predictions
    # Reason: RMSE penalizes large errors more than MAE, important for sales
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Calculate additional metrics for context
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    
    print(f"\n=== Model Performance ===")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Show sample predictions vs actual
    print("\n=== Sample Predictions ===")
    print("Actual vs Predicted (first 5 test samples):")
    for i in range(min(5, len(y_test))):
        print(f"  Actual: ${y_test[i]:,.2f}  |  Predicted: ${y_pred[i]:,.2f}  |  Error: ${abs(y_test[i] - y_pred[i]):,.2f}")
    
    return model, rmse


def save_models(kmeans, scaler, rf_model, model_dir='models'):
    """
    Save trained models to disk using joblib
    
    Joblib is preferred over pickle for scikit-learn models because:
    1. More efficient for large numpy arrays
    2. Better compression
    3. Faster serialization
    
    Args:
        kmeans: Trained K-Means model
        scaler: Fitted StandardScaler
        rf_model: Trained Random Forest model
        model_dir: Directory to save models
    """
    print(f"\n=== Saving Models ===")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save customer segmentation models
    # We save both the K-Means model and the scaler because we need to
    # transform new data the same way during inference
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"✓ Saved K-Means model to {model_dir}/kmeans_model.pkl")
    print(f"✓ Saved StandardScaler to {model_dir}/scaler.pkl")
    
    # Save sales forecasting model
    joblib.dump(rf_model, os.path.join(model_dir, 'sales_forecast_model.pkl'))
    print(f"✓ Saved Random Forest model to {model_dir}/sales_forecast_model.pkl")
    
    print("\nAll models saved successfully!")


def load_models(model_dir='models'):
    """
    Load previously saved models from disk
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        tuple: (kmeans, scaler, rf_model)
    """
    print(f"\n=== Loading Models ===")
    
    kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    rf_model = joblib.load(os.path.join(model_dir, 'sales_forecast_model.pkl'))
    
    print("✓ All models loaded successfully!")
    
    return kmeans, scaler, rf_model


def main():
    """
    Main execution function to train all models
    Run this after cleaning the data with data_loader.py
    """
    print("=" * 70)
    print("MARKET INTELLIGENCE PLATFORM - MACHINE LEARNING ENGINE")
    print("=" * 70)
    
    # Load cleaned data
    print("\n[Step 1/5] Loading cleaned data...")
    if not os.path.exists('data/cleaned_data.csv'):
        raise FileNotFoundError(
            "cleaned_data.csv not found! Please run data_loader.py first."
        )
    
    df = pd.read_csv('data/cleaned_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"✓ Loaded {len(df)} rows")
    
    # Task 1: Customer Segmentation
    print("\n[Step 2/5] Customer Segmentation...")
    rfm = create_rfm_features(df)
    rfm_segmented, kmeans, scaler, silhouette = customer_segmentation(rfm, n_clusters=3)
    
    # Save segmented customers for later use
    rfm_segmented.to_csv('data/customer_segments.csv', index=False)
    print("\n✓ Customer segments saved to data/customer_segments.csv")
    
    # Task 2: Sales Forecasting
    print("\n[Step 3/5] Sales Forecasting...")
    X_train, y_train, X_test, y_test, daily_sales = prepare_sales_forecasting_data(
        df, window_size=30, forecast_days=7
    )
    rf_model, rmse = sales_forecasting(X_train, y_train, X_test, y_test)
    
    # Save daily sales for visualization
    daily_sales.to_csv('data/daily_sales.csv', index=False)
    print("\n✓ Daily sales saved to data/daily_sales.csv")
    
    # Task 3: Save Models
    print("\n[Step 4/5] Saving Models...")
    save_models(kmeans, scaler, rf_model)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Customer Segmentation:")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"  - Number of Segments: 3 (Low, Mid, High value)")
    print(f"\n✓ Sales Forecasting:")
    print(f"  - RMSE: ${rmse:,.2f}")
    print(f"  - Forecast Window: 7 days based on 30-day history")
    print(f"\n✓ Models saved in: models/")
    print(f"  - kmeans_model.pkl")
    print(f"  - scaler.pkl")
    print(f"  - sales_forecast_model.pkl")
    print("\nReady for deployment! ✨")
    print("=" * 70)


if __name__ == "__main__":
    main()
