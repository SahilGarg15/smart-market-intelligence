"""
FastAPI Backend for Smart Market Intelligence Platform
Serves customer segmentation and sales forecasting models
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Initialize FastAPI app
app = FastAPI(
    title="Smart Market Intelligence API",
    description="API for customer segmentation and sales forecasting",
    version="1.0.0"
)

# Global variables to store loaded models and data
kmeans_model = None
scaler = None
sales_model = None
daily_sales_df = None


@app.on_event("startup")
async def load_models_and_data():
    """
    Load saved models and data when the API starts
    This runs once at startup to avoid loading on every request
    
    Startup Check: Fail fast if models don't exist
    Why? Better to prevent the API from starting than to serve 503 errors
    on every request. This gives clear feedback during deployment.
    """
    global kmeans_model, scaler, sales_model, daily_sales_df
    
    # Pre-flight check: Verify all required files exist BEFORE loading
    # Reason: Fail fast with helpful error message instead of cryptic FileNotFoundError
    required_files = [
        'models/kmeans_model.pkl',
        'models/scaler.pkl',
        'models/sales_forecast_model.pkl',
        'data/daily_sales.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        error_msg = (
            "\n" + "="*70 + "\n"
            "STARTUP FAILED: Required model files are missing!\n"
            "="*70 + "\n"
            f"Missing files:\n"
        )
        for f in missing_files:
            error_msg += f"  ❌ {f}\n"
        error_msg += (
            "\nREQUIRED ACTION:\n"
            "  1. Run: python ml_engine.py\n"
            "  2. Wait for training to complete\n"
            "  3. Restart this API server\n"
            "\nWhy this check exists:\n"
            "  Without models, the API would return 503 errors on every request.\n"
            "  Better to fail at startup with a clear error message.\n"
            "="*70
        )
        print(error_msg)
        raise RuntimeError(error_msg)
    
    try:
        # Load customer segmentation models
        # Reason: We need both the KMeans model and scaler to transform new data
        kmeans_model = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✓ Customer segmentation models loaded")
        
        # Load sales forecasting model
        sales_model = joblib.load('models/sales_forecast_model.pkl')
        print("✓ Sales forecasting model loaded")
        
        # Load daily sales data for forecasting
        # Reason: We need historical data to make predictions using the sliding window approach
        daily_sales_df = pd.read_csv('data/daily_sales.csv')
        daily_sales_df['Date'] = pd.to_datetime(daily_sales_df['Date'])
        print(f"✓ Daily sales data loaded ({len(daily_sales_df)} days)")
        
        print("\n" + "="*70)
        print("API STARTUP SUCCESSFUL - All models loaded and ready")
        print("="*70 + "\n")
        
    except Exception as e:
        error_msg = (
            f"\nFATAL ERROR during model loading: {e}\n"
            "The API cannot start without models. Please run ml_engine.py first."
        )
        print(error_msg)
        raise RuntimeError(error_msg)


# Pydantic Models for Request/Response Validation

class CustomerInput(BaseModel):
    """
    Input schema for customer segmentation
    Client sends RFM values to get customer segment prediction
    """
    Recency: float = Field(..., description="Days since last purchase", ge=0)
    Frequency: float = Field(..., description="Number of purchases", ge=1)
    Monetary: float = Field(..., description="Total amount spent", ge=0)
    
    class Config:
        # Example for API documentation (Swagger UI)
        schema_extra = {
            "example": {
                "Recency": 10,
                "Frequency": 15,
                "Monetary": 2500.50
            }
        }


class SegmentResponse(BaseModel):
    """Response schema for customer segmentation"""
    cluster_id: int
    segment_name: str
    rfm_input: dict


class ForecastResponse(BaseModel):
    """Response schema for sales forecast"""
    forecast_days: int
    predictions: list


# API Endpoints

@app.get("/")
async def root():
    """
    Root endpoint - returns API status
    """
    return {"message": "Smart Market Intelligence API is Online"}


@app.post("/segment", response_model=SegmentResponse)
async def segment_customer(customer: CustomerInput):
    """
    Predict customer segment based on RFM values
    
    Process:
    1. Validate input values (no negatives, realistic ranges)
    2. Log-transform RFM values (reduces skewness in data)
    3. Scale using saved scaler (standardizes features)
    4. Predict cluster using KMeans model
    
    Args:
        customer: CustomerInput with Recency, Frequency, Monetary
        
    Returns:
        Cluster ID and segment name
    """
    if kmeans_model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run ml_engine.py first."
        )
    
    # Input Validation: Check for negative values
    # Why? Negative RFM values are logically impossible:
    # - Recency: Cannot be negative days since last purchase
    # - Frequency: Cannot have negative number of purchases
    # - Monetary: Cannot spend negative money (this isn't a refund API)
    # Better to reject bad input than produce garbage predictions
    if customer.Recency < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: Recency cannot be negative (got {customer.Recency}). "
                   "Recency represents days since last purchase and must be >= 0."
        )
    
    if customer.Frequency < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: Frequency cannot be negative (got {customer.Frequency}). "
                   "Frequency represents number of purchases and must be >= 1."
        )
    
    if customer.Monetary < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: Monetary cannot be negative (got {customer.Monetary}). "
                   "Monetary represents total spend and must be >= 0."
        )
    
    try:
        # Create RFM array from input
        rfm_values = np.array([[
            customer.Recency,
            customer.Frequency,
            customer.Monetary
        ]])
        
        # Log transform the RFM values
        # Reason: RFM data is often right-skewed (few high spenders, many low spenders)
        # Log transformation makes the distribution more normal for better clustering
        rfm_log = np.log1p(rfm_values)
        
        # Scale the log-transformed values using the saved scaler
        # Reason: KMeans uses Euclidean distance, so features must be on same scale
        rfm_scaled = scaler.transform(rfm_log)
        
        # Predict the cluster
        cluster_id = int(kmeans_model.predict(rfm_scaled)[0])
        
        # Map cluster IDs to business-friendly names
        # Cluster interpretation based on RFM patterns:
        # 0 = Loyal customers (low recency, high frequency/monetary)
        # 1 = At-Risk customers (high recency, low frequency)
        # 2 = Whale customers (very high monetary value)
        segment_mapping = {
            0: 'Loyal',
            1: 'At-Risk',
            2: 'Whale'
        }
        
        segment_name = segment_mapping.get(cluster_id, f'Segment_{cluster_id}')
        
        return SegmentResponse(
            cluster_id=cluster_id,
            segment_name=segment_name,
            rfm_input={
                "Recency": customer.Recency,
                "Frequency": customer.Frequency,
                "Monetary": customer.Monetary
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors) as-is
        raise
    except Exception as e:
        # Catch unexpected errors and log them
        # Why detailed error messages?
        # - Helps debug issues in production
        # - Distinguishes between model errors vs data errors
        # - Provides context for monitoring/alerting systems
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "input_values": {
                "Recency": customer.Recency,
                "Frequency": customer.Frequency,
                "Monetary": customer.Monetary
            }
        }
        print(f"Segmentation error: {error_details}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}. Check API logs for details."
        )


@app.get("/forecast/{days}", response_model=ForecastResponse)
async def forecast_sales(days: int = 7):
    """
    Forecast sales for the next X days using a Sliding Window approach
    
    Sliding Window Approach:
    - We use the last 30 days of actual sales as the initial window
    - To predict Day 1: Use days [-30 to -1]
    - To predict Day 2: Use days [-29 to 0], where Day 0 is our Day 1 prediction
    - To predict Day 3: Use days [-28 to 1], where Day 1 is our Day 2 prediction
    - And so on...
    
    This means each prediction uses the most recent 30 values, including 
    previously predicted values. This is necessary because we don't have 
    future actual data, so we use our own predictions as inputs.
    
    Forecast Horizon Limit: Maximum 30 days
    Why? Prediction errors compound in sliding window approach:
    - Day 1: Uses 100% actual data
    - Day 15: Uses 50% actual, 50% predicted
    - Day 30: Uses 0% actual, 100% predicted (high uncertainty!)
    Beyond 30 days, errors accumulate exponentially and predictions become unreliable.
    
    Args:
        days: Number of days to forecast (1-30, default 7)
        
    Returns:
        List of predictions with dates and predicted sales
    """
    if sales_model is None or daily_sales_df is None:
        raise HTTPException(
            status_code=503,
            detail="Model or data not loaded. Please run ml_engine.py first."
        )
    
    # Input Validation: Limit forecast horizon to 30 days
    # Why this specific limit?
    # 1. Model trained on 30-day windows (expects 30 features)
    # 2. Sliding window error compounds: each prediction uses previous predictions
    # 3. Beyond 30 days, we're predicting entirely from predicted data (unreliable)
    # 4. Industry best practice: Don't forecast beyond 1 month for daily retail sales
    if days < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid forecast horizon: {days} days. Minimum is 1 day."
        )
    
    if days > 30:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid forecast horizon: {days} days exceeds maximum of 30 days. "
                "Why this limit? Prediction errors compound in the sliding window approach. "
                "Beyond 30 days, forecasts use only predicted data (no actual sales), "
                "making them highly unreliable. For longer forecasts, consider monthly aggregation."
            )
        )
    
    try:
        # Get the last 30 days of actual sales as our starting window
        # Reason: Our model was trained on 30-day windows, so it expects 30 features
        window_size = 30
        last_30_days = daily_sales_df['Sales'].tail(window_size).values.tolist()
        
        # Get the last date in our data to calculate future dates
        last_date = daily_sales_df['Date'].max()
        
        predictions = []
        
        # Iteratively predict one day at a time using the sliding window
        for i in range(days):
            # Take the last 30 values as features for the model
            # Initially, these are all actual values
            # As we predict, we include our own predictions
            X_predict = np.array(last_30_days[-window_size:]).reshape(1, -1)
            
            # Predict the next day's sales
            predicted_sales = sales_model.predict(X_predict)[0]
            
            # Calculate the date for this prediction
            predicted_date = last_date + timedelta(days=i+1)
            
            # Store the prediction
            predictions.append({
                "date": predicted_date.strftime('%Y-%m-%d'),
                "predicted_sales": round(float(predicted_sales), 2)
            })
            
            # Slide the window: append the new prediction to our history
            # This prediction will be used as input for the next day's forecast
            last_30_days.append(predicted_sales)
        
        return ForecastResponse(
            forecast_days=days,
            predictions=predictions
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors) as-is
        raise
    except Exception as e:
        # Catch unexpected errors with detailed context
        # Why comprehensive error handling?
        # - Forecasting can fail due to data issues (NaN values, missing dates)
        # - Model errors (corrupted .pkl file, version mismatch)
        # - Numerical errors (overflow, underflow in calculations)
        # We want to know WHICH type of error occurred for debugging
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "forecast_days": days,
            "data_available": len(daily_sales_df) if daily_sales_df is not None else 0
        }
        print(f"Forecast error: {error_details}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=(
                f"Forecast generation failed: {str(e)}. "
                "This could be due to insufficient historical data or model issues. "
                "Check API logs for details."
            )
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns status of loaded models and data
    """
    return {
        "status": "healthy",
        "kmeans_loaded": kmeans_model is not None,
        "scaler_loaded": scaler is not None,
        "sales_model_loaded": sales_model is not None,
        "data_loaded": daily_sales_df is not None,
        "data_points": len(daily_sales_df) if daily_sales_df is not None else 0
    }


# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    # Start the API server
    # host="0.0.0.0" makes it accessible from other machines
    # port=8000 is the default FastAPI port
    # reload=True automatically restarts when code changes (for development)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
