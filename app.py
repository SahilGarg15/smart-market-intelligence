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
    """
    global kmeans_model, scaler, sales_model, daily_sales_df
    
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
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load models or data: {e}")
        print("Please run ml_engine.py first to train and save models")


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
    1. Log-transform RFM values (reduces skewness in data)
    2. Scale using saved scaler (standardizes features)
    3. Predict cluster using KMeans model
    
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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
    
    Args:
        days: Number of days to forecast (default 7)
        
    Returns:
        List of predictions with dates and predicted sales
    """
    if sales_model is None or daily_sales_df is None:
        raise HTTPException(
            status_code=503,
            detail="Model or data not loaded. Please run ml_engine.py first."
        )
    
    if days < 1 or days > 30:
        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 30"
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


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
