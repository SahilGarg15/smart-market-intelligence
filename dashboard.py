"""
Streamlit Dashboard for Smart Market Intelligence Platform
Interactive frontend for data visualization and predictions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Market Intelligence Platform",
    page_icon="📊",
    layout="wide"  # Use full browser width for better visualization
)

# API endpoint configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-online {
        color: #00cc00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_status():
    """
    Check if the FastAPI backend is running
    Returns: bool - True if API is online, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def load_data():
    """
    Load the required CSV files for dashboard
    Uses st.cache_data to avoid reloading on every interaction
    """
    try:
        daily_sales = pd.read_csv('data/daily_sales.csv')
        daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
        
        customer_segments = pd.read_csv('data/customer_segments.csv')
        
        return daily_sales, customer_segments
    except FileNotFoundError as e:
        st.error(f"⚠️ Data file not found: {e}")
        st.info("Please run ml_engine.py first to generate the required data files.")
        return None, None


# Header Section
st.markdown('<h1 class="main-header">📊 Smart Market Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Retail Analytics</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Control Panel")

# Check API status
api_online = check_api_status()
if api_online:
    st.sidebar.markdown('<p class="status-online">✅ System Status: Online</p>', unsafe_allow_html=True)
else:
    st.sidebar.warning("⚠️ System Status: API Offline")
    st.sidebar.info("Start the API with: uvicorn app:app --reload")

st.sidebar.markdown("---")

# File uploader placeholder
st.sidebar.subheader("📂 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload New Sales Data",
    type=['csv'],
    help="Upload CSV file with new sales data (placeholder for future implementation)"
)

if uploaded_file is not None:
    st.sidebar.success("File uploaded! (Processing not yet implemented)")

st.sidebar.markdown("---")
st.sidebar.info("💡 Use the tabs below to explore data and make predictions")

# Load data
daily_sales, customer_segments = load_data()

# Create tabs for different sections
tab1, tab2 = st.tabs(["📈 Business Overview", "🔮 Prediction Lab"])

# ============================================================================
# TAB 1: Business Overview
# ============================================================================
with tab1:
    if daily_sales is None or customer_segments is None:
        st.warning("Data not available. Please run ml_engine.py first.")
    else:
        st.header("Business Overview")
        
        # Calculate key metrics
        # Reason: These metrics give a quick snapshot of business performance
        total_revenue = daily_sales['Sales'].sum()
        avg_order_value = daily_sales['Sales'].mean()
        total_customers = len(customer_segments)
        
        # Metrics Row - Display key business metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="💰 Total Revenue",
                value=f"${total_revenue:,.2f}",
                help="Sum of all daily sales in the dataset"
            )
        
        with col2:
            st.metric(
                label="📊 Avg Daily Sales",
                value=f"${avg_order_value:,.2f}",
                help="Average revenue per day"
            )
        
        with col3:
            st.metric(
                label="👥 Total Customers",
                value=f"{total_customers:,}",
                help="Number of unique customers in the segmentation data"
            )
        
        st.markdown("---")
        
        # Charts Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Line Chart: Daily Sales over time
            st.subheader("📈 Daily Sales Trend")
            
            # Create interactive line chart with Plotly
            # Reason: Plotly provides zoom, pan, and hover features for better exploration
            fig_sales = px.line(
                daily_sales,
                x='Date',
                y='Sales',
                title='Sales Over Time',
                labels={'Sales': 'Daily Revenue ($)', 'Date': 'Date'}
            )
            
            # Customize chart appearance
            fig_sales.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            fig_sales.update_traces(line_color='#1f77b4', line_width=2)
            
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            # Pie Chart: Customer Segments distribution
            st.subheader("🎯 Customer Segments")
            
            # Count customers per segment
            segment_counts = customer_segments['Segment'].value_counts().reset_index()
            segment_counts.columns = ['SegmentID', 'Count']
            
            # Map cluster IDs to names for better readability
            # Reason: "Loyal" is more meaningful than "Segment 0"
            segment_mapping = {0: 'Loyal', 1: 'At-Risk', 2: 'Whale'}
            segment_counts['Segment'] = segment_counts['SegmentID'].map(segment_mapping)
            
            # Create pie chart
            fig_pie = px.pie(
                segment_counts,
                values='Count',
                names='Segment',
                title='Customer Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Additional insights
        st.markdown("---")
        st.subheader("📋 Segment Statistics")
        
        # Calculate average RFM values per segment
        segment_stats = customer_segments.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CustomerID': 'count'
        }).round(2)
        
        segment_stats.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Customer Count']
        segment_stats.index = segment_stats.index.map(segment_mapping)
        
        st.dataframe(segment_stats, use_container_width=True)


# ============================================================================
# TAB 2: Prediction Lab
# ============================================================================
with tab2:
    st.header("🔮 Prediction Lab")
    st.write("Interact with machine learning models via the API")
    
    # Section A: Customer Segmentation
    st.subheader("👤 Section A: Check Customer Segment")
    st.write("Enter customer RFM values to predict their segment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Recency slider
        # Reason: Sliders provide intuitive input for numerical values
        recency = st.slider(
            "📅 Recency (days since last purchase)",
            min_value=0,
            max_value=365,
            value=30,
            help="Lower is better - recent customers are more engaged"
        )
    
    with col2:
        # Frequency slider
        frequency = st.slider(
            "🔄 Frequency (number of purchases)",
            min_value=1,
            max_value=100,
            value=10,
            help="Higher is better - frequent buyers are loyal"
        )
    
    with col3:
        # Monetary slider
        monetary = st.slider(
            "💵 Monetary (total spent $)",
            min_value=0.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            help="Higher is better - big spenders are valuable"
        )
    
    # Analyze button
    if st.button("🔍 Analyze Customer", type="primary"):
        if not api_online:
            st.error("⚠️ API is offline. Please start the API server first.")
        else:
            try:
                # Prepare the request payload
                payload = {
                    "Recency": float(recency),
                    "Frequency": float(frequency),
                    "Monetary": float(monetary)
                }
                
                # Send POST request to the /segment endpoint
                # Reason: We call the API to get real-time predictions from the trained model
                response = requests.post(
                    f"{API_BASE_URL}/segment",
                    json=payload,
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display the prediction result
                    cluster_id = result['cluster_id']
                    segment_name = result['segment_name']
                    
                    # Create a nice colored box based on segment
                    # Reason: Visual feedback makes results more engaging
                    colors = {
                        'Loyal': 'green',
                        'At-Risk': 'orange',
                        'Whale': 'blue'
                    }
                    color = colors.get(segment_name, 'gray')
                    
                    st.success(f"### 🎯 Analysis Complete!")
                    st.markdown(f"**Segment:** :{color}[{segment_name}] (Cluster {cluster_id})")
                    
                    # Provide segment-specific insights
                    insights = {
                        'Loyal': '✨ This is a **Loyal Customer**! They purchase frequently and recently. Focus on retention programs.',
                        'At-Risk': '⚠️ This is an **At-Risk Customer**. They haven\'t purchased recently. Consider re-engagement campaigns.',
                        'Whale': '🐋 This is a **Whale Customer**! High spending value. Provide VIP treatment and exclusive offers.'
                    }
                    
                    st.info(insights.get(segment_name, "Customer analyzed successfully."))
                    
                    # Show detailed response
                    with st.expander("📊 View API Response"):
                        st.json(result)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Connection Error: Could not reach the API. Make sure it's running.")
                st.code(str(e))
    
    st.markdown("---")
    
    # Section B: Sales Forecasting
    st.subheader("📊 Section B: Future Sales Forecast")
    st.write("Predict future sales using the sliding window approach")
    
    # Days slider
    forecast_days = st.slider(
        "📅 Days to Forecast",
        min_value=7,
        max_value=30,
        value=7,
        help="Number of days to predict into the future"
    )
    
    # Generate Forecast button
    if st.button("🔮 Generate Forecast", type="primary"):
        if not api_online:
            st.error("⚠️ API is offline. Please start the API server first.")
        else:
            try:
                # Send GET request to the /forecast endpoint
                # Reason: We call the API to get multi-day sales predictions
                with st.spinner(f"Forecasting sales for the next {forecast_days} days..."):
                    response = requests.get(
                        f"{API_BASE_URL}/forecast/{forecast_days}",
                        timeout=10
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result['predictions']
                    
                    # Convert predictions to DataFrame for easier plotting
                    forecast_df = pd.DataFrame(predictions)
                    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                    
                    st.success(f"✅ Forecast generated for {result['forecast_days']} days!")
                    
                    # Display forecast as line chart
                    fig_forecast = px.line(
                        forecast_df,
                        x='date',
                        y='predicted_sales',
                        title=f'Sales Forecast - Next {forecast_days} Days',
                        labels={
                            'predicted_sales': 'Predicted Sales ($)',
                            'date': 'Date'
                        },
                        markers=True  # Show data points
                    )
                    
                    # Customize appearance
                    fig_forecast.update_traces(
                        line_color='#ff7f0e',
                        line_width=3,
                        marker=dict(size=8)
                    )
                    
                    fig_forecast.update_layout(
                        hovermode='x unified',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Show summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_forecast = forecast_df['predicted_sales'].mean()
                        st.metric("📊 Avg Predicted Sales", f"${avg_forecast:,.2f}")
                    
                    with col2:
                        total_forecast = forecast_df['predicted_sales'].sum()
                        st.metric("💰 Total Forecast Revenue", f"${total_forecast:,.2f}")
                    
                    with col3:
                        max_forecast = forecast_df['predicted_sales'].max()
                        st.metric("📈 Peak Day Sales", f"${max_forecast:,.2f}")
                    
                    # Show detailed data table
                    with st.expander("📋 View Detailed Forecast Data"):
                        # Format the dataframe for display
                        display_df = forecast_df.copy()
                        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                        display_df['predicted_sales'] = display_df['predicted_sales'].apply(lambda x: f"${x:,.2f}")
                        display_df.columns = ['Date', 'Predicted Sales']
                        
                        st.dataframe(display_df, use_container_width=True)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Connection Error: Could not reach the API.")
                st.code(str(e))
            except Exception as e:
                st.error(f"⚠️ Error processing forecast: {str(e)}")


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Smart Market Intelligence Platform | Built with Streamlit & FastAPI | "
    "Final Year Project 2026"
    "</p>",
    unsafe_allow_html=True
)
