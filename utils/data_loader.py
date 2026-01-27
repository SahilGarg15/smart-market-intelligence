"""
Phase 1: Data Loader and Cleaner
Loads the Online Retail dataset, cleans it, and saves to cleaned_data.csv
"""

import pandas as pd
import os


def validate_data(df):
    """
    Validate raw data before cleaning to ensure pipeline fails fast on bad data
    
    Guardrails to ensure pipeline fails fast on bad data:
    - Check if dataframe is empty (no rows)
    - Check for negative UnitPrice (indicates data corruption)
    - Check for unrealistic Quantity values (>100,000 in single transaction)
    
    Why this matters:
    Instead of silently processing bad data and getting garbage predictions,
    we catch issues early and fail with clear error messages. This prevents
    downstream model training on corrupted data.
    
    Args:
        df: Raw dataframe to validate
        
    Raises:
        ValueError: If validation fails with specific reason
    """
    # Check 1: Empty dataset
    # Reason: Cannot train models or perform analysis on empty data
    if df.empty or len(df) == 0:
        raise ValueError(
            "Dataset is empty! Cannot proceed with empty dataframe. "
            "Please check if the CSV file loaded correctly."
        )
    
    # Check 2: Negative UnitPrice (data corruption indicator)
    # Reason: Prices should always be positive. Negative prices suggest:
    # - Data entry errors
    # - Corrupted file during download/transfer
    # - Wrong column mapping
    if 'UnitPrice' in df.columns:
        negative_prices = df[df['UnitPrice'] < 0]
        if len(negative_prices) > 0:
            raise ValueError(
                f"Found {len(negative_prices)} rows with negative UnitPrice! "
                f"Prices should be positive. This indicates data corruption. "
                f"Examples: {negative_prices['UnitPrice'].head().tolist()}"
            )
    
    # Check 3: Unrealistic Quantity values (>100,000)
    # Reason: While we filter negatives later, extreme values suggest:
    # - Data entry errors (e.g., someone typed 1000000 instead of 100)
    # - Fraudulent transactions
    # - Wholesale/bulk orders that might skew analysis
    # We use 100,000 as threshold based on retail industry standards
    if 'Quantity' in df.columns:
        unrealistic_qty = df[df['Quantity'] > 100000]
        if len(unrealistic_qty) > 0:
            raise ValueError(
                f"Found {len(unrealistic_qty)} rows with Quantity > 100,000! "
                f"This is unrealistic for retail transactions and suggests data errors. "
                f"Max quantity found: {df['Quantity'].max():,.0f}. "
                f"Please review the data source."
            )
    
    print("✓ Data validation passed: No critical issues found")


def load_and_clean_data(input_filepath='data/Online Retail.csv', 
                        output_filepath='data/cleaned_data.csv'):
    """
    Load, clean, and save the Online Retail dataset
    
    Steps:
    1. Load the CSV file
    2. Remove rows with missing CustomerIDs
    3. Remove rows with negative Quantity (returns)
    4. Create TotalAmount column
    5. Convert InvoiceDate to datetime
    6. Print summary statistics
    7. Save cleaned data
    
    Args:
        input_filepath: Path to raw data file
        output_filepath: Path to save cleaned data
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("=" * 60)
    print("PHASE 1: DATA LOADING AND CLEANING")
    print("=" * 60)
    
    # Step 1: Load the dataset
    print(f"\n[1/6] Loading data from {input_filepath}...")
    
    # Check if file exists
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(
            f"Dataset not found at {input_filepath}. "
            "Please download the Online Retail Dataset and place it in the data/ folder."
        )
    
    # Read CSV with proper encoding
    # encoding='latin-1' handles special characters in product descriptions
    df = pd.read_csv(input_filepath, encoding='latin-1')
    print(f"✓ Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    
    # Validate data before processing (fail fast on bad data)
    print(f"\n[1.5/6] Validating data quality...")
    validate_data(df)
    
    # Show initial data info
    print(f"\nInitial data shape: {df.shape}")
    print(f"Initial missing values:\n{df.isnull().sum()}")
    
    # Step 2: Remove rows with missing CustomerID
    print(f"\n[2/6] Removing rows with missing CustomerID...")
    initial_rows = len(df)
    
    # Drop rows where CustomerID is NaN
    # Reason: CustomerID is essential for customer segmentation analysis
    df = df.dropna(subset=['CustomerID'])
    
    removed = initial_rows - len(df)
    print(f"✓ Removed {removed:,} rows with missing CustomerID")
    print(f"  Remaining: {len(df):,} rows")
    
    # Step 3: Remove rows with negative Quantity (returns)
    print(f"\n[3/6] Removing rows with negative Quantity (returns)...")
    initial_rows = len(df)
    
    # Keep only positive quantities
    # Reason: Negative quantities represent returns, not actual sales
    df = df[df['Quantity'] > 0]
    
    removed = initial_rows - len(df)
    print(f"✓ Removed {removed:,} rows with negative Quantity")
    print(f"  Remaining: {len(df):,} rows")
    
    # Step 4: Create TotalAmount column
    print(f"\n[4/6] Creating TotalAmount column (Quantity × UnitPrice)...")
    
    # Calculate total amount for each line item
    # This represents the revenue generated per transaction line
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    print(f"✓ Created TotalAmount column")
    print(f"  Total revenue in dataset: ${df['TotalAmount'].sum():,.2f}")
    
    # Step 5: Convert InvoiceDate to datetime
    print(f"\n[5/6] Converting InvoiceDate to datetime format...")
    
    # Convert to datetime for time-based analysis
    # Reason: Pandas datetime type enables date operations and time series analysis
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print(f"✓ Converted InvoiceDate to datetime")
    print(f"  Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    # Step 6: Print summary of cleaned data
    print(f"\n[6/6] Summary of cleaned data:")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print("=" * 60)
    
    # Step 7: Save cleaned data
    print(f"\nSaving cleaned data to {output_filepath}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_filepath, index=False)
    
    print(f"✓ Cleaned data saved successfully!")
    print(f"  File size: {os.path.getsize(output_filepath) / 1024 / 1024:.2f} MB")
    
    return df


if __name__ == "__main__":
    """
    Run this script directly to load and clean the data
    Usage: python utils/data_loader.py
    """
    # Load and clean the data
    cleaned_df = load_and_clean_data(
        input_filepath='data/Online Retail.csv',
        output_filepath='data/cleaned_data.csv'
    )
    
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE!")
    print("=" * 60)
    print(f"Cleaned dataset ready with {len(cleaned_df):,} rows")
    print("Next step: Run feature engineering or model training scripts")
    print("=" * 60)
