"""
Phase 1: Data Loader and Cleaner
Loads the Online Retail dataset, cleans it, and saves to cleaned_data.csv
"""

import pandas as pd
import os


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
