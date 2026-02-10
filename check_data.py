import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("🔍 Checking data for inf/large values...")

# Check raw data
raw_path = Path('data/raw/5g_telemetry.csv')
if raw_path.exists():
    df_raw = pd.read_csv(raw_path)
    print(f"\n📊 RAW DATA: {len(df_raw)} rows")
    print(f"Columns: {list(df_raw.columns)}")
    
    # Check for inf/NaN
    print("\nChecking for issues in raw data:")
    for col in df_raw.select_dtypes(include=[np.number]).columns:
        if df_raw[col].isnull().any():
            print(f"  ❌ {col}: {df_raw[col].isnull().sum()} NaN values")
        if np.isinf(df_raw[col]).any():
            print(f"  ❌ {col}: Contains INF values")
        if df_raw[col].abs().max() > 1e6:
            print(f"  ⚠️  {col}: Very large values (max: {df_raw[col].max():.2f})")

# Check processed data
proc_path = Path('data/processed/processed_data.pkl')
if proc_path.exists():
    with open(proc_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\n📊 PROCESSED DATA:")
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    
    # Check training data
    X_train = data['X_train']
    print("\nChecking for issues in processed data:")
    
    # Check each column
    for col in X_train.columns:
        col_data = X_train[col]
        
        # Check for NaN
        nan_count = col_data.isnull().sum()
        if nan_count > 0:
            print(f"  ❌ {col}: {nan_count} NaN values")
        
        # Check for infinite
        if hasattr(col_data, '__array__'):
            try:
                if np.isinf(col_data.values).any():
                    print(f"  ❌ {col}: Contains INF values")
            except:
                pass
        
        # Check for extremely large values
        if hasattr(col_data, 'max'):
            try:
                max_val = col_data.max()
                if abs(max_val) > 1e6:
                    print(f"  ⚠️  {col}: Very large value (max: {max_val:.2e})")
            except:
                pass

print("\n" + "="*50)
print("DATA CHECK COMPLETE")