"""
Unit tests for data processing pipeline
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.make_dataset import DataProcessor

def test_data_processor_initialization():
    """Test DataProcessor initialization"""
    processor = DataProcessor(
        raw_data_path='data/raw/5g_telemetry.csv',
        processed_dir='data/processed_test'
    )
    
    assert processor.raw_data_path.exists()
    assert processor.processed_dir.exists()
    print("✅ DataProcessor initialization test passed")

def test_load_data():
    """Test data loading"""
    processor = DataProcessor(
        raw_data_path='data/raw/5g_telemetry.csv',
        processed_dir='data/processed_test'
    )
    
    df = processor.load_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'timestamp' in df.columns
    assert 'cell_id' in df.columns
    assert 'anomaly' in df.columns
    
    print("✅ Data loading test passed")
    return df

def test_feature_engineering():
    """Test feature engineering"""
    processor = DataProcessor(
        raw_data_path='data/raw/5g_telemetry.csv',
        processed_dir='data/processed_test'
    )
    
    df = processor.load_data()
    df_engineered = processor.engineer_features(df)
    
    assert len(df_engineered.columns) > len(df.columns)
    assert df_engineered.isnull().sum().sum() == 0
    
    # Check that rolling features were created
    rolling_cols = [col for col in df_engineered.columns if 'rolling' in col]
    assert len(rolling_cols) > 0
    
    print("✅ Feature engineering test passed")
    return df_engineered

def test_data_split():
    """Test data splitting"""
    processor = DataProcessor(
        raw_data_path='data/raw/5g_telemetry.csv',
        processed_dir='data/processed_test'
    )
    
    df = processor.load_data()
    df = processor.engineer_features(df)
    X_train, X_test, y_train, y_test, feature_cols = processor.split_data(df)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(feature_cols) == len(X_train.columns)
    
    # Check no overlap in time
    # (This would require timestamps in X_train/X_test)
    
    print("✅ Data splitting test passed")

def test_full_pipeline():
    """Test complete pipeline"""
    processor = DataProcessor(
        raw_data_path='data/raw/5g_telemetry.csv',
        processed_dir='data/processed_test'
    )
    
    processed_data = processor.process(save_output=False)
    
    assert 'X_train' in processed_data
    assert 'X_test' in processed_data
    assert 'y_train' in processed_data
    assert 'y_test' in processed_data
    assert 'feature_names' in processed_data
    assert 'metadata' in processed_data
    
    print("✅ Full pipeline test passed")

def main():
    """Run all tests"""
    print("="*50)
    print("RUNNING DATA PIPELINE TESTS")
    print("="*50)
    
    try:
        test_data_processor_initialization()
        df = test_load_data()
        df_engineered = test_feature_engineering()
        test_data_split()
        test_full_pipeline()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✅")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)