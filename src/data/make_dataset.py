"""
Data processing pipeline for 5G network telemetry
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import pickle
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, raw_data_path, processed_dir):
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and validate raw data"""
        logger.info(f"Loading data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'cell_id', 'anomaly']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} records, {df['cell_id'].nunique()} cells")
        return df
    
    def validate_data(self, df):
        """Perform data validation"""
        logger.info("Validating data...")
        
        # Check for missing values
        missing_pct = df.isnull().mean() * 100
        if missing_pct.any():
            logger.warning(f"Columns with missing values:\n{missing_pct[missing_pct > 0]}")
        
        # Check data types
        logger.info(f"Data types:\n{df.dtypes}")
        
        # Check anomaly distribution
        anomaly_pct = df['anomaly'].mean() * 100
        logger.info(f"Anomaly rate: {anomaly_pct:.2f}%")
        
        return True
    
    def engineer_features(self, df):
        """Create features for anomaly detection"""
        logger.info("Engineering features...")
        
        # Sort by cell and time
        df = df.sort_values(['cell_id', 'timestamp']).reset_index(drop=True)
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling statistics for each cell (window = 6 hours)
        features_df = []
        
        for cell_id, group in df.groupby('cell_id'):
            group = group.set_index('timestamp').sort_index()
            
            # Metrics to create rolling features for
            kpi_cols = [
                'rrc_connected_users',
                'dl_throughput_mbps', 
                'ul_throughput_mbps',
                'latency_ms',
                'handover_success_rate',
                'radio_connection_failures',
                'packet_loss_rate'
            ]
            
            for col in kpi_cols:
                if col in group.columns:
                    # Rolling statistics
                    group[f'{col}_rolling_mean_6h'] = group[col].rolling('6H').mean()
                    group[f'{col}_rolling_std_6h'] = group[col].rolling('6H').std()
                    group[f'{col}_rolling_min_6h'] = group[col].rolling('6H').min()
                    group[f'{col}_rolling_max_6h'] = group[col].rolling('6H').max()
                    
                    # Change from previous period
                    group[f'{col}_diff_1h'] = group[col].diff(periods=6)  # 6 samples = 1 hour
                    group[f'{col}_pct_change_1h'] = group[col].pct_change(periods=6)
            
            # Cell-specific normalization
            for col in kpi_cols:
                if col in group.columns:
                    mean_val = group[col].mean()
                    std_val = group[col].std()
                    if std_val > 0:
                        group[f'{col}_zscore'] = (group[col] - mean_val) / std_val
            
            features_df.append(group.reset_index())
        
        # Combine all cells
        df = pd.concat(features_df, ignore_index=True)
        
        # Fill NaN values created by rolling windows
        df = df.ffill().bfill()
        
        # Remove any remaining NaN
        df = df.dropna()
        
        logger.info(f"After feature engineering: {len(df)} records, {len(df.columns)} columns")
        return df
    
    def split_data(self, df, test_size=0.2):
        """Split data into train and test sets (time-based split)"""
        logger.info("Splitting data...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Time-based split
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Separate features and target
        target_col = 'anomaly'
        exclude_cols = ['timestamp', 'cell_id', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Number of features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def process(self, save_output=True):
        """Main processing pipeline"""
        logger.info("="*50)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*50)
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Validate
        self.validate_data(df)
        
        # 3. Feature engineering
        df = self.engineer_features(df)
        
        # 4. Split data
        X_train, X_test, y_train, y_test, feature_cols = self.split_data(df)
        
        # 5. Prepare output
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'metadata': {
                'n_samples_total': len(df),
                'n_features': len(feature_cols),
                'n_cells': df['cell_id'].nunique(),
                'time_range_start': df['timestamp'].min(),
                'time_range_end': df['timestamp'].max(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_anomaly_rate': y_train.mean(),
                'test_anomaly_rate': y_test.mean(),
            }
        }
        
        # 6. Save processed data
        if save_output:
            output_path = self.processed_dir / 'processed_data.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            # Also save as CSV for inspection
            train_df = X_train.copy()
            train_df['anomaly'] = y_train.values
            train_df.to_csv(self.processed_dir / 'train_data.csv', index=False)
            
            test_df = X_test.copy()
            test_df['anomaly'] = y_test.values
            test_df.to_csv(self.processed_dir / 'test_data.csv', index=False)
            
            logger.info(f"Processed data saved to {output_path}")
        
        # 7. Print summary
        logger.info("\n" + "="*50)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info("="*50)
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Features created: {len(feature_cols)}")
        logger.info(f"Train set: {len(X_train)} samples ({y_train.mean():.2%} anomalies)")
        logger.info(f"Test set: {len(X_test)} samples ({y_test.mean():.2%} anomalies)")
        logger.info(f"Feature examples: {feature_cols[:5]}...")
        
        return processed_data

def main():
    """Main function to run the pipeline"""
    processor = DataProcessor(
        raw_data_path='data/raw/5g_telemetry.csv',
        processed_dir='data/processed'
    )
    
    return processor.process()

if __name__ == "__main__":
    data = main()