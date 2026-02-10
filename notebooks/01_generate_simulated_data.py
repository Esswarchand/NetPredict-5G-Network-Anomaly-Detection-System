"""
Generate simulated 5G network telemetry data for NetPredict project
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

def create_5g_dataset(n_cells=20, n_days=30, samples_per_hour=12, anomaly_rate=0.05):
    """
    Generate realistic 5G network KPIs with anomalies
    
    Parameters:
    - n_cells: Number of cell towers
    - n_days: Number of days of data
    - samples_per_hour: Data points per hour
    - anomaly_rate: Percentage of anomalies (0.05 = 5%)
    """
    
    data = []
    cell_ids = [f'CELL_{i:03d}' for i in range(1, n_cells + 1)]
    
    start_time = datetime.now() - timedelta(days=n_days)
    
    print(f"Generating {n_days} days of 5G data for {n_cells} cells...")
    
    for day in range(n_days):
        for hour in range(24):
            for minute in range(0, 60, 60 // samples_per_hour):
                for cell_id in cell_ids:
                    ts = start_time + timedelta(
                        days=day, 
                        hours=hour, 
                        minutes=minute
                    )
                    
                    # Determine if this is an anomaly
                    is_anomaly = random.random() < anomaly_rate
                    
                    if not is_anomaly:
                        # Normal network behavior
                        kpi = {
                            'timestamp': ts,
                            'cell_id': cell_id,
                            'rrc_connected_users': random.randint(50, 200),
                            'dl_throughput_mbps': random.uniform(80, 120),
                            'ul_throughput_mbps': random.uniform(40, 80),
                            'latency_ms': random.uniform(5, 15),
                            'handover_success_rate': random.uniform(0.98, 1.0),
                            'radio_connection_failures': random.randint(0, 5),
                            'packet_loss_rate': random.uniform(0.0, 0.01),
                            'cpu_utilization': random.uniform(30, 60),
                            'memory_utilization': random.uniform(40, 70),
                            'anomaly': 0
                        }
                    else:
                        # Anomalous behavior
                        kpi = {
                            'timestamp': ts,
                            'cell_id': cell_id,
                            'rrc_connected_users': random.randint(300, 500),
                            'dl_throughput_mbps': random.uniform(10, 30),
                            'ul_throughput_mbps': random.uniform(10, 20),
                            'latency_ms': random.uniform(50, 200),
                            'handover_success_rate': random.uniform(0.7, 0.9),
                            'radio_connection_failures': random.randint(20, 50),
                            'packet_loss_rate': random.uniform(0.1, 0.3),
                            'cpu_utilization': random.uniform(80, 95),
                            'memory_utilization': random.uniform(85, 98),
                            'anomaly': 1
                        }
                    
                    data.append(kpi)
    
    df = pd.DataFrame(data)
    return df

def add_temporal_patterns(df):
    """Add realistic temporal patterns to the data"""
    
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Peak hours (9 AM - 9 PM) have higher usage
    peak_hours = (df['hour'] >= 9) & (df['hour'] <= 21)
    
    # Increase traffic during peak hours for normal data
    normal_peak = (df['anomaly'] == 0) & peak_hours
    df.loc[normal_peak, 'rrc_connected_users'] = df.loc[normal_peak, 'rrc_connected_users'] * 1.5
    df.loc[normal_peak, 'dl_throughput_mbps'] = df.loc[normal_peak, 'dl_throughput_mbps'] * 1.2
    df.loc[normal_peak, 'ul_throughput_mbps'] = df.loc[normal_peak, 'ul_throughput_mbps'] * 1.3
    
    # Weekends have different patterns
    weekend = df['is_weekend'] == 1
    df.loc[weekend & (df['anomaly'] == 0), 'rrc_connected_users'] *= 0.8  # Less traffic
    df.loc[weekend & (df['anomaly'] == 0), 'dl_throughput_mbps'] *= 1.3   # More video streaming
    
    return df

def main():
    """Main function to generate and save dataset"""
    
    # Create data directory if it doesn't exist
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    df = create_5g_dataset(
        n_cells=15,  # Reduced for faster processing
        n_days=60,   # 2 months of data
        samples_per_hour=6,  # Every 10 minutes
        anomaly_rate=0.04    # 4% anomalies
    )
    
    # Add temporal patterns
    df = add_temporal_patterns(df)
    
    # Save to CSV
    output_path = data_dir / '5g_telemetry.csv'
    df.to_csv(output_path, index=False)
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total records: {len(df):,}")
    print(f"Number of cells: {df['cell_id'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Anomaly rate: {df['anomaly'].mean():.2%}")
    print(f"File saved: {output_path}")
    
    # Show sample
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    # Show anomaly distribution
    print("\nAnomaly distribution:")
    print(df['anomaly'].value_counts())
    
    return df

if __name__ == "__main__":
    df = main()