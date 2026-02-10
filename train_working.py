"""
WORKING VERSION - Simple training without Optuna issues
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import pickle
import mlflow
import mlflow.sklearn
from datetime import datetime
import os
from pathlib import Path

print("🚀 NETPREDICT - SIMPLE WORKING TRAINING")
print("="*60)

# Setup directories
Path("models").mkdir(exist_ok=True)
Path("mlflow/artifacts").mkdir(parents=True, exist_ok=True)

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlflow/mlruns.db")
mlflow.set_experiment("5g_anomaly_detection")

def load_and_clean_data():
    """Load and clean data"""
    print("📊 Loading data...")
    
    # Check if raw data exists
    if not Path("data/raw/5g_telemetry.csv").exists():
        print("❌ No raw data found!")
        print("Run: python notebooks/01_generate_simulated_data.py")
        return None
    
    # Load raw data
    df = pd.read_csv("data/raw/5g_telemetry.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Raw data: {len(df)} rows, {df['anomaly'].mean():.2%} anomalies")
    
    # Simple feature engineering
    print("🔧 Creating simple features...")
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Group by cell and calculate simple statistics
    features = []
    for col in ['rrc_connected_users', 'dl_throughput_mbps', 'ul_throughput_mbps', 'latency_ms']:
        df[f'{col}_mean'] = df.groupby('cell_id')[col].transform('mean')
        df[f'{col}_std'] = df.groupby('cell_id')[col].transform('std')
    
    # Select features
    feature_cols = [
        'rrc_connected_users', 'dl_throughput_mbps', 'ul_throughput_mbps', 'latency_ms',
        'handover_success_rate', 'radio_connection_failures', 'packet_loss_rate',
        'cpu_utilization', 'memory_utilization',
        'hour', 'day_of_week',
        'rrc_connected_users_mean', 'rrc_connected_users_std',
        'dl_throughput_mbps_mean', 'dl_throughput_mbps_std',
    ]
    
    # Remove any columns with inf or NaN
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].mean())
    
    # Prepare X and y
    X = df[feature_cols]
    y = df['anomaly']
    
    # Split data (time-based)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Train anomalies: {y_train.mean():.2%}, Test anomalies: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_simple_xgboost(X_train, y_train, X_test, y_test):
    """Train a simple XGBoost model"""
    print("🌲 Training XGBoost model...")
    
    # Simple parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'auc',
        'use_label_encoder': False
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"✅ Model trained!")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    
    return model, auc, report

def main():
    """Main function"""
    
    # Load data
    data = load_and_clean_data()
    if data is None:
        return
    
    X_train, X_test, y_train, y_test, feature_cols = data
    
    # Start MLflow run
    run_name = f"simple_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # Train model
        model, auc, report = train_simple_xgboost(X_train, y_train, X_test, y_test)
        
        # Log parameters
        mlflow.log_params({
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_features': len(feature_cols),
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        })
        
        # Log metrics
        mlflow.log_metrics({
            'auc_roc': auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy']
        })
        
        # Log tags
        mlflow.set_tag('project', 'NetPredict')
        mlflow.set_tag('model', 'XGBoost')
        mlflow.set_tag('purpose', '5G Anomaly Detection')
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = "models/xgb_anomaly_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = "models/feature_importance.csv"
        importance.to_csv(importance_path, index=False)
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"📈 Feature importance saved to: {importance_path}")
        print(f"🔬 MLflow Run ID: {run_id}")
        
        # Show top features
        print("\n📊 Top 10 most important features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': model,
        'auc': auc,
        'run_id': run_id,
        'importance': importance
    }

if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "="*60)
        print("🎉 TRAINING SUCCESSFUL!")
        print("="*60)
        print(f"\nAUC-ROC: {result['auc']:.4f}")
        print(f"Run ID: {result['run_id']}")
        print("\n📺 View results:")
        print("mlflow ui --backend-store-uri sqlite:///mlflow/mlruns.db")
        print("Then open: http://localhost:5000")