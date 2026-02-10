"""
Training pipeline for 5G anomaly detection with MLflow tracking
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import pickle
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.make_dataset import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path='configs/mlflow_config.yaml'):
        self.config = self.load_config(config_path)
        self.setup_mlflow()
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_path):
        """Load configuration file"""
        config_path = Path(config_path)
        if not config_path.exists():
            # Create default config
            default_config = {
                'tracking': {
                    'uri': 'sqlite:///mlflow/mlruns.db',
                    'artifact_location': './mlflow/artifacts'
                },
                'experiment': {
                    'name': '5g_anomaly_detection',
                    'tags': {
                        'project': 'NetPredict',
                        'domain': 'Telecom',
                        'team': 'MLOps'
                    }
                },
                'model_registry': {
                    'name': 'NetworkAnomalyModel',
                    'stages': ['Staging', 'Production', 'Archived']
                },
                'training': {
                    'n_trials': 30,
                    'cv_folds': 3,
                    'random_state': 42
                }
            }
            
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
            
            return default_config
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_mlflow(self):
        """Initialize MLflow tracking server"""
        mlflow.set_tracking_uri(self.config['tracking']['uri'])
        
        # Create experiment if it doesn't exist
        experiment_name = self.config['experiment']['name']
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=self.config['tracking']['artifact_location']
            )
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Create mlflow directory structure
        mlflow_dir = Path('mlflow')
        mlflow_dir.mkdir(exist_ok=True)
        (mlflow_dir / 'artifacts').mkdir(exist_ok=True)
        
        return experiment_id
    
    def load_processed_data(self):
        """Load processed data from pipeline"""
        logger.info("Loading processed data...")
        
        try:
            # Try to load from pickle
            processed_path = Path('data/processed/processed_data.pkl')
            if processed_path.exists():
                with open(processed_path, 'rb') as f:
                    processed_data = pickle.load(f)
            else:
                # Run data processor
                processor = DataProcessor(
                    raw_data_path='data/raw/5g_telemetry.csv',
                    processed_dir='data/processed'
                )
                processed_data = processor.process()
            
            logger.info(f"Loaded data: {len(processed_data['X_train'])} train, "
                       f"{len(processed_data['X_test'])} test samples")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    # In train.py, add this after loading data:
def clean_data(self, X_train, X_test):
    """Clean data by removing inf and capping extreme values"""
    # Replace inf with NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with column means
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # Use training means
    
    # Cap extreme values (99.9th percentile)
    for col in X_train.columns:
        upper_limit = X_train[col].quantile(0.999)
        lower_limit = X_train[col].quantile(0.001)
        X_train[col] = X_train[col].clip(lower=lower_limit, upper=upper_limit)
        X_test[col] = X_test[col].clip(lower=lower_limit, upper=upper_limit)
    
    return X_train, X_test

# Then in your training function:

    
    def objective(self, trial, X_train, y_train):
        """Optuna objective function for hyperparameter optimization"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'random_state': self.config['training']['random_state'],
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'auc'
        }
        
        # Cross-validation
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(
            n_splits=self.config['training']['cv_folds'],
            shuffle=True,
            random_state=self.config['training']['random_state']
        )
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Predict and calculate ROC-AUC
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            from sklearn.metrics import roc_auc_score
            score = roc_auc_score(y_val_fold, y_pred_proba)
            cv_scores.append(score)
            
            # Report intermediate score for pruning
            trial.report(score, fold)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(cv_scores)
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials=None):
        """Run hyperparameter optimization with Optuna"""
        if n_trials is None:
            n_trials = self.config['training']['n_trials']
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name='xgboost_5g_anomaly',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best trial value: {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        return study.best_params, study.best_trial.value
    
    def train_final_model(self, X_train, y_train, X_test, y_test, best_params):
        """Train final model with best parameters"""
        logger.info("Training final model with best parameters...")
        
        # Add fixed parameters
        final_params = best_params.copy()
        final_params.update({
            'random_state': self.config['training']['random_state'],
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'auc'
        })
        
        # Train model
        model = xgb.XGBClassifier(**final_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import (roc_auc_score, precision_score, 
                                   recall_score, f1_score, 
                                   classification_report, confusion_matrix)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean()
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return model, metrics, report, cm
    
    def log_to_mlflow(self, run_name, model, metrics, params, feature_names, 
                     feature_importance, report, cm):
        """Log everything to MLflow"""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # Log tags
            for key, value in self.config['experiment']['tags'].items():
                mlflow.set_tag(key, value)
            mlflow.set_tag('model_type', 'XGBoost')
            mlflow.set_tag('task', 'binary_classification')
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            importance_path = self.models_dir / 'feature_importance.csv'
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
            
            # Log classification report
            report_path = self.models_dir / 'classification_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(str(report_path))
            
            # Log confusion matrix
            cm_path = self.models_dir / 'confusion_matrix.json'
            with open(cm_path, 'w') as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(str(cm_path))
            
            # Log dataset info
            dataset_info = {
                'n_features': len(feature_names),
                'feature_names': feature_names[:10]  # First 10 only
            }
            mlflow.log_dict(dataset_info, "dataset_info.json")
            
            # Save model locally
            model_path = self.models_dir / 'xgb_anomaly_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path))
            
            logger.info(f"Logged to MLflow: run_id={run_id}")
            
            return run_id
    
    def register_model(self, run_id, model_name=None, stage="Staging"):
        """Register model in MLflow Model Registry"""
        if model_name is None:
            model_name = self.config['model_registry']['name']
        
        model_uri = f"runs:/{run_id}/model"
        
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Create model version
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )
            logger.info(f"Created model version {model_version.version}")
            
            # Transition to stage
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            logger.info(f"Model version {model_version.version} transitioned to {stage}")
            
            return model_version.version
            
        except Exception as e:
            logger.warning(f"Model registration failed (might already exist): {e}")
            return None
    
    def train(self, n_trials=None):
        """Main training pipeline"""
        logger.info("="*50)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*50)
        
        # 1. Load data
        processed_data = self.load_processed_data()
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        feature_names = processed_data['feature_names']

        X_train, X_test = self.clean_data(X_train, X_test)
        
        # Log dataset info
        mlflow.log_param('n_train_samples', len(X_train))
        mlflow.log_param('n_test_samples', len(X_test))
        mlflow.log_param('n_features', len(feature_names))
        mlflow.log_param('train_anomaly_rate', y_train.mean())
        mlflow.log_param('test_anomaly_rate', y_test.mean())
        
        # 2. Hyperparameter optimization
        best_params, best_score = self.optimize_hyperparameters(
            X_train, y_train, n_trials
        )
        
        # 3. Train final model
        model, metrics, report, cm = self.train_final_model(
            X_train, y_train, X_test, y_test, best_params
        )
        
        # 4. Get feature importance
        feature_importance = model.feature_importances_
        
        # 5. Log to MLflow
        run_name = f"xgboost_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = self.log_to_mlflow(
            run_name=run_name,
            model=model,
            metrics=metrics,
            params=best_params,
            feature_names=feature_names,
            feature_importance=feature_importance,
            report=report,
            cm=cm
        )
        
        # 6. Register model
        model_version = self.register_model(run_id)
        
        # 7. Save model artifacts
        artifacts = {
            'model': model,
            'run_id': run_id,
            'metrics': metrics,
            'params': best_params,
            'feature_names': feature_names,
            'feature_importance': pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False),
            'model_version': model_version
        }
        
        # Save artifacts
        artifacts_path = self.models_dir / 'training_artifacts.pkl'
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*50)
        logger.info(f"Best AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"MLflow Run ID: {run_id}")
        logger.info(f"Model saved to: {self.models_dir}/")
        
        if model_version:
            logger.info(f"Model registered as version: {model_version}")
        
        # Print top features
        top_features = artifacts['feature_importance'].head(10)
        logger.info("\nTop 10 features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return artifacts

def main():
    """Main function"""
    # For quick testing, reduce number of trials
    trainer = ModelTrainer()
    
    # Create config if doesn't exist
    config_path = Path('configs/mlflow_config.yaml')
    if not config_path.exists():
        logger.info("Creating default MLflow config...")
    
    # Train with reduced trials for testing
    artifacts = trainer.train(n_trials=10)  # Use 10 for testing, 30+ for real
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. View MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow/mlruns.db")
    print("2. Check saved model: models/xgb_anomaly_model.pkl")
    print("3. Check artifacts: models/training_artifacts.pkl")
    print("\nTo start API: uvicorn src.api.main:app --reload")
    
    return artifacts

if __name__ == "__main__":
    artifacts = main()