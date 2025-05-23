from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging
import sys
import os
import tempfile
import json
import io
import base64
from waitress import serve
from mlflow.tracking import MlflowClient
import pickle
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set up MLflow tracking
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
MODEL_TYPE = "random_forest"

# Ensure MLflow tracking directory exists
def ensure_mlflow_directory():
    # Extract the local path from the tracking URI
    if MLFLOW_TRACKING_URI.startswith('file:///'):
        tracking_dir = MLFLOW_TRACKING_URI[7:]  # Remove 'file:///'
    else:
        # Default location
        tracking_dir = '/app/mlruns'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(tracking_dir):
        logger.warning(f"MLflow tracking directory {tracking_dir} does not exist. Creating it.")
        try:
            Path(tracking_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created MLflow tracking directory: {tracking_dir}")
        except Exception as e:
            logger.error(f"Error creating MLflow tracking directory: {str(e)}")
    else:
        logger.info(f"MLflow tracking directory already exists: {tracking_dir}")
    
    # Set proper permissions
    try:
        os.chmod(tracking_dir, 0o777)  # Full permissions
        logger.info(f"Set permissions on MLflow tracking directory")
    except Exception as e:
        logger.error(f"Error setting permissions on MLflow directory: {str(e)}")

# Initialize MLflow tracking
def init_mlflow():
    try:
        # Ensure directory exists
        ensure_mlflow_directory()
        
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Test MLflow connection
        client = MlflowClient()
        experiment_names = [exp.name for exp in client.list_experiments()]
        logger.info(f"Connected to MLflow. Available experiments: {experiment_names}")
        
        # Create model-specific experiment if it doesn't exist
        experiment_name = f"{MODEL_TYPE}_experiment"
        if experiment_name not in experiment_names:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Created {experiment_name} MLflow experiment")
    except Exception as e:
        logger.error(f"Error initializing MLflow: {str(e)}")

# Initialize MLflow on startup
init_mlflow()

MODEL_NAME = "random_forest"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', MODEL_NAME)
os.makedirs(MODELS_DIR, exist_ok=True)

class RandomForestTrainer:
    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.best_model = None
        self.best_model_info = None
        self.label_encoder = LabelEncoder()
        
        # Define model and parameters
        self.model_config = {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__class_weight': ['balanced', None]
            },
            'scoring': 'accuracy',  # Default scoring
            'refit': 'accuracy'    # Could be 'f1' or other metric depending on need
        }
    
    def _handle_imbalance(self, X, y):
        """Handle imbalanced datasets using SMOTE, conditionally based on dataset size"""
        # Skip SMOTE for larger datasets to save computation
        if X.shape[0] > 1000:
            logger.info("Dataset is large, skipping SMOTE to reduce computation")
            return X, y
            
        try:
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info("Applied SMOTE to handle class imbalance")
            return X_resampled, y_resampled
        except Exception as e:
            logger.warning(f"Could not apply SMOTE: {str(e)}. Using original data.")
            return X, y
    
    def train(self, X_train, X_test, y_train, y_test):
        """Train random forest model and track with MLflow"""
        logger.info("Starting random forest model training")
        
        # Handle imbalanced datasets
        X_train_balanced, y_train_balanced = self._handle_imbalance(X_train, y_train)
        
        # Encode labels if they're not numeric
        if not np.issubdtype(y_train.dtype, np.number):
            y_train_balanced = self.label_encoder.fit_transform(y_train_balanced)
            y_test = self.label_encoder.transform(y_test)
        
        with mlflow.start_run(run_name=MODEL_NAME) as run:
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', self.model_config['model'])
                ])
                
                # Perform grid search with cross-validation - REDUCED FOLDS
                grid_search = GridSearchCV(
                    pipeline,
                    self.model_config['params'],
                    cv=3,  # Reduced from 5 to 3 folds
                    scoring=self.model_config['scoring'],
                    refit=self.model_config['refit'],
                    n_jobs=-1,
                    verbose=1
                )
                
                # Train model
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                # Get best model
                self.best_model = grid_search.best_estimator_
                
                # Make predictions
                y_pred = self.best_model.predict(X_test)
                
                # Calculate metrics and convert numpy types to Python native types
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted')),
                    'recall': float(recall_score(y_test, y_pred, average='weighted')),
                    'f1': float(f1_score(y_test, y_pred, average='weighted')),
                    'cv_score_mean': float(grid_search.best_score_),
                    'cv_score_std': float(grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_])
                }
                
                # Log the metrics for debugging
                logger.info(f"Model metrics: {metrics}")
                
                # Log parameters and metrics
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics(metrics)
                
                # Store model and info
                model_info = {
                    'model': self.best_model,
                    'metrics': metrics,
                    'model_name': MODEL_NAME,
                    'params': grid_search.best_params_,
                    'label_encoder': self.label_encoder if not np.issubdtype(y_train.dtype, np.number) else None
                }
                
                self.best_model_info = model_info
                logger.info(f"Completed training {MODEL_NAME}")
                
                return model_info
                
            except Exception as e:
                logger.error(f"Error training {MODEL_NAME}: {str(e)}")
                raise
    
    def save_model(self, output_dir=MODELS_DIR):
        """Save trained model to disk"""
        if self.best_model_info is None:
            raise ValueError("No trained model to save. Train a model first.")
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_path = output_path / 'model.joblib'
        joblib.dump(self.best_model_info, model_path)
        
        return str(model_path)
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        model_info = joblib.load(model_path)
        self.best_model_info = model_info
        self.best_model = model_info['model']
        return model_info

# Initialize model trainer
model_trainer = RandomForestTrainer()

@app.route('/reset', methods=['POST'])
def reset_model():
    """Reset the model trainer to ensure fresh training for each dataset"""
    global model_trainer
    logger.info("Received request to reset model trainer - creating new instance")
    model_trainer = RandomForestTrainer()
    return jsonify({
        "status": "success",
        "message": "Model trainer has been reset and will train from scratch"
    })

@app.route('/health')
def health():
    return jsonify({
        "service": "random_forest_model_api",
        "status": "healthy"
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Train a Random Forest model on the provided data
    
    Expected JSON input:
    {
        "data": {...},  # Features in JSON format
        "target": [...],  # Target variable
        "test_size": 0.2  # Optional
    }
    
    Returns:
    {
        "model": {...},  # Trained model info
        "saved_model": "...",  # Saved model path
        "message": "Model trained successfully"
    }
    """
    try:
        request_data = request.json
        
        if not request_data or 'data' not in request_data or 'target' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data' or 'target'."}), 400
        
        # Convert JSON to DataFrame/array
        try:
            X = pd.DataFrame.from_dict(request_data['data'])
            y = np.array(request_data['target'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Set test size if provided
        if 'test_size' in request_data:
            model_trainer.test_size = float(request_data['test_size'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=model_trainer.test_size, random_state=42
        )
        
        # Train model
        model_info = model_trainer.train(X_train, X_test, y_train, y_test)
        
        # Save model
        saved_model = model_trainer.save_model()
        
        # Prepare response - ensure metrics are native Python types
        clean_metrics = {}
        for metric_name, metric_value in model_info['metrics'].items():
            if isinstance(metric_value, (np.float64, np.float32, np.int64, np.int32)):
                clean_metrics[metric_name] = float(metric_value)
            else:
                clean_metrics[metric_name] = metric_value
        
        response_model_info = {
            'model_name': model_info['model_name'],
            'metrics': clean_metrics,
            'params': str(model_info['params'])
        }
        
        # Log the response for debugging
        logger.info(f"Train response metrics: {clean_metrics}")
        
        return jsonify({
            "model": response_model_info,
            "saved_model": saved_model,
            "message": "Model trained successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in train endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@app.route('/model_info', methods=['GET'])
def model_info():
    """Get detailed information about the trained model"""
    try:
        model_path = os.path.join(MODELS_DIR, 'model.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found. Train model first."}), 404
        
        # Load model
        model_info = model_trainer.load_model(model_path)
        
        # Create a clean response (without the actual model object)
        # Ensure metrics are numeric and properly serialized
        clean_metrics = {}
        for metric_name, metric_value in model_info['metrics'].items():
            if isinstance(metric_value, (np.float64, np.float32, np.int64, np.int32)):
                # Convert numpy types to Python native types
                clean_metrics[metric_name] = float(metric_value)
            else:
                clean_metrics[metric_name] = metric_value
                
        response = {
            'model_name': model_info['model_name'],
            'metrics': clean_metrics,
            'params': str(model_info['params']),
            'task': 'classification'
        }
        
        # Add debug log to verify metrics
        logger.info(f"Returning model info with metrics: {clean_metrics}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in model_info endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download the trained model file"""
    try:
        model_path = os.path.join(MODELS_DIR, 'model.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        
        return send_file(model_path, as_attachment=True, download_name='random_forest_model.joblib')
    
    except Exception as e:
        logger.error(f"Error in download_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5012))
    serve(app, host='0.0.0.0', port=port) 