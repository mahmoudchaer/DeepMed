from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'logistic_regression_model')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_NAME = "logistic_regression"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', MODEL_NAME)
os.makedirs(MODELS_DIR, exist_ok=True)

class LogisticRegressionTrainer:
    def __init__(self, test_size=0.2):
        self.model = None
        self.best_model = None
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Define model with hyperparameters for grid search - REDUCED
        self.model_config = {
            'model': LogisticRegression(multi_class='ovr'),
            'params': {
                'classifier__C': [0.1, 1, 10],  # Reduced from 6 options to 3
                'classifier__solver': ['saga'],  # Using only the most versatile solver
                'classifier__max_iter': [5000],  # Fixed value instead of multiple options
                'classifier__class_weight': ['balanced']
            },
            'scoring': ['accuracy', 'precision_weighted', 'recall_weighted'],  # Removed f1
            'refit': 'accuracy'
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
        """Train logistic regression model and track with MLflow"""
        logger.info("Starting logistic regression model training")
        
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
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted'),
                    'cv_score_mean': grid_search.best_score_,
                    'cv_score_std': grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_]
                }
                
                # Log parameters and metrics
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(self.best_model, MODEL_NAME)
                
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
model_trainer = LogisticRegressionTrainer()

@app.route('/health')
def health():
    return jsonify({
        "service": "logistic_regression_model_api",
        "status": "healthy"
    })

@app.route('/train', methods=['POST'])
def train():
    """
    Train a logistic regression model on the provided data
    
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
        
        # Prepare response
        response_model_info = {
            'model_name': model_info['model_name'],
            'metrics': model_info['metrics'],
            'params': str(model_info['params'])
        }
        
        return jsonify({
            "model": response_model_info,
            "saved_model": saved_model,
            "message": "Model trained successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in train endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using the trained model
    
    Expected JSON input:
    {
        "data": {...}  # Features in JSON format
    }
    
    Returns:
    {
        "predictions": [...],  # List of predictions
        "probabilities": [...],  # List of probability distributions for each class
        "message": "Predictions made successfully"
    }
    """
    try:
        # Check if model exists
        model_path = os.path.join(MODELS_DIR, 'model.joblib')
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found. Train model first."}), 404
        
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'."}), 400
        
        # Convert JSON to DataFrame
        try:
            data = pd.DataFrame.from_dict(request_data['data'])
        except Exception as e:
            return jsonify({"error": f"Failed to convert JSON to DataFrame: {str(e)}"}), 400
        
        # Load model and make predictions
        model_info = model_trainer.load_model(model_path)
        model = model_info['model']
        
        predictions = model.predict(data)
        probabilities = model.predict_proba(data).tolist()
        
        # If classification with label encoder, decode predictions
        if 'label_encoder' in model_info and model_info['label_encoder'] is not None:
            label_encoder = model_info['label_encoder']
            mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
            decoded_predictions = [mapping[pred] for pred in predictions]
            predictions = decoded_predictions
        
        # Convert NumPy values to native Python types
        converted_predictions = []
        for p in predictions:
            if isinstance(p, (np.integer, np.floating)):
                converted_value = float(p)
                if abs(converted_value - int(converted_value)) < 1e-10:
                    converted_value = int(converted_value)
            else:
                converted_value = str(p)
            converted_predictions.append(converted_value)
        
        return jsonify({
            "predictions": converted_predictions,
            "probabilities": probabilities,
            "message": "Predictions made successfully"
        })
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
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
        response = {
            'model_name': model_info['model_name'],
            'metrics': model_info['metrics'],
            'params': str(model_info['params']),
            'task': 'classification'
        }
        
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
        
        return send_file(model_path, as_attachment=True, download_name='logistic_regression_model.joblib')
    
    except Exception as e:
        logger.error(f"Error in download_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5010))
    serve(app, host='0.0.0.0', port=port) 