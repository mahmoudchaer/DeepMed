from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
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
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'random_forest_regressor_model')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

MODEL_NAME = "random_forest_regressor"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', MODEL_NAME)
os.makedirs(MODELS_DIR, exist_ok=True)

class RandomForestRegressorTrainer:
    def __init__(self, test_size=0.2):
        self.model = None
        self.best_model = None
        self.test_size = test_size
        self.scaler = StandardScaler()
        
        # Define model with hyperparameters for grid search - REDUCED to avoid excessive training time
        self.model_config = {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__bootstrap': [True, False]
            },
            'scoring': ['neg_mean_squared_error', 'r2'],
            'refit': 'r2'  # Optimize for R-squared
        }
    
    def train(self, X_train, X_test, y_train, y_test):
        """Train random forest regressor model and track with MLflow"""
        logger.info("Starting random forest regressor model training")
        
        with mlflow.start_run(run_name=MODEL_NAME) as run:
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', self.model_config['model'])
                ])
                
                # Grid search for hyperparameter optimization
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=self.model_config['params'],
                    cv=5,
                    scoring=self.model_config['scoring'],
                    refit=self.model_config['refit'],
                    n_jobs=-1,
                    verbose=0
                )
                
                # Train the model
                grid_search.fit(X_train, y_train)
                self.best_model = grid_search.best_estimator_
                
                # Log best hyperparameters
                best_params = {
                    f"param_{k.replace('classifier__', '')}": v 
                    for k, v in grid_search.best_params_.items()
                }
                
                mlflow.log_params(best_params)
                
                # Make predictions on test set
                y_pred = self.best_model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Log metrics
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae
                }
                mlflow.log_metrics(metrics)
                
                # Log the model
                mlflow.sklearn.log_model(self.best_model, "model")
                
                # Log feature importances
                forest = self.best_model.named_steps['classifier']
                feature_importances = forest.feature_importances_
                feature_names = X_train.columns.tolist()
                
                importance_dict = {
                    f"importance_{feature_names[i]}": importance
                    for i, importance in enumerate(feature_importances)
                }
                mlflow.log_metrics(importance_dict)
                
                # Save the model locally
                self.save_model()
                
                logger.info(f"Model training completed with R² = {r2:.4f}, MSE = {mse:.4f}")
                logger.info(f"Best hyperparameters: {grid_search.best_params_}")
                
                return {
                    'status': 'success',
                    'run_id': run.info.run_id,
                    'metrics': metrics,
                    'params': best_params,
                    'feature_importances': dict(zip(feature_names, feature_importances.tolist()))
                }
                
            except Exception as e:
                logger.error(f"Error during model training: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e)
                }
    
    def save_model(self, output_dir=MODELS_DIR):
        """Save the trained model to disk"""
        if self.best_model is None:
            logger.error("No model to save. Train the model first.")
            return False
        
        model_path = os.path.join(output_dir, f"{MODEL_NAME}_model.joblib")
        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to {model_path}")
        return True
    
    def load_model(self, model_path):
        """Load a trained model from disk"""
        try:
            self.best_model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

@app.route('/health')
def health():
    """Check health of the service"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME
    })

@app.route('/train', methods=['POST'])
def train():
    """Train a random forest regressor model"""
    try:
        # Parse request data
        data = request.json
        
        if 'data' not in data or 'target' not in data:
            return jsonify({
                'status': 'error',
                'error': 'Missing required fields: data and target'
            }), 400
        
        # Convert data to appropriate format
        features = data['data']
        target = data['target']
        
        # Convert dict of lists to DataFrame
        if isinstance(features, dict):
            X = pd.DataFrame(features)
        # Convert list of dicts to DataFrame
        elif isinstance(features, list) and isinstance(features[0], dict):
            X = pd.DataFrame(features)
        else:
            return jsonify({
                'status': 'error',
                'error': 'Invalid data format'
            }), 400
        
        y = np.array(target)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        trainer = RandomForestRegressorTrainer()
        result = trainer.train(X_train, X_test, y_train, y_test)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in train endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with a trained model"""
    try:
        # Parse request data
        data = request.json
        
        if 'data' not in data:
            return jsonify({
                'status': 'error',
                'error': 'Missing required field: data'
            }), 400
        
        # Get model name (if provided) or use the latest model
        model_name = data.get('model', 'latest')
        
        # Convert data to appropriate format
        features = data['data']
        
        # Convert dict of lists to DataFrame
        if isinstance(features, dict):
            X = pd.DataFrame(features)
        # Convert list of dicts to DataFrame
        elif isinstance(features, list) and isinstance(features[0], dict):
            X = pd.DataFrame(features)
        else:
            return jsonify({
                'status': 'error',
                'error': 'Invalid data format'
            }), 400
        
        # Load the model
        model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_model.joblib")
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'error': 'No trained model available. Train a model first.'
            }), 400
        
        trainer = RandomForestRegressorTrainer()
        if not trainer.load_model(model_path):
            return jsonify({
                'status': 'error',
                'error': 'Failed to load model'
            }), 500
        
        # Make predictions
        predictions = trainer.best_model.predict(X)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions.tolist()
        })
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about available models"""
    try:
        models = []
        
        # Check if we have a saved model
        model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_model.joblib")
        if os.path.exists(model_path):
            # Get the most recent mlflow run to get metrics
            client = mlflow.tracking.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
                order_by=["attributes.start_time DESC"]
            )
            
            if runs:
                # Get the most recent run
                run = runs[0]
                
                # Extract feature importances from logged metrics
                feature_importances = {}
                for metric_name, metric_value in run.data.metrics.items():
                    if metric_name.startswith('importance_'):
                        feature_name = metric_name.replace('importance_', '')
                        feature_importances[feature_name] = metric_value
                
                # Add model info
                models.append({
                    'name': MODEL_NAME,
                    'type': 'random_forest_regressor',
                    'path': model_path,
                    'created_at': run.info.start_time,
                    'metrics': {
                        'mse': run.data.metrics.get('mse', None),
                        'rmse': run.data.metrics.get('rmse', None),
                        'r2': run.data.metrics.get('r2', None),
                        'mae': run.data.metrics.get('mae', None)
                    },
                    'params': {
                        param_key.replace('param_', ''): param_value
                        for param_key, param_value in run.data.params.items()
                    },
                    'feature_importances': feature_importances
                })
        
        return jsonify({
            'status': 'success',
            'model_type': MODEL_NAME,
            'models': models
        })
    
    except Exception as e:
        logger.error(f"Error in model_info endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download the trained model file"""
    try:
        model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_model.joblib")
        
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'error': 'Model file not found'
            }), 404
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"{MODEL_NAME}_model.joblib",
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        logger.error(f"Error in download_model endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 6013))
    logger.info(f"Starting Random Forest Regressor API on port {PORT}")
    serve(app, host="0.0.0.0", port=PORT)