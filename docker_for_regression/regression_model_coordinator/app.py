from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import time
import uuid
import requests
import traceback
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from datetime import datetime
from waitress import serve
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Get environment variables
IS_DOCKER = os.environ.get('IS_DOCKER', 'false').lower() == 'true'
LINEAR_REGRESSION_URL = os.environ.get('LINEAR_REGRESSION_URL', 'http://localhost:5041')
LASSO_REGRESSION_URL = os.environ.get('LASSO_REGRESSION_URL', 'http://localhost:5042')
RIDGE_REGRESSION_URL = os.environ.get('RIDGE_REGRESSION_URL', 'http://localhost:5043')
RANDOM_FOREST_REGRESSION_URL = os.environ.get('RANDOM_FOREST_REGRESSION_URL', 'http://localhost:5044')
KNN_REGRESSION_URL = os.environ.get('KNN_REGRESSION_URL', 'http://localhost:5045')
XGBOOST_REGRESSION_URL = os.environ.get('XGBOOST_REGRESSION_URL', 'http://localhost:5046')

# Database connection settings
MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_PORT = os.environ.get('MYSQL_PORT')
MYSQL_DB = os.environ.get('MYSQL_DB')

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create the default experiment if it doesn't exist
try:
    # Set up default experiment
    default_experiment = mlflow.get_experiment_by_name("Default")
    if not default_experiment:
        logger.info("Creating default MLflow experiment")
        mlflow.create_experiment("Default")
    
    # Set up regression experiment
    regression_experiment = mlflow.get_experiment_by_name("regression_experiment")
    if not regression_experiment:
        logger.info("Creating regression_experiment MLflow experiment")
        mlflow.create_experiment("regression_experiment")
except Exception as e:
    logger.error(f"Error setting up MLflow experiments: {str(e)}")
    logger.warning("Continuing without MLflow experiment setup")

app = Flask(__name__)

def configure_db_connection():
    """Configure the database connection with proper error handling"""
    # In the current setup, database connections are failing
    # Return None to disable database operations
    logger.warning("Database operations disabled in model coordinator. Models will be saved to blob storage only.")
    return None

class RegressionModelCoordinator:
    def __init__(self):
        """Initialize the model coordinator with services for regression models"""
        self.model_services = {
            'linear_regression': LINEAR_REGRESSION_URL,
            'lasso_regression': LASSO_REGRESSION_URL,
            'ridge_regression': RIDGE_REGRESSION_URL,
            'random_forest_regression': RANDOM_FOREST_REGRESSION_URL,
            'knn_regression': KNN_REGRESSION_URL,
            'xgboost_regression': XGBOOST_REGRESSION_URL
        }
        
        # Set up database connection
        self.db_connection = configure_db_connection()
    
    def _is_service_available(self, url):
        """Check if a service is available"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_available_services(self):
        """Get a list of available model services"""
        available_services = {}
        for model, url in self.model_services.items():
            available_services[model] = self._is_service_available(url)
        return available_services
    
    def _train_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a specific regression model using its service"""
        model_url = self.model_services.get(model_name)
        if not model_url or not self._is_service_available(model_url):
            logger.warning(f"Model service {model_name} is not available")
            return None
        
        try:
            # Convert data to format expected by the model service
            train_data = {
                'X_train': X_train.to_dict(orient='list') if isinstance(X_train, pd.DataFrame) else X_train,
                'y_train': y_train.tolist() if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train,
                'X_test': X_test.to_dict(orient='list') if isinstance(X_test, pd.DataFrame) else X_test,
                'y_test': y_test.tolist() if isinstance(y_test, (pd.DataFrame, pd.Series)) else y_test
            }
            
            # Call model service
            logger.info(f"Training {model_name} model")
            response = requests.post(
                f"{model_url}/train",
                json=train_data,
                timeout=600  # 10 minute timeout for training
            )
            
            if response.status_code != 200:
                logger.error(f"Error training {model_name} model: {response.text}")
                return None
            
            result = response.json()
            return result
        except Exception as e:
            logger.error(f"Error training {model_name} model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _save_models_to_db(self, user_id, run_id, model_results):
        """Save model metadata to database"""
        if not self.db_connection:
            logger.warning("Database connection not available, skipping model saving")
            return False
        
        try:
            # Save the best models to the database
            saved_models = []
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get the models to save (either top models or all models)
            models_to_save = model_results.get('results', [])
            
            # If we have performance metrics, sort by R-squared to select best models
            if models_to_save and 'r2' in models_to_save[0]:
                # Sort models by R-squared (descending)
                models_to_save = sorted(models_to_save, key=lambda x: x.get('r2', 0), reverse=True)
                
                # Optionally limit to top N models (e.g., top 4)
                models_to_save = models_to_save[:4]
            
            for model in models_to_save:
                model_name = model.get('name', 'unknown')
                model_url = model.get('model_url')
                accuracy = model.get('r2', 0)
                
                if not model_url:
                    logger.warning(f"Model {model_name} has no URL, skipping")
                    continue
                
                # Create SQL query
                query = text("""
                    INSERT INTO training_models 
                    (name, accuracy, url, parameters, run_id, user_id, created_at, confusion_matrix, metrics) 
                    VALUES (:name, :accuracy, :url, :parameters, :run_id, :user_id, :created_at, :confusion_matrix, :metrics)
                """)
                
                # Prepare parameters
                metrics = {
                    'r2': model.get('r2', 0),
                    'rmse': model.get('rmse', 0),
                    'mae': model.get('mae', 0),
                    'mse': model.get('mse', 0)
                }
                
                parameters = {
                    'name': model_name,
                    'accuracy': accuracy,
                    'url': model_url,
                    'parameters': json.dumps(model.get('parameters', {})),
                    'run_id': run_id,
                    'user_id': user_id,
                    'created_at': timestamp,
                    'confusion_matrix': json.dumps([]),  # No confusion matrix for regression
                    'metrics': json.dumps(metrics)
                }
                
                # Execute query
                with self.db_connection.connect() as conn:
                    conn.execute(query, parameters)
                
                saved_models.append(model_name)
                logger.info(f"Saved model {model_name} to database for run {run_id}")
            
            return saved_models
        except Exception as e:
            logger.error(f"Error saving models to database: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def train_models(self, X, y, test_size=0.2, run_id=None, user_id=None, run_name=None):
        """Train multiple regression models in parallel"""
        logger.info(f"Starting regression model training with {len(X.columns)} features")
        
        # Convert inputs to pandas if they're not already
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, dict):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
        
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        logger.info(f"Split data into {len(X_train)} training samples and {len(X_test)} test samples")
        
        # Get available model services
        available_services = self._get_available_services()
        logger.info(f"Available model services: {available_services}")
        
        # Create a unique run ID for this training session
        run_uuid = str(uuid.uuid4())
        mlflow_experiment_name = "regression_experiment"
        mlflow_run_id = run_uuid
        
        # Initialize results list - this will store model training results
        results = []
        
        # ----- Direct Model Training (No MLflow) -----
        # Train each model directly without MLflow to avoid experiment issues
        for model_name, available in available_services.items():
            if available:
                logger.info(f"Training {model_name}...")
                start_time = time.time()
                model_result = self._train_model(model_name, X_train, y_train, X_test, y_test)
                end_time = time.time()
                
                if model_result:
                    model_result['training_time'] = end_time - start_time
                    results.append(model_result)
                    logger.info(f"Successfully trained {model_name} with R² = {model_result.get('metrics', {}).get('r2', 0):.4f}")
                else:
                    logger.error(f"Failed to train {model_name}")
        
        # Find the best model based on R-squared
        best_model = None
        if results:
            # Sort results by R² score (descending)
            sorted_results = sorted(results, key=lambda x: x.get('metrics', {}).get('r2', 0) if isinstance(x.get('metrics'), dict) else 0, reverse=True)
            best_model = sorted_results[0] if sorted_results else None
            
            # Print top models summary
            logger.info("Top regression models:")
            for i, model in enumerate(sorted_results[:4]):
                r2 = model.get('metrics', {}).get('r2', 0) if isinstance(model.get('metrics'), dict) else model.get('r2', 0)
                logger.info(f"{i+1}. {model.get('name', 'unknown')}: R² = {r2:.4f}")
        else:
            logger.warning("No regression models were successfully trained")
        
        # Generate some sample predictions for visualization using the best model
        predictions = []
        if best_model and X_test is not None and y_test is not None:
            try:
                # Use a small subset of test data for visualization
                vis_size = min(50, len(X_test))
                X_vis = X_test.head(vis_size) if hasattr(X_test, 'head') else X_test[:vis_size]
                y_vis = y_test.head(vis_size) if hasattr(y_test, 'head') else y_test[:vis_size]
                
                # Get predictions from the best model
                best_model_url = best_model.get('model_url')
                if best_model_url:
                    # Extract base URL from model_url
                    base_url = best_model_url.split('/saved_models')[0] if '/saved_models' in best_model_url else ''
                    service_url = f"http://{best_model.get('name')}/predict" if not base_url else f"{base_url}/predict"
                    
                    pred_response = requests.post(
                        service_url,
                        json={
                            'X': X_vis.to_dict(orient='list') if isinstance(X_vis, pd.DataFrame) else X_vis
                        },
                        timeout=30
                    )
                    
                    if pred_response.status_code == 200:
                        pred_data = pred_response.json()
                        y_pred = pred_data.get('predictions', [])
                        
                        # Format predictions for visualization
                        for i in range(len(y_vis)):
                            actual = y_vis.iloc[i] if hasattr(y_vis, 'iloc') else y_vis[i]
                            predicted = y_pred[i] if i < len(y_pred) else None
                            
                            if predicted is not None:
                                predictions.append({
                                    'actual': float(actual),
                                    'predicted': float(predicted),
                                    'error': float(actual) - float(predicted)
                                })
            except Exception as e:
                logger.error(f"Error generating sample predictions: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Create a database-independent run_id if we don't have one
        if not run_id:
            db_run_id = int(time.time())
        else:
            db_run_id = run_id
            
        # Prepare the final response
        response = {
            'run_id': db_run_id,
            'mlflow_run_id': mlflow_run_id,
            'results': results,
            'best_model': best_model,
            'available_services': available_services,
            'predictions': predictions
        }
        
        # For app_regression.py to use ensure_regression_models_saved
        if results:
            logger.info(f"Successful training completed with {len(results)} models")
        
        return response


# Create a coordinator instance
model_coordinator = RegressionModelCoordinator()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with service availability information"""
    # Get available model services directly without creating a new coordinator
    available_services = model_coordinator._get_available_services()
    
    # Don't attempt database connection during health check to avoid timeouts
    db_status = "connected" if model_coordinator.db_connection is not None else "disconnected"
    
    return jsonify({
        "status": "healthy",
        "service": "regression_model_coordinator",
        "model_services": available_services,
        "database_status": db_status
    })

@app.route('/train', methods=['POST'])
def train():
    """Train multiple regression models"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data or 'target' not in request_data:
            return jsonify({"error": "Missing required data fields"}), 400
        
        # Extract data and options
        data = request_data['data']
        target = request_data['target']
        test_size = request_data.get('test_size', 0.2)
        user_id = request_data.get('user_id')
        run_id = request_data.get('run_id')
        run_name = request_data.get('run_name')
        
        # Convert data to DataFrame
        X = pd.DataFrame(data)
        y = pd.Series(target)
        
        # Log basic info
        logger.info(f"Received training request from user {user_id}")
        logger.info(f"Data contains {len(X)} records with {len(X.columns)} features")
        
        # Skip database checks as they're failing - proceed directly to training
        logger.warning("No database connection available, models won't be saved to database")
        
        # Train models
        try:
            results = model_coordinator.train_models(X, y, test_size, run_id, user_id, run_name)
            return jsonify(results)
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Training error: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5040))
    logger.info(f"Starting Regression Model Coordinator Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 