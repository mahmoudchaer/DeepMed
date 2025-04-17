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
if IS_DOCKER:
    # If running in Docker, use the container names for networking
    MYSQL_HOST = os.environ.get('MYSQL_HOST')
else:
    # Fallback for local development
    MYSQL_HOST = os.environ.get('MYSQL_HOST')

# Other database settings
MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_PORT = os.environ.get('MYSQL_PORT')
MYSQL_DB = os.environ.get('MYSQL_DB')

# Log database configuration (without password)
if all([MYSQL_USER, MYSQL_HOST, MYSQL_PORT, MYSQL_DB]):
    logger.info(f"Database configuration: mysql+pymysql://{MYSQL_USER}:***@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")
else:
    logger.warning("Database environment variables not fully configured. Database operations may not work.")

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = Flask(__name__)

# Add a function to configure the database connection
def configure_db_connection():
    """Configure the database connection with proper error handling"""
    # Check if all required environment variables are set
    if not all([MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB]):
        logger.warning("Missing database environment variables. Database operations will not work.")
        return None
    
    try:
        # URL encode the password to handle special characters
        encoded_password = urllib.parse.quote_plus(str(MYSQL_PASSWORD))
        
        # Create connection string
        connection_string = f"mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
        
        # Create the engine
        db_engine = create_engine(connection_string)
        
        # Test the connection with retries
        max_retries = 5
        retry_delay = 5  # seconds
        
        for retry in range(max_retries):
            try:
                # Test connection with a simple query
                with db_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    
                logger.info("Database connection established successfully")
                return db_engine
            except Exception as e:
                logger.error(f"Database connection attempt {retry+1}/{max_retries} failed: {str(e)}")
                if retry < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        logger.error("Failed to connect to database after maximum retries")
        return None
    except Exception as e:
        logger.error(f"Error configuring database: {str(e)}")
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
        
        # Start MLflow run if not already in one
        run_uuid = str(uuid.uuid4())
        experiment_name = f"regression_{run_name}" if run_name else f"regression_{run_uuid[:8]}"
        
        # Set up MLflow tracking - catch exceptions
        try:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name or f"regression_run_{run_uuid[:8]}") as run:
                mlflow_run_id = run.info.run_id
                logger.info(f"Started MLflow run with ID: {mlflow_run_id}")
                
                # Log dataset info
                try:
                    mlflow.log_param("num_samples", len(X))
                    mlflow.log_param("num_features", len(X.columns))
                    mlflow.log_param("test_size", test_size)
                except Exception as e:
                    logger.error(f"Error logging parameters to MLflow: {str(e)}")
                
                # Train models in "parallel" (sequentially for now, but could be parallelized)
                results = []
                for model_name, available in available_services.items():
                    if available:
                        start_time = time.time()
                        model_result = self._train_model(model_name, X_train, y_train, X_test, y_test)
                        end_time = time.time()
                        
                        if model_result:
                            model_result['training_time'] = end_time - start_time
                            results.append(model_result)
                            
                            # Try to log metrics to MLflow
                            if 'metrics' in model_result:
                                try:
                                    for metric_name, metric_value in model_result['metrics'].items():
                                        mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)
                                except Exception as e:
                                    logger.error(f"Error logging metrics to MLflow: {str(e)}")
                
                # Find the best model based on R-squared
                best_model = None
                if results:
                    best_model = max(results, key=lambda x: x.get('r2', 0))
                    
                    # Log best model to MLflow
                    try:
                        mlflow.log_param("best_model", best_model.get('name'))
                        mlflow.log_metric("best_r2", best_model.get('r2', 0))
                        mlflow.log_metric("best_rmse", best_model.get('rmse', 0))
                    except Exception as e:
                        logger.error(f"Error logging best model to MLflow: {str(e)}")
        except Exception as e:
            logger.error(f"Error with MLflow tracking: {str(e)}")
            mlflow_run_id = str(uuid.uuid4())  # Use a generated ID as fallback
            results = []
            
            # Still train models even if MLflow failed
            for model_name, available in available_services.items():
                if available:
                    start_time = time.time()
                    model_result = self._train_model(model_name, X_train, y_train, X_test, y_test)
                    end_time = time.time()
                    
                    if model_result:
                        model_result['training_time'] = end_time - start_time
                        results.append(model_result)
            
            # Find the best model based on R-squared
            best_model = None
            if results:
                best_model = max(results, key=lambda x: x.get('r2', 0))
        
        logger.info(f"Trained {len(results)} regression models")
        
        # Save training run info to database if database is available
        saved_best_models = None
        db_run_id = run_id
        
        if self.db_connection and user_id:
            try:
                # If we need to create a new run in the database
                if not db_run_id:
                    try:
                        # Create the training run
                        query = text("""
                            INSERT INTO training_runs (user_id, run_name, created_at, prompt)
                            VALUES (:user_id, :run_name, :created_at, :prompt)
                        """)
                        
                        with self.db_connection.connect() as conn:
                            result = conn.execute(
                                query, 
                                {
                                    'user_id': user_id,
                                    'run_name': run_name or f"Regression Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'prompt': None
                                }
                            )
                            conn.commit()
                            
                            # Get the last inserted ID
                            last_id_query = text("SELECT LAST_INSERT_ID()")
                            result = conn.execute(last_id_query)
                            db_run_id = result.scalar()
                            logger.info(f"Created training run with ID {db_run_id}")
                    except Exception as e:
                        logger.error(f"Error creating training run: {str(e)}")
                        logger.error(traceback.format_exc())
                
                # Save models to database if we have a run_id
                if db_run_id:
                    saved_best_models = self._save_models_to_db(user_id, db_run_id, {'results': results})
                    logger.info(f"Saved models to database for run {db_run_id}: {saved_best_models}")
            except Exception as e:
                logger.error(f"Database operations failed: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("Database connection not available or user_id not provided, skipping database operations")
            # Use a fake run ID for the response
            if not db_run_id:
                db_run_id = int(time.time())
        
        # Generate some sample predictions for visualization
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
                    pred_response = requests.post(
                        f"{best_model_url.split('/saved_models')[0]}/predict",
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
        
        # Prepare the final response
        response = {
            'run_id': db_run_id,
            'mlflow_run_id': mlflow_run_id,
            'results': results,
            'best_model': best_model,
            'available_services': available_services,
            'predictions': predictions,
            'saved_best_models': saved_best_models
        }
        
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
        
        # Check if MySQL is available before proceeding
        db_available = False
        if model_coordinator.db_connection:
            try:
                # Test the connection
                with model_coordinator.db_connection.connect() as conn:
                    conn.execute(text("SELECT 1"))
                db_available = True
                logger.info("Database is available for model training")
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")
                logger.warning("Will proceed with training but models won't be saved to database")
        else:
            logger.warning("No database connection available, models won't be saved to database")
        
        # Train models
        try:
            results = model_coordinator.train_models(X, y, test_size, run_id if db_available else None, user_id if db_available else None, run_name)
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