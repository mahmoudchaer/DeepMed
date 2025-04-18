from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import io
import time
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from waitress import serve
import uuid
from datetime import datetime
import traceback

# Try to import storage module for Azure Blob Storage
try:
    from storage import upload_to_blob, get_blob_url, download_blob
    BLOB_STORAGE_AVAILABLE = True
    logging.info("Azure Blob Storage integration available")
except ImportError:
    BLOB_STORAGE_AVAILABLE = False
    logging.warning("Azure Blob Storage integration not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Enable verbose logging if environment variable is set
if os.environ.get('VERBOSE_LOGGING', 'false').lower() in ('true', '1', 't', 'yes'):
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("Verbose logging enabled")

logger = logging.getLogger(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'regression_experiment')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Define directories
SAVED_MODELS_DIR = os.environ.get('SAVED_MODELS_DIR', '/app/saved_models/xgboost_regression')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

app = Flask(__name__)

class XGBoostRegressionModel:
    def __init__(self):
        """Initialize the XGBoost regression model"""
        # Default hyperparameters - will be tuned
        self.learning_rate = 0.1
        self.n_estimators = 100
        self.max_depth = 5
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.min_child_weight = 1
        self.gamma = 0
        self.reg_alpha = 0
        self.reg_lambda = 1
        self.objective = 'reg:squarederror'
        
        self.model = None
        self.is_trained = False
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train an XGBoost Regression model with hyperparameter tuning"""
        logger.info(f"Training XGBoost Regression model with {X_train.shape[1]} features and hyperparameter tuning")
        
        # Start timer
        start_time = time.time()
        
        # Get a unique experiment ID for all the runs
        experiment_id = int(time.time())
        
        # Start MLflow parent run
        with mlflow.start_run(run_name=f"xgboost_regression_tuning_{experiment_id}") as parent_run:
            # Log basic information
            mlflow.log_params({
                "model_type": "xgboost_regression",
                "tuning": True,
                "tuning_method": "randomized_search",
                "num_features": X_train.shape[1],
                "num_samples": X_train.shape[0]
            })
            
            try:
                # Define the hyperparameter search space
                param_space = {
                    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
                    'n_estimators': [50, 100, 150, 200, 300],
                    'max_depth': [3, 4, 5, 6, 7, 9],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5, 7],
                    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
                    'reg_alpha': [0, 0.001, 0.01, 0.1, 1],
                    'reg_lambda': [0.01, 0.1, 1, 10]
                }
                
                # Configure base model
                base_model = xgb.XGBRegressor(
                    objective=self.objective,
                    random_state=42,
                    tree_method='hist'  # 'hist' is faster than 'exact' for large datasets
                )
                
                # Configure RandomizedSearchCV
                n_iter = min(20, 3 * len(X_train) // 1000 + 5)  # Scale iterations based on dataset size
                logger.info(f"Running RandomizedSearchCV with {n_iter} iterations")
                
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_space,
                    n_iter=n_iter,
                    scoring='r2',
                    cv=3,
                    verbose=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Perform the search
                search.fit(X_train, y_train)
                
                # Log all evaluated models to MLflow
                all_results = []
                for i, params in enumerate(search.cv_results_['params']):
                    r2 = search.cv_results_['mean_test_score'][i]
                    fit_time = search.cv_results_['mean_fit_time'][i]
                    
                    result = {
                        **params,
                        'test_r2': r2,
                        'fit_time': fit_time
                    }
                    all_results.append(result)
                    
                    # Log top models to MLflow
                    if i < 10:  # Only log top models to avoid clutter
                        with mlflow.start_run(run_name=f"xgboost_model_{i+1}", nested=True) as run:
                            mlflow.log_params(params)
                            mlflow.log_metric("cv_r2", r2)
                            mlflow.log_metric("fit_time", fit_time)
                
                # Set best model parameters
                self.model = search.best_estimator_
                best_params = search.best_params_
                
                # Update instance variables with best parameters
                self.learning_rate = best_params.get('learning_rate', self.learning_rate)
                self.n_estimators = best_params.get('n_estimators', self.n_estimators)
                self.max_depth = best_params.get('max_depth', self.max_depth)
                self.subsample = best_params.get('subsample', self.subsample)
                self.colsample_bytree = best_params.get('colsample_bytree', self.colsample_bytree)
                self.min_child_weight = best_params.get('min_child_weight', self.min_child_weight)
                self.gamma = best_params.get('gamma', self.gamma)
                self.reg_alpha = best_params.get('reg_alpha', self.reg_alpha)
                self.reg_lambda = best_params.get('reg_lambda', self.reg_lambda)
                
                # Make predictions on test set
                y_pred_train = self.model.predict(X_train)
                y_pred_test = self.model.predict(X_test)
                
                # Calculate final metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                # Get feature importances
                feature_importances = self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else []
                
                # Log final metrics to parent run
                mlflow.log_params(best_params)
                mlflow.log_metrics({
                    "best_train_r2": train_r2,
                    "best_test_r2": test_r2,
                    "best_test_rmse": test_rmse,
                    "best_test_mae": test_mae,
                    "best_test_mse": test_mse
                })
                
                # Log feature importances if available
                if feature_importances:
                    for i, importance in enumerate(feature_importances):
                        mlflow.log_metric(f"feature_importance_{i}", importance)
                
                # Calculate total training time
                train_time = time.time() - start_time
                logger.info(f"Total hyperparameter tuning time: {train_time:.2f} seconds")
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Best test R²: {test_r2:.4f}")
                
                # Get model parameters
                model_params = {
                    'learning_rate': self.learning_rate,
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'subsample': self.subsample,
                    'colsample_bytree': self.colsample_bytree,
                    'min_child_weight': self.min_child_weight,
                    'gamma': self.gamma,
                    'reg_alpha': self.reg_alpha,
                    'reg_lambda': self.reg_lambda,
                    'objective': self.objective,
                    'feature_importances': feature_importances
                }
                
                # Generate timestamp and unique ID for filenames
                timestamp = int(time.time())
                unique_id = str(uuid.uuid4())[:8]
                model_basename = f"xgboost_regression_{timestamp}_{unique_id}"
                
                # Save best model to disk
                local_model_path = f"{SAVED_MODELS_DIR}/{model_basename}.joblib"
                joblib.dump(self.model, local_model_path)
                logger.info(f"Best model saved to {local_model_path}")
                
                # Save model to Blob Storage if available
                model_url = None
                if BLOB_STORAGE_AVAILABLE:
                    try:
                        logger.info("Attempting to save model to Azure Blob Storage...")
                        # Serialize model to memory
                        model_bytes = io.BytesIO()
                        joblib.dump(self.model, model_bytes)
                        model_bytes.seek(0)
                        
                        # Upload to blob storage
                        blob_filename = f"{model_basename}.joblib"
                        logger.debug(f"Uploading model as {blob_filename}")
                        blob_url = upload_to_blob(model_bytes, blob_filename)
                        
                        if blob_url:
                            logger.info(f"Model successfully uploaded to blob storage: {blob_url}")
                            model_url = blob_url
                        else:
                            logger.error("Failed to upload model to blob storage - null response from upload_to_blob")
                    except Exception as e:
                        logger.error(f"Error uploading model to blob storage: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning("Azure Blob Storage not available, skipping cloud upload")
                
                # If blob storage upload failed or isn't available, use local path
                if not model_url:
                    model_url = f"/saved_models/xgboost_regression/{model_basename}.joblib"
                    logger.info(f"Using local file path as model URL: {model_url}")
                
                # Log the best model in MLflow
                mlflow.sklearn.log_model(self.model, "best_model")
                logger.info("Best model logged to MLflow successfully")
                
                # Set trained flag
                self.is_trained = True
                
                # Prepare and return results
                return {
                    'name': 'xgboost_regression',
                    'parameters': model_params,
                    'metrics': {
                        'train_r2': float(train_r2),
                        'r2': float(test_r2),
                        'rmse': float(test_rmse),
                        'mae': float(test_mae),
                        'mse': float(test_mse)
                    },
                    'tuning_results': [
                        {
                            'learning_rate': float(r.get('learning_rate', 0)),
                            'n_estimators': int(r.get('n_estimators', 0)),
                            'max_depth': int(r.get('max_depth', 0)),
                            'test_r2': float(r.get('test_r2', 0)),
                            'fit_time': float(r.get('fit_time', 0))
                        } for r in sorted(all_results, key=lambda x: x.get('test_r2', 0), reverse=True)[:10]  # Only return top 10
                    ],
                    'model_url': model_url,
                    'training_time': train_time
                }
                
            except Exception as e:
                logger.error(f"Error in XGBoost training: {str(e)}")
                logger.error(traceback.format_exc())
                # Re-raise to be caught by the endpoint handler
                raise
    
    def predict(self, X):
        """Make predictions with the trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        return y_pred.tolist()
    
    def load_model(self, model_path):
        """Load a saved model"""
        try:
            # Check if model path is a URL (starts with http or https)
            if model_path.startswith(('http://', 'https://')):
                if BLOB_STORAGE_AVAILABLE:
                    logger.info(f"Downloading model from blob storage: {model_path}")
                    
                    # Create a temporary file path
                    temp_path = os.path.join(SAVED_MODELS_DIR, f"temp_model_{int(time.time())}.joblib")
                    
                    # Download the model
                    download_success = download_blob(model_path, temp_path)
                    
                    if download_success:
                        logger.info(f"Model downloaded successfully to {temp_path}")
                        full_path = temp_path
                    else:
                        logger.error(f"Failed to download model from {model_path}")
                        return False
                else:
                    logger.error("Cannot download model from URL - Azure Blob Storage not available")
                    return False
            else:
                # Local file
                full_path = model_path
                if not model_path.startswith('/'):
                    full_path = os.path.join(SAVED_MODELS_DIR, os.path.basename(model_path))
            
            logger.info(f"Loading model from {full_path}")
            self.model = joblib.load(full_path)
            self.is_trained = True
            logger.info(f"Model loaded successfully from {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# Create model instance
regression_model = XGBoostRegressionModel()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "xgboost_regression",
        "is_trained": regression_model.is_trained,
        "blob_storage_available": BLOB_STORAGE_AVAILABLE
    })

@app.route('/train', methods=['POST'])
def train():
    """Train the model with provided data"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'X_train' not in request_data or 'y_train' not in request_data:
            return jsonify({"error": "Missing required training data"}), 400
        
        # Extract training and testing data
        X_train = request_data['X_train']
        y_train = request_data['y_train']
        X_test = request_data['X_test']
        y_test = request_data['y_test']
        
        # Convert to DataFrame/Series if they're lists/dictionaries
        if isinstance(X_train, dict):
            X_train = pd.DataFrame(X_train)
        elif isinstance(X_train, list):
            X_train = pd.DataFrame(X_train)
            
        if isinstance(X_test, dict):
            X_test = pd.DataFrame(X_test)
        elif isinstance(X_test, list):
            X_test = pd.DataFrame(X_test)
            
        if isinstance(y_train, list):
            y_train = pd.Series(y_train)
            
        if isinstance(y_test, list):
            y_test = pd.Series(y_test)
        
        # Train the model
        result = regression_model.train(X_train, y_train, X_test, y_test)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with the trained model"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'X' not in request_data:
            return jsonify({"error": "Missing input data"}), 400
        
        # Extract prediction data
        X = request_data['X']
        
        # Convert to DataFrame if it's a list/dictionary
        if isinstance(X, dict):
            X = pd.DataFrame(X)
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Check if model is trained
        if not regression_model.is_trained:
            # Check if a specific model file is requested
            model_url = request_data.get('model_url')
            if model_url:
                if not regression_model.load_model(model_url):
                    return jsonify({"error": "Failed to load specified model"}), 400
            else:
                return jsonify({"error": "Model is not trained and no model URL specified"}), 400
        
        # Make predictions
        predictions = regression_model.predict(X)
        
        return jsonify({
            "predictions": predictions
        })
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a saved model"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'model_url' not in request_data:
            return jsonify({"error": "Missing model URL"}), 400
        
        # Extract model URL
        model_url = request_data['model_url']
        
        # Load the model
        success = regression_model.load_model(model_url)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Model loaded from {model_url}"
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Failed to load model from {model_url}"
            }), 400
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download the trained model file"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Download model request received")
    
    try:
        if not regression_model.is_trained:
            logger.warning(f"[{request_id}] Model is not trained, cannot download")
            return jsonify({"error": "Model is not trained"}), 400
            
        # Find the most recent model file
        model_path = None
        latest_time = 0
        
        # Log available models
        if os.path.exists(SAVED_MODELS_DIR):
            model_files = [f for f in os.listdir(SAVED_MODELS_DIR) if f.endswith('.joblib')]
            logger.info(f"[{request_id}] Found {len(model_files)} model files in {SAVED_MODELS_DIR}")
            for file in model_files:
                file_path = os.path.join(SAVED_MODELS_DIR, file)
                file_size = os.path.getsize(file_path)
                file_time = os.path.getmtime(file_path)
                logger.debug(f"[{request_id}] Model file: {file}, Size: {file_size} bytes, Modified: {datetime.fromtimestamp(file_time).isoformat()}")
        else:
            logger.warning(f"[{request_id}] Models directory {SAVED_MODELS_DIR} does not exist")
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            logger.info(f"[{request_id}] Created models directory {SAVED_MODELS_DIR}")
        
        for file in os.listdir(SAVED_MODELS_DIR):
            if file.startswith('xgboost_regression_') and file.endswith('.joblib'):
                file_path = os.path.join(SAVED_MODELS_DIR, file)
                # Get file creation time
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    model_path = file_path
        
        if not model_path:
            logger.error(f"[{request_id}] No model file found in {SAVED_MODELS_DIR}")
            return jsonify({"error": "No model file found"}), 404
            
        # Log model details
        model_size = os.path.getsize(model_path)
        logger.info(f"[{request_id}] Sending model file: {model_path}, Size: {model_size} bytes")
        
        # Verify file is readable
        try:
            with open(model_path, 'rb') as f:
                # Read first few bytes to verify file is accessible
                header = f.read(10)
                logger.debug(f"[{request_id}] File header (hex): {header.hex()}")
        except Exception as read_err:
            logger.error(f"[{request_id}] Error reading model file: {str(read_err)}")
            return jsonify({"error": f"Cannot read model file: {str(read_err)}"}), 500
        
        # Verify it's a valid joblib file
        try:
            import joblib
            # Just check if it can be loaded, we don't need the actual model
            joblib.load(model_path)
            logger.info(f"[{request_id}] Model file validated as a valid joblib file")
        except Exception as joblib_err:
            logger.warning(f"[{request_id}] Model file may not be valid joblib: {str(joblib_err)}")
            # Continue anyway, as it could be an issue with joblib version
        
        # Return the model file
        logger.info(f"[{request_id}] Sending model file as attachment")
        response = send_file(
            model_path, 
            as_attachment=True, 
            download_name='xgboost_regression_model.joblib',
            mimetype='application/octet-stream'
        )
        
        # Add additional headers to ensure proper download
        response.headers['Content-Length'] = str(model_size)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        logger.info(f"[{request_id}] Model file sent successfully with headers: {dict(response.headers)}")
        
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in download_model endpoint: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5046))
    logger.info(f"Starting XGBoost Regression Model Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 