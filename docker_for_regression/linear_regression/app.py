from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import time
import io
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
SAVED_MODELS_DIR = os.environ.get('SAVED_MODELS_DIR', '/app/saved_models/linear_regression')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

app = Flask(__name__)

class LinearRegressionModel:
    def __init__(self):
        """Initialize the linear regression model"""
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train a Linear Regression model"""
        logger.info(f"Training Linear Regression model with {X_train.shape[1]} features")
        
        # Start timer
        start_time = time.time()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        train_time = time.time() - start_time
        logger.info(f"Model trained in {train_time:.2f} seconds")
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        
        logger.info(f"Training R²: {train_r2:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        
        # Set trained flag
        self.is_trained = True
        
        # Get model parameters
        model_params = {
            'coefficients': self.model.coef_.tolist() if hasattr(self.model.coef_, 'tolist') else self.model.coef_.astype(float).tolist(),
            'intercept': float(self.model.intercept_)
        }
        
        # Generate timestamp and unique ID for filenames
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        model_basename = f"linear_regression_{timestamp}_{unique_id}"
        
        # Save best model to disk
        local_model_path = f"{SAVED_MODELS_DIR}/{model_basename}.joblib"
        joblib.dump(self.model, local_model_path)
        logger.info(f"Model saved to {local_model_path}")
        
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
            model_url = f"/saved_models/linear_regression/{model_basename}.joblib"
            logger.info(f"Using local file path as model URL: {model_url}")
        
        # Log with MLflow
        with mlflow.start_run(run_name="linear_regression") as run:
            # Log parameters
            mlflow.log_params({
                "model_type": "linear_regression",
                "num_features": X_train.shape[1]
            })
            
            # Log metrics
            mlflow.log_metrics({
                "train_r2": train_r2,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_mse": test_mse
            })
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
            logger.info("Model logged to MLflow successfully")
        
        # Return training results
        return {
            'name': 'linear_regression',
            'parameters': model_params,
            'metrics': {
                'train_r2': float(train_r2),
                'r2': float(test_r2),
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'mse': float(test_mse)
            },
            'model_url': model_url,
            'training_time': train_time
        }
    
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
regression_model = LinearRegressionModel()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "linear_regression",
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
        import traceback
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
        import traceback
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
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    """Download the trained model file"""
    try:
        if not regression_model.is_trained:
            return jsonify({"error": "Model is not trained"}), 400
            
        # Find the most recent model file
        model_path = None
        latest_time = 0
        
        for file in os.listdir(SAVED_MODELS_DIR):
            if file.startswith('linear_regression_') and file.endswith('.joblib'):
                file_path = os.path.join(SAVED_MODELS_DIR, file)
                # Get file creation time
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    model_path = file_path
        
        if not model_path:
            return jsonify({"error": "No model file found"}), 404
            
        # Return the model file
        return send_file(model_path, as_attachment=True, download_name='linear_regression_model.joblib')
        
    except Exception as e:
        logger.error(f"Error in download_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5041))
    logger.info(f"Starting Linear Regression Model Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 