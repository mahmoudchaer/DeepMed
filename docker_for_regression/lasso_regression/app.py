from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from waitress import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/mlruns')
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'regression_experiment')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Define directories
SAVED_MODELS_DIR = os.environ.get('SAVED_MODELS_DIR', '/app/saved_models/lasso_regression')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

app = Flask(__name__)

class LassoRegressionModel:
    def __init__(self):
        """Initialize the Lasso regression model"""
        # Default alpha value - will be tuned
        self.alpha = 0.1
        self.model = None
        self.is_trained = False
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train a Lasso Regression model with hyperparameter tuning"""
        logger.info(f"Training Lasso Regression model with {X_train.shape[1]} features and hyperparameter tuning")
        
        # Start timer
        start_time = time.time()
        
        # Define the hyperparameter search space
        alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
        
        # Initialize variables to track best model
        best_model = None
        best_alpha = None
        best_score = -float('inf')  # Initialize with worst possible score
        
        # Track all results for MLflow
        all_results = []
        
        # Get a unique experiment ID for all the runs
        experiment_id = int(time.time())
        
        # Start MLflow parent run
        with mlflow.start_run(run_name=f"lasso_regression_tuning_{experiment_id}") as parent_run:
            mlflow.log_param("experiment_type", "lasso_regression_alpha_search")
            mlflow.log_param("num_features", X_train.shape[1])
            
            # Try each alpha value
            for alpha in alpha_values:
                # Log each attempt as a child run
                with mlflow.start_run(run_name=f"lasso_alpha_{alpha}", nested=True) as child_run:
                    logger.info(f"Trying alpha={alpha}")
                    
                    # Create and train the model
                    model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    
                    logger.info(f"Alpha={alpha}, Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}")
                    
                    # Store results
                    result = {
                        'alpha': alpha,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'test_mse': test_mse,
                        'model': model
                    }
                    all_results.append(result)
                    
                    # Log parameters and metrics to MLflow
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_metrics({
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "test_rmse": test_rmse,
                        "test_mae": test_mae,
                        "test_mse": test_mse
                    })
                    
                    # Track best model
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = model
                        best_alpha = alpha
            
            # Log the best model parameters to the parent run
            mlflow.log_param("best_alpha", best_alpha)
            mlflow.log_metric("best_test_r2", best_score)
            
            # Set the best model as our model
            self.model = best_model
            self.alpha = best_alpha
            
            # Get model parameters
            model_params = {
                'coefficients': self.model.coef_.tolist() if hasattr(self.model.coef_, 'tolist') else self.model.coef_.astype(float).tolist(),
                'intercept': float(self.model.intercept_),
                'alpha': float(self.alpha),
                'n_nonzero_coefs': int(np.sum(self.model.coef_ != 0))
            }
            
            # Calculate training time
            train_time = time.time() - start_time
            logger.info(f"Hyperparameter tuning completed in {train_time:.2f} seconds")
            logger.info(f"Best alpha={best_alpha} with test R²={best_score:.4f}")
            
            # Save best model to disk
            model_filename = f"{SAVED_MODELS_DIR}/lasso_regression_{int(time.time())}.joblib"
            joblib.dump(self.model, model_filename)
            logger.info(f"Best model saved to {model_filename}")
            
            # Get the model URL (for retrieval)
            model_url = f"/saved_models/lasso_regression/lasso_regression_{int(time.time())}.joblib"
            
            # Log the best model in the parent run
            mlflow.sklearn.log_model(self.model, "best_model")
            logger.info("Best model logged to MLflow successfully")
        
        # Set trained flag
        self.is_trained = True
        
        # Find metrics for best model
        best_result = next(r for r in all_results if r['alpha'] == best_alpha)
        
        # Return training results
        return {
            'name': 'lasso_regression',
            'parameters': model_params,
            'metrics': {
                'train_r2': float(best_result['train_r2']),
                'r2': float(best_result['test_r2']),
                'rmse': float(best_result['test_rmse']),
                'mae': float(best_result['test_mae']),
                'mse': float(best_result['test_mse'])
            },
            'tuning_results': [
                {
                    'alpha': float(r['alpha']),
                    'test_r2': float(r['test_r2']),
                    'test_rmse': float(r['test_rmse'])
                } for r in all_results
            ],
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
            full_path = model_path
            if not model_path.startswith('/'):
                full_path = os.path.join(SAVED_MODELS_DIR, os.path.basename(model_path))
            
            self.model = joblib.load(full_path)
            self.is_trained = True
            logger.info(f"Model loaded from {full_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

# Create model instance
regression_model = LassoRegressionModel()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "lasso_regression",
        "is_trained": regression_model.is_trained
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
            if file.startswith('lasso_regression_') and file.endswith('.joblib'):
                file_path = os.path.join(SAVED_MODELS_DIR, file)
                # Get file creation time
                file_time = os.path.getmtime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    model_path = file_path
        
        if not model_path:
            return jsonify({"error": "No model file found"}), 404
            
        # Return the model file
        return send_file(model_path, as_attachment=True, download_name='lasso_regression_model.joblib')
        
    except Exception as e:
        logger.error(f"Error in download_model endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5042))
    logger.info(f"Starting Lasso Regression Model Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 