from flask import Flask, request, jsonify
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
import xgboost as xgb
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
        self.objective = 'reg:squarederror'
        
        self.model = None
        self.is_trained = False
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train an XGBoost Regression model with hyperparameter tuning"""
        logger.info(f"Training XGBoost Regression model with {X_train.shape[1]} features and hyperparameter tuning")
        
        # Start timer
        start_time = time.time()
        
        # Define the hyperparameter search space
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # Create combinations of parameters to try (limit to prevent too many)
        param_combinations = [
            {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 7, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5, 'subsample': 1.0, 'colsample_bytree': 0.8},
            {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 1.0},
            {'learning_rate': 0.2, 'n_estimators': 100, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 7, 'subsample': 1.0, 'colsample_bytree': 1.0}
        ]
        
        # Initialize variables to track best model
        best_model = None
        best_params = None
        best_score = -float('inf')  # Initialize with worst possible score
        
        # Track all results for MLflow
        all_results = []
        
        # Get a unique experiment ID for all the runs
        experiment_id = int(time.time())
        
        # Start MLflow parent run
        with mlflow.start_run(run_name=f"xgboost_regression_tuning_{experiment_id}") as parent_run:
            # Log basic information
            mlflow.log_params({
                "model_type": "xgboost_regression",
                "tuning": True,
                "num_features": X_train.shape[1],
                "num_samples": X_train.shape[0],
                "num_combinations": len(param_combinations)
            })
            
            # Test each parameter combination
            for params in param_combinations:
                # Create a descriptive run name
                run_name = f"xgboost_lr_{params['learning_rate']}_est_{params['n_estimators']}_depth_{params['max_depth']}"
                
                # Create a nested run for this parameter set
                with mlflow.start_run(run_name=run_name, nested=True) as run:
                    logger.info(f"Trying parameters: {params}")
                    
                    # Create and train the model with these parameters
                    model = xgb.XGBRegressor(
                        learning_rate=params['learning_rate'],
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        objective=self.objective,
                        random_state=42
                    )
                    
                    # Train the model and measure time
                    fit_start_time = time.time()
                    model.fit(X_train, y_train)
                    fit_time = time.time() - fit_start_time
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    
                    # Get feature importances
                    feature_importances = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else []
                    
                    # Store results
                    result = {
                        **params,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'test_mse': test_mse,
                        'fit_time': fit_time
                    }
                    all_results.append(result)
                    
                    logger.info(f"Parameters: {params}")
                    logger.info(f"Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
                    
                    # Log parameters and metrics to MLflow
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        "train_r2": train_r2,
                        "test_r2": test_r2,
                        "test_rmse": test_rmse,
                        "test_mae": test_mae,
                        "test_mse": test_mse,
                        "fit_time": fit_time
                    })
                    
                    # Log feature importances if available
                    if feature_importances:
                        for i, importance in enumerate(feature_importances):
                            mlflow.log_metric(f"feature_importance_{i}", importance)
                    
                    # Track best model
                    if test_r2 > best_score:
                        best_score = test_r2
                        best_model = model
                        best_params = params.copy()
            
            # Log the best model parameters to the parent run
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_test_r2", best_score)
            
            # Set the best model as our model
            self.model = best_model
            self.learning_rate = best_params['learning_rate']
            self.n_estimators = best_params['n_estimators']
            self.max_depth = best_params['max_depth']
            self.subsample = best_params['subsample']
            self.colsample_bytree = best_params['colsample_bytree']
            
            # Get best feature importances
            feature_importances = self.model.feature_importances_.tolist() if hasattr(self.model, 'feature_importances_') else []
            
            # Calculate total training time
            train_time = time.time() - start_time
            logger.info(f"Total hyperparameter tuning time: {train_time:.2f} seconds")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best test R²: {best_score:.4f}")
            
            # Get model parameters
            model_params = {
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'objective': self.objective,
                'feature_importances': feature_importances
            }
            
            # Save best model to disk
            model_filename = f"{SAVED_MODELS_DIR}/xgboost_regression_{int(time.time())}.joblib"
            joblib.dump(self.model, model_filename)
            logger.info(f"Best model saved to {model_filename}")
            
            # Get the model URL (for retrieval)
            model_url = f"/saved_models/xgboost_regression/xgboost_regression_{int(time.time())}.joblib"
            
            # Log the best model in the parent run
            mlflow.sklearn.log_model(self.model, "best_model")
            logger.info("Best model logged to MLflow successfully")
        
        # Set trained flag
        self.is_trained = True
        
        # Find best metrics
        best_result = next(r for r in all_results if (
            r['learning_rate'] == best_params['learning_rate'] and
            r['n_estimators'] == best_params['n_estimators'] and
            r['max_depth'] == best_params['max_depth'] and
            r['subsample'] == best_params['subsample'] and
            r['colsample_bytree'] == best_params['colsample_bytree']
        ))
        
        # Return training results
        return {
            'name': 'xgboost_regression',
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
                    'learning_rate': float(r['learning_rate']),
                    'n_estimators': int(r['n_estimators']),
                    'max_depth': int(r['max_depth']),
                    'subsample': float(r['subsample']),
                    'colsample_bytree': float(r['colsample_bytree']),
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
regression_model = XGBoostRegressionModel()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "xgboost_regression",
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5046))
    logger.info(f"Starting XGBoost Regression Model Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 