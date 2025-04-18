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
from sklearn.model_selection import RandomizedSearchCV
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
                
                # Return training results
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
                            'subsample': float(r.get('subsample', 0)),
                            'colsample_bytree': float(r.get('colsample_bytree', 0)),
                            'min_child_weight': int(r.get('min_child_weight', 1)),
                            'gamma': float(r.get('gamma', 0)),
                            'reg_alpha': float(r.get('reg_alpha', 0)),
                            'reg_lambda': float(r.get('reg_lambda', 1)),
                            'test_r2': float(r.get('test_r2', 0))
                        } for r in all_results[:10]  # Only include top 10 models
                    ],
                    'model_url': model_url,
                    'training_time': train_time
                }
                
            except Exception as e:
                logger.error(f"Error during hyperparameter tuning: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Fallback to basic model in case tuning fails
                logger.info("Falling back to default model without tuning")
                
                self.model = xgb.XGBRegressor(
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    objective=self.objective,
                    random_state=42
                )
                
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = self.model.predict(X_train)
                y_pred_test = self.model.predict(X_test)
                
                # Calculate metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                test_mae = mean_absolute_error(y_test, y_pred_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                
                # Save model
                model_filename = f"{SAVED_MODELS_DIR}/xgboost_regression_{int(time.time())}.joblib"
                joblib.dump(self.model, model_filename)
                model_url = f"/saved_models/xgboost_regression/xgboost_regression_{int(time.time())}.joblib"
                
                # Set trained flag
                self.is_trained = True
                
                # Return results
                return {
                    'name': 'xgboost_regression',
                    'parameters': {
                        'learning_rate': self.learning_rate,
                        'n_estimators': self.n_estimators,
                        'max_depth': self.max_depth,
                        'subsample': self.subsample,
                        'colsample_bytree': self.colsample_bytree,
                        'objective': self.objective
                    },
                    'metrics': {
                        'train_r2': float(train_r2),
                        'r2': float(test_r2),
                        'rmse': float(test_rmse),
                        'mae': float(test_mae),
                        'mse': float(test_mse)
                    },
                    'model_url': model_url,
                    'training_time': time.time() - start_time
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