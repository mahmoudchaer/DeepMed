from flask import Flask, request, jsonify
import requests
import json
import concurrent.futures
import logging
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
import shutil
from pathlib import Path
from waitress import serve
import joblib  # Add this import for saving models
import datetime
import pandas as pd

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
        
        # Create default experiment if none exists
        if not experiment_names:
            mlflow.create_experiment("default")
            logger.info("Created default MLflow experiment")
    except Exception as e:
        logger.error(f"Error initializing MLflow: {str(e)}")

# Initialize MLflow on startup
init_mlflow()

# Directory for saving the best models
BEST_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'best_models')
os.makedirs(BEST_MODELS_DIR, exist_ok=True)

# Model services configuration - REGRESSION MODELS
MODEL_SERVICES = {
    'linear_regression': {'url': 'http://linear_regression:6010', 'local_url': 'http://localhost:6010'},
    'ridge_regression': {'url': 'http://ridge_regression:6011', 'local_url': 'http://localhost:6011'},
    'lasso_regression': {'url': 'http://lasso_regression:6012', 'local_url': 'http://localhost:6012'},
    'random_forest_regressor': {'url': 'http://random_forest_regressor:6013', 'local_url': 'http://localhost:6013'},
    'svr': {'url': 'http://svr:6014', 'local_url': 'http://localhost:6014'}
}

# Flag to determine if running in Docker or locally
IS_DOCKER = os.getenv('IS_DOCKER', 'false').lower() == 'true'

def get_service_url(service_name):
    """Get the appropriate URL based on whether running in Docker or locally"""
    if IS_DOCKER:
        return MODEL_SERVICES[service_name]['url']
    else:
        return MODEL_SERVICES[service_name]['local_url']

@app.route('/health')
def health():
    """Check health of all model services"""
    results = {}
    
    for service_name in MODEL_SERVICES:
        service_url = get_service_url(service_name)
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            results[service_name] = {
                'status': 'up' if response.status_code == 200 else 'error',
                'details': response.json()
            }
        except Exception as e:
            results[service_name] = {
                'status': 'down',
                'error': str(e)
            }
    
    # Overall status is determined by the status of all services
    all_up = all(service['status'] == 'up' for service in results.values())
    
    return jsonify({
        'status': 'healthy' if all_up else 'degraded',
        'services': results
    })

def train_model(service_name, train_data):
    """Train a specific model service with the provided data"""
    service_url = get_service_url(service_name)
    
    try:
        response = requests.post(
            f"{service_url}/train",
            json=train_data,
            timeout=600  # 10 minutes timeout for training
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error training {service_name}: {response.text}")
            return {
                'status': 'error',
                'error': f"Training failed with status code {response.status_code}",
                'details': response.text if response.text else 'No error details provided'
            }
    except Exception as e:
        logger.error(f"Exception when training {service_name}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

@app.route('/train', methods=['POST'])
def train_models():
    """Train all regression models in parallel"""
    try:
        # Get training data from request
        train_data = request.json
        
        # Validate input data
        if not validate_data(train_data):
            return jsonify({
                'status': 'error',
                'error': 'Invalid data format. Please provide "data" and "target" fields.'
            }), 400
        
        # Log input data summary
        features = train_data.get('data', {})
        target = train_data.get('target', [])
        feature_count = len(features.keys()) if isinstance(features, dict) else len(features[0]) if features else 0
        sample_count = len(list(features.values())[0]) if isinstance(features, dict) else len(features) if features else 0
        
        logger.info(f"Training all regression models with {sample_count} samples and {feature_count} features")
        
        # Train models in parallel
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_SERVICES)) as executor:
            # Submit training tasks
            future_to_model = {
                executor.submit(train_model, model_name, train_data): model_name
                for model_name in MODEL_SERVICES
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    results[model_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        # Determine best model based on different metrics
        best_models = {}
        
        # Get all models that trained successfully
        successful_models = {
            model_name: result for model_name, result in results.items()
            if result.get('status') == 'success' and 'metrics' in result
        }
        
        # Find best model for MSE (lower is better)
        if successful_models:
            mse_scores = {
                model_name: result['metrics'].get('mse', float('inf'))
                for model_name, result in successful_models.items()
            }
            best_mse_model = min(mse_scores.items(), key=lambda x: x[1])[0]
            best_models['mse'] = best_mse_model
        
            # Find best model for R^2 (higher is better)
            r2_scores = {
                model_name: result['metrics'].get('r2', -float('inf'))
                for model_name, result in successful_models.items()
            }
            best_r2_model = max(r2_scores.items(), key=lambda x: x[1])[0]
            best_models['r2'] = best_r2_model
            
            # Find best model for RMSE (lower is better)
            rmse_scores = {
                model_name: result['metrics'].get('rmse', float('inf'))
                for model_name, result in successful_models.items()
            }
            best_rmse_model = min(rmse_scores.items(), key=lambda x: x[1])[0]
            best_models['rmse'] = best_rmse_model
            
            # Save the best models
            save_best_models(best_models, successful_models)
        
        return jsonify({
            'status': 'success',
            'results': results,
            'best_models': best_models
        })
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def save_best_models(best_models, model_results):
    """Save only the best models based on different metrics"""
    try:
        # Clear previous best models directory
        if os.path.exists(BEST_MODELS_DIR):
            for file in os.listdir(BEST_MODELS_DIR):
                file_path = os.path.join(BEST_MODELS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        # Download and save best models for each metric
        for metric, model_name in best_models.items():
            # Get the model's run ID from results
            run_id = model_results[model_name].get('run_id')
            if not run_id:
                logger.warning(f"No run ID found for {model_name}, cannot save model for {metric}")
                continue
            
            try:
                # Get the model service URL
                service_url = get_service_url(model_name)
                
                # Download the model file from the model service
                response = requests.get(
                    f"{service_url}/download_model",
                    timeout=60
                )
                
                if response.status_code == 200:
                    # Save the model with a metric-specific name
                    model_filename = f"best_model_for_{metric}.joblib"
                    model_path = os.path.join(BEST_MODELS_DIR, model_filename)
                    
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Saved best model for {metric} from {model_name} to {model_path}")
                    
                    # Also create a metadata file
                    metadata = {
                        'metric': metric,
                        'model_type': model_name,
                        'value': model_results[model_name]['metrics'].get(metric),
                        'run_id': run_id,
                        'saved_at': str(datetime.datetime.now())
                    }
                    
                    metadata_path = os.path.join(BEST_MODELS_DIR, f"best_model_for_{metric}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                else:
                    logger.error(f"Failed to download model for {metric} from {model_name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error saving best model for {metric} from {model_name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in save_best_models: {str(e)}")

def predict_with_model(service_name, model_name, data):
    """Make predictions with a specific model service"""
    service_url = get_service_url(service_name)
    
    try:
        response = requests.post(
            f"{service_url}/predict",
            json={
                'data': data,
                'model': model_name
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error predicting with {service_name}: {response.text}")
            return {
                'status': 'error',
                'error': f"Prediction failed with status code {response.status_code}",
                'details': response.text
            }
    except Exception as e:
        logger.error(f"Exception when predicting with {service_name}: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using trained regression models"""
    try:
        # Get prediction request data
        request_data = request.json
        
        # Validate request data
        if 'data' not in request_data:
            return jsonify({
                'status': 'error',
                'error': 'Missing "data" field in request'
            }), 400
        
        # Get models to use for prediction (r2, mse, rmse, or all)
        models_to_use = request_data.get('models', ['r2'])
        if not isinstance(models_to_use, list):
            models_to_use = [models_to_use]
        
        # Get model info to identify best models
        model_info_response = get_model_services()
        model_info = json.loads(model_info_response.get_data(as_text=True))
        
        # Filter out services without trained models
        available_services = {}
        for service_name, service_info in model_info['services'].items():
            if service_info['models']:
                available_services[service_name] = service_info
        
        if not available_services:
            return jsonify({
                'status': 'error',
                'error': 'No trained models available for prediction'
            }), 400
        
        # Check if we can load from best models directory first
        best_model_available = False
        prediction_tasks = []
        
        # Try to load from best models directory if requested
        for metric in models_to_use:
            if metric in ['r2', 'mse', 'rmse']:
                metadata_path = os.path.join(BEST_MODELS_DIR, f"best_model_for_{metric}_metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        service_name = metadata['model_type']
                        model_name = f"best_model_for_{metric}"
                        prediction_tasks.append((service_name, model_name, metric))
                        best_model_available = True
                    except Exception as e:
                        logger.error(f"Error loading metadata for best {metric} model: {str(e)}")
        
        # If no best models available, fall back to the previous approach
        if not best_model_available:
            # Determine which models to use
            # For each metric (r2, mse, rmse), find the best model
            best_models = {}
            for metric in ['r2', 'mse', 'rmse']:
                metric_values = {}
                
                for service_name, service_info in available_services.items():
                    for model in service_info['models']:
                        model_metrics = model.get('metrics', {})
                        
                        # For r2, higher is better; for mse and rmse, lower is better
                        if metric == 'r2':
                            metric_val = model_metrics.get(metric, -float('inf'))
                            if metric_val == float('inf') or metric_val == -float('inf'):
                                continue
                            # Find maximum r2
                            if metric_val > metric_values.get(service_name, -float('inf')):
                                metric_values[service_name] = metric_val
                                best_models[metric] = (service_name, model['name'])
                        else:
                            # For mse and rmse, lower is better
                            metric_val = model_metrics.get(metric, float('inf'))
                            if metric_val == float('inf') or metric_val == -float('inf'):
                                continue
                            # Find minimum mse/rmse
                            if metric_val < metric_values.get(service_name, float('inf')):
                                metric_values[service_name] = metric_val
                                best_models[metric] = (service_name, model['name'])
            
            # Add prediction tasks for requested metrics
            for metric in models_to_use:
                if metric in best_models:
                    service_name, model_name = best_models[metric]
                    prediction_tasks.append((service_name, model_name, metric))
                elif metric == 'all':
                    # Use all available models
                    for service_name, service_info in available_services.items():
                        if service_info['models']:
                            model_name = service_info['models'][0]['name']  # Use the first model from each service
                            prediction_tasks.append((service_name, model_name, service_name))
        
        if not prediction_tasks:
            return jsonify({
                'status': 'error',
                'error': f'No suitable models found for metrics: {models_to_use}'
            }), 400
        
        # Make predictions with selected models
        prediction_results = {}
        for service_name, model_name, result_key in prediction_tasks:
            # Check if this is a best model from our directory
            if model_name.startswith("best_model_for_"):
                # Load from our best models directory
                model_path = os.path.join(BEST_MODELS_DIR, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    try:
                        # Load the model
                        model = joblib.load(model_path)
                        
                        # Convert input data to DataFrame
                        input_data = request_data['data']
                        if isinstance(input_data, dict):
                            X = pd.DataFrame(input_data)
                        elif isinstance(input_data, list) and isinstance(input_data[0], dict):
                            X = pd.DataFrame(input_data)
                        
                        # Make prediction
                        predictions = model.predict(X)
                        
                        prediction_results[result_key] = {
                            'status': 'success',
                            'model': model_name,
                            'predictions': predictions.tolist()
                        }
                        continue
                    except Exception as e:
                        logger.error(f"Error predicting with local best model {model_name}: {str(e)}")
                        # Fall back to using the service
            
            # Use the service for prediction
            result = predict_with_model(service_name, model_name, request_data['data'])
            prediction_results[result_key] = result
        
        return jsonify({
            'status': 'success',
            'predictions': prediction_results
        })
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about all available regression models"""
    results = {}
    
    for service_name in MODEL_SERVICES:
        service_url = get_service_url(service_name)
        try:
            response = requests.get(f"{service_url}/model_info", timeout=10)
            if response.status_code == 200:
                results[service_name] = response.json()
            else:
                results[service_name] = {
                    'status': 'error',
                    'error': f"Failed to get model info with status code {response.status_code}"
                }
        except Exception as e:
            results[service_name] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Process results to identify the best models for each metric
    best_models = {}
    
    # Get services with trained models
    services_with_models = {
        service_name: service_info
        for service_name, service_info in results.items()
        if service_info.get('status') == 'success' and service_info.get('models')
    }
    
    # Find best models for each metric
    for metric in ['r2', 'mse', 'rmse']:
        metric_values = {}
        
        for service_name, service_info in services_with_models.items():
            for model in service_info['models']:
                model_metrics = model.get('metrics', {})
                
                # For r2, higher is better; for mse and rmse, lower is better
                if metric == 'r2':
                    metric_val = model_metrics.get(metric, -float('inf'))
                    if metric_val == float('inf') or metric_val == -float('inf'):
                        continue
                    # Find maximum r2
                    if metric_val > metric_values.get((service_name, model['name']), -float('inf')):
                        metric_values[(service_name, model['name'])] = metric_val
                else:
                    # For mse and rmse, lower is better
                    metric_val = model_metrics.get(metric, float('inf'))
                    if metric_val == float('inf') or metric_val == -float('inf'):
                        continue
                    # Find minimum mse/rmse
                    if metric_val < metric_values.get((service_name, model['name']), float('inf')):
                        metric_values[(service_name, model['name'])] = metric_val
        
        # Determine best model for this metric
        if metric_values:
            if metric == 'r2':
                best_key = max(metric_values.items(), key=lambda x: x[1])[0]
            else:
                best_key = min(metric_values.items(), key=lambda x: x[1])[0]
            
            service_name, model_name = best_key
            metric_value = metric_values[best_key]
            best_models[metric] = {
                'service': service_name,
                'model': model_name,
                'value': metric_value
            }
    
    # Add information about the saved best models
    best_models_info = {}
    if os.path.exists(BEST_MODELS_DIR):
        for metric in ['r2', 'mse', 'rmse']:
            metadata_path = os.path.join(BEST_MODELS_DIR, f"best_model_for_{metric}_metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    best_models_info[metric] = metadata
                except Exception as e:
                    logger.error(f"Error reading metadata for {metric} best model: {str(e)}")
    
    return jsonify({
        'status': 'success',
        'services': results,
        'best_models': best_models,
        'saved_best_models': best_models_info
    })

def get_model_services():
    """Helper function to get model information (used by other endpoints)"""
    return model_info()

def validate_data(data):
    """Validate the input data format"""
    if not isinstance(data, dict):
        return False
    
    if 'data' not in data or 'target' not in data:
        return False
    
    # Check if data is in the right format (dict of lists or list of dicts)
    features = data['data']
    target = data['target']
    
    if isinstance(features, dict):
        # Format 1: {feature1: [1,2,3], feature2: [4,5,6]}
        feature_lengths = [len(values) for values in features.values()]
        if not feature_lengths or not all(length == feature_lengths[0] for length in feature_lengths):
            return False
        if len(target) != feature_lengths[0]:
            return False
    elif isinstance(features, list):
        # Format 2: [{feature1: 1, feature2: 4}, {feature1: 2, feature2: 5}]
        if not features or not isinstance(features[0], dict):
            return False
        if len(target) != len(features):
            return False
    else:
        return False
    
    return True

if __name__ == "__main__":
    PORT = int(os.environ.get('PORT', 6020))
    logger.info(f"Starting Model Coordinator API on port {PORT}")
    serve(app, host="0.0.0.0", port=PORT) 