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

# Model services configuration
MODEL_SERVICES = {
    'logistic_regression': {'url': 'http://logistic_regression:5010', 'local_url': 'http://localhost:5010'},
    'decision_tree': {'url': 'http://decision_tree:5011', 'local_url': 'http://localhost:5011'},
    'random_forest': {'url': 'http://random_forest:5012', 'local_url': 'http://localhost:5012'},
    'svm': {'url': 'http://svm:5013', 'local_url': 'http://localhost:5013'},
    'knn': {'url': 'http://knn:5014', 'local_url': 'http://localhost:5014'},
    'naive_bayes': {'url': 'http://naive_bayes:5015', 'local_url': 'http://localhost:5015'}
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
    """Health check endpoint"""
    service_statuses = {}
    
    for service_name, service_info in MODEL_SERVICES.items():
        url = get_service_url(service_name)
        try:
            response = requests.get(f"{url}/health", timeout=2)
            service_statuses[service_name] = "healthy" if response.status_code == 200 else "unhealthy"
        except requests.RequestException:
            service_statuses[service_name] = "unreachable"
    
    all_healthy = all(status == "healthy" for status in service_statuses.values())
    
    return jsonify({
        "service": "model_coordinator_api",
        "status": "healthy" if all_healthy else "degraded",
        "model_services": service_statuses
    })

def train_model(service_name, train_data):
    """Train a specific model using its service"""
    url = get_service_url(service_name)
    logger.info(f"Training {service_name} model")
    
    try:
        response = requests.post(
            f"{url}/train",
            json=train_data,
            timeout=600  # 10 minutes timeout for training
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully trained {service_name} model")
            return service_name, response.json()
        else:
            logger.error(f"Failed to train {service_name} model: {response.text}")
            return service_name, {"error": response.text, "status_code": response.status_code}
    
    except requests.RequestException as e:
        logger.error(f"Error training {service_name} model: {str(e)}")
        return service_name, {"error": str(e)}

@app.route('/train', methods=['POST'])
def train_models():
    """Coordinate the training of multiple machine learning models"""
    try:
        # Get data from request
        data = request.json
        print(f"Received training request with data keys: {data.keys()}")
        
        if not data or 'data' not in data or 'target' not in data:
            return jsonify({'error': 'Missing required data fields'}), 400
        
        # Enhanced debug logging
        print(f"Training data shape: {len(data['data'].keys())} features, target shape: {len(data['target'])} samples")
        print(f"Available features: {list(data['data'].keys())}")
        print(f"First few target values: {data['target'][:5]}")
        
        # Get test_size parameter (default to 0.2)
        test_size = data.get('test_size', 0.2)
        print(f"Using test_size: {test_size}")
        
        # Validate data
        if not validate_data(data):
            print("ERROR: Data validation failed")
            return jsonify({'error': 'Invalid data format'}), 400
        
        # Transform data to format needed by model APIs
        X_data = data['data']
        y_data = data['target']
        
        # Get available model services
        model_services = get_model_services()
        print(f"Available model services: {list(model_services.keys())}")
        
        # Train models in parallel
        models_results = []
        errors = []
        
        # For each model service
        for model_name, model_url in model_services.items():
            try:
                print(f"Attempting to train {model_name} model at {model_url}")
                
                # Create payload for model training
                payload = {
                    'data': X_data,
                    'target': y_data,
                    'test_size': test_size
                }
                
                # Send training request to model service
                response = requests.post(f"{model_url}/train", json=payload, timeout=120)
                
                # Check response
                if response.status_code == 200:
                    model_result = response.json()
                    print(f"Successfully trained {model_name} model. Metrics: {model_result.get('metrics', {})}")
                    models_results.append({
                        'model': model_result
                    })
                else:
                    error_message = f"Error training {model_name} model: {response.text}"
                    print(error_message)
                    errors.append(error_message)
            except Exception as e:
                error_message = f"Exception training {model_name} model: {str(e)}"
                print(error_message)
                errors.append(error_message)
        
        # Check if any models were trained successfully
        if not models_results:
            all_errors = '; '.join(errors)
            print(f"ERROR: Failed to train any models. Errors: {all_errors}")
            return jsonify({'error': f'Failed to train any models: {all_errors}'}), 500
        
        # Return combined results
        return jsonify({
            'models': models_results,
            'errors': errors
        })
        
    except Exception as e:
        print(f"ERROR in train_models: {str(e)}")
        return jsonify({'error': str(e)}), 500

def predict_with_model(service_name, model_name, data):
    """Make predictions using a specific model service"""
    url = get_service_url(model_name)
    logger.info(f"Making predictions with {model_name} model")
    
    try:
        response = requests.post(
            f"{url}/predict",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            result['model_name'] = model_name
            return service_name, result
        else:
            logger.error(f"Failed to get predictions from {model_name}: {response.text}")
            return service_name, {"error": response.text, "status_code": response.status_code}
    
    except requests.RequestException as e:
        logger.error(f"Error getting predictions from {model_name}: {str(e)}")
        return service_name, {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Get predictions from the requested models
    
    Expected JSON input:
    {
        "data": {...},  # Features in JSON format
        "models": ["accuracy", "precision", "recall"]  # Optional, default is all three
    }
    
    Returns:
    {
        "predictions": {
            "accuracy": {...},
            "precision": {...},
            "recall": {...}
        },
        "message": "Predictions made successfully"
    }
    """
    try:
        request_data = request.json
        
        if not request_data or 'data' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data'."}), 400
        
        # Default to all three metrics if not specified
        requested_models = request_data.get('models', ['accuracy', 'precision', 'recall'])
        
        # Get model info for best models
        model_mapping = {
            'accuracy': None,
            'precision': None,
            'recall': None
        }
        
        # For each requested model type, find which model service has the best metrics
        for metric in requested_models:
            try:
                # Check the health status of all services
                health_response = health()
                health_data = json.loads(health_response.data)
                
                # Get the service with the best metric
                service_name = None
                best_metric_value = -1
                
                # Call model_info on each service to find the best for this metric
                for model_name, status in health_data['model_services'].items():
                    if status != "healthy":
                        continue
                    
                    url = get_service_url(model_name)
                    try:
                        info_response = requests.get(f"{url}/model_info", timeout=5)
                        if info_response.status_code == 200:
                            model_info = info_response.json()
                            current_metric_value = model_info.get('metrics', {}).get(metric, 0)
                            
                            if current_metric_value > best_metric_value:
                                best_metric_value = current_metric_value
                                service_name = model_name
                    except:
                        continue
                
                if service_name:
                    model_mapping[metric] = service_name
            
            except Exception as e:
                logger.error(f"Error determining best model for {metric}: {str(e)}")
                continue
        
        # Make predictions with the selected models
        predictions = {}
        futures = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for metric, model_name in model_mapping.items():
                if model_name and metric in requested_models:
                    futures.append(
                        executor.submit(predict_with_model, metric, model_name, request_data)
                    )
            
            for future in concurrent.futures.as_completed(futures):
                metric, result = future.result()
                if 'error' not in result:
                    predictions[metric] = {
                        'model_name': result['model_name'],
                        'predictions': result['predictions'],
                        'probabilities': result.get('probabilities')
                    }
        
        if not predictions:
            return jsonify({
                "error": "No models available for prediction. Train models first."
            }), 404
        
        return jsonify({
            "predictions": predictions,
            "message": "Predictions made successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about all available models"""
    try:
        model_info = {}
        
        # Query each service for model info
        for model_name, service_info in MODEL_SERVICES.items():
            url = get_service_url(model_name)
            try:
                response = requests.get(f"{url}/model_info", timeout=5)
                if response.status_code == 200:
                    model_info[model_name] = response.json()
                else:
                    model_info[model_name] = {"status": "unavailable", "error": response.text}
            except requests.RequestException as e:
                model_info[model_name] = {"status": "unreachable", "error": str(e)}
        
        # Determine best models based on metrics
        best_models = {}
        
        # Models with valid metrics
        valid_models = {
            name: info for name, info in model_info.items() 
            if 'metrics' in info and isinstance(info['metrics'], dict)
        }
        
        if valid_models:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            for metric in metrics:
                try:
                    best_model = max(
                        valid_models.items(),
                        key=lambda x: x[1]['metrics'].get(metric, 0)
                    )
                    best_models[metric] = {
                        'model_name': best_model[0],
                        'value': best_model[1]['metrics'].get(metric, 0)
                    }
                except:
                    continue
        
        return jsonify({
            "models": model_info,
            "best_models": best_models
        })
        
    except Exception as e:
        logger.error(f"Error in model_info endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Get available model services
def get_model_services():
    """Get available model services"""
    # In a local development environment with Docker Compose
    services = {
        "logistic_regression": os.getenv("LOGISTIC_REGRESSION_URL", "http://logistic_regression:5010"),
        "decision_tree": os.getenv("DECISION_TREE_URL", "http://decision_tree:5011"),
        "random_forest": os.getenv("RANDOM_FOREST_URL", "http://random_forest:5012"),
        "svm": os.getenv("SVM_URL", "http://svm:5013"),
        "knn": os.getenv("KNN_URL", "http://knn:5014"),
        "naive_bayes": os.getenv("NAIVE_BAYES_URL", "http://naive_bayes:5015"),
    }
    
    # Debug the model service URLs
    print("MODEL SERVICE URLS:")
    for name, url in services.items():
        print(f"  {name}: {url}")
    
    # Filter for available services
    available_services = {}
    for name, url in services.items():
        try:
            # Test connectivity with a short timeout
            print(f"Testing connectivity to {name} at {url}")
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"  - {name} is AVAILABLE")
                available_services[name] = url
            else:
                print(f"  - {name} returned status code {response.status_code}")
        except Exception as e:
            print(f"  - {name} is UNAVAILABLE: {str(e)}")
    
    print(f"Found {len(available_services)} available model services: {list(available_services.keys())}")
    return available_services

def validate_data(data):
    """Validate data format"""
    try:
        print("Validating data format...")
        
        # Check basic structure
        if 'data' not in data or 'target' not in data:
            print("ERROR: Missing 'data' or 'target' keys")
            return False
        
        # Check that data is a dictionary of features
        if not isinstance(data['data'], dict):
            print(f"ERROR: 'data' should be a dictionary, got {type(data['data'])}")
            return False
        
        # Check that target is a list
        if not isinstance(data['target'], list):
            print(f"ERROR: 'target' should be a list, got {type(data['target'])}")
            return False
        
        # Check that there is data
        if not data['data'] or not data['target']:
            print("ERROR: Empty data or target")
            return False
        
        # Check that all feature values are lists
        for feature, values in data['data'].items():
            if not isinstance(values, list):
                print(f"ERROR: Feature '{feature}' values should be a list, got {type(values)}")
                return False
            
            # Check that all feature lists have the same length
            if len(values) != len(data['target']):
                print(f"ERROR: Feature '{feature}' has {len(values)} values, but target has {len(data['target'])} values")
                return False
        
        # Check for non-numeric values
        for feature, values in data['data'].items():
            for i, value in enumerate(values):
                if value is not None and not isinstance(value, (int, float)):
                    print(f"ERROR: Non-numeric value in feature '{feature}' at index {i}: {value} (type: {type(value)})")
                    return False
        
        print("Data validation PASSED")
        return True
    
    except Exception as e:
        print(f"Exception during data validation: {str(e)}")
        return False

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5020))
    
    # Set Docker flag based on environment
    IS_DOCKER = os.environ.get('IS_DOCKER', 'false').lower() == 'true'
    logger.info(f"Running in {'Docker' if IS_DOCKER else 'local'} mode")
    
    serve(app, host='0.0.0.0', port=port) 