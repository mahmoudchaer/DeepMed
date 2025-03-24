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
import time
import io
import uuid
import pandas as pd
import numpy as np
import sys

# Add parent directory to path for imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

# Import for Azure Blob Storage
from storage import upload_to_blob, get_blob_url, delete_blob

# Import database models
from db.users import db, User, TrainingRun, TrainingModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure the database
def configure_db():
    """Configure the database connection"""
    try:
        # Get database credentials from environment variables
        MYSQL_USER = os.getenv("MYSQL_USER")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
        MYSQL_HOST = os.getenv("MYSQL_HOST")
        MYSQL_PORT = os.getenv("MYSQL_PORT")
        MYSQL_DB = os.getenv("MYSQL_DB")
        
        # Check if all required environment variables are set
        if not all([MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB]):
            logger.warning("Missing database environment variables. Database operations will not work.")
            return
        
        # URL encode the password to handle special characters - fix the encoding issue
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(str(MYSQL_PASSWORD))
        
        # Configure SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        # Initialize the database
        db.init_app(app)
        logger.info("Database configured successfully")
    except Exception as e:
        logger.error(f"Error configuring database: {str(e)}")

# Configure the database on startup
configure_db()

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
        
        # Test MLflow connection - using search_experiments instead of list_experiments
        client = MlflowClient()
        try:
            # For newer MLflow versions
            experiment_names = [exp.name for exp in client.search_experiments()]
            logger.info(f"Connected to MLflow. Available experiments: {experiment_names}")
        except AttributeError:
            # Fallback for older MLflow versions
            try:
                experiment_names = [exp.name for exp in client.list_experiments()]
                logger.info(f"Connected to MLflow. Available experiments: {experiment_names}")
            except AttributeError:
                logger.warning("Could not list MLflow experiments. Proceeding anyway.")
        
        # Create default experiment if not found
        default_experiment = mlflow.get_experiment_by_name("default")
        if default_experiment is None:
            mlflow.create_experiment("default")
            logger.info("Created default MLflow experiment")
        else:
            logger.info("Default MLflow experiment already exists")
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

# Database functions for model storage
def create_training_run(user_id, run_name):
    """Create a new training run entry in the database"""
    try:
        with app.app_context():
            # Explicitly set SQLALCHEMY_TRACK_MODIFICATIONS to avoid errors
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            
            # Check if user exists
            user = None
            try:
                user = User.query.get(user_id)
            except Exception as e:
                logger.warning(f"Could not query user: {str(e)}")
            
            # Create training run
            training_run = TrainingRun(
                user_id=user_id,
                run_name=run_name
            )
            
            try:
                db.session.add(training_run)
                db.session.commit()
                logger.info(f"Created training run {training_run.id} for user {user_id}")
                return training_run.id
            except Exception as db_error:
                db.session.rollback()
                logger.error(f"Database error creating training run: {str(db_error)}")
                # Create a placeholder ID for non-DB environments
                return int(time.time())
    except Exception as e:
        logger.error(f"Error creating training run: {str(e)}")
        # Return a timestamp as a fallback ID
        return int(time.time())

def save_model_to_blob(model_data, model_name, metric_name=None):
    """Save model to Azure Blob Storage and return the URL"""
    try:
        # Generate a unique filename
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        if metric_name:
            filename = f"{model_name}_{metric_name}_{timestamp}_{unique_id}.joblib"
        else:
            filename = f"{model_name}_{timestamp}_{unique_id}.joblib"
            
        # Convert model to bytes
        model_bytes = io.BytesIO()
        import joblib
        joblib.dump(model_data, model_bytes)
        model_bytes.seek(0)
        
        # Upload to blob storage
        blob_url = upload_to_blob(model_bytes, filename)
        if blob_url:
            logger.info(f"Saved model {model_name} to blob storage: {blob_url}")
            return blob_url, filename
        else:
            logger.error(f"Failed to save model {model_name} to blob storage")
            return None, None
    except Exception as e:
        logger.error(f"Error saving model to blob: {str(e)}")
        return None, None

def save_model_to_db(user_id, run_id, model_name, model_url):
    """Save model reference to database"""
    try:
        with app.app_context():
            # Explicitly set SQLALCHEMY_TRACK_MODIFICATIONS to avoid errors
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            
            model_record = TrainingModel(
                user_id=user_id,
                run_id=run_id,
                model_name=model_name,
                model_url=model_url
            )
            
            try:
                db.session.add(model_record)
                db.session.commit()
                logger.info(f"Saved model {model_name} to database with ID {model_record.id}")
                return True
            except Exception as db_error:
                db.session.rollback()
                logger.error(f"Database error saving model: {str(db_error)}")
                # Still return True since the model is in blob storage even if DB failed
                return True
    except Exception as e:
        logger.error(f"Error saving model to database: {str(e)}")
        # We still return True because the blob storage worked even if DB failed
        return True

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
        
        # Get user_id and run_name (required for database tracking)
        user_id = data.get('user_id')
        run_name = data.get('run_name', f"training_run_{time.strftime('%Y%m%d_%H%M%S')}")
        
        if not user_id:
            return jsonify({'error': 'Missing user_id in request'}), 400
            
        # Create training run in database
        run_id = create_training_run(user_id, run_name)
        if not run_id:
            return jsonify({'error': 'Failed to create training run in database'}), 500
        
        # Enhanced debug logging
        print(f"Training data shape: {len(data['data'].keys())} features, target shape: {len(data['target'])} samples")
        print(f"Available features: {list(data['data'].keys())}")
        print(f"First few target values: {data['target'][:5]}")
        print(f"Created training run with ID {run_id} for user {user_id}")
        
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
                    
                    # Ensure metrics are properly processed
                    if 'model' in model_result and 'metrics' in model_result['model']:
                        metrics = model_result['model']['metrics']
                        print(f"Successfully trained {model_name} model. Metrics: {metrics}")
                        
                        # Debug - print types of metrics values
                        for metric_name, metric_value in metrics.items():
                            print(f"Metric {metric_name} is of type {type(metric_value)}")
                            
                        # Ensure metrics are numeric
                        clean_metrics = {}
                        for metric_name, metric_value in metrics.items():
                            try:
                                if isinstance(metric_value, str):
                                    clean_metrics[metric_name] = float(metric_value)
                                else:
                                    clean_metrics[metric_name] = metric_value
                            except (ValueError, TypeError):
                                clean_metrics[metric_name] = 0
                                print(f"WARNING: Could not convert metric {metric_name}={metric_value} to float")
                        
                        # Replace original metrics with clean metrics
                        model_result['model']['metrics'] = clean_metrics
                        
                        # Save model to blob storage
                        model_data = model_result['model']
                        blob_url, blob_filename = save_model_to_blob(model_data, model_name)
                        
                        if blob_url:
                            # Save model reference to database
                            save_model_to_db(user_id, run_id, model_name, blob_url)
                    
                    models_results.append({
                        'model': model_result['model']
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
        
        # Find best models for each metric
        best_models = find_best_models(models_results)
        saved_best_models = save_best_models(best_models, user_id, run_id)
        
        # Return combined results
        return jsonify({
            'models': models_results,
            'best_models': best_models,
            'saved_best_models': saved_best_models,
            'run_id': run_id,
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
                    model_data = response.json()
                    
                    # Ensure metrics are properly processed
                    if 'metrics' in model_data and isinstance(model_data['metrics'], dict):
                        metrics = model_data['metrics']
                        
                        # Debug - print metrics
                        print(f"Model {model_name} metrics: {metrics}")
                        
                        # Ensure metrics are numeric
                        clean_metrics = {}
                        for metric_name, metric_value in metrics.items():
                            try:
                                if isinstance(metric_value, str):
                                    clean_metrics[metric_name] = float(metric_value)
                                else:
                                    clean_metrics[metric_name] = metric_value
                            except (ValueError, TypeError):
                                clean_metrics[metric_name] = 0
                                print(f"WARNING: Could not convert metric {metric_name}={metric_value} to float")
                        
                        # Replace original metrics with clean metrics
                        model_data['metrics'] = clean_metrics
                    
                    model_info[model_name] = model_data
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
                    # Find the best model for this metric
                    # Convert all metric values to float for proper comparison
                    best_model = max(
                        valid_models.items(),
                        key=lambda x: float(x[1]['metrics'].get(metric, 0))
                    )
                    best_models[metric] = {
                        'model_name': best_model[0],
                        'value': float(best_model[1]['metrics'].get(metric, 0))
                    }
                except:
                    continue
                    
            # Print best models for debugging
            print(f"Best models by metric: {best_models}")
        
        return jsonify({
            'models': model_info,
            'best_models': best_models
        })
    except Exception as e:
        print(f"ERROR in model_info: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
    """Validate the data structure for training"""
    try:
        if not isinstance(data, dict):
            return False
        
        if 'data' not in data or 'target' not in data:
            return False
            
        features = data['data']
        target = data['target']
        
        # Validate features is a dict with keys as feature names
        if not isinstance(features, dict):
            return False
        
        # Check if each feature has the same length
        feature_lengths = []
        for feature_name, feature_values in features.items():
            if not isinstance(feature_values, list):
                return False
            feature_lengths.append(len(feature_values))
            
        # All features should have the same length
        if len(set(feature_lengths)) != 1:
            return False
        
        # Target should be a list with same length as features
        if not isinstance(target, list) or len(target) != feature_lengths[0]:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return False

def find_best_models(models_results):
    """Find the best models for each metric"""
    try:
        # Metrics we're interested in
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        best_models = {}
        
        # Extract models with their metrics
        all_models = {}
        for model_result in models_results:
            if 'model' in model_result:
                model_info = model_result['model']
                model_name = model_info.get('model_name', 'unknown')
                if 'metrics' in model_info:
                    all_models[model_name] = model_info['metrics']
        
        # Find best model for each metric
        for metric in metrics:
            best_metric_value = -1
            best_model_name = None
            
            for model_name, model_metrics in all_models.items():
                if metric in model_metrics:
                    metric_value = float(model_metrics[metric])
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_model_name = model_name
            
            if best_model_name:
                best_models[metric] = {
                    'model_name': best_model_name,
                    'value': best_metric_value
                }
        
        logger.info(f"Best models by metric: {best_models}")
        return best_models
    except Exception as e:
        logger.error(f"Error finding best models: {str(e)}")
        return {}

def save_best_models(best_models, user_id, run_id):
    """Save the best models to blob storage and database"""
    try:
        saved_models = {}
        
        # For each best model by metric
        for metric, model_info in best_models.items():
            model_name = model_info['model_name']
            
            # Find the model data in the trained models
            for service_name, service_url in get_model_services().items():
                if service_name == model_name:
                    try:
                        # Get model data from the model service
                        info_response = requests.get(f"{service_url}/model_info", timeout=5)
                        if info_response.status_code == 200:
                            model_data = info_response.json()
                            
                            # Save as best model for this metric
                            display_name = f"best_model_for_{metric}"
                            blob_url, blob_filename = save_model_to_blob(model_data, model_name, metric)
                            
                            if blob_url:
                                # Save to database
                                saved = save_model_to_db(user_id, run_id, display_name, blob_url)
                                if saved:
                                    saved_models[metric] = {
                                        'model_name': model_name,
                                        'display_name': display_name,
                                        'url': blob_url,
                                        'filename': blob_filename
                                    }
                    except Exception as e:
                        logger.error(f"Error saving best model for {metric}: {str(e)}")
        
        return saved_models
    except Exception as e:
        logger.error(f"Error in save_best_models: {str(e)}")
        return {}

@app.route('/predict_with_blob', methods=['POST'])
def predict_with_blob():
    """
    Get predictions using models stored in blob storage
    
    Expected JSON input:
    {
        "data": {...},  # Features in JSON format
        "model_url": "...",  # URL to the model in blob storage
        "user_id": 123  # Optional - for access control
    }
    
    Returns:
    {
        "predictions": [...],
        "probabilities": [...],
        "message": "Predictions made successfully"
    }
    """
    try:
        request_data = request.json
        
        if not request_data or 'data' not in request_data or 'model_url' not in request_data:
            return jsonify({"error": "Invalid request. Missing 'data' or 'model_url'."}), 400
        
        model_url = request_data['model_url']
        user_id = request_data.get('user_id')
        
        # Optional access control if user_id is provided
        if user_id:
            # Check if user has access to this model
            with app.app_context():
                model_record = TrainingModel.query.filter_by(model_url=model_url, user_id=user_id).first()
                if not model_record:
                    return jsonify({"error": "Access denied or model not found"}), 403
        
        # Download the model from blob storage
        try:
            # Convert URL to filename
            # Example URL: https://accountname.blob.core.windows.net/container/filename
            model_filename = model_url.split('/')[-1]
            
            # Create a temporary file to store the model
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_model_path = os.path.join(temp_dir, model_filename)
            
            try:
                # Download the model
                import requests
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                with open(temp_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Load the model
                import joblib
                model_data = joblib.load(temp_model_path)
                
                # Make predictions
                X = pd.DataFrame.from_dict(request_data['data'])
                
                # Get the actual model from the model data
                if 'model' in model_data:
                    model = model_data['model']
                else:
                    model = model_data
                    
                # Make predictions
                predictions = model.predict(X)
                
                # Get probabilities if available
                probabilities = None
                try:
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X).tolist()
                except Exception as prob_error:
                    logger.warning(f"Could not get probabilities: {str(prob_error)}")
                
                return jsonify({
                    "predictions": predictions.tolist() if isinstance(predictions, (np.ndarray, pd.Series)) else predictions,
                    "probabilities": probabilities,
                    "message": "Predictions made successfully"
                })
                
            except Exception as e:
                logger.error(f"Error making predictions with blob model: {str(e)}")
                return jsonify({"error": f"Error making predictions: {str(e)}"}), 500
            finally:
                # Always clean up the temporary directory
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temporary directory: {str(cleanup_error)}")
                
        except Exception as e:
            logger.error(f"Error setting up for prediction: {str(e)}")
            return jsonify({"error": f"Error setting up for prediction: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Error in predict_with_blob endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5020))
    
    # Set Docker flag based on environment
    IS_DOCKER = os.environ.get('IS_DOCKER', 'false').lower() == 'true'
    logger.info(f"Running in {'Docker' if IS_DOCKER else 'local'} mode")
    
    serve(app, host='0.0.0.0', port=port) 