# Import docker_secrets to load environment variables
try:
    import docker_secrets
    print("Successfully imported docker_secrets adapter")
except ImportError as e:
    print(f"Could not import docker_secrets adapter: {str(e)}")

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
import pymysql

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
        # Get database credentials from environment variables (no defaults)
        MYSQLUSER = os.getenv("MYSQLUSER")
        MYSQLPASSWORD = os.getenv("MYSQLPASSWORD")
        MYSQLHOST = os.getenv("MYSQLHOST")
        MYSQLPORT = os.getenv("MYSQLPORT")
        MYSQLDB = os.getenv("MYSQLDB")
        
        # Log the database configuration (without password)
        logger.info(f"Database configuration: mysql+pymysql://{MYSQLUSER}:***@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}")
        
        # Check if all required environment variables are set
        if not all([MYSQLUSER, MYSQLPASSWORD, MYSQLHOST, MYSQLPORT, MYSQLDB]):
            logger.warning("Missing database environment variables. Database operations will not work.")
            return
        
        # URL encode the password to handle special characters
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(str(MYSQLPASSWORD))
        
        # Configure SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQLUSER}:{encoded_password}@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}'
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
        # Create a custom app context
        ctx = app.app_context()
        ctx.push()
        
        try:
            # Ensure SQLALCHEMY_TRACK_MODIFICATIONS is set
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            
            # Verify database connection
            try:
                # Simple database ping to verify connection
                db.session.execute("SELECT 1")
                logger.info("Database connection verified successfully")
            except Exception as db_error:
                logger.error(f"Database connection failed: {str(db_error)}")
                return int(time.time())  # Return timestamp as fallback ID
            
            # Create training run
            training_run = TrainingRun(
                user_id=user_id,
                run_name=run_name,
                prompt=None
            )
            
            # Add and commit in separate try blocks to pinpoint errors
            try:
                db.session.add(training_run)
                logger.info(f"Added training run to session")
            except Exception as add_error:
                logger.error(f"Error adding training run to session: {str(add_error)}")
                db.session.rollback()
                return int(time.time())
                
            try:
                db.session.commit()
                logger.info(f"Committed training run with ID {training_run.id}")
                # Return the actual generated ID
                return training_run.id
            except Exception as commit_error:
                logger.error(f"Error committing training run: {str(commit_error)}")
                db.session.rollback()
                return int(time.time())
        finally:
            # Always pop the context
            ctx.pop()
            
    except Exception as e:
        logger.error(f"Error in create_training_run: {str(e)}")
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

def save_model_to_db(user_id, run_id, model_name, model_url, filename=None, metric_name=None, metric_value=None):
    """Save model reference to database"""
    try:
        # Create a custom app context
        ctx = app.app_context()
        ctx.push()
        
        try:
            # Ensure SQLALCHEMY_TRACK_MODIFICATIONS is set
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            
            # Force using host.docker.internal for testing
            MYSQLUSER = os.getenv("MYSQLUSER")
            MYSQLPASSWORD = os.getenv("MYSQLPASSWORD")
            MYSQLPORT = os.getenv("MYSQLPORT")
            MYSQLDB = os.getenv("MYSQLDB")
            
            # Hardcode the host for testing
            MYSQLHOST = "host.docker.internal"
            
            # URL encode the password
            import urllib.parse
            encoded_password = urllib.parse.quote_plus(str(MYSQLPASSWORD))
            
            # Configure connection string with hardcoded host
            app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQLUSER}:{encoded_password}@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}'
            
            # Verify database connection
            try:
                # Simple database ping to verify connection
                db.session.execute("SELECT 1")
                logger.info("Database connection verified successfully")
            except Exception as db_error:
                logger.error(f"Database connection failed: {str(db_error)}")
                # Still return True since the model is in blob storage even if DB failed
                return True
            
            # If filename is None, extract it from the URL
            if filename is None and model_url:
                # Extract filename from URL: https://accountname.blob.core.windows.net/container/filename
                filename = model_url.split('/')[-1]
            
            # Create model record with metric information
            model_record = TrainingModel(
                user_id=user_id,
                run_id=run_id,
                model_name=model_name,
                model_url=model_url,
                file_name=filename,
                metric_name=metric_name,
                metric_value=metric_value
            )
            
            # Add and commit in separate try blocks for better error identification
            try:
                db.session.add(model_record)
                logger.info(f"Added model {model_name} to session with metric {metric_name}={metric_value}")
            except Exception as add_error:
                logger.error(f"Error adding model to session: {str(add_error)}")
                db.session.rollback()
                return True
                
            try:
                db.session.commit()
                logger.info(f"Committed model {model_name} with ID {model_record.id}")
                return True
            except Exception as commit_error:
                logger.error(f"Error committing model: {str(commit_error)}")
                db.session.rollback()
                return True
        finally:
            # Always pop the context
            ctx.pop()
            
    except Exception as e:
        logger.error(f"Error in save_model_to_db: {str(e)}")
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
    
    # Add unique identifier to force fresh training
    train_data['force_new_training'] = True
    train_data['unique_id'] = f"{service_name}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    try:
        # First, reset the model service to clear any previous state
        reset_response = requests.post(f"{url}/reset", timeout=10)
        if reset_response.status_code == 200:
            logger.info(f"Successfully reset {service_name} model service")
        else:
            logger.warning(f"Could not reset {service_name} model service: {reset_response.status_code}")
            
        # Now train on fresh model
        logger.info(f"Starting NEW training for {service_name} with unique ID {train_data['unique_id']}")
        response = requests.post(
            f"{url}/train",
            json=train_data,
            timeout=600  # 10 minutes timeout for training
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully trained NEW {service_name} model")
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
            
        # CRITICAL FIX: Ensure we always create a new run_id based on timestamp
        # to guarantee unique models for each dataset upload
        run_id = int(time.time())
        logger.info(f"ENFORCING NEW MODELS: Created unique run_id {run_id} for this dataset upload")
        
        # Create training run in database with the new run_id
        db_run_id = create_training_run(user_id, run_name)
        if not db_run_id:
            logger.warning(f"Failed to create training run in database, using timestamp run_id {run_id}")
        else:
            # Use the database-created run_id if available
            run_id = db_run_id
            logger.info(f"Verified unique training run with ID {run_id} was created successfully")
        
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
        
        # If data cleaning is needed, send to data cleaner API
        DATA_CLEANER_URL = os.getenv("DATA_CLEANER_URL", "http://data_cleaner:5001")
        cleaning_prompt = None
        if DATA_CLEANER_URL:
            try:
                print(f"Sending data to data cleaner at {DATA_CLEANER_URL}")
                
                # Check if we have a previous prompt to use for consistent cleaning
                previous_prompt = data.get('prompt')
                
                # Prepare payload for data cleaner
                cleaner_payload = {
                    'data': X_data,
                    'target_column': 'target'  # We need to know which column is the target
                }
                
                # Add previous prompt if available
                if previous_prompt:
                    cleaner_payload['prompt'] = previous_prompt
                
                # Send to data cleaner
                response = requests.post(
                    f"{DATA_CLEANER_URL}/clean",
                    json=cleaner_payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    cleaner_result = response.json()
                    X_data = cleaner_result['data']
                    
                    # Extract the cleaning prompt but save it later
                    print("Data cleaning completed successfully")
                    cleaning_prompt = cleaner_result.get('prompt')
                    if cleaning_prompt:
                        print(f"Received cleaning prompt - length: {len(cleaning_prompt)}")
                        print(f"First 100 chars: {cleaning_prompt[:100]}...")
                    else:
                        print("No cleaning prompt was returned from data cleaner")
                else:
                    print(f"Data cleaner error: {response.text}")
            except Exception as e:
                print(f"Error with data cleaner: {str(e)}")
        
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
                        
                        # Enhanced debugging for metrics
                        print(f"DETAILED METRICS FOR {model_name}:")
                        for metric_name, metric_value in metrics.items():
                            print(f"  - {metric_name}: {metric_value} (type: {type(metric_value)})")
                        
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
                        
                        # Store model data for later saving if it's one of the best
                        model_data = model_result['model']
                    
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
        
        # Successfully trained models - find and save the best
        print(f"Successfully trained {len(models_results)} models")
        
        # Verify metrics for all model services - this is for debugging
        available_services = get_model_services()
        verify_model_metrics(available_services)
        
        # Find the absolute best models for each metric across all architectures
        best_models = find_best_models(models_results)
        print(f"Found {len(best_models)} best models by metric")
        
        # Save only the 4 best models (one for each metric) to Azure Blob Storage
        saved_best_models = save_best_models(best_models, user_id, run_id)
        print(f"Saved {len(saved_best_models)} best models to Azure Blob Storage")
        
        # Now save the cleaning prompt after all models are trained
        if cleaning_prompt and run_id:
            print(f"Now saving prompt to database for run_id {run_id}")
            result = save_cleaning_prompt(run_id, cleaning_prompt)
            print(f"Prompt save result: {'Success' if result else 'Failed'}")
        
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
    """Find the absolute best models for each metric across all architectures"""
    try:
        # Only include the 4 main metrics we want to save
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        best_models = {}
        
        # First, collect all models with their metrics
        all_models = {}
        
        logger.info(f"Finding absolute best models across all architectures for metrics: {metrics}")
        logger.info(f"Processing results from {len(models_results)} model architectures")
        
        for model_index, model_result in enumerate(models_results):
            if 'model' in model_result:
                # Handle the nested structure
                model_info = model_result['model']
                
                # Extract the actual model name (architecture name)
                if 'model' in model_info and 'model_name' in model_info['model']:
                    model_name = model_info['model']['model_name']
                elif 'model_name' in model_info:
                    model_name = model_info['model_name']
                else:
                    model_name = f"unknown_model_{model_index}"
                
                # Extract metrics
                metrics_data = {}
                
                # Try different paths to find metrics
                if 'model' in model_info and 'metrics' in model_info['model']:
                    # Common case - metrics in nested structure
                    raw_metrics = model_info['model']['metrics']
                    logger.info(f"Found metrics in nested structure for {model_name}")
                elif 'metrics' in model_info:
                    # Alternative path
                    raw_metrics = model_info['metrics']
                    logger.info(f"Found metrics in direct structure for {model_name}")
                else:
                    logger.warning(f"No metrics found for {model_name}")
                    continue
                
                # Process and clean the metrics
                for metric in metrics:
                    if metric in raw_metrics:
                        # Ensure metric value is a float
                        try:
                            metric_value = float(raw_metrics[metric])
                            metrics_data[metric] = metric_value
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid {metric} value for {model_name}: {raw_metrics[metric]}")
                            metrics_data[metric] = 0.0
                
                # Only store if we have metrics
                if metrics_data:
                    all_models[model_name] = metrics_data
                    logger.info(f"Processed {model_name} with metrics: {metrics_data}")
        
        # Log all collected models
        logger.info(f"Collected metrics for {len(all_models)} models:")
        for model_name, model_metrics in all_models.items():
            metric_values = [f"{m}: {v:.4f}" for m, v in model_metrics.items()]
            logger.info(f"  - {model_name}: {', '.join(metric_values)}")
        
        # Find best model for each metric across all architectures
        for metric in metrics:
            best_metric_value = -1
            best_model_name = None
            
            for model_name, model_metrics in all_models.items():
                if metric in model_metrics:
                    metric_value = model_metrics[metric]
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_model_name = model_name
            
            if best_model_name:
                best_models[metric] = {
                    'model_name': best_model_name,
                    'value': best_metric_value
                }
                logger.info(f"Best model for {metric}: {best_model_name} with value {best_metric_value:.4f}")
        
        return best_models
    except Exception as e:
        logger.error(f"Error finding best models: {str(e)}")
        return {}

def save_best_models(best_models, user_id, run_id):
    """Save only the 4 absolute best models to blob storage and database - one for each metric"""
    try:
        saved_models = {}
        metrics_to_save = ['accuracy', 'precision', 'recall', 'f1']
        
        logger.info(f"Saving only the absolute best models for metrics: {metrics_to_save}")
        logger.info(f"Best models identified: {list(best_models.keys())}")
        logger.info(f"Ensuring completely new models are created for run_id: {run_id}")
        
        # Only process the 4 main metrics we care about
        for metric in metrics_to_save:
            if metric not in best_models:
                logger.warning(f"No best model found for metric: {metric}")
                continue
                
            model_info = best_models[metric]
            model_name = model_info['model_name']
            metric_value = model_info['value']
            
            logger.info(f"Processing best {metric} model: {model_name} with value {metric_value}")
            
            # Find the model data for this architecture in the trained models
            for service_name, service_url in get_model_services().items():
                if service_name == model_name:
                    try:
                        # First get model metadata for reference
                        logger.info(f"Fetching model metadata from {service_name} for metric {metric}")
                        info_response = requests.get(f"{service_url}/model_info", timeout=5)
                        
                        if info_response.status_code != 200:
                            logger.error(f"Failed to get model info from {service_name}: {info_response.text}")
                            continue
                        
                        # Get the actual model binary data using download_model endpoint
                        logger.info(f"Downloading model binary from {service_name}")
                        download_response = requests.get(f"{service_url}/download_model", timeout=60)
                        
                        if download_response.status_code != 200:
                            logger.error(f"Failed to download model from {service_name}: {download_response.text}")
                            continue
                            
                        # Save as best model for this metric
                        display_name = f"best_model_for_{metric}"
                        logger.info(f"Saving {model_name} as {display_name}")
                        
                        # Generate a unique filename with the run_id to ensure uniqueness per dataset
                        timestamp = int(time.time())
                        unique_id = str(uuid.uuid4())[:8]
                        filename = f"{model_name}_{metric}_{run_id}_{timestamp}_{unique_id}.joblib"
                        
                        # Upload to blob storage directly
                        model_bytes = io.BytesIO(download_response.content)
                        blob_url = upload_to_blob(model_bytes, filename)
                        
                        if blob_url:
                            # Save to database with metric information
                            saved = save_model_to_db(user_id, run_id, display_name, blob_url, filename, metric, metric_value)
                            if saved:
                                saved_models[metric] = {
                                    'model_name': model_name,
                                    'display_name': display_name,
                                    'url': blob_url,
                                    'filename': filename,
                                    'value': metric_value
                                }
                                logger.info(f"Successfully saved {display_name} to blob: {blob_url}")
                            else:
                                logger.error(f"Failed to save {display_name} to database")
                        else:
                            logger.error(f"Failed to save {display_name} to blob storage")
                    except Exception as e:
                        logger.error(f"Error saving best model for {metric}: {str(e)}")
        
        if not saved_models:
            logger.warning("No models were saved to blob storage!")
        else:
            logger.info(f"Successfully saved {len(saved_models)} best models to blob storage")
            for metric, model in saved_models.items():
                logger.info(f"  - {metric}: {model['model_name']} (score: {model['value']})")
                
        return saved_models
    except Exception as e:
        logger.error(f"Error in save_best_models: {str(e)}")
        return {}

# Helper function to verify model metrics (for debugging)
def verify_model_metrics(model_services):
    """Verify and debug metrics for each model service"""
    results = {}
    
    logger.info("Verifying metrics from all model services...")
    
    for model_name, service_url in model_services.items():
        try:
            logger.info(f"Requesting metrics from {model_name} at {service_url}")
            response = requests.get(f"{service_url}/model_info", timeout=5)
            
            if response.status_code == 200:
                model_info = response.json()
                
                # Determine where metrics are stored in the response
                metrics_data = None
                if 'metrics' in model_info:
                    metrics_data = model_info['metrics']
                    logger.info(f"Found metrics directly in model_info for {model_name}")
                elif 'model' in model_info and 'metrics' in model_info['model']:
                    metrics_data = model_info['model']['metrics']
                    logger.info(f"Found metrics in nested model for {model_name}")
                
                if metrics_data:
                    # Log detailed metrics
                    logger.info(f"Metrics for {model_name}:")
                    for metric_name, metric_value in metrics_data.items():
                        logger.info(f"  - {metric_name}: {metric_value} (type: {type(metric_value).__name__})")
                    
                    # Store metrics for comparison
                    results[model_name] = metrics_data
                else:
                    logger.warning(f"No metrics found for {model_name}")
            else:
                logger.error(f"Error from {model_name}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error verifying metrics for {model_name}: {str(e)}")
    
    # Compare metrics across models
    if results:
        logger.info("=== METRICS COMPARISON ACROSS MODELS ===")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            best_model = None
            best_value = -1
            
            logger.info(f"\nComparison for {metric}:")
            for model_name, metrics in results.items():
                if metric in metrics:
                    try:
                        value = float(metrics[metric])
                        logger.info(f"  - {model_name}: {value:.4f}")
                        
                        if value > best_value:
                            best_value = value
                            best_model = model_name
                    except:
                        logger.warning(f"  - {model_name}: {metrics[metric]} (invalid format)")
            
            if best_model:
                logger.info(f"  → Best for {metric}: {best_model} ({best_value:.4f})")
            else:
                logger.info(f"  → No valid models found for {metric}")
    
    return results

@app.route('/init_db', methods=['GET'])
def init_database():
    """Initialize database tables - for troubleshooting only"""
    try:
        # Create a custom app context
        ctx = app.app_context()
        ctx.push()
        
        try:
            # Create all tables
            db.create_all()
            logger.info("Database tables created successfully")
            
            # Verify that we can access the tables
            try:
                # Try to count users
                user_count = User.query.count()
                # Try to count training runs
                run_count = TrainingRun.query.count()
                # Try to count models
                model_count = TrainingModel.query.count()
                
                return jsonify({
                    "success": True,
                    "message": "Database initialized successfully",
                    "stats": {
                        "users": user_count,
                        "training_runs": run_count,
                        "models": model_count
                    }
                })
            except Exception as query_error:
                logger.error(f"Error querying tables: {str(query_error)}")
                return jsonify({
                    "success": False,
                    "error": f"Error querying tables: {str(query_error)}"
                }), 500
        finally:
            # Always pop the context
            ctx.pop()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/db_diagnostics', methods=['GET'])
def db_diagnostics():
    """Run database diagnostics to help troubleshoot connectivity issues"""
    try:
        # Create a custom app context
        ctx = app.app_context()
        ctx.push()
        
        diagnostics = {
            "database_config": {
                "host": os.getenv("MYSQLHOST", "Not set"),
                "port": os.getenv("MYSQLPORT", "Not set"),
                "user": os.getenv("MYSQLUSER", "Not set"),
                "database": os.getenv("MYSQLDB", "Not set"),
                "password": "********" if os.getenv("MYSQLPASSWORD") else "Not set"
            },
            "connection_string": app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured'),
            "blob_storage": {
                "account": os.getenv("AZURESTORAGEACCOUNT", "Not set"),
                "container": os.getenv("AZURECONTAINER", "Not set"),
                "key_configured": "Yes" if os.getenv("AZURESTORAGEKEY") else "No"
            }
        }
        
        # Test database connection
        try:
            # Try a simple query
            db.session.execute("SELECT 1")
            diagnostics["database_connection"] = "Success"
            
            # Check if tables exist
            diagnostics["tables"] = {}
            
            # Check User table
            try:
                user_count = User.query.count()
                diagnostics["tables"]["users"] = {"exists": True, "count": user_count}
            except Exception as e:
                diagnostics["tables"]["users"] = {"exists": False, "error": str(e)}
                
            # Check TrainingRun table
            try:
                run_count = TrainingRun.query.count()
                diagnostics["tables"]["training_runs"] = {"exists": True, "count": run_count}
            except Exception as e:
                diagnostics["tables"]["training_runs"] = {"exists": False, "error": str(e)}
                
            # Check TrainingModel table
            try:
                model_count = TrainingModel.query.count()
                diagnostics["tables"]["training_models"] = {"exists": True, "count": model_count}
            except Exception as e:
                diagnostics["tables"]["training_models"] = {"exists": False, "error": str(e)}
                
        except Exception as e:
            diagnostics["database_connection"] = f"Failed: {str(e)}"
            
        # Always pop the context
        ctx.pop()
        
        return jsonify(diagnostics)
    except Exception as e:
        logger.error(f"Error running database diagnostics: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

def save_cleaning_prompt(run_id, prompt):
    """
    Save the cleaning prompt using SQLAlchemy with the same approach as save_model_to_db
    """
    try:
        # Create a custom app context
        ctx = app.app_context()
        ctx.push()
        
        try:
            # Force using host.docker.internal for testing
            MYSQLUSER = os.getenv("MYSQLUSER")
            MYSQLPASSWORD = os.getenv("MYSQLPASSWORD")
            MYSQLPORT = os.getenv("MYSQLPORT")
            MYSQLDB = os.getenv("MYSQLDB")
            
            # Always hardcode the host to host.docker.internal
            MYSQLHOST = "host.docker.internal"
            
            print(f"DEBUG: Directly connecting to MySQL at {MYSQLHOST}:{MYSQLPORT}")
            
            # Create a direct connection to verify it's accessible
            try:
                # Direct connection test
                connection = pymysql.connect(
                    host=MYSQLHOST,
                    user=MYSQLUSER,
                    password=MYSQLPASSWORD,
                    database=MYSQLDB,
                    port=int(MYSQLPORT),
                    connect_timeout=5
                )
                
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    print(f"DEBUG: Direct MySQL connection test successful")
                    
                    # Update directly with raw PyMySQL
                    cursor.execute("UPDATE training_run SET prompt = %s WHERE id = %s", (prompt, run_id))
                    connection.commit()
                    print(f"DEBUG: Directly updated prompt for run {run_id}")
                    connection.close()
                    return True
                    
            except Exception as direct_error:
                print(f"DEBUG: Direct MySQL connection failed: {str(direct_error)}")
                # Fall through to SQLAlchemy method
            
            # URL encode the password
            import urllib.parse
            encoded_password = urllib.parse.quote_plus(str(MYSQLPASSWORD))
            
            # Configure connection string with hardcoded host
            new_uri = f'mysql+pymysql://{MYSQLUSER}:{encoded_password}@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}'
            print(f"DEBUG: Setting SQLAlchemy URI to: mysql+pymysql://{MYSQLUSER}:***@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}")
            app.config['SQLALCHEMY_DATABASE_URI'] = new_uri
            
            # Re-initialize the db with the app to apply the new connection string
            print("DEBUG: Re-initializing database connection")
            db.init_app(app)
            
            # Verify database connection
            try:
                # Simple database ping to verify connection
                db.session.execute("SELECT 1")
                print("DEBUG: SQLAlchemy database connection verified successfully")
            except Exception as db_error:
                print(f"DEBUG: SQLAlchemy database connection failed: {str(db_error)}")
                # Return True to allow process to continue
                return True
            
            # Find the training run
            training_run = TrainingRun.query.get(run_id)
            if not training_run:
                print(f"DEBUG: Training run {run_id} not found in database")
                return True
                
            # Update the prompt
            print(f"DEBUG: Found training run {run_id}, updating prompt")
            training_run.prompt = prompt
            
            try:
                db.session.commit()
                print(f"DEBUG: Successfully committed prompt update for run {run_id}")
                return True
            except Exception as commit_error:
                print(f"DEBUG: Error committing prompt update: {str(commit_error)}")
                db.session.rollback()
                return True
        finally:
            # Always pop the context
            ctx.pop()
    except Exception as e:
        print(f"DEBUG: Error in save_cleaning_prompt: {str(e)}")
        # Still return True to allow process to continue
        return True

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5020))
    
    # Set Docker flag based on environment
    IS_DOCKER = os.environ.get('IS_DOCKER', 'false').lower() == 'true'
    logger.info(f"Running in {'Docker' if IS_DOCKER else 'local'} mode")
    
    # Initialize database tables on startup
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
            
            # Verify tables by counting entries
            user_count = User.query.count()
            run_count = TrainingRun.query.count()
            model_count = TrainingModel.query.count()
            logger.info(f"Database stats: {user_count} users, {run_count} training runs, {model_count} models")
            
            # MLflow setup
            mlflow_client = MlflowClient()
            experiment_names = [exp.name for exp in mlflow_client.list_experiments()]
            logger.info(f"MLflow experiments: {experiment_names}")
    except Exception as e:
        logger.error(f"Error initializing database tables: {str(e)}")
        logger.warning("Some database operations may not work correctly.")
    
    serve(app, host='0.0.0.0', port=port) 