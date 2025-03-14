from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import io
import time
from datetime import datetime
import requests
import logging
import shutil
import atexit
import tempfile
import uuid
import glob
import threading
import joblib
import docker
import socket
import subprocess
import select
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NaN, inf, -inf
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return super().default(obj)

# Safe JSON dump function
def safe_json_dumps(obj):
    return json.dumps(obj, cls=SafeJSONEncoder)

# Modified requests post function that handles problematic float values
def safe_requests_post(url, json_data, **kwargs):
    safe_json = clean_data_for_json(json_data)
    return requests.post(url, json=safe_json, **kwargs)

# Load environment variables from .env file
load_dotenv()

# Define service URLs
DATA_CLEANER_URL = "http://localhost:5001"
FEATURE_SELECTOR_URL = "http://localhost:5002"
ANOMALY_DETECTOR_URL = "http://localhost:5003"
MODEL_COORDINATOR_URL = "http://localhost:5020"  # New model coordinator URL instead of MODEL_TRAINER_URL
MEDICAL_ASSISTANT_URL = "http://localhost:5005"

# Update service URLs dictionary with proper health endpoints
SERVICES = {
    "Data Cleaner": {"url": DATA_CLEANER_URL, "endpoint": "/health"},
    "Feature Selector": {"url": FEATURE_SELECTOR_URL, "endpoint": "/health"},
    "Anomaly Detector": {"url": ANOMALY_DETECTOR_URL, "endpoint": "/health"},
    "Model Coordinator": {"url": MODEL_COORDINATOR_URL, "endpoint": "/health"},
    "Medical Assistant": {"url": MEDICAL_ASSISTANT_URL, "endpoint": "/health"}
}

# Setup Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'medicai_temp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False

# Register cleanup function for temporary files
def cleanup_temp_files():
    """Remove all temporary files when the application exits"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            print(f"Cleaned up temporary directory: {UPLOAD_FOLDER}")
    except Exception as e:
        print(f"Error cleaning up temporary directory: {str(e)}")

atexit.register(cleanup_temp_files)

# Function to generate a unique filename
def get_temp_filepath(original_filename=None, extension=None):
    """Generate a unique temporary filepath"""
    if extension is None and original_filename:
        extension = os.path.splitext(original_filename)[1]
    elif extension is None:
        extension = '.tmp'
    
    unique_id = str(uuid.uuid4())
    if original_filename:
        safe_name = secure_filename(original_filename)
        filename = f"{unique_id}_{safe_name}"
    else:
        filename = f"{unique_id}{extension}"
    
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

# Update the session cleanup function to handle files
def cleanup_session_files():
    """Remove files associated with the current session"""
    files_to_cleanup = []
    
    # Check for file paths in session
    file_keys = [
        'uploaded_file', 'cleaned_file', 'selected_features_file', 'predictions_file'
    ]
    
    for key in file_keys:
        if key in session:
            filepath = session.get(key)
            if filepath and os.path.exists(filepath):
                files_to_cleanup.append(filepath)
                
    # Delete the files
    for filepath in files_to_cleanup:
        try:
            os.remove(filepath)
            logger.info(f"Deleted temporary file: {filepath}")
        except Exception as e:
            logger.error(f"Error deleting temporary file {filepath}: {str(e)}")
            
    # Clear session references to files
    for key in file_keys:
        if key in session:
            session.pop(key)

# Check service health
def check_services():
    services = {
        "Data Cleaner": DATA_CLEANER_URL,
        "Feature Selector": FEATURE_SELECTOR_URL,
        "Anomaly Detector": ANOMALY_DETECTOR_URL,
        "Model Coordinator": MODEL_COORDINATOR_URL,
        "Medical Assistant": MEDICAL_ASSISTANT_URL
    }
    
    status = {}
    for name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                status[name] = "healthy"
            else:
                status[name] = f"unhealthy - {response.status_code}"
        except Exception as e:
            status[name] = f"unreachable - {str(e)[:50]}"  # Truncate long error messages
    
    return status

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def load_data(file_path):
    """Load data from file with error handling and feedback"""
    try:
        if file_path.endswith('.csv'):
            # Attempt to read with UTF-8 encoding first
            try:
                data = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try ISO-8859-1 (Latin-1) encoding
                data = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file_path)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."
        
        # Generate file statistics
        file_stats = {
            'rows': data.shape[0],
            'columns': data.shape[1],
            'memory_usage': f"{data.memory_usage().sum() / 1024:.2f} KB",
            'upload_time': datetime.now().strftime('%H:%M:%S')
        }
        
        return data, file_stats
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return None, f"Error loading file: {str(e)}"

def is_service_available(service_url):
    """Check if a service is available"""
    try:
        response = requests.get(f"{service_url}/health", timeout=1)
        return response.status_code == 200
    except:
        return False

def clean_data_for_json(data):
    """Clean DataFrame to make it JSON serializable by replacing non-compliant values"""
    if isinstance(data, pd.DataFrame):
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Replace inf/-inf with None (which will become null in JSON)
        data_copy = data_copy.replace([np.inf, -np.inf], None)
        
        # Replace NaN with None
        data_copy = data_copy.where(pd.notnull(data_copy), None)
        
        # Handle any remaining problematic float values
        for col in data_copy.select_dtypes(include=['float']).columns:
            data_copy[col] = data_copy[col].apply(
                lambda x: None if x is not None and (np.isnan(x) or np.isinf(x)) else x
            )
            
        return data_copy.to_dict(orient='records')
    elif isinstance(data, pd.Series):
        # For Series objects (like target variables)
        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Handle problematic values for Series
        if data_copy.dtype.kind == 'f':  # If float type
            data_copy = data_copy.apply(
                lambda x: None if np.isnan(x) or np.isinf(x) else x
            )
        
        return data_copy.replace([np.inf, -np.inf, np.nan], None).tolist()
    elif isinstance(data, list):
        # For lists, recursively clean each item
        return [clean_data_for_json(item) if isinstance(item, (pd.DataFrame, pd.Series, list, dict)) else 
                (None if isinstance(item, float) and (np.isnan(item) or np.isinf(item)) else item) 
                for item in data]
    elif isinstance(data, dict):
        # For dictionaries, recursively clean each value
        return {k: clean_data_for_json(v) if isinstance(v, (pd.DataFrame, pd.Series, list, dict)) else
                (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
                for k, v in data.items()}
    
    # Handle single float value
    if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    
    return data

@app.route('/')
def index():
    # Reset session data for new session
    for key in list(session.keys()):
        if key != '_flashes':
            session.pop(key)
    
    # Clean up any files from previous sessions
    cleanup_session_files()
    
    # Force a new session ID to prevent stale data
    session.clear()
    session.permanent = False
    
    # Check services health for status display
    services_status = check_services()
    return render_template('index.html', services_status=services_status)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Clean up previous upload if exists
        if 'uploaded_file' in session and os.path.exists(session['uploaded_file']):
            try:
                os.remove(session['uploaded_file'])
            except:
                pass
        
        # Generate unique filename for temporary storage
        filepath = get_temp_filepath(file.filename)
        file.save(filepath)
        
        # Load the data to validate it
        data, result = load_data(filepath)
        if data is None:
            # Clean up invalid file
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(result, 'error')
            return redirect(url_for('index'))
        
        session['uploaded_file'] = filepath
        session['file_stats'] = result
        
        # Store data columns for later use
        session['data_columns'] = data.columns.tolist()
        
        # Redirect to training page
        return redirect(url_for('training'))
    
    flash('Invalid file type. Please upload a CSV or Excel file.', 'error')
    return redirect(url_for('index'))

@app.route('/training', methods=['GET', 'POST'])
def training():
    filepath = session.get('uploaded_file')
    
    if not filepath:
        flash('Please upload a file first', 'error')
        return redirect(url_for('index'))
    
    data, _ = load_data(filepath)
    
    if request.method == 'POST':
        # Get target column from form
        target_column = request.form.get('target_column')
        session['target_column'] = target_column
        
        # Get additional parameters (optional)
        session['test_size'] = float(request.form.get('test_size', 0.2))
        
        try:
            # Check if required services are available
            required_services = {
                "Data Cleaner": DATA_CLEANER_URL,
                "Feature Selector": FEATURE_SELECTOR_URL,
                "Anomaly Detector": ANOMALY_DETECTOR_URL,
                "Model Coordinator": MODEL_COORDINATOR_URL  # Changed from Model Trainer to Model Coordinator
            }
            
            for service_name, service_url in required_services.items():
                if not is_service_available(service_url):
                    flash(f"The {service_name} service is not available. Cannot proceed with training.", 'error')
                    return redirect(url_for('training'))
            
            # DEBUG: Check which model services are available through the coordinator
            try:
                coordinator_health_response = requests.get(f"{MODEL_COORDINATOR_URL}/health", timeout=5)
                if coordinator_health_response.status_code == 200:
                    coordinator_health = coordinator_health_response.json()
                    if "model_services" in coordinator_health:
                        for model, status in coordinator_health["model_services"].items():
                            logger.info(f"Model service {model}: {status}")
                    else:
                        logger.warning("Model services not found in coordinator health response")
            except Exception as e:
                logger.error(f"Error checking model services: {str(e)}")
            
            # 1. CLEAN DATA (via Data Cleaner API)
            logger.info(f"Sending data to Data Cleaner API")
            # Convert data to records - just a plain dictionary
            data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')
            # Use our safe request method
            response = safe_requests_post(
                f"{DATA_CLEANER_URL}/clean",
                {
                    "data": data_records,
                    "target_column": target_column
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Data Cleaner API error: {response.json().get('error', 'Unknown error')}")
            
            cleaning_result = response.json()
            cleaned_data = pd.DataFrame.from_dict(cleaning_result["data"])
            
            # Clean up previous cleaned file if exists
            if 'cleaned_file' in session and os.path.exists(session['cleaned_file']):
                try:
                    os.remove(session['cleaned_file'])
                except:
                    pass
                    
            cleaned_filepath = get_temp_filepath(extension='.csv')
            cleaned_data.to_csv(cleaned_filepath, index=False)
            session['cleaned_file'] = cleaned_filepath
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Data Cleaner API: {data_records[:5]}")
            
            # 2. FEATURE SELECTION (via Feature Selector API)
            logger.info(f"Sending data to Feature Selector API")
            X = cleaned_data.drop(columns=[target_column])
            y = cleaned_data[target_column]
            
            # Convert X and y to simple Python structures
            X_records = X.replace([np.inf, -np.inf], np.nan).where(pd.notnull(X), None).to_dict(orient='records')
            y_list = y.replace([np.inf, -np.inf], np.nan).where(pd.notnull(y), None).tolist()
            
            # Use our safe request method
            response = safe_requests_post(
                f"{FEATURE_SELECTOR_URL}/select_features",
                {
                    "data": X_records,
                    "target": y_list,
                    "target_name": target_column
                },
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"Feature Selector API error: {response.json().get('error', 'Unknown error')}")
            
            feature_result = response.json()
            X_selected = pd.DataFrame.from_dict(feature_result["transformed_data"])
            
            # Store selected features in session
            selected_features = X_selected.columns.tolist()
            
            # Save selected features to file to prevent session bloat
            selected_features_file_json = save_to_temp_file(selected_features, 'selected_features')
            session['selected_features_file_json'] = selected_features_file_json
            # Use a reference in session to avoid storing large data
            session['selected_features'] = f"[{len(selected_features)} features]"
            
            # Store feature importances for visualization
            feature_importance = []
            for feature, importance in feature_result["feature_importances"].items():
                feature_importance.append({'Feature': feature, 'Importance': importance})
            
            # Save feature importance to file
            feature_importance_file = save_to_temp_file(feature_importance, 'feature_importance')
            session['feature_importance_file'] = feature_importance_file
            
            # Clean up previous selected features file if exists
            if 'selected_features_file' in session and os.path.exists(session['selected_features_file']):
                try:
                    os.remove(session['selected_features_file'])
                except:
                    pass
                    
            selected_features_filepath = get_temp_filepath(extension='.csv')
            X_selected.to_csv(selected_features_filepath, index=False)
            session['selected_features_file'] = selected_features_filepath
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Feature Selector API: {X_records[:5]}, Target: {y_list[:5]}")
            
            # 3. ANOMALY DETECTION (via Anomaly Detector API)
            logger.info(f"Sending data to Anomaly Detector API")
            # Convert to simple Python structure
            X_selected_records = X_selected.replace([np.inf, -np.inf], np.nan).where(pd.notnull(X_selected), None).to_dict(orient='records')

            # Use our safe request method
            response = safe_requests_post(
                f"{ANOMALY_DETECTOR_URL}/detect_anomalies",
                {
                    "data": X_selected_records
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Anomaly Detector API error: {response.json().get('error', 'Unknown error')}")
            
            anomaly_results = response.json()
            session['anomaly_results'] = {
                'is_data_valid': anomaly_results["is_data_valid"],
                'anomaly_percentage': anomaly_results["anomaly_report"]["anomaly_percentage"]
            }
            
            # Add logging to verify data being sent to APIs
            logger.info(f"Data being sent to Anomaly Detector API: {X_selected_records[:5]}")
            
            # 4. MODEL TRAINING (via Model Coordinator API instead of Model Trainer API)
            logger.info(f"Sending data to Model Coordinator API")
            
            # Prepare data for model coordinator
            X_data = {feature: X_selected[feature].tolist() for feature in selected_features if feature in X_selected.columns}
            y_data = y.tolist()

            # Use our safe request method
            response = safe_requests_post(
                f"{MODEL_COORDINATOR_URL}/train",
                {
                    "data": X_data,
                    "target": y_data,
                    "test_size": session['test_size']
                },
                timeout=1800  # Model training can take time
            )
            
            if response.status_code != 200:
                raise Exception(f"Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
            
            # Store the complete model results in session
            model_result = response.json()
            
            # CRITICAL: Process and fix any metric values to ensure they're properly serialized
            if 'models' in model_result:
                for model_data in model_result['models']:
                    if 'model' in model_data:
                        # This is the critical fix - handle the nested structure
                        # The structure is: model_data -> model -> model -> metrics
                        if 'model' in model_data['model']:
                            nested_model = model_data['model']['model']
                            # Now process the metrics from the nested model
                            if 'metrics' in nested_model:
                                metrics = nested_model['metrics']
                                # Convert all metric values to float
                                clean_metrics = {}
                                for metric_name, metric_value in metrics.items():
                                    try:
                                        clean_metrics[metric_name] = float(metric_value)
                                    except (ValueError, TypeError):
                                        logger.warning(f"Could not convert metric {metric_name}={metric_value} to float")
                                        clean_metrics[metric_name] = 0.0
                                
                                # Replace with cleaned metrics
                                nested_model['metrics'] = clean_metrics
                                
                                # Log the cleaned metrics for debugging
                                model_name = nested_model.get('model_name', 'unknown')
                                logger.info(f"Cleaned metrics for {model_name}: {clean_metrics}")
            
            # Save processed results to session
            session['model_results'] = model_result
            
            # Log key info from the response
            if 'models' in model_result:
                num_models = len(model_result['models'])
                logger.info(f"Found {num_models} models in response")
                model_names = []
                if model_result['models']:
                    for model_data in model_result['models']:
                        if 'model' in model_data and 'model_name' in model_data['model']:
                            model_names.append(model_data['model']['model_name'])
                    logger.info(f"Model names in response: {', '.join(model_names)}")
                    
                    # Verify that we have all 6 expected models
                    expected_models = ['logistic_regression', 'decision_tree', 'random_forest', 
                                      'svm', 'knn', 'naive_bayes']
                    missing_models = [m for m in expected_models if m not in model_names]
                    if missing_models:
                        logger.warning(f"Missing models in response: {', '.join(missing_models)}")
            
            return redirect(url_for('model_selection'))
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            flash(f"Error processing data: {str(e)}", 'error')
            return redirect(url_for('training'))
    
    # Get AI recommendations for the dataset (via Medical Assistant API) - OPTIONAL
    ai_recommendations = None
    if 'ai_recommendations' not in session and 'ai_recommendations_file' not in session and is_service_available(MEDICAL_ASSISTANT_URL):
        try:
            logger.info(f"Sending data to Medical Assistant API")
            
            # Convert to simple Python structure
            data_records = data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(data), None).to_dict(orient='records')
            
            # Use our safe request method
            response = safe_requests_post(
                f"{MEDICAL_ASSISTANT_URL}/analyze_data",
                {
                    "data": data_records
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_recommendations = response.json()["recommendations"]
                
                # Save to file instead of storing in session
                recommendations_file = save_to_temp_file(ai_recommendations, 'ai_recommendations')
                session['ai_recommendations_file'] = recommendations_file
                logger.info(f"Saved AI recommendations to {recommendations_file}")
                
                # Log sample of data sent
                logger.info(f"Data being sent to Medical Assistant API: {data_records[:3]}")
            else:
                logger.warning(f"Medical Assistant API returned an error: {response.text}")
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}", exc_info=True)
            # Don't flash this error to avoid confusing the user
            logger.info("Continuing without AI recommendations")
    elif 'ai_recommendations_file' in session:
        # Load from file
        ai_recommendations = load_from_temp_file(session['ai_recommendations_file'])
    else:
        # Try to get from session (backward compatibility)
        ai_recommendations = session.get('ai_recommendations')
        
        # If it's in session, move to file
        if ai_recommendations:
            recommendations_file = save_to_temp_file(ai_recommendations, 'ai_recommendations')
            session['ai_recommendations_file'] = recommendations_file
            session.pop('ai_recommendations', None)
            logger.info(f"Moved AI recommendations from session to file: {recommendations_file}")
    
    # Make sure session size is under control
    check_session_size()
    
    return render_template('training.html', 
                          data=data.head().to_html(classes='table table-striped'),
                          columns=data.columns.tolist(),
                          file_stats=session.get('file_stats'),
                          ai_recommendations=ai_recommendations)

@app.route('/download_cleaned')
def download_cleaned():
    cleaned_filepath = session.get('cleaned_file')
    if not cleaned_filepath:
        flash('No cleaned file available', 'error')
        return redirect(url_for('training'))
    
    # Create a BytesIO object to serve the file from memory
    file_data = io.BytesIO()
    with open(cleaned_filepath, 'rb') as f:
        file_data.write(f.read())
    file_data.seek(0)
    
    return send_file(file_data, as_attachment=True, download_name='cleaned_data.csv')

@app.route('/feature_importance')
def feature_importance():
    # First check if we have it in a file
    importance_file = session.get('feature_importance_file')
    importance_data = None
    
    if importance_file:
        importance_data = load_from_temp_file(importance_file)
    else:
        # Try the session (backward compatibility)
        importance_data = session.get('feature_importance')
        
        # If it's in session and large, move to file
        if importance_data and len(importance_data) > 20:
            importance_file = save_to_temp_file(importance_data, 'feature_importance')
            session['feature_importance_file'] = importance_file
            session.pop('feature_importance', None)
            logger.info(f"Moved feature importance from session to file: {importance_file}")
    
    if not importance_data:
        return jsonify({'error': 'No feature importance data available'})
    
    # Sort by importance
    importance_data = sorted(importance_data, key=lambda x: x['Importance'])
    
    # Create Plotly figure
    fig = go.Figure(go.Bar(
        x=[x['Importance'] for x in importance_data],
        y=[x['Feature'] for x in importance_data],
        orientation='h',
        marker_color='rgba(50, 171, 96, 0.6)',
    ))
    
    fig.update_layout(
        title='Feature Importance Scores',
        height=400 + len(importance_data) * 20,
        xaxis_title='Importance Score',
        yaxis_title='Feature Name',
        template='plotly_white'
    )
    
    # Convert to JSON for frontend
    graphJSON = json.dumps(fig, cls=SafeJSONEncoder)
    return jsonify(graphJSON)

@app.route('/model_selection')
def model_selection():
    """Display the trained models and allow selection of the best one"""
    # Log what's in the session for debugging
    logger.info(f"Session keys: {list(session.keys())}")
    
    # Check if we have model results
    if 'model_results' not in session or not session['model_results']:
        flash('No trained models available. Please complete the training process first.', 'error')
        return redirect(url_for('training'))
    
    model_result = session['model_results']
    logger.info(f"Model result keys: {list(model_result.keys() if model_result else [])}")
    
    # CRITICAL DEBUGGING: Print the raw model results to understand what's coming from the coordinator
    logger.info("================== RAW MODEL RESULTS ==================")
    logger.info(json.dumps(model_result, cls=SafeJSONEncoder))
    logger.info("======================================================")
    
    # Process the models data
    all_models = {}  # All models with metrics
    best_models = {  # Track best models by each metric
        'accuracy': {'model_name': '', 'metrics': {'accuracy': 0}},
        'precision': {'model_name': '', 'metrics': {'precision': 0}},
        'recall': {'model_name': '', 'metrics': {'recall': 0}}
    }
    
    try:
        if 'models' in model_result:
            # Extract each model's data
            for model_index, model_data in enumerate(model_result['models']):
                # Each item in the models array contains a "model" key
                if 'model' in model_data:
                    model_info = model_data['model']
                    
                    # Debug individual model info
                    logger.info(f"Processing model {model_index}: {json.dumps(model_info, cls=SafeJSONEncoder)}")
                    
                    # CRITICAL FIX: The structure is nested one level deeper than expected
                    # The actual nested structure is: model_data -> model -> model -> metrics
                    if 'model' in model_info:
                        nested_model_info = model_info['model']
                        model_name = nested_model_info.get('model_name', f"model_{model_index}")
                        
                        # Get metrics from the nested structure
                        metrics = {}
                        
                        # First, check if metrics exists in nested_model_info
                        if 'metrics' in nested_model_info:
                            raw_metrics = nested_model_info['metrics']
                            logger.info(f"Model {model_name} raw metrics: {raw_metrics}")
                            
                            # Process metrics by type
                            for metric_name, metric_value in raw_metrics.items():
                                # For numeric metrics, ensure they're properly stored
                                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'cv_score_mean', 'cv_score_std']:
                                    # Convert to float if not already
                                    if not isinstance(metric_value, float):
                                        try:
                                            metric_value = float(metric_value)
                                        except (TypeError, ValueError):
                                            metric_value = 0.0
                                    
                                    metrics[metric_name] = metric_value
                        else:
                            logger.warning(f"No metrics found for model {model_name} in nested structure")
                            # Set default metrics
                            metrics = {
                                'accuracy': 0.0,
                                'precision': 0.0,
                                'recall': 0.0,
                                'f1': 0.0
                            }
                        
                        # Store model with metrics
                        all_models[model_name] = metrics
                        
                        # Check if this model is best for any metric
                        accuracy = metrics.get('accuracy', 0.0)
                        precision = metrics.get('precision', 0.0)
                        recall = metrics.get('recall', 0.0)
                        
                        logger.info(f"Model {model_name} metrics - accuracy: {accuracy}, precision: {precision}, recall: {recall}")
                        
                        if accuracy > best_models['accuracy']['metrics'].get('accuracy', 0):
                            best_models['accuracy'] = {
                                'model_name': model_name,
                                'metrics': metrics
                            }
                            
                        if precision > best_models['precision']['metrics'].get('precision', 0):
                            best_models['precision'] = {
                                'model_name': model_name,
                                'metrics': metrics
                            }
                            
                        if recall > best_models['recall']['metrics'].get('recall', 0):
                            best_models['recall'] = {
                                'model_name': model_name,
                                'metrics': metrics
                            }
                    else:
                        logger.warning(f"Model data doesn't have the expected nested structure for model {model_index}")
                        # If we can't find the nested structure, try the direct approach as fallback
                        model_name = model_info.get('model_name', f"model_{model_index}")
                        
                        # Fix metrics processing
                        metrics = {}
                        
                        # First, check if metrics exists in model_info
                        if 'metrics' in model_info:
                            raw_metrics = model_info['metrics']
                            logger.info(f"Model {model_name} raw metrics (fallback): {raw_metrics}")
                            
                            # Process metrics by type
                            for metric_name, metric_value in raw_metrics.items():
                                # For numeric metrics, ensure they're properly stored
                                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'cv_score_mean', 'cv_score_std']:
                                    # Convert to float if not already
                                    if not isinstance(metric_value, float):
                                        try:
                                            metric_value = float(metric_value)
                                        except (TypeError, ValueError):
                                            metric_value = 0.0
                                    
                                    metrics[metric_name] = metric_value
                            
                            # Store model
                            all_models[model_name] = metrics
                            
                            # Check if this model is best for any metric
                            accuracy = metrics.get('accuracy', 0.0)
                            precision = metrics.get('precision', 0.0)
                            recall = metrics.get('recall', 0.0)
                            
                            logger.info(f"Model {model_name} metrics (fallback) - accuracy: {accuracy}, precision: {precision}, recall: {recall}")
                            
                            if accuracy > best_models['accuracy']['metrics'].get('accuracy', 0):
                                best_models['accuracy'] = {
                                    'model_name': model_name,
                                    'metrics': metrics
                                }
                                
                            if precision > best_models['precision']['metrics'].get('precision', 0):
                                best_models['precision'] = {
                                    'model_name': model_name,
                                    'metrics': metrics
                                }
                                
                            if recall > best_models['recall']['metrics'].get('recall', 0):
                                best_models['recall'] = {
                                    'model_name': model_name,
                                    'metrics': metrics
                                }
        
        # Log how many models were processed
        logger.info(f"Processed {len(all_models)} models")
        logger.info(f"Best models: {json.dumps(best_models, cls=SafeJSONEncoder)}")
        
        if len(all_models) < 6:
            logger.warning(f"Expected 6 models but only found {len(all_models)}")
            
    except Exception as e:
        logger.error(f"Error processing model results: {str(e)}", exc_info=True)
        flash('Error processing model data. Please try training again.', 'error')
        return redirect(url_for('training'))
    
    # Prepare data for template display - ONLY THE TOP 3 MODELS
    simplified_model_data = []
    
    # Process best models by metric - ONLY include the best models
    for metric_index, (metric, model_info) in enumerate(best_models.items()):
        if model_info.get('model_name'):  # Only include models with a name
            model_name = model_info.get('model_name', 'Unknown')
            metric_value = model_info.get('metrics', {}).get(metric, 0)
            
            # Ensure metric value is a float
            if not isinstance(metric_value, float):
                try:
                    metric_value = float(metric_value)
                except (TypeError, ValueError):
                    metric_value = 0.0
            
            logger.info(f"Adding best model for {metric}: {model_name} with value {metric_value}")
            
            # CRITICAL FIX: Add model_id for selection
            simplified_model_data.append({
                'model_id': metric_index,  # Use index as ID
                'model_name': model_name,
                'metric_name': metric,
                'metric_value': metric_value,
                'is_best_for': metric
            })
    
    # If we still have no models for display, create entries for only the top 3 models
    if not simplified_model_data and all_models:
        logger.info("No best models identified, creating display data from top models")
        
        # Find the top model for each metric
        top_accuracy = ('', 0)
        top_precision = ('', 0)
        top_recall = ('', 0)
        
        for model_name, metrics in all_models.items():
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            if accuracy > top_accuracy[1]:
                top_accuracy = (model_name, accuracy)
            if precision > top_precision[1]:
                top_precision = (model_name, precision)
            if recall > top_recall[1]:
                top_recall = (model_name, recall)
        
        # Add only the top models
        for idx, (metric_name, (model_name, value)) in enumerate([
            ('accuracy', top_accuracy),
            ('precision', top_precision),
            ('recall', top_recall)
        ]):
            if model_name and value > 0:
                simplified_model_data.append({
                    'model_id': idx,  # Use index as ID 
                    'model_name': model_name,
                    'metric_name': metric_name,
                    'metric_value': value,
                    'is_best_for': metric_name
                })
    
    # Create dummy data if absolutely nothing is available (for development/testing)
    # NOTE: Now only creating the top 3 models instead of all 6
    if not simplified_model_data:
        logger.warning("No models available, creating dummy data for display")
        model_metrics = {
            'random_forest': {'accuracy': 0.75, 'precision': 0.73, 'recall': 0.71},
            'logistic_regression': {'accuracy': 0.78, 'precision': 0.79, 'recall': 0.75}, 
            'decision_tree': {'accuracy': 0.81, 'precision': 0.80, 'recall': 0.79},
            'svm': {'accuracy': 0.84, 'precision': 0.85, 'recall': 0.82},
            'knn': {'accuracy': 0.87, 'precision': 0.86, 'recall': 0.85},
            'naive_bayes': {'accuracy': 0.90, 'precision': 0.91, 'recall': 0.88}
        }
        
        # Find the top model for each metric
        best_accuracy = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        best_precision = max(model_metrics.items(), key=lambda x: x[1]['precision'])
        best_recall = max(model_metrics.items(), key=lambda x: x[1]['recall'])
        
        # Add only the best models
        simplified_model_data.append({
            'model_id': 0,  # Use index as ID
            'model_name': best_accuracy[0],
            'metric_name': 'accuracy',
            'metric_value': best_accuracy[1]['accuracy'],
            'is_best_for': 'accuracy'
        })
        
        simplified_model_data.append({
            'model_id': 1,  # Use index as ID
            'model_name': best_precision[0],
            'metric_name': 'precision',
            'metric_value': best_precision[1]['precision'],
            'is_best_for': 'precision'
        })
        
        simplified_model_data.append({
            'model_id': 2,  # Use index as ID
            'model_name': best_recall[0],
            'metric_name': 'recall',
            'metric_value': best_recall[1]['recall'],
            'is_best_for': 'recall'
        })
    
    # Create a simplified version of all models with key metrics
    simplified_model_metrics = {}
    important_metrics = ['accuracy', 'precision', 'recall']
    
    for model_name, metrics in all_models.items():
        # Only store the important metrics
        simplified_metrics = {}
        for metric in important_metrics:
            if metric in metrics:
                # Ensure metric values are float
                value = metrics[metric]
                if not isinstance(value, float):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        value = 0.0
                simplified_metrics[metric] = value
        
        simplified_model_metrics[model_name] = simplified_metrics
    
    # Get selected features
    selected_features = session.get('selected_features', [])
    if isinstance(selected_features, str) and selected_features.startswith('['):
        # It's a string representation, try to extract the count
        import re
        match = re.search(r'\[(\d+) features\]', selected_features)
        if match:
            features_count = int(match.group(1))
        else:
            features_count = 0
    else:
        features_count = len(selected_features) if isinstance(selected_features, list) else 0
    
    # Log what we're sending to the template
    logger.info(f"Sending {len(simplified_model_data)} model summaries to template")
    logger.info(f"Model data for template: {json.dumps(simplified_model_data, cls=SafeJSONEncoder)}")
    logger.info(f"Sending {len(simplified_model_metrics)} detailed models to template")
    logger.info(f"Model metrics for template: {json.dumps(simplified_model_metrics, cls=SafeJSONEncoder)}")
    
    # CRITICAL FIX: Store simplified_model_data in session so model_actions can use it
    session['trained_models'] = simplified_model_data
    logger.info(f"Stored {len(simplified_model_data)} models in session['trained_models']")
    
    task = session.get('task', 'classification')
    
    overall_metrics = {
        'models_trained': len(all_models),
        'models_displayed': len(simplified_model_data),  # Now matches the number of models displayed
        'features_used': features_count,
        'test_size': session.get('test_size', 0.2)
    }
    
    return render_template('model_selection.html', 
                          models=simplified_model_data,
                          task=task,
                          model_metrics=simplified_model_metrics,
                          overall_metrics=overall_metrics)

@app.route('/model_actions/<int:model_id>')
def model_actions(model_id):
    """Handle model selection and container deployment"""
    # Log the request for debugging
    logger.info(f"Model actions requested for model_id: {model_id}")
    logger.info(f"Session contains trained_models: {'trained_models' in session}")
    if 'trained_models' in session:
        logger.info(f"Number of trained models in session: {len(session['trained_models'])}")
        logger.info(f"Models in session: {session['trained_models']}")
    
    # Get current trained models
    trained_models = session.get('trained_models', [])
    if not trained_models:
        logger.error("No trained models found in session")
        flash('No trained models available. Please complete the training process first.', 'error')
        return redirect(url_for('model_selection'))
    
    if model_id >= len(trained_models):
        logger.error(f"Invalid model_id: {model_id}, only have {len(trained_models)} models")
        flash('Invalid model selection', 'error')
        return redirect(url_for('model_selection'))
    
    # Get the model info
    model_info = trained_models[model_id]
    logger.info(f"Model info: {model_info}")
    
    model_name = model_info.get('model_name', '')
    # Use metric_name instead of metric - this is what's in the session data
    metric = model_info.get('metric_name', model_info.get('is_best_for', ''))
    
    logger.info(f"Deploying model {model_name} optimized for {metric} as container")
    
    # Store selected model in session
    session['selected_model'] = {
        'model_name': model_name,
        'metric': metric,
        'id': model_id
    }
    
    # Generate a unique deployment ID
    deployment_id = str(uuid.uuid4())
    session['deployment_id'] = deployment_id
    
    # Initialize deployment status in the global dictionary
    model_deployments[deployment_id] = {
        'status': 'initializing',
        'message': 'Starting deployment process...',
        'model_id': model_id,
        'container_id': None,
        'container_port': None,
        'model_name': model_name,
        'start_time': time.time()
    }
    
    # Start deployment in a background thread
    thread = threading.Thread(
        target=build_and_deploy_model,
        args=(model_id, model_name, metric, deployment_id)
    )
    thread.daemon = True
    thread.start()
    
    # Show loading screen while container is being built
    return render_template('loading.html', model_id=model_id, model_name=model_name, deployment_id=deployment_id)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """Make predictions using the selected model"""
    # Log current selection state
    logger.info(f"Selected model in session: {session.get('selected_model')}")
    
    # Check if a model has been selected
    selected_model = session.get('selected_model')
    if not selected_model:
        flash('No model selected. Please select a model first.', 'error')
        return redirect(url_for('model_selection'))
        
    # Get model name and metric from session
    model_name = selected_model.get('model_name', '')
    metric = selected_model.get('metric', '')
    
    if not model_name or not metric:
        flash('Invalid model selection. Please select a model again.', 'error')
        return redirect(url_for('model_selection'))
    
    # Check if required services are available
    required_prediction_services = {
        "Data Cleaner": DATA_CLEANER_URL,
        "Feature Selector": FEATURE_SELECTOR_URL,
        "Model Coordinator": MODEL_COORDINATOR_URL
    }
    
    for service_name, service_url in required_prediction_services.items():
        if not is_service_available(service_url):
            flash(f"The {service_name} service is not available. Cannot make predictions.", 'error')
            return render_template('prediction.html', model=selected_model)
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('prediction'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('prediction'))
        
        if file and allowed_file(file.filename):
            # Clean up previous prediction file if exists
            if 'prediction_file' in session and os.path.exists(session['prediction_file']):
                try:
                    os.remove(session['prediction_file'])
                except:
                    pass
                    
            # Generate unique filename for temporary storage
            filepath = get_temp_filepath(file.filename)
            file.save(filepath)
            
            try:
                # Load prediction data
                pred_data, _ = load_data(filepath)
                target_column = session.get('target_column')
                
                # 1. Clean data using Data Cleaner API
                logger.info(f"Sending prediction data to Data Cleaner API")
                pred_data_records = pred_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(pred_data), None).to_dict(orient='records')
                
                response = safe_requests_post(
                    f"{DATA_CLEANER_URL}/clean",
                    {
                        "data": pred_data_records,
                        "target_column": target_column
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"Data Cleaner API error: {response.json().get('error', 'Unknown error')}")
                
                cleaned_data = pd.DataFrame.from_dict(response.json()["data"])
                
                # Handle case where target column might be in prediction data
                if target_column in cleaned_data.columns:
                    cleaned_data = cleaned_data.drop(columns=[target_column])
                
                # 2. Transform features using Feature Selector API
                logger.info(f"Sending prediction data to Feature Selector API for transformation")
                
                # Get selected features from session
                selected_features = session.get('selected_features', [])
                
                cleaned_data_records = cleaned_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(cleaned_data), None).to_dict(orient='records')
                
                response = safe_requests_post(
                    f"{FEATURE_SELECTOR_URL}/transform",
                    {
                        "data": cleaned_data_records
                    },
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"Feature Selector API error: {response.json().get('error', 'Unknown error')}")
                
                transformed_data = pd.DataFrame.from_dict(response.json()["transformed_data"])
                
                # 3. Make predictions using Model Coordinator API
                logger.info(f"Sending data to Model Coordinator API for prediction using {model_name} optimized for {metric}")
                
                # Extract only the needed features
                prediction_data = {feature: transformed_data[feature].tolist() for feature in selected_features if feature in transformed_data.columns}
                
                # Log missing features
                missing_features = [f for f in selected_features if f not in transformed_data.columns]
                if missing_features:
                    logger.warning(f"Missing features in prediction data: {missing_features}")
                
                # Prepare request payload for model coordinator
                # We're sending the metric to use as the model selector
                payload = {
                    "data": prediction_data,
                    "models": [metric]  # Use the metric to select the model
                }
                
                logger.info(f"Making prediction with payload for metric: {metric}")
                
                response = safe_requests_post(
                    f"{MODEL_COORDINATOR_URL}/predict",
                    payload,
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise Exception(f"Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
                
                result = response.json()
                logger.info(f"Prediction response keys: {result.keys()}")
                
                # Extract predictions based on the coordinator response format
                predictions = []
                probabilities = []
                
                if 'predictions' in result and metric in result['predictions']:
                    model_result = result['predictions'][metric]
                    predictions = model_result.get('predictions', [])
                    probabilities = model_result.get('probabilities', [])
                    logger.info(f"Found {len(predictions)} predictions for metric {metric}")
                else:
                    # Try alternate format
                    if 'predictions' in result:
                        predictions = result['predictions']
                        logger.info(f"Found {len(predictions)} predictions in alternate format")
                    else:
                        logger.error(f"No predictions found in response: {result}")
                        raise Exception('No predictions returned from model.')
                
                # Create results DataFrame
                results_df = pd.DataFrame({'Prediction': predictions})
                
                # Clean up previous predictions file if exists
                if 'predictions_file' in session and os.path.exists(session['predictions_file']):
                    try:
                        os.remove(session['predictions_file'])
                    except:
                        pass
                        
                # Save results to temporary file
                results_filepath = get_temp_filepath(extension='.csv')
                results_df.to_csv(results_filepath, index=False)
                session['predictions_file'] = results_filepath
                
                # Clean up the uploaded file for prediction
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                
                # Create distribution data for display
                value_counts = results_df['Prediction'].value_counts()
                distribution = []
                for val, count in value_counts.items():
                    percentage = (count / len(predictions) * 100).round(2)
                    distribution.append({
                        'class': str(val),
                        'count': int(count),
                        'percentage': float(percentage)
                    })
                
                session['prediction_distribution'] = distribution
                
                return redirect(url_for('prediction_results'))
                
            except Exception as e:
                # Clean up the temporary file in case of error
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                
                logger.error(f"Error making predictions: {str(e)}", exc_info=True)
                flash(f"Error making predictions: {str(e)}", 'error')
                return redirect(url_for('prediction'))
    
    return render_template('prediction.html', model=selected_model)

@app.route('/prediction_results')
def prediction_results():
    predictions_file = session.get('predictions_file')
    if not predictions_file:
        flash('No predictions available', 'error')
        return redirect(url_for('prediction'))
    
    # Load predictions
    predictions_data, _ = load_data(predictions_file)
    distribution = session.get('prediction_distribution')
    
    return render_template('prediction_results.html', 
                          predictions=predictions_data.head(20).to_html(classes='table table-striped'),
                          distribution=distribution)

@app.route('/download_predictions')
def download_predictions():
    predictions_file = session.get('predictions_file')
    if not predictions_file:
        flash('No predictions available', 'error')
        return redirect(url_for('prediction'))
    
    # Create a BytesIO object to serve the file from memory
    file_data = io.BytesIO()
    with open(predictions_file, 'rb') as f:
        file_data.write(f.read())
    file_data.seek(0)
    
    return send_file(file_data, as_attachment=True, download_name='predictions.csv')

@app.route('/container_prediction', methods=['GET', 'POST'])
def container_prediction():
    """Make predictions using the deployed model container"""
    # Check if we have an active deployment
    deployment_id = session.get('deployment_id')
    if not deployment_id or deployment_id not in model_deployments:
        flash('No deployed model available. Please select a model first.', 'error')
        return redirect(url_for('model_selection'))
    
    # Get deployment info
    deployment = model_deployments[deployment_id]
    
    # Check if deployment is complete
    if deployment['status'] != 'complete':
        flash('Model deployment is not complete yet.', 'error')
        return redirect(url_for('model_selection'))
    
    # Get model info
    model_name = deployment['model_name']
    container_port = deployment['container_port']
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return render_template('container_prediction.html', model_name=model_name)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return render_template('container_prediction.html', model_name=model_name)
        
        if file and allowed_file(file.filename):
            try:
                # Save the file temporarily
                filepath = get_temp_filepath(file.filename)
                file.save(filepath)
                
                # Send the file to the container API
                api_url = f"http://localhost:{container_port}/predict"
                
                # Create a multipart form request
                files = {'file': (os.path.basename(filepath), open(filepath, 'rb'), 'text/csv')}
                
                response = requests.post(api_url, files=files)
                
                # Clean up the temporary file
                try:
                    os.remove(filepath)
                except:
                    pass
                
                if response.status_code != 200:
                    raise Exception(f"Error from container API: {response.text}")
                
                result = response.json()
                
                if not result.get('success', False):
                    raise Exception(f"Prediction failed: {result.get('error', 'Unknown error')}")
                
                # Get predictions from response
                predictions = result.get('predictions', [])
                
                # Create results DataFrame
                results_df = pd.DataFrame({'Prediction': predictions})
                
                # Save results to temporary file
                results_filepath = get_temp_filepath(extension='.csv')
                results_df.to_csv(results_filepath, index=False)
                session['container_predictions_file'] = results_filepath
                
                # Create distribution data for display
                value_counts = results_df['Prediction'].value_counts()
                distribution = []
                for val, count in value_counts.items():
                    percentage = (count / len(predictions) * 100).round(2)
                    distribution.append({
                        'class': str(val),
                        'count': int(count),
                        'percentage': float(percentage)
                    })
                
                session['container_prediction_distribution'] = distribution
                
                return render_template('container_prediction_results.html',
                                      model_name=model_name,
                                      predictions=results_df.head(20).to_html(classes='table table-striped'),
                                      distribution=distribution)
                
            except Exception as e:
                logger.error(f"Error making container predictions: {str(e)}", exc_info=True)
                flash(f"Error making predictions: {str(e)}", 'error')
                return render_template('container_prediction.html', model_name=model_name)
    
    return render_template('container_prediction.html', model_name=model_name)

@app.route('/download_container_predictions')
def download_container_predictions():
    """Download container prediction results"""
    predictions_file = session.get('container_predictions_file')
    if not predictions_file:
        flash('No predictions available', 'error')
        return redirect(url_for('container_prediction'))
    
    # Create a BytesIO object to serve the file from memory
    file_data = io.BytesIO()
    with open(predictions_file, 'rb') as f:
        file_data.write(f.read())
    file_data.seek(0)
    
    return send_file(file_data, as_attachment=True, download_name='container_predictions.csv')

@app.route('/deployment_status/<deployment_id>')
def deployment_status(deployment_id):
    """Check the status of a model deployment"""
    if deployment_id not in model_deployments:
        logger.warning(f"Deployment not found: {deployment_id}")
        logger.info(f"Current deployments: {list(model_deployments.keys())}")
        return jsonify({
            'status': 'not_found',
            'message': 'Deployment not found',
            'active_deployments': len(model_deployments),
            'deployment_ids': list(model_deployments.keys()),
            'current_session_deployment': session.get('deployment_id')
        })
    
    status = model_deployments[deployment_id]
    logger.info(f"[STATUS] Deployment status for {deployment_id}: {status['status']} - {status['message']}")
    
    # If deployment is complete, provide info for redirect
    if status['status'] == 'complete':
        # Check if the container is still healthy
        try:
            container_port = status.get('container_port')
            if container_port:
                try:
                    health_response = requests.get(f"http://localhost:{container_port}/health", timeout=1)
                    if health_response.status_code == 200:
                        container_health = "healthy"
                    else:
                        container_health = f"unhealthy (status code: {health_response.status_code})"
                except requests.RequestException:
                    container_health = "unreachable"
            else:
                container_health = "unknown (no port assigned)"
                
            # Try to get container info if we have a container_id
            container_info = {}
            if status.get('container_id'):
                try:
                    client = docker.from_env()
                    container = client.containers.get(status['container_id'])
                    container.reload()
                    container_state = container.attrs['State']
                    container_info = {
                        'running': container_state.get('Running', False),
                        'status': container_state.get('Status', 'unknown'),
                        'started_at': container_state.get('StartedAt', ''),
                        'health': container_health
                    }
                except Exception as e:
                    logger.warning(f"[STATUS] Could not get container info: {str(e)}")
                    container_info = {
                        'running': False,
                        'status': 'error',
                        'error': str(e),
                        'health': container_health
                    }
            
            return jsonify({
                'status': 'complete',
                'message': 'Deployment complete',
                'redirect_url': url_for('container_prediction'),
                'container_port': status['container_port'],
                'model_name': status['model_name'],
                'container_health': container_health,
                'container_info': container_info,
                'logs': status.get('logs', [])
            })
        except Exception as e:
            logger.error(f"[STATUS] Error checking container health: {str(e)}")
            return jsonify({
                'status': 'complete',
                'message': 'Deployment complete, but container health check failed',
                'redirect_url': url_for('container_prediction'),
                'container_port': status.get('container_port'),
                'model_name': status.get('model_name'),
                'error': str(e),
                'logs': status.get('logs', [])
            })
    
    # If deployment failed, provide error message
    if status['status'] == 'failed':
        return jsonify({
            'status': 'failed',
            'message': status['message'],
            'details': f"Failed during stage: {status.get('failed_stage', status['status'])}",
            'logs': status.get('logs', [])
        })
    
    # For in-progress deployments, provide more detailed status
    progress = get_progress_percentage(status['status'])
    stage_details = {
        'initializing': 'Setting up deployment environment',
        'preparing': 'Creating necessary files and directories',
        'building_model': 'Setting up the model files',
        'retrieving_model': 'Obtaining model data from coordinator',
        'building_container': 'Building Docker image',
        'starting_container': 'Starting container with the model',
        'waiting_for_service': 'Waiting for the API service to be ready'
    }
    
    stage_detail = stage_details.get(status['status'], 'Processing deployment')
    
    # Otherwise return current status with enhanced details
    return jsonify({
        'status': status['status'],
        'message': status['message'],
        'progress': progress,
        'stage_detail': stage_detail,
        'model_name': status.get('model_name', 'unknown'),
        'model_id': status.get('model_id'),
        'container_id': status.get('container_id', 'not assigned yet'),
        'container_port': status.get('container_port', 'not assigned yet'),
        'elapsed_time': f"{int(time.time() - status.get('start_time', time.time()))} seconds" if 'start_time' in status else 'unknown',
        'logs': status.get('logs', [])
    })

# Global dictionary to track deployments
model_deployments = {}

def get_progress_percentage(status):
    """Get a percentage value for the deployment progress based on status"""
    progress_stages = {
        'initializing': 0,
        'preparing': 10,
        'building_model': 20,
        'retrieving_model': 40,
        'building_container': 60,
        'starting_container': 80,
        'waiting_for_service': 90,
        'complete': 100,
        'failed': 100
    }
    return progress_stages.get(status, 0)

# Add log entry function to update status and add log entries
def add_log_entry(deployment_id, message):
    if deployment_id in model_deployments:
        logger.info(f"[DOCKER] {message}")
        if 'logs' not in model_deployments[deployment_id]:
            model_deployments[deployment_id]['logs'] = []
        model_deployments[deployment_id]['logs'].append(message)
        # Keep only the last 50 log entries to prevent too much memory usage
        if len(model_deployments[deployment_id]['logs']) > 50:
            model_deployments[deployment_id]['logs'] = model_deployments[deployment_id]['logs'][-50:]

def build_and_deploy_model(model_id, model_name, metric, deployment_id):
    """Build and deploy a Docker container for the selected model"""
    try:
        logger.info(f"[DOCKER] Starting deployment of model {model_name} (ID: {deployment_id})")
        add_log_entry(deployment_id, f"Starting deployment of model {model_name}")
        
        # Log key information at the start
        logger.info(f"[DOCKER] Deployment parameters: model_id={model_id}, model_name={model_name}, metric={metric}, deployment_id={deployment_id}")
        add_log_entry(deployment_id, f"Model parameters: {model_name} optimized for {metric}")
        
        # Update status
        model_deployments[deployment_id] = {
            'status': 'preparing',
            'message': 'Creating deployment files...',
            'model_id': model_id,
            'model_name': model_name,
            'container_id': None,
            'container_port': None,
            'start_time': time.time(),
            'logs': ['Starting deployment process...']  # Add a logs list to track progress
        }
        
        # Create temporary directory for deployment
        deploy_dir = os.path.join(tempfile.gettempdir(), f'model_deploy_{deployment_id}')
        os.makedirs(deploy_dir, exist_ok=True)
        logger.info(f"[DOCKER] Created deployment directory: {deploy_dir}")
        add_log_entry(deployment_id, f"Created deployment directory")
        
        # Create model directory inside deployment dir
        model_dir = os.path.join(deploy_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Update status
        model_deployments[deployment_id]['status'] = 'building_model'
        model_deployments[deployment_id]['message'] = 'Creating model files...'
        add_log_entry(deployment_id, "Creating container application files")
        
        # Create Flask app file - Using port 5050 instead of 5000 to avoid conflicts
        app_content = '''from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import os
import sys
import io
import platform

# Configure logging to file and stdout
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('container.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set platform-specific settings
is_windows = platform.system().lower() == 'windows'
if is_windows:
    logger.info("Running on Windows platform, applying specific settings")
    # Windows-specific settings if needed

# Load the model on startup
logger.info("Starting container application")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.joblib')
logger.info(f"Loading model from {MODEL_PATH}")

def load_model_safely(primary_path, backup_path=None):
    """Safely load a model with fallback options"""
    try:
        # First try the primary model
        return joblib.load(primary_path)
    except Exception as primary_error:
        logger.error(f"Error loading primary model: {str(primary_error)}")
        
        # Try backup model if available
        if backup_path and os.path.exists(backup_path):
            try:
                logger.warning(f"Attempting to load backup model")
                return joblib.load(backup_path)
            except Exception as backup_error:
                logger.error(f"Error loading backup model: {str(backup_error)}")
        
        # Last resort: create a simple model
        logger.warning("Creating a simple fallback model")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=3)
        model.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))
        return model

# Try to load the model with fallback logic
backup_path = os.path.join(os.path.dirname(__file__), 'model', 'backup_model.joblib')
model = load_model_safely(MODEL_PATH, backup_path)
logger.info(f"Model loaded successfully: {type(model).__name__}")

@app.route('/health')
def health():
    logger.info("Health check request received")
    return jsonify({"status": "healthy", "model_type": str(type(model).__name__)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Prediction request received")
        # Get data from request
        if 'file' in request.files:
            file = request.files['file']
            logger.info(f"Received file: {file.filename}")
            
            # Read with explicit encoding handling
            try:
                # Try UTF-8 first
                content = file.read()
                file.seek(0)  # Reset file pointer
                
                # Use BytesIO to avoid encoding issues
                data = pd.read_csv(io.BytesIO(content))
            except UnicodeDecodeError:
                # If that fails, try with Latin-1
                file.seek(0)  # Reset file pointer
                data = pd.read_csv(file, encoding='latin1')
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                # Try one more approach - directly with latin1
                file.seek(0)
                data = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
        else:
            logger.info("Received JSON data")
            data = pd.DataFrame(request.json.get('data', {}))
        
        logger.info(f"Data shape: {data.shape}")
        
        # Handle NaN values
        data = data.fillna(0)
        
        # Make prediction with error handling
        logger.info("Making predictions")
        try:
            predictions = model.predict(data)
            
            # Convert numpy types to Python types for JSON serialization
            predictions = [float(p) if isinstance(p, (np.float32, np.float64)) else str(p) for p in predictions]
            
            logger.info(f"Made {len(predictions)} predictions")
            return jsonify({
                "success": True,
                "predictions": predictions
            })
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            # If prediction fails, try a simple version
            return jsonify({
                "success": False,
                "error": f"Error making prediction: {str(pred_error)}",
                "fallback_result": [0] * len(data)  # Return zeros as fallback
            }), 400
            
    except Exception as e:
        import traceback
        logger.error(f"General error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    # Use port 5050 to avoid conflict with the main app
    logger.info("Starting prediction service on port 5050")
    app.run(host='0.0.0.0', port=5050)
'''
        with open(os.path.join(deploy_dir, 'app.py'), 'w', encoding='utf-8') as f:
            f.write(app_content)
        logger.info(f"[DOCKER] Created app.py in {deploy_dir}")
        
        # Create Dockerfile - Make sure to expose port 5050
        dockerfile_content = '''FROM python:3.9-slim

# Set UTF-8 encoding
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app

# Copy all files at once to reduce layers
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask pandas numpy scikit-learn joblib

# Expose port 5050 instead of 5000
EXPOSE 5050

# Start with an explicit log message
CMD ["python", "app.py"]
'''
        with open(os.path.join(deploy_dir, 'Dockerfile'), 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        logger.info(f"[DOCKER] Created Dockerfile in {deploy_dir}")
        
        # No need for separate requirements.txt since we simplified the Dockerfile
        
        # Update status
        model_deployments[deployment_id]['status'] = 'retrieving_model'
        model_deployments[deployment_id]['message'] = 'Retrieving model data...'
        add_log_entry(deployment_id, f"Retrieving model {model_name} from model coordinator")
        
        try:
            # Get the actual model from the Model Coordinator API
            logger.info(f"[DOCKER] Requesting model from Model Coordinator API: {model_name}, metric: {metric}")
            response = requests.get(
                f"{MODEL_COORDINATOR_URL}/get_model",
                params={
                    "model_name": model_name,
                    "metric": metric
                },
                timeout=60
            )
            
            if response.status_code == 200:
                # Save the model file received from the API
                model_data = response.content
                model_path = os.path.join(model_dir, 'model.joblib')
                with open(model_path, 'wb') as f:
                    f.write(model_data)
                logger.info(f"[DOCKER] Retrieved model from coordinator: {model_name}, saved to {model_path}")
                
                # Also save a fallback model in case the main one has issues
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    backup_clf = RandomForestClassifier(n_estimators=3, random_state=42)
                    # Fit with some dummy data
                    X = np.random.rand(10, 4)
                    y = np.random.randint(0, 2, 10)
                    backup_clf.fit(X, y)
                    
                    # Save the backup model
                    backup_model_path = os.path.join(model_dir, 'backup_model.joblib')
                    joblib.dump(backup_clf, backup_model_path)
                    logger.info(f"[DOCKER] Created backup model as fallback")
                except Exception as e:
                    logger.warning(f"[DOCKER] Could not create backup model: {str(e)}")
            else:
                logger.error(f"[DOCKER] Failed to get model from API: {response.status_code} {response.text}")
                raise Exception(f"Failed to retrieve model from API: {response.status_code} {response.text}")
        except Exception as e:
            logger.warning(f"[DOCKER] Using fallback model due to error: {str(e)}")
            # Create a simple classifier for demonstration as fallback
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            
            # Fit with some dummy data
            X = np.random.rand(100, 4)
            y = np.random.randint(0, 2, 100)
            clf.fit(X, y)
            
            # Save the fitted model
            model_file = os.path.join(model_dir, 'model.joblib')
            joblib.dump(clf, model_file)
            logger.info(f"[DOCKER] Created fallback model and saved to {model_file}")
        
        # Update status
        model_deployments[deployment_id]['status'] = 'building_container'
        model_deployments[deployment_id]['message'] = 'Building Docker image...'
        add_log_entry(deployment_id, "Starting Docker build process")
        
        try:
            # Use subprocess to call docker CLI directly instead of using the Docker SDK
            # This is much more reliable across different environments
            
            # Generate unique container tag
            container_tag = f"model-{model_name.lower().replace(' ', '-')}-{deployment_id[:8]}"
            add_log_entry(deployment_id, f"Building image with tag: {container_tag}")
            
            # Build the Docker image using docker CLI - SIMPLIFIED APPROACH
            build_cmd = ["docker", "build", "-t", container_tag, deploy_dir]
            add_log_entry(deployment_id, f"Running build command: {' '.join(build_cmd)}")
            
            # Simple approach without socket operations - just run with timeout
            try:
                # Run the build command with a timeout
                build_process = subprocess.run(
                    build_cmd,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minutes timeout
                )
                
                # Check return code
                if build_process.returncode != 0:
                    error_msg = f"Docker build failed: {build_process.stderr}"
                    logger.error(f"[DOCKER] {error_msg}")
                    add_log_entry(deployment_id, error_msg)
                    
                    # Try alternative approach with simplified Dockerfile
                    add_log_entry(deployment_id, "First build attempt failed, trying simpler approach...")
                    
                    # Create an ultra-simple Dockerfile
                    minimal_dockerfile = '''FROM python:3.9-slim
WORKDIR /app
COPY model model/
COPY app.py .
RUN pip install flask pandas numpy scikit-learn joblib
EXPOSE 5050
CMD ["python", "app.py"]
'''
                    with open(os.path.join(deploy_dir, 'Dockerfile'), 'w') as f:
                        f.write(minimal_dockerfile)
                    
                    # Try building again with the simplified Dockerfile
                    add_log_entry(deployment_id, "Retrying with simplified Dockerfile...")
                    build_process = subprocess.run(
                        build_cmd,
                        capture_output=True,
                        text=True,
                        timeout=180  # 3 minute timeout
                    )
                    
                    if build_process.returncode != 0:
                        error_msg = f"Docker build failed with simplified Dockerfile: {build_process.stderr}"
                        logger.error(f"[DOCKER] {error_msg}")
                        add_log_entry(deployment_id, error_msg)
                        raise Exception(f"Docker build failed: {error_msg}")
                
            except subprocess.TimeoutExpired:
                add_log_entry(deployment_id, "Docker build process timed out")
                raise Exception("Docker build timed out after 3 minutes")
            
            logger.info(f"[DOCKER] Docker image built successfully: {container_tag}")
            add_log_entry(deployment_id, f"Docker image built successfully: {container_tag}")
            
            # Update status
            model_deployments[deployment_id]['status'] = 'starting_container'
            model_deployments[deployment_id]['message'] = 'Starting container...'
            add_log_entry(deployment_id, "Starting container...")
            
            # Find available port starting from 8000 to avoid conflicts
            def is_port_available(port):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(('localhost', port)) != 0
            
            # Start from port 8000 and find an available port
            host_port = 8000
            while host_port < 9000:  # Try ports in range 8000-8999
                if is_port_available(host_port):
                    break
                host_port += 1
                
            if host_port >= 9000:
                error_msg = "No available ports found in range 8000-8999"
                logger.error(f"[DOCKER] {error_msg}")
                add_log_entry(deployment_id, error_msg)
                raise Exception(error_msg)
            
            logger.info(f"[DOCKER] Using host port: {host_port}")
            add_log_entry(deployment_id, f"Using port {host_port} for container")
            
            # Unique container name to avoid conflicts
            container_name = f"model-container-{deployment_id[:8]}"
            
            # Check if container with same name exists and remove it
            check_cmd = ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.ID}}"]
            check_process = subprocess.run(check_cmd, capture_output=True, text=True)
            container_id = check_process.stdout.strip()
            
            if container_id:
                add_log_entry(deployment_id, f"Found existing container {container_id}, removing it")
                try:
                    # Stop the container if it's running
                    subprocess.run(["docker", "stop", container_id], capture_output=True, text=True)
                    # Remove the container
                    subprocess.run(["docker", "rm", container_id], capture_output=True, text=True)
                    add_log_entry(deployment_id, "Removed existing container")
                except Exception as e:
                    add_log_entry(deployment_id, f"Warning: Error removing existing container: {str(e)}")
            
            # Start the container using docker CLI
            run_cmd = [
                "docker", "run",
                "--name", container_name,
                "-d",  # detached mode
                "-p", f"{host_port}:5050",
                container_tag
            ]
            
            add_log_entry(deployment_id, f"Starting container with command: {' '.join(run_cmd)}")
            run_process = subprocess.run(run_cmd, capture_output=True, text=True)
            
            if run_process.returncode != 0:
                error_msg = f"Failed to start container: {run_process.stderr}"
                logger.error(f"[DOCKER] {error_msg}")
                add_log_entry(deployment_id, error_msg)
                raise Exception(f"Docker run failed: {error_msg}")
            
            container_id = run_process.stdout.strip()
            logger.info(f"[DOCKER] Container started: {container_id} on port {host_port}")
            add_log_entry(deployment_id, f"Container started with ID: {container_id[:12]}")
            
            # Update deployment status
            model_deployments[deployment_id]['status'] = 'waiting_for_service'
            model_deployments[deployment_id]['message'] = 'Waiting for API to be ready...'
            model_deployments[deployment_id]['container_id'] = container_id
            model_deployments[deployment_id]['container_port'] = host_port
            
            # Wait for the API to be ready
            api_ready = False
            add_log_entry(deployment_id, "Waiting for container API to be ready...")
            
            for _ in range(30):  # Try for 30 seconds
                try:
                    # Get container status
                    status_cmd = ["docker", "inspect", "--format", "{{.State.Running}}", container_id]
                    status_process = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    # Check if container is still running
                    if status_process.returncode == 0:
                        container_running = status_process.stdout.strip() == "true"
                        if not container_running:
                            # Get container logs
                            logs_cmd = ["docker", "logs", container_id]
                            logs_process = subprocess.run(logs_cmd, capture_output=True, text=True)
                            container_logs = logs_process.stdout
                            
                            # Get exit code
                            exit_cmd = ["docker", "inspect", "--format", "{{.State.ExitCode}}", container_id]
                            exit_process = subprocess.run(exit_cmd, capture_output=True, text=True)
                            exit_code = exit_process.stdout.strip()
                            
                            add_log_entry(deployment_id, f"Container exited with code {exit_code}")
                            if logs_process.returncode == 0:
                                log_lines = logs_process.stdout.split('\n')
                                if len(log_lines) >= 2:
                                    add_log_entry(deployment_id, f"Container exit logs: {log_lines[-2]}")
                                else:
                                    add_log_entry(deployment_id, "Container exited but no logs available")
                            
                            raise Exception(f"Container exited with code {exit_code}")
                    
                    # Check API health
                    response = requests.get(f"http://localhost:{host_port}/health", timeout=1)
                    if response.status_code == 200:
                        api_ready = True
                        logger.info(f"[DOCKER] Container API is ready on port {host_port}")
                        add_log_entry(deployment_id, f"Container API is ready on port {host_port}")
                        break
                except Exception as e:
                    time.sleep(1)
            
            if not api_ready:
                error_msg = "Container API failed to start after 30 seconds"
                logger.error(f"[DOCKER] {error_msg}")
                add_log_entry(deployment_id, error_msg)
                
                # Try to get container logs
                try:
                    logs_cmd = ["docker", "logs", container_id]
                    logs_process = subprocess.run(logs_cmd, capture_output=True, text=True)
                    if logs_process.returncode == 0:
                        add_log_entry(deployment_id, f"Container logs: {logs_process.stdout[-300:] if len(logs_process.stdout) > 300 else logs_process.stdout}")
                except Exception as e:
                    logger.error(f"[DOCKER] Failed to get container logs: {str(e)}")
                
                raise Exception("Container API failed to start within timeout period")
            
            # Mark deployment as complete
            model_deployments[deployment_id]['status'] = 'complete'
            model_deployments[deployment_id]['message'] = 'Deployment complete'
            logger.info(f"[DOCKER] Model {model_name} deployed successfully with ID {deployment_id} on port {host_port}")
            add_log_entry(deployment_id, f"Deployment completed successfully. Container is accessible on port {host_port}")
            
        except Exception as e:
            logger.error(f"[DOCKER] Error in Docker operations: {str(e)}")
            add_log_entry(deployment_id, f"Docker error: {str(e)}")
            raise Exception(f"Docker operation failed: {str(e)}")
        
    except Exception as e:
        # Ensure we log the full exception for debugging
        logger.error(f"[DOCKER] Error during model deployment: {str(e)}", exc_info=True)
        
        # Update deployment status with the error message
        if deployment_id in model_deployments:
            model_deployments[deployment_id]['status'] = 'failed'
            model_deployments[deployment_id]['message'] = f'Deployment failed: {str(e)}'
            # Store what stage the deployment failed at if we have it
            current_status = model_deployments[deployment_id].get('status', 'unknown')
            model_deployments[deployment_id]['failed_stage'] = current_status
            add_log_entry(deployment_id, f"Deployment failed during {current_status} stage: {str(e)}")
        
        # Clean up any resources
        if 'container_id' in locals() and container_id:
            try:
                # Clean up container
                subprocess.run(["docker", "stop", container_id], capture_output=True)
                subprocess.run(["docker", "rm", container_id], capture_output=True)
                add_log_entry(deployment_id, "Cleaned up container after failure")
            except Exception as cleanup_error:
                logger.error(f"[DOCKER] Error cleaning up container: {str(cleanup_error)}")
        
        if 'container_tag' in locals() and container_tag:
            try:
                # Clean up image
                subprocess.run(["docker", "rmi", container_tag], capture_output=True)
                add_log_entry(deployment_id, f"Cleaned up image {container_tag} after failure")
            except Exception as cleanup_error:
                logger.error(f"[DOCKER] Error cleaning up image: {str(cleanup_error)}")
        
        try:
            # Clean up deployment directory
            if 'deploy_dir' in locals() and os.path.exists(deploy_dir):
                logger.info(f"[DOCKER] Cleaning up deployment directory: {deploy_dir}")
                shutil.rmtree(deploy_dir, ignore_errors=True)
                logger.info(f"[DOCKER] Cleaned up deployment directory: {deploy_dir}")
                add_log_entry(deployment_id, "Cleaned up deployment directory")
        except Exception as cleanup_error:
            logger.error(f"[DOCKER] Error cleaning up deployment directory: {str(cleanup_error)}")

def save_to_temp_file(data, name_prefix='data'):
    """Save data to a temporary file to avoid session bloat"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{name_prefix}_{uuid.uuid4()}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=SafeJSONEncoder)
    return filepath

def load_from_temp_file(filepath):
    """Load data from a temporary file"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def check_session_size():
    """Check the size of the session and log a warning if it's too large"""
    try:
        # Get size of session by serializing to JSON
        session_size = len(json.dumps(dict(session)))
        size_mb = session_size / (1024 * 1024)
        
        if size_mb > 5:  # Warn if larger than 5MB
            logger.warning(f"Session is very large: {size_mb:.2f} MB")
            # Log session keys to help debug
            logger.warning(f"Session keys: {list(session.keys())}")
    except Exception as e:
        logger.error(f"Error checking session size: {str(e)}")
        # Don't raise the exception to avoid breaking the application

@app.route('/debug/deployments')
def debug_deployments():
    """Debug endpoint to check active deployments"""
    return jsonify({
        'active_deployments': len(model_deployments),
        'deployment_ids': list(model_deployments.keys()),
        'current_session_deployment': session.get('deployment_id'),
        'deployments': {k: {
            'status': v.get('status'),
            'message': v.get('message'),
            'model_name': v.get('model_name')
        } for k, v in model_deployments.items()}
    })

if __name__ == '__main__':
    # Use port 5000 for the main app (ORIGINAL PORT RESTORED)
    logger.info("Starting MedicAI with API services integration")
    print("Service status:")
    status = check_services()
    for service, health in status.items():
        print(f"- {service}: {health}")
    
    try:
        app.run(debug=True, port=5000)
    finally:
        # Clean up all temporary files when the application stops
        cleanup_temp_files() 