from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash, make_response
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
import secrets  # Import for generating secure tokens
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models.users import db, User

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
app.config['SESSION_PERMANENT'] = False  # Changed from True to False
app.config['REMEMBER_COOKIE_DURATION'] = None  # Don't remember user login
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24  # 24 hours

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepmedver.db'  # Using SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

# Initialize login manager before the routes
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'
login_manager.session_protection = None  # Disable session protection completely

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID"""
    return User.query.get(int(user_id))

# Add context processor to make current_user available in all templates
@app.context_processor
def inject_user():
    context = {'current_user': current_user}
    
    # Add logout token if user is authenticated
    if current_user.is_authenticated and 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    if current_user.is_authenticated and 'logout_token' in session:
        context['logout_token'] = session['logout_token']
    
    return context

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
    """Root route - always check authentication first"""
    # First, check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the application.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Only clear data-related keys, but preserve authentication
    data_keys = ['uploaded_file', 'cleaned_file', 'selected_features_file', 
                'predictions_file', 'file_stats', 'data_columns']
    
    for key in list(session.keys()):
        if key in data_keys:
            session.pop(key)
    
    # Clean up any files from previous sessions
    cleanup_session_files()
    
    # Check services health for status display
    services_status = check_services()
    return render_template('index.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    # Double check authentication - ensure user is logged in
    if not current_user.is_authenticated:
        logger.warning("Upload attempted without authentication")
        flash('Please log in to upload files.', 'warning')
        return redirect(url_for('login'))
        
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
@login_required
def training():
    filepath = session.get('uploaded_file')
    
    if not filepath:
        # If accessed directly without upload, show a friendly message
        flash('Please upload a file first to start training.', 'info')
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
            X_data = {feature: X_selected[feature].tolist() for feature in selected_features}
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
@login_required
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
@login_required
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
    
    # Only include the model with best accuracy
    if best_models['accuracy'].get('model_name'):  # Only include model with a name
        model_name = best_models['accuracy'].get('model_name', 'Unknown')
        metric_value = best_models['accuracy'].get('metrics', {}).get('accuracy', 0)
        
        # Ensure metric value is a float
        if not isinstance(metric_value, float):
            try:
                metric_value = float(metric_value)
            except (TypeError, ValueError):
                metric_value = 0.0
        
        logger.info(f"Adding best accuracy model: {model_name} with value {metric_value}")
        
        simplified_model_data.append({
            'model_name': model_name,
            'metric_name': 'accuracy',
            'metric_value': metric_value,
            'is_best_for': 'accuracy'
        })
    
    # If we still have no models for display, create entries for only the best accuracy model
    if not simplified_model_data and all_models:
        logger.info("No best models identified, creating display data from top model")
        
        # Find the top model by accuracy
        top_accuracy = ('', 0)
        
        for model_name, metrics in all_models.items():
            accuracy = metrics.get('accuracy', 0)
            
            if accuracy > top_accuracy[1]:
                top_accuracy = (model_name, accuracy)
        
        # Add only the top accuracy model
        model_name, value = top_accuracy
        if model_name and value > 0:
            simplified_model_data.append({
                'model_name': model_name,
                'metric_name': 'accuracy',
                'metric_value': value,
                'is_best_for': 'accuracy'
            })
    
    # Create dummy data if absolutely nothing is available (for development/testing)
    # NOTE: Now only creating the best accuracy model instead of top 3
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
        
        # Find the top model by accuracy
        best_accuracy = max(model_metrics.items(), key=lambda x: x[1]['accuracy'])
        
        # Add only the best accuracy model
        simplified_model_data.append({
            'model_name': best_accuracy[0],
            'metric_name': 'accuracy',
            'metric_value': best_accuracy[1]['accuracy'],
            'is_best_for': 'accuracy'
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

@app.route('/select_model/<model_name>/<metric>')
@login_required
def select_model(model_name, metric):
    """Select a model for predictions"""
    # Check if we have model results
    if 'model_results' not in session:
        flash('No trained models available. Please train models first.', 'error')
        return redirect(url_for('training'))
    
    # Log the selection
    logger.info(f"Selecting model {model_name} optimized for {metric}")
    
    # Store selected model details in session
    session['selected_model'] = {
        'model_name': model_name,
        'metric': metric
    }
    
    # Log success
    logger.info(f"Successfully selected model {model_name} with metric {metric}")
    
    flash(f'Selected {model_name} model optimized for {metric}', 'success')
    return redirect(url_for('prediction'))

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
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
@login_required
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
@login_required
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

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    # Check if Medical Assistant API is available
    if not is_service_available(MEDICAL_ASSISTANT_URL):
        flash('Medical Assistant service is not available.', 'error')
        return render_template('chat.html', messages=[])
    
    # Initialize chat history in session if not present
    if 'messages' not in session:
        session['messages'] = []
    
    if request.method == 'POST':
        prompt = request.form.get('prompt')
        if prompt:
            # Add user message to chat history
            session['messages'].append({
                'role': 'user',
                'content': prompt
            })
            
            # Get AI response via API
            try:
                response = safe_requests_post(
                    f"{MEDICAL_ASSISTANT_URL}/chat",
                    {
                        "message": prompt,
                        "session_id": f"session_{id(session)}"  # Create a unique session ID
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    ai_response = response.json()["response"]
                    
                    # Add assistant response to chat history
                    session['messages'].append({
                        'role': 'assistant',
                        'content': ai_response
                    })
                    # Add logging to verify data being sent to APIs
                    logger.info(f"Data being sent to Medical Assistant Chat API: {prompt}")
                else:
                    flash(f"Error communicating with AI assistant: {response.text}", 'error')
                
            except Exception as e:
                logger.error(f"Error communicating with AI assistant: {str(e)}", exc_info=True)
                flash(f"Error communicating with AI assistant: {str(e)}", 'error')
    
    return render_template('chat.html', messages=session.get('messages', []))

@app.route('/clear_chat')
@login_required
def clear_chat():
    if 'messages' in session:
        session.pop('messages')
        
        # Also clear on the API side
        if is_service_available(MEDICAL_ASSISTANT_URL):
            try:
                safe_requests_post(
                    f"{MEDICAL_ASSISTANT_URL}/clear_chat",
                    {"session_id": f"session_{id(session)}"},
                    timeout=10
                )
            except:
                pass  # Ignore errors in clearing remote chat history
            
    return redirect(url_for('chat'))

@app.route('/service_status')
def service_status():
    """Check the status of all services"""
    services_status = check_services()
    
    # If model coordinator is available, also check model services
    if services_status.get("Model Coordinator") == "healthy":
        try:
            response = requests.get(f"{MODEL_COORDINATOR_URL}/health", timeout=5)
            if response.status_code == 200:
                coordinator_data = response.json()
                if "model_services" in coordinator_data:
                    for model_name, status in coordinator_data["model_services"].items():
                        services_status[f"Model - {model_name}"] = "healthy" if status == "healthy" else "unhealthy"
        except Exception as e:
            logger.error(f"Error getting model service status: {str(e)}")
    
    return jsonify(services_status)

# Function to save data to a temporary file
def save_to_temp_file(data, prefix='data'):
    """Save data to a temporary file and return the filepath"""
    filepath = get_temp_filepath(extension='.json')
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=SafeJSONEncoder)
    return filepath

# Function to load data from a temporary file
def load_from_temp_file(filepath):
    """Load data from a temporary file"""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading data from file {filepath}: {str(e)}")
        return None

# Add a helper function to estimate session size
def get_session_size():
    """Estimate the current session size in bytes"""
    try:
        import pickle
        import sys
        
        # Try to pickle the session data to estimate its size
        session_copy = dict(session)
        pickled = pickle.dumps(session_copy)
        size = sys.getsizeof(pickled)
        logger.info(f"Current session size estimate: {size / 1024:.2f} KB")
        return size
    except Exception as e:
        logger.error(f"Error estimating session size: {str(e)}")
        return 0

# Add a function to monitor and trim session if needed
def check_session_size(max_size=3000000):  # ~3MB limit
    """Check if session is too large and trim it if needed"""
    size = get_session_size()
    if size > max_size:
        logger.warning(f"Session size ({size/1024:.2f} KB) exceeds limit ({max_size/1024:.2f} KB). Moving data to files.")
        
        # Move large data to files
        large_keys = ['ai_recommendations', 'anomaly_results', 'raw_model_result', 'models', 'all_models', 'feature_importance']
        for key in large_keys:
            if key in session and session[key]:
                try:
                    # Save to file and store only the path
                    data = session[key]
                    file_path = save_to_temp_file(data, key)
                    session[f"{key}_file"] = file_path
                    # Remove the large data from session
                    session.pop(key)
                    logger.info(f"Moved {key} to file: {file_path}")
                except Exception as e:
                    logger.error(f"Error moving {key} to file: {str(e)}")
        
        # Check if we need to move feature data too
        if 'selected_features' in session and len(session['selected_features']) > 100:
            selected_features = session['selected_features']
            file_path = save_to_temp_file(selected_features, 'selected_features')
            session['selected_features_file_json'] = file_path
            session['selected_features'] = f"[{len(selected_features)} features saved to file]"
            logger.info(f"Moved selected_features to file: {file_path}")
            
        logger.info(f"Session size after optimization: {get_session_size() / 1024:.2f} KB")

@app.route('/login', methods=['GET', 'POST'])
@app.route('/login/<path:action>', methods=['GET', 'POST'])
def login(action=None):
    """User login page with action parameter to handle different cases"""
    logger.info(f"Login route accessed with method: {request.method}, action: {action}")
    
    # Force showing login page if action is 'force'
    if action == 'force':
        logger.info("Force login page display requested")
        logout_user()
        session.clear()
        flash('You have been logged out. Please log in again.', 'info')
        return render_template('login.html')
    
    # If already logged in, redirect to index
    if current_user.is_authenticated:
        logger.info("User already authenticated, redirecting to index")
        return redirect('/')
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user exists and verify password
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            # Never set permanent session
            session.permanent = False
            
            # Log in user but DO NOT remember them
            login_user(user, remember=False)
            
            # Flash a success message
            flash('Login successful!', 'success')
            
            # Simple redirect to index
            return redirect('/')
        else:
            flash('Invalid email or password.', 'danger')
    
    # Simple template response
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    # Check if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        name = request.form.get('name', '')
        # Split name into first and last name
        name_parts = name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please login or use a different email.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(email=email, first_name=first_name, last_name=last_name)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating account: {str(e)}', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@login_manager.unauthorized_handler
def unauthorized():
    """Most direct unauthorized handler possible"""
    # Clear all session data
    session.clear()
    # Add message
    flash('Please log in to access this page.', 'warning')
    # Direct redirect to login
    return redirect('/login')

@app.route('/force_logout', methods=['POST'])
def force_logout():
    """Emergency force logout route"""
    logger.info("Emergency force logout triggered")
    
    # First, perform normal logout actions
    if current_user.is_authenticated:
        logout_user()
    
    # Clear the entire session
    session.clear()
    
    # Set response
    response = make_response('')
    
    # Destroy all cookies
    for cookie in request.cookies:
        response.delete_cookie(cookie)
    
    # Set no-cache headers
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    # Return an empty response since the JavaScript will handle the redirect
    return response

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """User logout with absolute most direct approach possible"""
    # Add debug logging
    logger.info(f"Logout route accessed with method: {request.method}")
    
    # Perform logout actions
    logout_user()
    session.clear()
    flash('You have been logged out.', 'info')
    
    # Different approach based on request method
    if request.method == 'POST':
        # For POST requests (form submission), use 303 See Other to force GET on redirect
        return redirect('/login', code=303)
    else:
        # For GET requests, use the JavaScript approach as fallback
        response = make_response("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/login">
            <title>Logging out...</title>
            <script>
                window.location.href = "/login";
            </script>
        </head>
        <body>
            <p>Logging out... <a href="/login">Click here</a> if you are not redirected.</p>
        </body>
        </html>
        """)
        
        # Add headers to prevent caching
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response

if __name__ == '__main__':
    print("Starting MedicAI with API services integration")
    print("Service status:")
    status = check_services()
    for service, health in status.items():
        print(f"- {service}: {health}")
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        print("Database tables created/verified")
    
    try:
        app.run(debug=True, port=5000)
    finally:
        # Clean up all temporary files when the application stops
        cleanup_temp_files() 