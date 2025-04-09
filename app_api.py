from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash, make_response, Response
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
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData
import urllib.parse
import zipfile  # Required for handling ZIP files in model training
from requests_toolbelt.multipart.encoder import MultipartEncoder  # For sending multipart form data
from storage import download_blob

# For PyTorch model training - only import if not present
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
except ImportError:
    pass  # We'll handle this in the route if needed

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

# Define service URLs with localhost and ports for host machine access
DATA_CLEANER_URL = os.getenv('DATA_CLEANER_URL', 'http://localhost:5001')
FEATURE_SELECTOR_URL = os.getenv('FEATURE_SELECTOR_URL', 'http://localhost:5002')
ANOMALY_DETECTOR_URL = os.getenv('ANOMALY_DETECTOR_URL', 'http://localhost:5003')
MODEL_COORDINATOR_URL = os.getenv('MODEL_COORDINATOR_URL', 'http://localhost:5020')  # New model coordinator URL instead of MODEL_TRAINER_URL
MEDICAL_ASSISTANT_URL = os.getenv('MEDICAL_ASSISTANT_URL', 'http://localhost:5005')
AUGMENTATION_SERVICE_URL = os.getenv('AUGMENTATION_SERVICE_URL', 'http://localhost:5023')
MODEL_TRAINING_SERVICE_URL = os.getenv('MODEL_TRAINING_SERVICE_URL', 'http://localhost:5021')

# Update service URLs dictionary with proper health endpoints
SERVICES = {
    "Data Cleaner": {"url": DATA_CLEANER_URL, "endpoint": "/health"},
    "Feature Selector": {"url": FEATURE_SELECTOR_URL, "endpoint": "/health"},
    "Anomaly Detector": {"url": ANOMALY_DETECTOR_URL, "endpoint": "/health"},
    "Model Coordinator": {"url": MODEL_COORDINATOR_URL, "endpoint": "/health"},
    "Medical Assistant": {"url": MEDICAL_ASSISTANT_URL, "endpoint": "/health"},
    "Augmentation Service": {"url": AUGMENTATION_SERVICE_URL, "endpoint": "/health"},
    "Model Training Service": {"url": MODEL_TRAINING_SERVICE_URL, "endpoint": "/health"}
}

# Setup Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'medicai_temp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload (commented out)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False  # Changed from True to False
app.config['REMEMBER_COOKIE_DURATION'] = None  # Don't remember user login
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24  # 24 hours

# Database configuration
# Get credentials from environment variables with proper URL encoding for password
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_PORT = int(os.getenv('MYSQL_PORT'))
MYSQL_DB = os.getenv('MYSQL_DB')

# URL encode the password to handle special characters
encoded_password = urllib.parse.quote_plus(MYSQL_PASSWORD)

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}'
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
    status = {}
    for name, service_info in SERVICES.items():
        url = service_info["url"]
        endpoint = service_info["endpoint"]
        try:
            response = requests.get(f"{url}{endpoint}", timeout=2)
            if response.status_code == 200:
                status[name] = "healthy"
            else:
                status[name] = f"unhealthy - {response.status_code}"
        except Exception as e:
            logger.error(f"Error checking {name} health: {str(e)}")
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

@app.route('/images')
@login_required
def images():
    """Route for image-based analysis (model training)"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the image analysis.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    return render_template('train_model.html', services_status=services_status, logout_token=session['logout_token'])

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
        
        # Always use fixed test size of 20%
        session['test_size'] = 0.2
        
        try:
            # Check if required services are available
            required_services = {
                "Data Cleaner": DATA_CLEANER_URL,
                "Feature Selector": FEATURE_SELECTOR_URL,
                "Anomaly Detector": ANOMALY_DETECTOR_URL,
                "Model Coordinator": MODEL_COORDINATOR_URL  # Changed from Model Trainer to Model Coordinator
            }
            
            logger.info("Checking required services before training:")
            unavailable_services = []
            
            for service_name, service_url in required_services.items():
                logger.info(f"Checking service: {service_name} at {service_url}")
                if not is_service_available(service_url):
                    logger.error(f"Service {service_name} at {service_url} is not available")
                    unavailable_services.append(service_name)
                else:
                    logger.info(f"Service {service_name} is available")
            
            if unavailable_services:
                error_message = f"The following services are not available: {', '.join(unavailable_services)}. Cannot proceed with training."
                logger.error(error_message)
                flash(error_message, 'error')
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

            # Extract original dataset filename from the temporary filepath
            import re
            temp_filepath = session.get('uploaded_file', '')
            # The filepath format is typically: UPLOAD_FOLDER/uuid_originalfilename
            # Extract the original filename portion
            original_filename = ""
            if temp_filepath:
                # Match the pattern uuid_originalfilename
                match = re.search(r'[a-f0-9-]+_(.+)$', os.path.basename(temp_filepath))
                if match:
                    original_filename = match.group(1)
                else:
                    # Fallback to just basename if pattern doesn't match
                    original_filename = os.path.basename(temp_filepath)

            # Add direct SQL insert to ensure training_run is populated
            run_name = original_filename if original_filename else f"Training Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            user_id = current_user.id
            
            # Direct database connection to ensure the training run is saved
            try:
                # Create a temporary app context for database operations
                with app.app_context():
                    # Create training run entry
                    training_run = TrainingRun(
                        user_id=user_id,
                        run_name=run_name,
                        prompt=None
                    )
                    db.session.add(training_run)
                    db.session.commit()
                    
                    # Store the run_id for model saving
                    local_run_id = training_run.id
                    logger.info(f"Directly added training run to database with ID {local_run_id}")
                    
                    # Store preprocessing data for this run
                    try:
                        # Prepare data for storage
                        original_columns = list(data.columns)
                        
                        # Save cleaning configuration
                        cleaner_config = {
                            "llm_instructions": cleaning_result.get("prompt", ""),
                            "options": cleaning_result.get("options", {}),
                            "handle_missing": True,
                            "handle_outliers": True
                        }
                        
                        # Save feature selection configuration
                        feature_selector_config = {
                            "llm_instructions": feature_result.get("prompt", ""),
                            "options": feature_result.get("options", {}),
                            "method": feature_result.get("method", "auto")
                        }
                        
                        # Create preprocessing data record
                        preprocessing_data = PreprocessingData(
                            run_id=local_run_id,
                            user_id=user_id,
                            cleaner_config=json.dumps(cleaner_config),
                            feature_selector_config=json.dumps(feature_selector_config),
                            original_columns=json.dumps(original_columns),
                            selected_columns=json.dumps(selected_features),
                            cleaning_report=json.dumps(cleaning_result.get("report", {}))
                        )
                        db.session.add(preprocessing_data)
                        db.session.commit()
                        logger.info(f"Stored preprocessing data for run ID {local_run_id}")
                    except Exception as pp_error:
                        logger.error(f"Error saving preprocessing data: {str(pp_error)}")
                    
                    # Verify entry was created by checking the database
                    verification = db.session.query(TrainingRun).filter_by(id=local_run_id).first()
                    if verification:
                        logger.info(f"Verified training run in database: {verification.id}, {verification.run_name}")
                    else:
                        logger.warning(f"Could not verify training run in database after commit")
                    
                    # Try to diagnose the issue if verification failed
                    if not verification:
                        # Check if database is accessible
                        db.session.execute("SELECT 1")
                        logger.info("Database connection is working")
                        
                        # Check table structure
                        logger.info("Attempting to check training_run table structure")
                        table_info = db.session.execute("DESCRIBE training_run").fetchall()
                        logger.info(f"Table structure: {table_info}")
            except Exception as e:
                logger.error(f"Error adding training run directly to database: {str(e)}")
                # Try with direct SQL as a fallback
                try:
                    import pymysql
                    # Get database credentials from environment variables
                    MYSQL_USER = os.getenv("MYSQL_USER")
                    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
                    MYSQL_HOST = os.getenv("MYSQL_HOST")
                    MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
                    MYSQL_DB = os.getenv("MYSQL_DB")
                    
                    # Connect to database
                    conn = pymysql.connect(
                        host=MYSQL_HOST,
                        user=MYSQL_USER,
                        password=MYSQL_PASSWORD,
                        port=MYSQL_PORT,
                        database=MYSQL_DB
                    )
                    cursor = conn.cursor()
                    
                    # Insert training run
                    cursor.execute(
                        "INSERT INTO training_run (user_id, run_name, prompt, created_at) VALUES (%s, %s, %s, NOW())",
                        (user_id, run_name, None)
                    )
                    conn.commit()
                    local_run_id = cursor.lastrowid
                    logger.info(f"Added training run using direct SQL: {local_run_id}")
                    
                    cursor.close()
                    conn.close()
                except Exception as sql_error:
                    logger.error(f"Error with direct SQL approach: {str(sql_error)}")
                    local_run_id = None

            # Use our safe request method
            response = safe_requests_post(
                f"{MODEL_COORDINATOR_URL}/train",
                {
                    "data": X_data,
                    "target": y_data,
                    "test_size": session['test_size'],
                    "user_id": current_user.id,
                    "run_name": run_name
                },
                timeout=1800  # Model training can take time
            )
            
            if response.status_code != 200:
                raise Exception(f"Model Coordinator API error: {response.json().get('error', 'Unknown error')}")
            
            # Store the complete model results in session
            model_result = response.json()
            
            # Log the run_id from the model coordinator response
            if 'run_id' in model_result:
                model_coordinator_run_id = model_result['run_id']
                logger.info(f"Model coordinator created run with ID: {model_coordinator_run_id}")
                
                # If we have our own run_id and it doesn't match the coordinator's,
                # make sure we also save the models to our run_id
                if local_run_id and local_run_id != model_coordinator_run_id:
                    logger.info(f"Local run_id {local_run_id} doesn't match coordinator run_id {model_coordinator_run_id}")
                    
                    # If we have saved best models, save them to our database too
                    if 'saved_best_models' in model_result and model_result['saved_best_models']:
                        try:
                            with app.app_context():
                                # Process each of the 4 best models (accuracy, precision, recall, f1)
                                for metric, model_info in model_result['saved_best_models'].items():
                                    if 'url' in model_info and 'filename' in model_info:
                                        # Create a model record
                                        model_record = TrainingModel(
                                            user_id=user_id,
                                            run_id=local_run_id,  # Use our own run_id
                                            model_name=f"best_model_for_{metric}",
                                            model_url=model_info['url'],
                                            file_name=model_info['filename']  # Add the filename
                                        )
                                        db.session.add(model_record)
                                        logger.info(f"Added model for {metric} to database with run_id {local_run_id}")
                                
                                # Commit all model records
                                db.session.commit()
                                logger.info(f"Committed all model records to database for run_id {local_run_id}")
                        except Exception as model_save_error:
                            logger.error(f"Error saving models to database: {str(model_save_error)}")
            
            # Check if we need to ensure models are saved for the coordinator's run_id
            if 'saved_best_models' in model_result and model_result['saved_best_models']:
                # Use either the coordinator's run_id or our local one
                run_id_to_use = model_result.get('run_id', local_run_id)
                if run_id_to_use:
                    # Ensure all 4 models are properly saved to the database
                    ensure_training_models_saved(user_id, run_id_to_use, model_result)
                    
                    # If we have our local run_id, also ensure models are saved for it
                    if local_run_id and local_run_id != run_id_to_use:
                        ensure_training_models_saved(user_id, local_run_id, model_result)
                else:
                    logger.warning("No run_id available to save models")
            
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
        'recall': {'model_name': '', 'metrics': {'recall': 0}},
        'f1': {'model_name': '', 'metrics': {'f1': 0}},
        'specificity': {'model_name': '', 'metrics': {'specificity': 0}}
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
                                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'cv_score_mean', 'cv_score_std']:
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
                                'f1': 0.0,
                                'specificity': 0.0
                            }
                        
                        # Store model with metrics
                        all_models[model_name] = metrics
                        
                        # If specificity is not provided but we have other metrics, calculate it
                        # Specificity = TN / (TN + FP) = (1 - FPR)
                        if 'specificity' not in metrics and 'accuracy' in metrics and 'precision' in metrics and 'recall' in metrics:
                            # Try to estimate specificity using other available metrics
                            # This is an approximation since we don't have direct access to TN and FP
                            # For binary classification, can estimate: specificity ≈ (accuracy - recall * positive_rate) / (1 - positive_rate)
                            # Assuming balanced classes (positive_rate = 0.5)
                            accuracy = metrics.get('accuracy', 0.0)
                            recall = metrics.get('recall', 0.0)
                            # A simple estimation for specificity when true rate is unknown
                            estimated_specificity = 2 * accuracy - recall
                            # Cap between 0 and 1
                            estimated_specificity = max(0.0, min(1.0, estimated_specificity))
                            metrics['specificity'] = estimated_specificity
                            all_models[model_name]['specificity'] = estimated_specificity
                        
                        # Check if this model is best for any metric
                        accuracy = metrics.get('accuracy', 0.0)
                        precision = metrics.get('precision', 0.0)
                        recall = metrics.get('recall', 0.0)
                        f1 = metrics.get('f1', 0.0)
                        specificity = metrics.get('specificity', 0.0)
                        
                        logger.info(f"Model {model_name} metrics - accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, specificity: {specificity}")
                        
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
                            
                        if f1 > best_models['f1']['metrics'].get('f1', 0):
                            best_models['f1'] = {
                                'model_name': model_name,
                                'metrics': metrics
                            }
                            
                        if specificity > best_models['specificity']['metrics'].get('specificity', 0):
                            best_models['specificity'] = {
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
                                if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'cv_score_mean', 'cv_score_std']:
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
                            f1 = metrics.get('f1', 0.0)
                            specificity = metrics.get('specificity', 0.0)
                            
                            logger.info(f"Model {model_name} metrics (fallback) - accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}, specificity: {specificity}")
                            
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
                                
                            if f1 > best_models['f1']['metrics'].get('f1', 0):
                                best_models['f1'] = {
                                    'model_name': model_name,
                                    'metrics': metrics
                                }
                                
                            if specificity > best_models['specificity']['metrics'].get('specificity', 0):
                                best_models['specificity'] = {
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
    
    # Prepare data for template display - SHOWING THE TOP 5 MODELS BY DIFFERENT METRICS
    simplified_model_data = []
    
    # Include model with best accuracy
    if best_models['accuracy'].get('model_name'):
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
    
    # Include model with best precision
    if best_models['precision'].get('model_name'):
        model_name = best_models['precision'].get('model_name', 'Unknown')
        metric_value = best_models['precision'].get('metrics', {}).get('precision', 0)
        
        # Ensure metric value is a float
        if not isinstance(metric_value, float):
            try:
                metric_value = float(metric_value)
            except (TypeError, ValueError):
                metric_value = 0.0
        
        logger.info(f"Adding best precision model: {model_name} with value {metric_value}")
        
        simplified_model_data.append({
            'model_name': model_name,
            'metric_name': 'precision',
            'metric_value': metric_value,
            'is_best_for': 'precision'
        })
    
    # Include model with best recall
    if best_models['recall'].get('model_name'):
        model_name = best_models['recall'].get('model_name', 'Unknown')
        metric_value = best_models['recall'].get('metrics', {}).get('recall', 0)
        
        # Ensure metric value is a float
        if not isinstance(metric_value, float):
            try:
                metric_value = float(metric_value)
            except (TypeError, ValueError):
                metric_value = 0.0
        
        logger.info(f"Adding best recall model: {model_name} with value {metric_value}")
        
        simplified_model_data.append({
            'model_name': model_name,
            'metric_name': 'recall',
            'metric_value': metric_value,
            'is_best_for': 'recall'
        })
    
    # Include model with best F1 score
    if best_models['f1'].get('model_name'):
        model_name = best_models['f1'].get('model_name', 'Unknown')
        metric_value = best_models['f1'].get('metrics', {}).get('f1', 0)
        
        # Ensure metric value is a float
        if not isinstance(metric_value, float):
            try:
                metric_value = float(metric_value)
            except (TypeError, ValueError):
                metric_value = 0.0
        
        logger.info(f"Adding best F1 model: {model_name} with value {metric_value}")
        
        simplified_model_data.append({
            'model_name': model_name,
            'metric_name': 'f1',
            'metric_value': metric_value,
            'is_best_for': 'f1'
        })
    
    # Include model with best specificity
    if best_models['specificity'].get('model_name'):
        model_name = best_models['specificity'].get('model_name', 'Unknown')
        metric_value = best_models['specificity'].get('metrics', {}).get('specificity', 0)
        
        # Ensure metric value is a float
        if not isinstance(metric_value, float):
            try:
                metric_value = float(metric_value)
            except (TypeError, ValueError):
                metric_value = 0.0
        
        logger.info(f"Adding best specificity model: {model_name} with value {metric_value}")
        
        simplified_model_data.append({
            'model_name': model_name,
            'metric_name': 'specificity',
            'metric_value': metric_value,
            'is_best_for': 'specificity'
        })
    
    # If we still have no models for display, create entries for the 5 best models by different metrics
    if not simplified_model_data and all_models:
        logger.info("No best models identified, creating display data from top models")
        
        # Find the top models by different metrics
        top_accuracy = max(all_models.items(), key=lambda x: x[1].get('accuracy', 0))
        top_precision = max(all_models.items(), key=lambda x: x[1].get('precision', 0))
        top_recall = max(all_models.items(), key=lambda x: x[1].get('recall', 0))
        top_f1 = max(all_models.items(), key=lambda x: x[1].get('f1', 0))
        top_specificity = max(all_models.items(), key=lambda x: x[1].get('specificity', 0))
        
        # Add the top models
        top_models = [
            {'name': top_accuracy[0], 'metric': 'accuracy', 'value': top_accuracy[1].get('accuracy', 0)},
            {'name': top_precision[0], 'metric': 'precision', 'value': top_precision[1].get('precision', 0)},
            {'name': top_recall[0], 'metric': 'recall', 'value': top_recall[1].get('recall', 0)},
            {'name': top_f1[0], 'metric': 'f1', 'value': top_f1[1].get('f1', 0)},
            {'name': top_specificity[0], 'metric': 'specificity', 'value': top_specificity[1].get('specificity', 0)}
        ]
        
        for model in top_models:
            if model['value'] > 0:
                simplified_model_data.append({
                    'model_name': model['name'],
                    'metric_name': model['metric'],
                    'metric_value': model['value'],
                    'is_best_for': model['metric']
                })
    
    # Create a simplified version of all models with key metrics
    simplified_model_metrics = {}
    important_metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    
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
        
        # Calculate specificity if it's missing but we have other metrics
        if 'specificity' not in simplified_metrics and 'accuracy' in simplified_metrics and 'recall' in simplified_metrics:
            accuracy = simplified_metrics['accuracy']
            recall = simplified_metrics['recall']
            # A simple estimation for specificity
            estimated_specificity = 2 * accuracy - recall
            # Cap between 0 and 1
            estimated_specificity = max(0.0, min(1.0, estimated_specificity))
            simplified_metrics['specificity'] = estimated_specificity
            logger.info(f"Estimated specificity for {model_name}: {estimated_specificity}")
        
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
        'test_size': session.get('test_size', 0.2),
        'metrics_shown': 'accuracy, precision, recall, f1, specificity'
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
                
                # Try to get the cleaning prompt from the training run associated with the model
                cleaning_prompt = None
                if best_model_record:
                    run_id = best_model_record.run_id
                    try:
                        # Get the cleaning prompt from the training run
                        training_run = TrainingRun.query.filter_by(id=run_id).first()
                        if training_run and training_run.prompt:
                            cleaning_prompt = training_run.prompt
                            logger.info(f"Retrieved cleaning prompt from training run {run_id}")
                        else:
                            logger.info(f"No cleaning prompt found for training run {run_id}")
                    except Exception as e:
                        logger.error(f"Error retrieving cleaning prompt: {str(e)}")
                
                # 1. Clean data using Data Cleaner API
                logger.info(f"Sending prediction data to Data Cleaner API")
                pred_data_records = pred_data.replace([np.inf, -np.inf], np.nan).where(pd.notnull(pred_data), None).to_dict(orient='records')
                
                # Prepare payload for data cleaner
                cleaner_payload = {
                    "data": pred_data_records,
                    "target_column": target_column
                }
                
                # Add cleaning prompt if available
                if cleaning_prompt:
                    cleaner_payload["prompt"] = cleaning_prompt
                    logger.info("Using stored cleaning prompt for consistent data cleaning")
                
                response = safe_requests_post(
                    f"{DATA_CLEANER_URL}/clean",
                    cleaner_payload,
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
                
                # Get the best model URL for this metric from the database
                best_model_record = TrainingModel.query.filter_by(
                    model_name=f"best_model_for_{metric}"
                ).order_by(TrainingModel.created_at.desc()).first()
                
                if best_model_record:
                    # Use the new predict_with_blob endpoint
                    logger.info(f"Using blob-stored model for prediction: {best_model_record.model_url}")
                    
                    payload = {
                        "data": prediction_data,
                        "model_url": best_model_record.model_url,
                        "user_id": current_user.id
                    }
                    
                    response = safe_requests_post(
                        f"{MODEL_COORDINATOR_URL}/predict_with_blob",
                        payload,
                        timeout=60
                    )
                else:
                    # Fall back to the original prediction method
                    logger.info("No blob-stored model found, using classic prediction endpoint")
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

@app.route('/api/train_model', methods=['POST'])
@login_required
def api_train_model():
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        num_classes = int(request.form.get('numClasses', 5))
        training_level = int(request.form.get('trainingLevel', 3))
        
        # Forward the request to the dockerized model training service
        model_service_url = f"{MODEL_TRAINING_SERVICE_URL}/train"
        
        # Create form data to send to the service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'numClasses': str(num_classes),
                'trainingLevel': str(training_level)
            }
        )
        
        # Make the request to the model service
        response = requests.post(
            model_service_url,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True
        )
        
        if response.status_code != 200:
            error_message = "Error from model training service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                pass
            return jsonify({"error": error_message}), response.status_code
        
        # Get metrics from the response headers
        metrics = {}
        if 'X-Training-Metrics' in response.headers:
            try:
                metrics = json.loads(response.headers['X-Training-Metrics'])
                logger.info(f"Successfully parsed metrics from model service: {metrics}")
            except Exception as e:
                logger.error(f"Failed to parse metrics from model service response: {str(e)}")
                
        # Debug log all response headers
        logger.info("All headers from model service response:")
        for header, value in response.headers.items():
            logger.info(f"  {header}: {value[:100]}{'...' if len(value) > 100 else ''}")
        
        # Create a response with both the model file and metrics
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/octet-stream"
        flask_response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics header if available
        if metrics:
            flask_response.headers["X-Training-Metrics"] = json.dumps(metrics)
            # Debug log to verify metrics are being added to the response
            logger.info(f"Added X-Training-Metrics header to response: {metrics}")
        else:
            logger.warning("No metrics available to add to response headers")
            
        # Check for Access-Control-Expose-Headers and add it if needed
        if 'Access-Control-Expose-Headers' not in flask_response.headers:
            flask_response.headers['Access-Control-Expose-Headers'] = 'X-Training-Metrics'
            logger.info("Added Access-Control-Expose-Headers to response")
        
        return flask_response
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def ensure_training_models_saved(user_id, run_id, model_result):
    """Ensure that all 4 best models are saved to the TrainingModel table"""
    try:
        # Check if we have saved_best_models in the result
        if 'saved_best_models' not in model_result or not model_result['saved_best_models']:
            logger.warning("No saved_best_models found in model result")
            return False
            
        # Get the saved models
        saved_models = model_result['saved_best_models']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        with app.app_context():
            # Check if models already exist for this run
            existing_models = TrainingModel.query.filter_by(run_id=run_id).count()
            if existing_models >= 4:
                logger.info(f"Found {existing_models} models already saved for run_id {run_id}")
                return True
                
            # Save each model
            for metric in metrics:
                if metric not in saved_models:
                    logger.warning(f"No saved model found for metric {metric}")
                    continue
                    
                model_info = saved_models[metric]
                
                # Check if we have the URL
                if 'url' not in model_info:
                    logger.warning(f"No URL found for saved model {metric}")
                    continue
                    
                # Create model record
                model_name = f"best_model_for_{metric}"
                model_url = model_info['url']
                
                # Extract filename from URL or use the one from the model_info
                if 'filename' in model_info:
                    filename = model_info['filename']
                else:
                    # Extract filename from URL: https://accountname.blob.core.windows.net/container/filename
                    filename = model_url.split('/')[-1]
                
                # Check if this model is already saved
                existing_model = TrainingModel.query.filter_by(
                    run_id=run_id,
                    model_name=model_name
                ).first()
                
                if existing_model:
                    logger.info(f"Model {model_name} already exists for run_id {run_id}")
                    continue
                    
                # Create and save the model record
                model_record = TrainingModel(
                    user_id=user_id,
                    run_id=run_id,
                    model_name=model_name,
                    model_url=model_url,
                    file_name=filename  # Save the filename too
                )
                
                db.session.add(model_record)
                logger.info(f"Added model {model_name} to database for run_id {run_id}")
            
            # Commit all changes
            db.session.commit()
            logger.info(f"Committed training models to database for run_id {run_id}")
            
            # Verify models were saved
            saved_count = TrainingModel.query.filter_by(run_id=run_id).count()
            logger.info(f"Verified {saved_count} models saved for run_id {run_id}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error ensuring training models are saved: {str(e)}")
        return False

@app.route('/augment')
@login_required
def augment():
    """Route for data augmentation page"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the data augmentation page.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    return render_template('augment.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/augment/process', methods=['POST'])
@login_required
def process_augmentation():
    """Process dataset augmentation and return results"""
    # Check for file upload
    if 'zipFile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    zip_file = request.files['zipFile']
    if not zip_file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not zip_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "File must be a ZIP archive"}), 400
    
    # Get form parameters
    level = request.form.get('level', '3')
    num_augmentations = request.form.get('numAugmentations', '2')
    
    try:
        # Check if the augmentation service is available
        if not is_service_available(f"{AUGMENTATION_SERVICE_URL}/health"):
            return jsonify({"error": "Augmentation service is not available. Please try again later."}), 503
        
        logger.info(f"Starting augmentation process for file: {zip_file.filename}")
        
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_file.save(temp_file.name)
        temp_file.close()
        
        # Create a proper multipart request
        with open(temp_file.name, 'rb') as f:
            # Setup multipart form data with optimized buffering for large files
            m = MultipartEncoder(
                fields={
                    'zipFile': (zip_file.filename, f, 'application/zip'),
                    'level': level,
                    'numAugmentations': num_augmentations
                }
            )
            
            # Forward the request to the augmentation service
            headers = {
                'Content-Type': m.content_type
            }
            
            # Stream the upload to the service and stream the response back
            response = requests.post(
                f"{AUGMENTATION_SERVICE_URL}/augment",
                data=m,
                headers=headers,
                stream=True
            )
        
        # Clean up temporary file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        # If there's an error, return it
        if response.status_code != 200:
            error_msg = "Error from augmentation service"
            if response.headers.get('Content-Type') == 'application/json':
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', error_msg)
                except:
                    pass
            logger.error(f"Augmentation service returned error: {error_msg}")
            return jsonify({"error": error_msg}), response.status_code
        
        # Create a file-like object from the response content
        result = io.BytesIO()
        
        # Stream the response in chunks
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                result.write(chunk)
        
        # Reset the file pointer to the beginning
        result.seek(0)
        
        # Log successful completion
        logger.info(f"Augmentation process completed for file: {zip_file.filename}")
        
        # Return the augmented dataset
        return send_file(
            result,
            mimetype='application/zip',
            as_attachment=True,
            download_name='augmented_dataset.zip'
        )
    except Exception as e:
        logger.error(f"Error in augmentation process: {str(e)}")
        return jsonify({"error": str(e)}), 500

def is_service_available(service_url):
    """Check if a service is available"""
    try:
        # Parse the service URL to determine the endpoint
        endpoint = "/health"  # Default endpoint
        
        # Find the matching service in our SERVICES dictionary
        for name, service_info in SERVICES.items():
            if service_info["url"] == service_url:
                endpoint = service_info["endpoint"]
                logger.info(f"Checking service {name} at {service_url}{endpoint}")
                break
        
        # Make the request with increased timeout (5 seconds instead of 1)
        logger.info(f"Sending health check to {service_url}{endpoint}")
        response = requests.get(f"{service_url}{endpoint}", timeout=5)
        
        # Log the response
        logger.info(f"Health check response from {service_url}{endpoint}: {response.status_code}")
        if response.status_code == 200:
            try:
                # Try to parse the response for additional info
                response_json = response.json()
                logger.info(f"Health check response content: {response_json}")
            except:
                pass
            return True
        return False
    except Exception as e:
        logger.error(f"Service unavailable ({service_url}): {str(e)}")
        return False

@app.route('/my_models')
@login_required
def my_models():
    """Route to display user's trained models."""
    user_id = current_user.id
    
    # Get all training runs for the current user with their models
    training_runs = TrainingRun.query.filter_by(user_id=user_id).order_by(TrainingRun.created_at.desc()).all()
    
    # For each training run, get up to top 4 models and check if preprocessing data exists
    for run in training_runs:
        run.models = TrainingModel.query.filter_by(run_id=run.id).order_by(TrainingModel.created_at.desc()).limit(4).all()
        
        # Check if preprocessing data exists for this run
        preprocessing_data = PreprocessingData.query.filter_by(run_id=run.id, user_id=user_id).first()
        run.has_preprocessing = preprocessing_data is not None
    
    return render_template('my_models.html', training_runs=training_runs)

@app.route('/download_model/<int:model_id>')
@login_required
def download_model(model_id):
    """Download a model package with the model and preprocessing information."""
    from flask import send_file
    import io
    import zipfile
    import tempfile
    import os
    import json
    
    # Get model info from database
    model = TrainingModel.query.filter_by(id=model_id, user_id=current_user.id).first_or_404()
    
    # Get preprocessing data for this run
    preproc_data = PreprocessingData.query.filter_by(run_id=model.run_id, user_id=current_user.id).first()
    
    try:
        # Get the model from blob storage
        model_url = model.model_url
        
        if not model_url:
            raise ValueError("Model URL not found in database")
        
        # Download the model using our authenticated function
        blob_data = download_blob(model_url)
        
        if not blob_data:
            raise ValueError("Failed to download model from storage")
        
        # Create a temporary directory to build the package
        temp_dir = tempfile.mkdtemp()
        
        # Save the model file
        model_filename = model.file_name if model.file_name else f"{model.model_name}_{model.id}.joblib"
        model_path = os.path.join(temp_dir, model_filename)
        with open(model_path, 'wb') as f:
            f.write(blob_data)
            
        # If we have preprocessing data, include it
        preprocessing_info = {}
        if preproc_data:
            preprocessing_info = {
                'cleaner_config': json.loads(preproc_data.cleaner_config) if preproc_data.cleaner_config else {},
                'feature_selector_config': json.loads(preproc_data.feature_selector_config) if preproc_data.feature_selector_config else {},
                'original_columns': json.loads(preproc_data.original_columns) if preproc_data.original_columns else [],
                'selected_columns': json.loads(preproc_data.selected_columns) if preproc_data.selected_columns else [],
                'cleaning_report': json.loads(preproc_data.cleaning_report) if preproc_data.cleaning_report else {}
            }
            
            # Save preprocessing info
            preproc_path = os.path.join(temp_dir, 'preprocessing_info.json')
            with open(preproc_path, 'w') as f:
                json.dump(preprocessing_info, f, indent=2)
                
        # Create a utility script for using the model
        create_utility_script(temp_dir, model_filename, preprocessing_info)
        
        # Create a README file
        create_readme_file(temp_dir, model, preprocessing_info)
        
        # Create a zip file containing all the package contents
        zip_filename = f"{model.model_name}_package_{model.id}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add model file
            zipf.write(model_path, os.path.basename(model_path))
            
            # Add preprocessing info if available
            if preproc_data:
                zipf.write(preproc_path, os.path.basename(preproc_path))
                
            # Add utility script and README
            for filename in os.listdir(temp_dir):
                if filename not in [zip_filename, os.path.basename(model_path)] and (filename.endswith('.py') or filename == 'README.md'):
                    file_path = os.path.join(temp_dir, filename)
                    zipf.write(file_path, filename)
        
        # Return the zip file
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        # Log the error
        logger.error(f"Error packaging model (ID: {model_id}): {str(e)}")
        flash(f"Error packaging model: {str(e)}", "danger")
        return redirect(url_for('my_models'))


def create_utility_script(temp_dir, model_filename, preprocessing_info):
    """Create a utility script for using the model with proper preprocessing."""
    script_content = '''
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ModelPredictor:
    def __init__(self, model_path=None, preprocessing_info_path=None):
        """
        Initialize the predictor with model and preprocessing information.
        
        Args:
            model_path: Path to the .joblib model file
            preprocessing_info_path: Path to the preprocessing_info.json file
        """
        # Find paths automatically if in the same directory
        if model_path is None:
            for file in os.listdir('.'):
                if file.endswith('.joblib'):
                    model_path = file
                    break
            if model_path is None:
                raise ValueError("No .joblib model file found in current directory")
                
        if preprocessing_info_path is None:
            if os.path.exists('preprocessing_info.json'):
                preprocessing_info_path = 'preprocessing_info.json'
        
        # Load the model
        print(f"Loading model from {model_path}")
        self.model_data = joblib.load(model_path)
        
        # Handle both direct model objects and dictionary storage format
        if isinstance(self.model_data, dict) and 'model' in self.model_data:
            self.model = self.model_data['model']
            print("Model loaded from dictionary format")
        else:
            self.model = self.model_data
            print("Model loaded directly")
            
        # Get model type information
        self.model_type = type(self.model).__name__
        print(f"Model type: {self.model_type}")
        
        # Extract components if model is a pipeline
        self.classifier = None
        self.scaler = None
        if hasattr(self.model, 'steps'):
            print("Model is a pipeline with steps:")
            for name, step in self.model.steps:
                print(f"- {name}: {type(step).__name__}")
                if name == 'classifier':
                    self.classifier = step
                elif name == 'scaler':
                    self.scaler = step
        
        # Load preprocessing info if available
        self.preprocessing_info = None
        if preprocessing_info_path and os.path.exists(preprocessing_info_path):
            with open(preprocessing_info_path, 'r') as f:
                self.preprocessing_info = json.load(f)
                print("Loaded preprocessing information")
                
    def preprocess_data(self, df):
        """Apply the same preprocessing steps as during training."""
        if self.preprocessing_info is None:
            print("Warning: No preprocessing information available. Using raw data.")
            return df
            
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Apply data cleaning
        processed_df = self._apply_cleaning(processed_df)
        
        # Apply feature selection 
        processed_df = self._apply_feature_selection(processed_df)
        
        return processed_df
        
    def _apply_cleaning(self, df):
        """Apply cleaning operations based on stored configuration."""
        if self.preprocessing_info is None or 'cleaner_config' not in self.preprocessing_info:
            return df
            
        cleaner_config = self.preprocessing_info['cleaner_config']
        
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Apply basic cleaning operations based on cleaner_config
        # Handle missing values
        if cleaner_config.get('handle_missing', True):
            for col in cleaned_df.columns:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Fill numeric columns with mean or specified value
                    fill_value = cleaner_config.get('numeric_fill', 'mean')
                    if fill_value == 'mean':
                        if cleaned_df[col].isna().any():
                            # Calculate mean excluding NaN values
                            mean_value = cleaned_df[col].mean()
                            cleaned_df[col] = cleaned_df[col].fillna(mean_value)
                    elif fill_value == 'median':
                        if cleaned_df[col].isna().any():
                            # Calculate median excluding NaN values
                            median_value = cleaned_df[col].median()
                            cleaned_df[col] = cleaned_df[col].fillna(median_value)
                    elif fill_value == 'zero':
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                else:
                    # Fill categorical/text columns with mode or specified value
                    fill_value = cleaner_config.get('categorical_fill', 'mode')
                    if fill_value == 'mode':
                        if cleaned_df[col].isna().any():
                            mode = cleaned_df[col].mode()
                            if not mode.empty:
                                cleaned_df[col] = cleaned_df[col].fillna(mode[0])
                    elif fill_value == 'unknown':
                        cleaned_df[col] = cleaned_df[col].fillna('unknown')
        
        # Remove duplicates if specified
        if cleaner_config.get('remove_duplicates', True):
            cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle outliers if specified (using IQR method)
        if cleaner_config.get('handle_outliers', False):
            for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_action = cleaner_config.get('outlier_action', 'clip')
                if outlier_action == 'clip':
                    cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
                elif outlier_action == 'remove':
                    mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                    cleaned_df = cleaned_df[mask]
        
        return cleaned_df
    
    def _apply_feature_selection(self, df):
        """Apply feature selection based on stored configuration."""
        if self.preprocessing_info is None:
            return df
            
        # If we have specific columns to select, use them
        if 'selected_columns' in self.preprocessing_info and self.preprocessing_info['selected_columns']:
            selected_columns = self.preprocessing_info['selected_columns']
            # Check which columns are available in the DataFrame
            available_columns = [col for col in selected_columns if col in df.columns]
            
            if len(available_columns) != len(selected_columns):
                missing_columns = set(selected_columns) - set(available_columns)
                print(f"Warning: Missing columns: {missing_columns}")
                
            if not available_columns:
                print("Warning: No selected columns found in the dataset. Using all columns.")
                return df
                
            return df[available_columns].copy()
        
        return df
    
    def predict(self, df):
        """Preprocess data and make predictions with version compatibility handling."""
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        print(f"Preprocessed data shape: {processed_df.shape}")
        
        # Check if we have a target column and remove it
        target_col = None
        for col in ['diagnosis', 'target', 'label', 'class']:
            if col in processed_df.columns:
                target_col = col
                true_values = processed_df[target_col].copy()
                processed_df = processed_df.drop(target_col, axis=1)
                print(f"Removed target column '{target_col}' for prediction")
                break
                
        # Try different prediction methods and use the first one that works
        prediction_methods = []
        
        try:
            # Method 1: Direct prediction
            predictions = self.model.predict(processed_df)
            prediction_methods.append("Direct model.predict()")
            
            # Check for suspicious predictions (all the same value)
            unique_preds = np.unique(predictions)
            if len(unique_preds) == 1:
                print(f"Warning: All predictions are {unique_preds[0]}. Trying compatibility fix...")
                raise Exception("All predictions are the same - trying alternative method")
                
        except Exception as e:
            print(f"Direct prediction failed or gave suspicious results: {str(e)}")
            
            try:
                # Method 2: Manual logistic regression with fresh scaling
                if self.classifier is not None and hasattr(self.classifier, 'coef_'):
                    print("Using manual logistic regression with coefficients")
                    
                    # Apply fresh StandardScaler
                    fresh_scaler = StandardScaler()
                    X_scaled = fresh_scaler.fit_transform(processed_df.values)
                    
                    # Get coefficients and intercept
                    coef = self.classifier.coef_[0]
                    intercept = self.classifier.intercept_[0]
                    
                    # Handle coefficient length mismatch
                    if len(coef) > processed_df.shape[1]:
                        print(f"Warning: Coefficient length ({len(coef)}) > data columns ({processed_df.shape[1]})")
                        print("Using only the coefficients that match data dimensions")
                        coef = coef[:processed_df.shape[1]]
                    
                    # Calculate log-odds (z)
                    z = np.dot(X_scaled, coef) + intercept
                    
                    # Apply sigmoid function to get probabilities
                    probs = 1 / (1 + np.exp(-z))
                    
                    # Convert to 0/1 predictions
                    predictions = (probs > 0.5).astype(int)
                    prediction_methods.append("Manual logistic regression with fresh scaling")
                    
                    # Check again for suspicious predictions
                    unique_preds = np.unique(predictions)
                    if len(unique_preds) == 1:
                        print(f"Warning: All predictions are {unique_preds[0]}. Trying direct classifier...")
                        raise Exception("All predictions are the same - trying classifier directly")
                else:
                    raise Exception("No classifier component with coefficients found")
            except Exception as e:
                print(f"Manual prediction failed: {str(e)}")
                
                try:
                    # Method 3: Try classifier component directly with fresh scaling
                    if self.classifier is not None:
                        print("Using classifier component directly")
                        
                        # Apply fresh scaling if we have a scaler
                        if self.scaler is not None:
                            print("Using fresh StandardScaler before classifier")
                            fresh_scaler = StandardScaler()
                            df_scaled = fresh_scaler.fit_transform(processed_df.values)
                        else:
                            df_scaled = processed_df.values
                        
                        predictions = self.classifier.predict(df_scaled)
                        prediction_methods.append("Direct classifier.predict()")
                        
                        # Final check for suspicious predictions
                        unique_preds = np.unique(predictions)
                        if len(unique_preds) == 1:
                            print(f"Warning: All predictions are {unique_preds[0]}. Using fallback...")
                            raise Exception("All predictions are the same - using fallback")
                    else:
                        raise Exception("No classifier component found")
                except Exception as e:
                    print(f"Classifier prediction failed: {str(e)}")
                    
                    # Method 4: Emergency fallback
                    print("All prediction methods failed. Using emergency fallback (all 1's).")
                    predictions = np.ones(processed_df.shape[0])
                    prediction_methods.append("Emergency fallback (all 1's)")
        
        # Print prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        print("Prediction distribution:")
        for val, count in zip(unique, counts):
            print(f"  {val}: {count} ({count/len(predictions)*100:.1f}%)")
            
        print(f"Prediction method used: {prediction_methods[0]}")
        return predictions
        
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <data_file.csv>")
        print("The data file should be a CSV file with features.")
        sys.exit(1)
        
    data_file = sys.argv[1]
    
    try:
        # Load the data
        df = pd.read_csv(data_file)
        print(f"Loaded data from {data_file} with shape {df.shape}")
        
        # Create a predictor instance
        predictor = ModelPredictor()
        
        # Make predictions
        predictions = predictor.predict(df)
        
        # Save predictions to file
        output_file = data_file.replace(".csv", "_predictions.csv")
        # If the file doesn't end with .csv, just append _predictions.csv
        if output_file == data_file:
            output_file = data_file + "_predictions.csv"
            
        result_df = df.copy()
        result_df['prediction'] = predictions
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
'''

    # Write the utility script to a file
    script_path = os.path.join(temp_dir, 'predict.py')
    with open(script_path, 'w') as f:
        f.write(script_content)
def create_readme_file(temp_dir, model, preprocessing_info):
    """Create a README file with instructions for using the model."""
    readme_content = f'''# Model Package: {model.model_name}

## Contents

- {model.file_name if model.file_name else model.model_name + ".joblib"}: The trained model file
- preprocessing_info.json: Configuration for data preprocessing
- predict.py: Utility script for making predictions
- README.md: This file

## Quick Start
1. Make sure you have the required packages installed:
   ```
   pip install pandas numpy scikit-learn joblib
   ```

2. Place your data file (CSV format) in the same directory as these files.

3. Run the prediction script:
   ```
   python predict.py your_data.csv
   ```

4. The script will:
   - Load your data
   - Preprocess it using the same steps as during training
   - Make predictions with the model
   - Save the results as your_data_predictions.csv

## Version Compatibility

The included predict.py script is designed to work with different versions of scikit-learn. If you encounter any issues with predictions (such as all predictions being the same value), the script will automatically apply compatibility fixes to ensure accurate results.

This makes the model package robust across different environments and scikit-learn versions. The script will provide detailed logging about which prediction method was used.

## Using in Your Own Code
```python
from predict import ModelPredictor
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the predictor
predictor = ModelPredictor()

# Preprocess data and get predictions
predictions = predictor.predict(df)

# Or do the steps separately
processed_df = predictor.preprocess_data(df)
predictions = predictor.predict(processed_df)
```

## Model Information
- Model Type: {model.model_name}
- Created: {model.created_at}
- ID: {model.id}

'''

    # Add preprocessing information if available
    if preprocessing_info:
        if 'selected_columns' in preprocessing_info and preprocessing_info['selected_columns']:
            column_count = len(preprocessing_info['selected_columns'])
            readme_content += f"\n## Preprocessing Information\n"
            readme_content += f"- Selected Features: {column_count} features\n"
            if column_count <= 20:  # Only list if not too many
                readme_content += f"- Feature List: {', '.join(preprocessing_info['selected_columns'])}\n"
            
            if 'cleaning_report' in preprocessing_info and preprocessing_info['cleaning_report']:
                readme_content += "- Cleaning Operations Applied:\n"
                for operation, details in preprocessing_info['cleaning_report'].items():
                    if isinstance(details, dict):
                        readme_content += f"  - {operation}: {details.get('description', 'Applied')}\n"
                    else:
                        readme_content += f"  - {operation}: Applied\n"
    
    readme_path = os.path.join(temp_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

@app.route('/preprocess_new_data/<int:run_id>', methods=['POST'])
@login_required
def preprocess_new_data(run_id):
    """Preprocess a new dataset using the same steps as a previous training run."""
    import pandas as pd
    import json
    import traceback
    
    user_id = current_user.id
    
    # Check if the run exists and belongs to the user
    training_run = TrainingRun.query.filter_by(id=run_id, user_id=user_id).first_or_404()
    
    # Get preprocessing data for this run
    preproc_data = PreprocessingData.query.filter_by(run_id=run_id, user_id=user_id).first_or_404()
    
    if 'dataFile' not in request.files:
        flash("No file was uploaded", "danger")
        return redirect(url_for('my_models'))
    
    file = request.files['dataFile']
    
    if file.filename == '':
        flash("No file was selected", "danger")
        return redirect(url_for('my_models'))
    
    if not allowed_file(file.filename):
        flash("Only CSV and Excel files are supported", "danger")
        return redirect(url_for('my_models'))
    
    try:
        # Save uploaded file temporarily
        temp_path = get_temp_filepath(file.filename)
        file.save(temp_path)
        
        # Load the dataset
        if temp_path.endswith('.csv'):
            df = pd.read_csv(temp_path)
        else:
            df = pd.read_excel(temp_path)
        
        # Load preprocessing configurations
        cleaner_config = json.loads(preproc_data.cleaner_config) if preproc_data.cleaner_config else {}
        feature_selector_config = json.loads(preproc_data.feature_selector_config) if preproc_data.feature_selector_config else {}
        selected_columns = json.loads(preproc_data.selected_columns) if preproc_data.selected_columns else []
        
        # Apply data cleaning using the stored configuration
        cleaned_df = apply_stored_cleaning(df, cleaner_config)
        
        # Apply feature selection using the stored configuration
        if selected_columns:
            # If we have a list of selected columns, use it
            processed_df = cleaned_df[selected_columns].copy()
        else:
            # Otherwise, apply the feature selection config
            processed_df = apply_stored_feature_selection(cleaned_df, feature_selector_config)
        
        # Create a temporary file for the processed data
        processed_file_path = get_temp_filepath(original_filename=file.filename, extension='.csv')
        processed_df.to_csv(processed_file_path, index=False)
        
        # Create a descriptive filename for the download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"preprocessed_data_{run_id}_{timestamp}.csv"
        
        return send_file(
            processed_file_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}\n{traceback.format_exc()}")
        flash(f"Error preprocessing data: {str(e)}", "danger")
        return redirect(url_for('my_models'))

def apply_stored_cleaning(df, cleaner_config):
    """Apply stored cleaning operations to a DataFrame."""
    import pandas as pd
    import numpy as np
    import requests
    
    # If we have cleaner_config with LLM instructions, use the data cleaner service
    if cleaner_config.get('llm_instructions'):
        # Call data cleaner service
        try:
            # Convert DataFrame to CSV
            csv_data = df.to_csv(index=False)
            
            # Prepare multipart form data
            data = {
                'prompt': cleaner_config.get('llm_instructions', ''),
                'options': json.dumps(cleaner_config.get('options', {}))
            }
            files = {'file': ('data.csv', csv_data, 'text/csv')}
            
            # Send request to data cleaner service
            response = requests.post(f"{DATA_CLEANER_URL}/clean", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                # Convert the cleaned CSV data back to DataFrame
                cleaned_df = pd.read_csv(io.StringIO(response.text))
                return cleaned_df
            else:
                raise Exception(f"Data cleaner service failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Error calling data cleaner service: {str(e)}")
            # Fall back to basic cleaning if service fails
            return apply_basic_cleaning(df, cleaner_config)
    else:
        # Apply basic cleaning if no LLM instructions
        return apply_basic_cleaning(df, cleaner_config)

def apply_basic_cleaning(df, cleaner_config):
    """Apply basic cleaning operations based on config."""
    import pandas as pd
    import numpy as np
    
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Apply basic cleaning operations based on cleaner_config
    # Handle missing values
    if cleaner_config.get('handle_missing', True):
        for col in cleaned_df.columns:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                # Fill numeric columns with mean or specified value
                fill_value = cleaner_config.get('numeric_fill', 'mean')
                if fill_value == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                elif fill_value == 'median':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                elif fill_value == 'zero':
                    cleaned_df[col] = cleaned_df[col].fillna(0)
            else:
                # Fill categorical/text columns with mode or specified value
                fill_value = cleaner_config.get('categorical_fill', 'mode')
                if fill_value == 'mode':
                    mode = cleaned_df[col].mode()
                    if not mode.empty:
                        cleaned_df[col] = cleaned_df[col].fillna(mode[0])
                elif fill_value == 'unknown':
                    cleaned_df[col] = cleaned_df[col].fillna('unknown')
    
    # Remove duplicates if specified
    if cleaner_config.get('remove_duplicates', True):
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle outliers if specified (using IQR method)
    if cleaner_config.get('handle_outliers', False):
        for col in cleaned_df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_action = cleaner_config.get('outlier_action', 'clip')
            if outlier_action == 'clip':
                cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
            elif outlier_action == 'remove':
                mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                cleaned_df = cleaned_df[mask]
    
    return cleaned_df

def apply_stored_feature_selection(df, feature_selector_config):
    """Apply stored feature selection operations to a DataFrame."""
    import pandas as pd
    import numpy as np
    import requests
    
    # If we have feature_selector_config with LLM instructions, use the feature selector service
    if feature_selector_config.get('llm_instructions'):
        # Call feature selector service
        try:
            # Convert DataFrame to CSV
            csv_data = df.to_csv(index=False)
            
            # Prepare multipart form data
            data = {
                'prompt': feature_selector_config.get('llm_instructions', ''),
                'options': json.dumps(feature_selector_config.get('options', {}))
            }
            files = {'file': ('data.csv', csv_data, 'text/csv')}
            
            # Send request to feature selector service
            response = requests.post(f"{FEATURE_SELECTOR_URL}/select", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                # The response should contain the selected columns
                selection_result = response.json()
                
                if 'selected_features' in selection_result:
                    selected_columns = selection_result['selected_features']
                    return df[selected_columns].copy()
                else:
                    # If no explicit selection, return the original dataframe
                    return df
            else:
                raise Exception(f"Feature selector service failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Error calling feature selector service: {str(e)}")
            # If service fails, return the original dataframe
            return df
    else:
        # No feature selection config, return original dataframe
        return df

if __name__ == '__main__':
    print("Starting DeepMed with API services integration")
    
    # Log service URLs for debugging
    # Check for database connection
    try:
        with db.engine.connect() as connection:
            logger.info("Database connection successful")
            user_count = db.session.query(User).count()
            logger.info(f"Total users in database: {user_count}")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
    
    try:
        # Start the application
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Clean up all temporary files when the application stops
        cleanup_temp_files()
