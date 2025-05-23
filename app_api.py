from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, make_response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import io
import time
from datetime import datetime
import requests
import logging
import shutil
import atexit
import tempfile
import uuid
import secrets as std_secrets
import urllib.parse
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
from db.users import db, User, TrainingRun, TrainingModel, PreprocessingData
# Import the chatbot module
from app_llm import register_chatbot_blueprint
# Import the Key Vault module
import keyvault

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

# Modified requests post function that handles problematic float values
def safe_requests_post(url, json_data, **kwargs):
    safe_json = clean_data_for_json(json_data)
    return requests.post(url, json=safe_json, **kwargs)

# Define service URLs with localhost and ports for host machine access
DATA_CLEANER_URL = 'http://localhost:5001'
FEATURE_SELECTOR_URL = 'http://localhost:5002'
ANOMALY_DETECTOR_URL = 'http://localhost:5003'
MODEL_COORDINATOR_URL = 'http://localhost:5020'
MEDICAL_ASSISTANT_URL = 'http://localhost:5005'
AUGMENTATION_SERVICE_URL = 'http://localhost:5023'
MODEL_TRAINING_SERVICE_URL = 'http://localhost:5021'

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
app.secret_key = keyvault.getenv('SECRETKEY')

# First try to use the default temp directory
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'medicai_temp')

# Function to setup and validate a temp directory
def setup_temp_dir(directory_path):
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        
        # Test write permissions by creating a test file
        test_file = os.path.join(directory_path, f"test_{uuid.uuid4()}.txt")
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        # Set permissions
        os.chmod(directory_path, 0o777)
        
        logger.info(f"Using temporary directory: {directory_path}")
        return directory_path
    except (PermissionError, OSError) as e:
        logger.error(f"Error setting up directory {directory_path}: {str(e)}")
        return None

# Try to set up the default temp directory
if not setup_temp_dir(UPLOAD_FOLDER):
    # Try alternative locations
    alternatives = [
        os.path.expanduser("~/medicai_temp"),  # User's home directory
        os.path.join(os.getcwd(), "temp"),     # Current working directory
        os.path.join("static", "temp")         # Static folder (should always be writable)
    ]
    
    for alt_dir in alternatives:
        logger.info(f"Trying alternate temporary directory: {alt_dir}")
        if setup_temp_dir(alt_dir):
            UPLOAD_FOLDER = alt_dir
            break
    else:
        # If all alternatives fail, log a critical error
        logger.critical("No writable temporary directory could be found!")
        # Use the static/temp as a last resort
        UPLOAD_FOLDER = os.path.join("static", "temp")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['REMEMBER_COOKIE_DURATION'] = None
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24  # 24 hours

# Database configuration
MYSQLUSER = keyvault.getenv('MYSQLUSER')
MYSQLPASSWORD = keyvault.getenv('MYSQLPASSWORD')
MYSQLHOST = keyvault.getenv('MYSQLHOST')
MYSQLPORT = int(keyvault.getenv('MYSQLPORT'))
MYSQLDB = keyvault.getenv('MYSQLDB')

# URL encode the password to handle special characters
encoded_password = urllib.parse.quote_plus(MYSQLPASSWORD)

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQLUSER}:{encoded_password}@{MYSQLHOST}:{MYSQLPORT}/{MYSQLDB}'
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
        session['logout_token'] = std_secrets.token_hex(16)
    
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
    # Make sure the upload folder exists with proper permissions
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.chmod(app.config['UPLOAD_FOLDER'], 0o777)  # Full permissions
            logger.info(f"Created missing upload folder: {app.config['UPLOAD_FOLDER']}")
        except Exception as e:
            logger.error(f"Failed to create upload folder: {str(e)}")
            # Fall back to user's home directory if temp folder creation fails
            fallback_dir = os.path.expanduser("~/medicai_temp")
            os.makedirs(fallback_dir, exist_ok=True)
            app.config['UPLOAD_FOLDER'] = fallback_dir
            logger.info(f"Using fallback directory: {fallback_dir}")
    
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
        'uploaded_file', 'cleaned_file', 'selected_features_file', 'predictions_file',
        'uploaded_file_regression', 'cleaned_file_regression', 'selected_features_regression_file', 
        'selected_features_regression_file_json', 'feature_importance_regression_file'
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
    """Check if the file type and size are allowed"""
    # Check file extension
    allowed_extensions = {'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def check_file_size(file, max_size_mb=2):
    """Check if file size is within the limit"""
    # Get the file size in bytes
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)  # Reset file pointer to beginning
    
    # Convert max_size_mb to bytes
    max_size_bytes = max_size_mb * 1024 * 1024
    
    return size <= max_size_bytes

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
    """Root route - renders the home page"""
    return render_template('home.html')

@app.route('/welcome')
def welcome():
    """Welcome page with generic intro message"""
    if not current_user.is_authenticated:
        flash('Please log in to access the application.', 'info')
        return redirect('/login', code=302)
        
    return render_template('welcome.html')

@app.route('/home')
def home():
    """Home page - accessible to everyone, logged in or not"""
    return render_template('home.html')

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
        
        # Validate password
        if not password:
            flash('Password is required.', 'danger')
            return redirect(url_for('register'))
            
        # Check password requirements
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('register'))
            
        if not any(c.isupper() for c in password):
            flash('Password must contain at least one uppercase letter.', 'danger')
            return redirect(url_for('register'))
            
        if not any(c.islower() for c in password):
            flash('Password must contain at least one lowercase letter.', 'danger')
            return redirect(url_for('register'))
            
        if not any(c.isdigit() for c in password):
            flash('Password must contain at least one number.', 'danger')
            return redirect(url_for('register'))
            
        if not any(c in '@$!%*?&' for c in password):
            flash('Password must contain at least one special character (@$!%*?&).', 'danger')
            return redirect(url_for('register'))
        
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

@app.route('/force_logout', methods=['POST'])
def force_logout():
    """Force logout endpoint for JavaScript calls"""
    # Perform the same logout actions
    logout_user()
    session.clear()
    
    # Return a success response
    return jsonify({'status': 'success', 'message': 'User logged out successfully'})

# Function to save data to a temporary file
def save_to_temp_file(data, prefix='data'):
    """Save data to a temporary file and return the filepath"""
    # First try the configured temp directory
    filepath = get_temp_filepath(extension='.json')
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=SafeJSONEncoder)
        return filepath
    except PermissionError as e:
        # If permission issue, try a fallback directory in user's home
        logger.error(f"Permission error saving to {filepath}: {str(e)}")
        try:
            fallback_dir = os.path.expanduser("~/medicai_temp")
            os.makedirs(fallback_dir, exist_ok=True)
            alt_filepath = os.path.join(fallback_dir, os.path.basename(filepath))
            with open(alt_filepath, 'w') as f:
                json.dump(data, f, cls=SafeJSONEncoder)
            logger.info(f"Successfully saved to fallback location: {alt_filepath}")
            return alt_filepath
        except Exception as inner_e:
            logger.error(f"Error saving to fallback location: {str(inner_e)}")
            # Last resort: try saving to the static directory
            try:
                static_dir = os.path.join('static', 'temp')
                os.makedirs(static_dir, exist_ok=True)
                static_filepath = os.path.join(static_dir, f"{uuid.uuid4()}.json")
                with open(static_filepath, 'w') as f:
                    json.dump(data, f, cls=SafeJSONEncoder)
                logger.info(f"Successfully saved to static directory: {static_filepath}")
                return static_filepath
            except Exception as last_e:
                logger.error(f"Critical error - failed to save file anywhere: {str(last_e)}")
                raise

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

@app.route('/about')
def about():
    """About page """
    return render_template('about.html')

@app.route('/privacy-policy')
def privacy_policy():
    """Privacy policy page"""
    return render_template('privacy_policy.html')

@app.route('/terms-of-service')
def terms_of_service():
    """Terms of service page"""
    return render_template('terms_of_service.html')

@app.route('/cookie-policy')
def cookie_policy():
    """Cookie policy page"""
    return render_template('cookie_policy.html')

@app.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('documentation.html')

@app.route('/blog')
def blog():
    """Blog page"""
    return render_template('blog.html')

@app.route('/case-studies')
def case_studies():
    """Case studies page"""
    return render_template('case_studies.html')

@app.route('/support')
def support():
    """Support page"""
    return render_template('support.html')

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

def is_service_available(service_url):
    """Check if a service is available"""
    try:
        response = requests.get(f"{service_url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Create a directory for downloads
DOWNLOADS_FOLDER = os.path.join('static', 'downloads')
if not os.path.exists(DOWNLOADS_FOLDER):
    os.makedirs(DOWNLOADS_FOLDER)

# Define the API endpoints
@app.route('/api/status/tabular_classification_training', methods=['GET'])
@login_required
def api_status_tabular_classification_training():
    user_id = current_user.id
    if user_id in classification_training_status:
        return jsonify(classification_training_status[user_id])
    else:
        # If not found, assume completed or not started
        return jsonify({
            'status': 'not_found',
            'overall_progress': 0,
            'overall_status': 'No training in progress'
        })

@app.route('/api/reset/tabular_classification_training', methods=['POST'])
@login_required
def api_reset_tabular_classification_training():
    user_id = current_user.id
    
    # Clear the training status
    if user_id in classification_training_status:
        classification_training_status.pop(user_id, None)
    
    # Clear training-related session data
    keys_to_remove = [
        'model_results', 'last_training_run_id', 'run_id', 'trained_models',
        'selected_features', 'cleaned_file', 'anomaly_results',
        'current_training_dataset_id', 'feature_importance_file',
        'selected_features_file', 'selected_features_file_json'
    ]
    
    for key in keys_to_remove:
        if key in session:
            session.pop(key, None)
    
    # Also remove any keys that might be from previous training sessions
    session_keys = list(session.keys())
    for key in session_keys:
        if key.startswith('old_') or key.endswith('_cached'):
            session.pop(key, None)
    
    return jsonify({'status': 'success'})

@app.route('/api/start/tabular_classification_training', methods=['POST'])
@login_required
def api_start_tabular_classification_training():
    user_id = current_user.id
    # Clear any existing training status first
    if user_id in classification_training_status:
        classification_training_status.pop(user_id, None)
        
    # Clear any previous training results
    keys_to_remove = [
        'model_results', 'last_training_run_id', 'run_id', 'trained_models',
        'selected_features', 'cleaned_file', 'anomaly_results',
        'current_training_dataset_id', 'feature_importance_file',
        'selected_features_file', 'selected_features_file_json'
    ]
    
    for key in keys_to_remove:
        if key in session:
            session.pop(key, None)
    
    # Initialize the training status
    classification_training_status[user_id] = {
        'status': 'in_progress',
        'overall_progress': 0,
        'overall_status': 'Training initialized. Preparing data...',
        'model_statuses': {
            'logistic-regression': {'progress': 0, 'status': 'Initializing...'},
            'decision-tree': {'progress': 0, 'status': 'Initializing...'},
            'random-forest': {'progress': 0, 'status': 'Initializing...'},
            'knn': {'progress': 0, 'status': 'Initializing...'},
            'svm': {'progress': 0, 'status': 'Initializing...'},
            'naive-bayes': {'progress': 0, 'status': 'Initializing...'}
        }
    }
    
    # In a real implementation, we would call the training process in a background task here
    # For now, we just return the initial status
    return jsonify(classification_training_status[user_id])

if __name__ == '__main__':
    # Import application modules
    from app_tabular import *
    from app_images import *
    from app_others import *

    
    # Register the chatbot blueprint
    app = register_chatbot_blueprint(app)
    
    # Ensure the database exists
    with app.app_context():
        # Try to create all tables if they don't exist
        try:
            db.create_all()
            print("Database tables created/verified")
        except Exception as e:
            print(f"Error creating database tables: {str(e)}")
    
    # Start the Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)