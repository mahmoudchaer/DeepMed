"""
MedicAI - Medical Image Analysis Application
"""

import os
import zipfile
import tempfile
import shutil
import json
import uuid
import logging
import secrets
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'dev-key-for-testing'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload
app.config['JSON_SORT_KEYS'] = False

# Create required directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
DATASET_FOLDER = os.path.join(os.getcwd(), 'datasets')
MODELS_FOLDER = os.path.join(os.getcwd(), 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Helper functions
def allowed_image_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_hyperparameters(total_images, num_categories):
    """Calculate optimal hyperparameters based on dataset size and complexity."""
    # Base hyperparameters
    hyperparams = {
        'img_size': 224,  # Default size for EfficientNet-B0
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 0.001,
        'augmentation_intensity': 'medium'  # low, medium, high
    }
    
    # Adjust batch size based on total images
    if total_images < 100:
        hyperparams['batch_size'] = 16
    elif total_images > 1000:
        hyperparams['batch_size'] = 64
    
    # Adjust epochs based on total images
    if total_images < 200:
        hyperparams['epochs'] = 10
    elif total_images > 2000:
        hyperparams['epochs'] = 30
    
    logger.info(f"Calculated hyperparameters: {hyperparams}")
    return hyperparams

# Routes
@app.route('/')
def index():
    logger.info("Accessing index page")
    return render_template('index.html')

@app.route('/images')
def images():
    """Main route for image analysis dashboard"""
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = {
        "Image Classification": "healthy",
        "Image Segmentation": "unhealthy",
        "Object Detection": "unhealthy"
    }
    
    # Get active training jobs if any
    active_job = session.get('active_training_job')
    job_status = None
    
    if active_job:
        job_id = active_job.get('job_id')
        if job_id:
            job_status = {
                'job_id': job_id,
                'status': 'pending'  # Placeholder
            }
    
    return render_template('images.html', 
                         services_status=services_status, 
                         active_job=active_job,
                         job_status=job_status,
                         logout_token=session.get('logout_token', ''))

@app.route('/images/upload', methods=['POST'])
def upload():
    """Handle zip file upload with categorized images"""
    logger.info("Upload endpoint called")
    logger.debug(f"Request details: files={list(request.files.keys())}")
    
    if 'imageFile' not in request.files:
        logger.error("No file part in the request")
        flash('No file part in the request', 'error')
        return redirect(url_for('images'))
    
    file = request.files['imageFile']
    if file.filename == '':
        logger.error("No file selected")
        flash('No file selected', 'error')
        return redirect(url_for('images'))
    
    logger.info(f"File uploaded: {file.filename}")
    
    if file and file.filename.endswith('.zip'):
        # Create a temporary directory to extract the zip file
        extract_dir = tempfile.mkdtemp(prefix='zip_extract_')
        zip_path = os.path.join(extract_dir, 'dataset.zip')
        
        try:
            # Save the zip file
            file.save(zip_path)
            logger.info(f"Saved zip file to {zip_path}")
            
            # Create directory for extraction
            extracted_dir = os.path.join(extract_dir, 'extracted')
            os.makedirs(extracted_dir, exist_ok=True)
            logger.debug(f"Created extraction directory: {extracted_dir}")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            logger.info(f"Extracted ZIP file to {extracted_dir}")
            
            # Get categories (folders)
            categories = [f for f in os.listdir(extracted_dir) 
                        if os.path.isdir(os.path.join(extracted_dir, f))]
            logger.info(f"Found {len(categories)} categories: {categories}")
            
            # Count images in each category
            category_counts = {}
            total_images = 0
            
            for category in categories:
                category_path = os.path.join(extracted_dir, category)
                image_files = [f for f in os.listdir(category_path) 
                             if allowed_image_file(f)]
                category_counts[category] = len(image_files)
                total_images += len(image_files)
            
            logger.info(f"Image counts per category: {category_counts}")
            logger.info(f"Total images: {total_images}")
            
            # Store the extracted path in session for later processing
            session['extracted_dataset_path'] = extracted_dir
            session['dataset_categories'] = categories
            session['analysis_type'] = request.form.get('analysisType', 'classification')
            
            # Create summary message
            category_summary = ", ".join([f"{cat} ({count})" for cat, count in category_counts.items()])
            
            logger.info(f"Extracted zip with {len(categories)} categories: {category_summary}")
            
            # If the analysis type is classification, start training an EfficientNet model
            if session['analysis_type'] == 'classification':
                try:
                    logger.info("Starting classification model training")
                    
                    # Calculate hyperparameters based on dataset size
                    hyperparams = calculate_hyperparameters(total_images, len(categories))
                    
                    # Simulating a successful API response
                    job_id = secrets.token_hex(8)
                    logger.info(f"Generated simulated job_id: {job_id}")
                    
                    simulated_response = {
                        'job_id': job_id,
                        'id': job_id,
                        'status_url': url_for('job_status', _external=True),
                        'message': 'Training job started successfully'
                    }
                    
                    # Store the job information in session
                    session['active_training_job'] = simulated_response
                    logger.info(f"Stored job info in session: {simulated_response}")
                    
                    flash(f'Dataset uploaded with {len(categories)} categories: {category_summary}. Started training model.', 'success')
                    
                except Exception as e:
                    logger.error(f"Error starting model training: {str(e)}", exc_info=True)
                    flash(f'Dataset uploaded but error starting model training: {str(e)}', 'error')
            else:
                # For other analysis types
                flash(f'Dataset uploaded with {len(categories)} categories: {category_summary}. Total images: {total_images}', 'success')
            
        except Exception as e:
            logger.error(f"Error processing zip file: {str(e)}", exc_info=True)
            flash(f'Error processing zip file: {str(e)}', 'error')
            # Clean up the temporary directory
            shutil.rmtree(extract_dir, ignore_errors=True)
            return redirect(url_for('images'))
        
        logger.info("Upload successful, redirecting to images page")
        return redirect(url_for('images'))
    else:
        logger.error(f"Invalid file type: {file.filename}")
        flash('Invalid file type. Please upload a ZIP file.', 'error')
        return redirect(url_for('images'))

@app.route('/images/process_features', methods=['POST'])
def process_features():
    """Handle client-side extracted features for privacy-preserving model training."""
    logger.info("Process features endpoint called")
    logger.debug(f"Request details: form keys={list(request.form.keys())}")
    
    try:
        analysis_type = request.form.get('analysisType', 'classification')
        logger.info(f"Analysis type: {analysis_type}")
        
        # Check if we got the feature data
        if 'featureData' not in request.form:
            logger.error("No feature data provided in request")
            return jsonify({'error': 'No feature data provided'}), 400
        
        # Get the feature data
        logger.debug("Parsing feature data")
        try:
            raw_feature_data = request.form.get('featureData')
            feature_data = json.loads(raw_feature_data)
            logger.info(f"Received feature data with {len(feature_data.get('categories', []))} categories")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing feature data: {str(e)}", exc_info=True)
            return jsonify({'error': f'Invalid feature data format: {str(e)}'}), 400
        
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        logger.info(f"Generated analysis ID: {analysis_id}")
        
        # Create a directory to store the feature data
        dataset_dir = os.path.join('images', 'data', 'datasets')
        feature_dir = os.path.join(dataset_dir, f"features_{analysis_id}")
        logger.debug(f"Creating feature directory: {feature_dir}")
        os.makedirs(feature_dir, exist_ok=True)
        
        # Save the feature data
        feature_file = os.path.join(feature_dir, 'features.json')
        logger.debug(f"Saving feature data to: {feature_file}")
        with open(feature_file, 'w') as f:
            json.dump(feature_data, f)
        logger.info(f"Feature data saved to {feature_file}")
        
        # Extract information from feature data
        categories = feature_data['categories']
        total_images = feature_data['totalImages']
        logger.info(f"Feature data contains {len(categories)} categories and {total_images} images")
        
        # Set dynamic hyperparameters based on dataset size
        hyperparams = calculate_hyperparameters(total_images, len(categories))
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        
        # Store job info in session
        session['active_training_job'] = {
            'id': job_id,
            'job_id': job_id,
            'analysis_id': analysis_id,
            'type': analysis_type,
            'status_url': url_for('job_status', _external=True)
        }
        logger.debug("Job info stored in session")
        
        return jsonify({
            'message': 'Features processed and training initiated',
            'redirect': url_for('training_status')
        })
            
    except Exception as e:
        logger.error(f"Error processing features: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing features: {str(e)}'}), 500

@app.route('/images/job_status')
def job_status():
    """Check the status of the current training job"""
    logger.info("Job status endpoint called")
    
    active_job = session.get('active_training_job')
    if not active_job:
        logger.error("No active training job found in session")
        return jsonify({
            'error': 'No active job found'
        }), 404
    
    job_id = active_job.get('job_id')
    if not job_id:
        logger.error("Invalid job information - no job_id found")
        return jsonify({
            'error': 'Invalid job information'
        }), 400
    
    logger.info(f"Checking status of job: {job_id}")
    
    # Simulate different job statuses
    import random
    simulated_statuses = ['initializing', 'preparing', 'training', 'evaluating', 'completed']
    simulated_idx = int(job_id.replace('-', '')[:8], 16) % len(simulated_statuses)
    simulated_status = simulated_statuses[simulated_idx]
    
    response = {
        'job_id': job_id,
        'status': simulated_status,
        'start_time': datetime.now().timestamp() - 300,  # Started 5 minutes ago
        'progress': {
            'status': simulated_status,
            'current_epoch': random.randint(1, 10),
            'total_epochs': 10,
            'accuracy': random.random(),
            'loss': random.random() * 0.5,
            'val_accuracy': random.random(),
            'val_loss': random.random() * 0.5,
            'message': f'Simulated status: {simulated_status}'
        }
    }
    
    return jsonify(response), 200

@app.route('/images/training_status')
def training_status():
    """Show training status page."""
    logger.info("Training status page requested")
    
    if 'active_training_job' not in session:
        flash('No active training job found', 'warning')
        return redirect(url_for('images'))
    
    return render_template('training_status.html')

@app.route('/service_status')
def service_status():
    """Return service status information"""
    return jsonify({
        "Image Classification": "healthy",
        "Image Segmentation": "unhealthy", 
        "Object Detection": "unhealthy",
        "Medical Assistant": "unhealthy",
        "Model Coordinator": "unhealthy"
    })

@app.route('/test-images')
def test_images():
    """Simple test page for images module"""
    return render_template('test_upload.html')

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000) 