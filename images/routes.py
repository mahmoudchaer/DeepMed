"""
Route definitions for the image analysis module.
"""

from flask import render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_login import login_required, current_user
import secrets
import os
import zipfile
import tempfile
import shutil
import json
import requests
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

from . import images_bp
from .utils import allowed_image_file, get_temp_filepath
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('images')

# Directory for temporary uploaded files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Temp directory: {TEMP_DIR}")

# Directory for datasets
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'datasets')
os.makedirs(DATASET_DIR, exist_ok=True)
logger.info(f"Dataset directory: {DATASET_DIR}")

# Directory for trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'trained_models')
os.makedirs(MODELS_DIR, exist_ok=True)
logger.info(f"Models directory: {MODELS_DIR}")

@images_bp.route('/', methods=['GET'])
@login_required
def index():
    """Main route for image analysis dashboard"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the image analysis.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    # Placeholder for actual service health checks
    services_status = {
        "Image Classification": "healthy",
        "Image Segmentation": "unhealthy",
        "Object Detection": "unhealthy"
    }
    
    # Get active training jobs if any
    active_job = session.get('active_training_job')
    
    # If there's an active job, check its status
    job_status = None
    if active_job:
        try:
            # For a real implementation, this would be a request to the model API
            # Here we just check if the job info file exists
            status_url = active_job.get('status_url')
            if status_url:
                # This would be a real HTTP request in production
                try:
                    # For demonstration, we just check if job_id exists
                    job_id = active_job.get('job_id')
                    if job_id:
                        job_status = {
                            'job_id': job_id,
                            'status': 'pending'  # Placeholder
                        }
                except Exception as e:
                    logger.error(f"Error checking job status: {str(e)}")
        except Exception as e:
            logger.error(f"Error retrieving job status: {str(e)}")
    
    return render_template('images.html', 
                         services_status=services_status, 
                         active_job=active_job,
                         job_status=job_status,
                         logout_token=session['logout_token'])

@images_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    """Handle zip file upload with categorized images"""
    if not current_user.is_authenticated:
        flash('Please log in to upload files.', 'warning')
        return redirect(url_for('login'))
    
    if 'imageFile' not in request.files:
        flash('No file part in the request', 'error')
        return redirect(url_for('images.index'))
    
    file = request.files['imageFile']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('images.index'))
    
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
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            
            # Get categories (folders)
            categories = [f for f in os.listdir(extracted_dir) 
                        if os.path.isdir(os.path.join(extracted_dir, f))]
            
            # Count images in each category
            category_counts = {}
            total_images = 0
            
            for category in categories:
                category_path = os.path.join(extracted_dir, category)
                image_files = [f for f in os.listdir(category_path) 
                             if allowed_image_file(f)]
                category_counts[category] = len(image_files)
                total_images += len(image_files)
            
            # Store the extracted path in session for later processing
            session['extracted_dataset_path'] = extracted_dir
            session['dataset_categories'] = categories
            session['analysis_type'] = request.form.get('analysisType', 'classification')
            
            # Create summary message
            category_summary = ", ".join([f"{cat} ({count})" for cat, count in category_counts.items()])
            
            logger.info(f"Extracted zip with {len(categories)} categories: {category_summary}")
            
            # If the analysis type is classification, start training an EfficientNet model
            if session['analysis_type'] == 'classification':
                # Send request to the EfficientNet-B0 model API
                try:
                    # Simulating a successful API response
                    job_id = secrets.token_hex(8)
                    simulated_response = {
                        'job_id': job_id,
                        'id': job_id,  # Add an 'id' property for consistency
                        'status_url': url_for('images.job_status', _external=True),
                        'message': 'Training job started successfully'
                    }
                    
                    # Store the job information in session
                    session['active_training_job'] = simulated_response
                    
                    flash(f'Dataset uploaded with {len(categories)} categories: {category_summary}. Started training EfficientNet-B0 model.', 'success')
                    
                except Exception as e:
                    logger.error(f"Error starting model training: {str(e)}")
                    flash(f'Dataset uploaded but error starting model training: {str(e)}', 'error')
            else:
                # For other analysis types
                flash(f'Dataset uploaded with {len(categories)} categories: {category_summary}. Total images: {total_images}', 'success')
            
        except Exception as e:
            logger.error(f"Error processing zip file: {str(e)}")
            flash(f'Error processing zip file: {str(e)}', 'error')
            # Clean up the temporary directory
            shutil.rmtree(extract_dir, ignore_errors=True)
            return redirect(url_for('images.index'))
            
        return redirect(url_for('images.index'))
    else:
        flash('Invalid file type. Please upload a ZIP file.', 'error')
        return redirect(url_for('images.index'))

@images_bp.route('/job_status', methods=['GET'])
@login_required
def job_status():
    """Check the status of the current training job"""
    # In a real implementation, this would forward the request to the model's status endpoint
    # For now, we'll just return simulated status data
    
    active_job = session.get('active_training_job')
    if not active_job:
        return jsonify({
            'error': 'No active job found'
        }), 404
    
    job_id = active_job.get('job_id')
    if not job_id:
        return jsonify({
            'error': 'Invalid job information'
        }), 400
    
    # In a real implementation, this would fetch the actual status from the model API
    # For now, we'll just return a simulated status
    import random
    
    # Simulate different job statuses based on job_id for demo purposes
    # In a real implementation, we would query the actual job status
    simulated_statuses = ['initializing', 'preparing', 'training', 'evaluating', 'completed']
    simulated_idx = int(job_id, 16) % len(simulated_statuses)
    simulated_status = simulated_statuses[simulated_idx]
    
    response = {
        'job_id': job_id,
        'status': simulated_status,
        'start_time': 1616703703.8231947,  # Simulated timestamp
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
    
    # If "completed", add download URLs
    if simulated_status == 'completed':
        response['download_urls'] = {
            'keras_model': url_for('images.download_model', format='keras', _external=True),
            'tflite_model': url_for('images.download_model', format='tflite', _external=True),
            'saved_model': url_for('images.download_model', format='saved_model', _external=True)
        }
    
    return jsonify(response), 200

@images_bp.route('/download_model', methods=['GET'])
@login_required
def download_model():
    """Placeholder for downloading a trained model"""
    format_type = request.args.get('format', 'keras')
    
    # In a real implementation, this would forward to the model's download endpoint
    # For now, just return a message
    return jsonify({
        'message': f'Model download not yet implemented. Requested format: {format_type}',
        'note': 'This is a placeholder. In a real implementation, you would get the actual model file.'
    }), 501  # 501 = Not Implemented 

@images_bp.route('/process_features', methods=['POST'])
def process_features():
    """Handle client-side extracted features for privacy-preserving model training."""
    logger.info("Process features endpoint called")
    
    try:
        analysis_type = request.form.get('analysisType', 'classification')
        logger.info(f"Analysis type: {analysis_type}")
        
        # Check if we got the feature data
        if 'featureData' not in request.form:
            logger.error("No feature data provided in request")
            return jsonify({'error': 'No feature data provided'}), 400
        
        # Get the feature data
        logger.debug("Parsing feature data")
        feature_data = json.loads(request.form.get('featureData'))
        logger.info(f"Received feature data with {len(feature_data.get('categories', []))} categories")
        
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        logger.info(f"Generated analysis ID: {analysis_id}")
        
        # Create a directory to store the feature data
        feature_dir = os.path.join(DATASET_DIR, f"features_{analysis_id}")
        logger.debug(f"Creating feature directory: {feature_dir}")
        os.makedirs(feature_dir, exist_ok=True)
        
        # Save the feature data
        feature_file = os.path.join(feature_dir, 'features.json')
        logger.debug(f"Saving feature data to: {feature_file}")
        with open(feature_file, 'w') as f:
            json.dump(feature_data, f)
        
        # Extract information from feature data
        categories = feature_data['categories']
        total_images = feature_data['totalImages']
        logger.info(f"Feature data contains {len(categories)} categories and {total_images} images")
        
        # Set dynamic hyperparameters based on dataset size
        hyperparams = calculate_hyperparameters(total_images, len(categories))
        logger.info(f"Calculated hyperparameters: {hyperparams}")
        
        # Store analysis info in session
        session['analysis'] = {
            'id': analysis_id,
            'type': analysis_type,
            'categories': {cat: feature_data['features'].get(cat, []) for cat in categories},
            'total_images': total_images,
            'hyperparams': hyperparams,
            'privacy_mode': True,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        logger.debug("Analysis info stored in session")
        
        if analysis_type == 'classification':
            logger.info("Starting feature-based EfficientNet training")
            # Initiate model training with features instead of raw images
            job_id = start_efficientnet_feature_training(feature_dir, hyperparams)
            logger.info(f"Feature-based training job started with ID: {job_id}")
            
            # Store job info in session
            session['active_training_job'] = {
                'id': job_id,
                'job_id': job_id,
                'analysis_id': analysis_id,
                'type': analysis_type,
                'status_url': f'/images/job_status?job_id={job_id}'
            }
            logger.debug("Job info stored in session")
            
            return jsonify({
                'message': 'Features processed and training initiated',
                'redirect': url_for('images.training_status')
            })
        else:
            logger.warning(f"Analysis type not yet implemented: {analysis_type}")
            return jsonify({
                'message': 'Features processed. This analysis type is not yet implemented.',
                'redirect': url_for('images.index')
            })
            
    except Exception as e:
        logger.error(f"Error processing features: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing features: {str(e)}'}), 500

def calculate_hyperparameters(total_images, num_categories):
    """
    Calculate optimal hyperparameters based on dataset size and complexity.
    
    Args:
        total_images (int): Total number of images in the dataset
        num_categories (int): Number of categories in the dataset
        
    Returns:
        dict: Dictionary of hyperparameters
    """
    logger.info(f"Calculating hyperparameters for {total_images} images in {num_categories} categories")
    
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
        logger.debug("Small dataset detected, reducing batch size")
        hyperparams['batch_size'] = 16
    elif total_images > 1000:
        logger.debug("Large dataset detected, increasing batch size")
        hyperparams['batch_size'] = 64
    
    # Adjust epochs based on total images
    if total_images < 200:
        logger.debug("Small dataset detected, reducing epochs")
        hyperparams['epochs'] = 10
    elif total_images > 2000:
        logger.debug("Large dataset detected, increasing epochs")
        hyperparams['epochs'] = 30
    
    # Adjust augmentation intensity based on dataset size
    if total_images < 300:
        logger.debug("Small dataset detected, increasing augmentation intensity")
        hyperparams['augmentation_intensity'] = 'high'  # More augmentation for small datasets
    elif total_images > 1000:
        logger.debug("Large dataset detected, reducing augmentation intensity")
        hyperparams['augmentation_intensity'] = 'low'   # Less augmentation for large datasets
    
    # Adjust learning rate based on number of categories
    if num_categories > 10:
        logger.debug("Many categories detected, reducing learning rate")
        hyperparams['learning_rate'] = 0.0005
    
    logger.info(f"Final hyperparameters: {hyperparams}")
    return hyperparams

def start_efficientnet_training(dataset_path, hyperparams):
    """
    Start a training job for the EfficientNet-B0 model.
    
    Args:
        dataset_path (str): Path to the dataset
        hyperparams (dict): Hyperparameters for training
        
    Returns:
        str: Job ID
    """
    logger.info(f"Starting EfficientNet training with dataset: {dataset_path}")
    logger.debug(f"Hyperparameters: {hyperparams}")
    
    # This would normally make an API call to the model service
    # For now, we'll simulate it
    job_id = str(uuid.uuid4())
    logger.info(f"Generated job ID: {job_id}")
    
    # In a real implementation, this would be:
    try:
        logger.info("Attempting to call model API")
        # response = requests.post(
        #     'http://localhost:5100/models/efficientnet_b0/train',
        #     json={
        #         'dataset_path': dataset_path,
        #         'hyperparameters': hyperparams
        #     }
        # )
        # job_id = response.json().get('job_id')
        # logger.info(f"API response: {response.json()}")
        
        # For now, simulate the API call
        logger.info("Using simulated API response")
    except Exception as e:
        logger.error(f"Error calling model API: {str(e)}", exc_info=True)
    
    return job_id

def start_efficientnet_feature_training(feature_dir, hyperparams):
    """
    Start a training job for the EfficientNet-B0 model using pre-extracted features.
    
    Args:
        feature_dir (str): Path to the directory containing feature data
        hyperparams (dict): Hyperparameters for training
        
    Returns:
        str: Job ID
    """
    logger.info(f"Starting feature-based EfficientNet training with features from: {feature_dir}")
    logger.debug(f"Hyperparameters: {hyperparams}")
    
    # This would normally make an API call to the model service
    # For now, we'll simulate it
    job_id = str(uuid.uuid4())
    logger.info(f"Generated job ID: {job_id}")
    
    # In a real implementation, this would be:
    try:
        logger.info("Attempting to call model API for feature training")
        # response = requests.post(
        #     'http://localhost:5100/models/efficientnet_b0/train_features',
        #     json={
        #         'feature_path': feature_dir,
        #         'hyperparameters': hyperparams
        #     }
        # )
        # job_id = response.json().get('job_id')
        # logger.info(f"API response: {response.json()}")
        
        # For now, simulate the API call
        logger.info("Using simulated API response")
    except Exception as e:
        logger.error(f"Error calling model API for feature training: {str(e)}", exc_info=True)
    
    return job_id

@images_bp.route('/training_status')
def training_status():
    """Show training status page."""
    logger.info("Training status page requested")
    
    if 'active_training_job' not in session:
        logger.error("No active training job found in session")
        flash('No active training job found', 'warning')
        return redirect(url_for('images.index'))
    
    logger.debug(f"Rendering training status page for job: {session['active_training_job'].get('id')}")
    return render_template('training_status.html')

@images_bp.route('/download_model')
def download_model():
    """Download the trained model."""
    logger.info("Download model endpoint called")
    
    if 'active_training_job' not in session:
        logger.error("No active job found in session")
        flash('No active job found', 'warning')
        return redirect(url_for('images.index'))
    
    job_id = session['active_training_job'].get('id')
    model_format = request.args.get('format', 'keras')
    logger.info(f"Download requested for job {job_id} in format {model_format}")
    
    try:
        logger.info("Attempting to call model API for model download")
        # This would normally make an API call to the model service
        # For now, we'll return a placeholder
        # In a real implementation, this would be:
        # response = requests.get(
        #     f'http://localhost:5100/models/efficientnet_b0/jobs/{job_id}/model?format={model_format}',
        #     stream=True
        # )
        # if response.ok:
        #     return send_file(
        #         io.BytesIO(response.content),
        #         as_attachment=True,
        #         download_name=f'efficientnet_b0_{job_id}.{model_format}',
        #         mimetype='application/octet-stream'
        #     )
        
        logger.info("Model download API not implemented yet")
        return "Model download API not implemented yet"
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error downloading model: {str(e)}'}), 500

# Add this at the end of the file
logger.info("Images routes module loaded successfully") 