"""
API interface for the EfficientNet-B0 model.

This module provides routes for training and downloading the model.
"""

import os
import uuid
import threading
import time
import logging
import json
import sys
import traceback
from flask import Blueprint, request, jsonify, send_file, current_app
import shutil

# Set up comprehensive logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('efficientnet_api.log')
    ]
)
logger = logging.getLogger('efficientnet_api')
logger.info("EfficientNet-B0 API module initializing")

# Try both ways to import the model
try:
    # First try relative import
    from .model import EfficientNetB0Classifier
    logger.info("Imported model using relative import")
except ImportError:
    # Then try direct import
    from model import EfficientNetB0Classifier
    logger.info("Imported model using direct import")

# Create a Blueprint for the EfficientNet-B0 API routes
logger.info("Creating EfficientNet-B0 Blueprint")
efficientnet_bp = Blueprint('efficientnet', __name__, url_prefix='/models/efficientnet_b0')

# Dictionary to store active training jobs
active_jobs = {}
logger.info("Active jobs dictionary initialized")

# Directory for storing trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'trained_models')
os.makedirs(MODELS_DIR, exist_ok=True)
logger.info(f"Models directory: {MODELS_DIR}")

@efficientnet_bp.route('/train', methods=['POST'])
def train_model():
    """
    Start training an EfficientNet-B0 model on the provided dataset.
    
    Expected JSON payload:
    {
        "dataset_path": "/path/to/dataset",
        "hyperparameters": {
            "img_size": 224,          // Optional, default 224
            "batch_size": 32,         // Optional, default 32
            "epochs": 20,             // Optional, default 20
            "learning_rate": 0.001,   // Optional, default 0.001
            "augmentation_intensity": "medium"  // Optional, default "medium"
        }
    }
    
    Returns:
        JSON with job_id and status_url
    """
    logger.info("Train model endpoint called")
    
    try:
        # Parse request data
        data = request.get_json()
        logger.debug(f"Request data: {data}")
        
        # Check if dataset_path is provided
        if 'dataset_path' not in data:
            logger.error("Missing required parameter: dataset_path")
            return jsonify({'error': 'dataset_path is required'}), 400
            
        dataset_path = data['dataset_path']
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check if dataset path exists
        if not os.path.isdir(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return jsonify({'error': f'Dataset path not found: {dataset_path}'}), 404
        
        # Extract hyperparameters
        hyperparams = data.get('hyperparameters', {})
        logger.info(f"Hyperparameters: {hyperparams}")
        
        # Create a unique output directory
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        
        output_dir = os.path.join(MODELS_DIR, f'efficientnet_b0_{job_id}')
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")
        
        # Create model instance
        logger.info("Initializing EfficientNetB0Classifier")
        model = EfficientNetB0Classifier(
            dataset_path=dataset_path,
            output_dir=output_dir,
            img_size=hyperparams.get('img_size', 224),
            batch_size=hyperparams.get('batch_size', 32),
            epochs=hyperparams.get('epochs', 20),
            learning_rate=hyperparams.get('learning_rate', 0.001),
            augmentation_intensity=hyperparams.get('augmentation_intensity', 'medium')
        )
        logger.debug("Model initialized successfully")
        
        # Store job info
        active_jobs[job_id] = {
            'id': job_id,
            'status': 'initializing',
            'model': model,
            'start_time': time.time(),
            'dataset_path': dataset_path,
            'output_dir': output_dir,
            'thread': None
        }
        logger.debug(f"Job info stored with ID: {job_id}")
        
        # Start training in a separate thread
        logger.info(f"Starting training thread for job {job_id}")
        thread = threading.Thread(
            target=_run_training_job,
            args=(job_id,),
            daemon=True
        )
        thread.start()
        
        active_jobs[job_id]['thread'] = thread
        active_jobs[job_id]['status'] = 'processing'
        logger.info(f"Training thread started for job {job_id}")
        
        # Return job ID and status URL
        response = {
            'job_id': job_id,
            'status': 'processing',
            'status_url': f'/models/efficientnet_b0/jobs/{job_id}/status'
        }
        logger.info(f"Returning response for job {job_id}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error starting training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@efficientnet_bp.route('/train_features', methods=['POST'])
def train_model_with_features():
    """
    Start training an EfficientNet-B0 model using pre-extracted features.
    
    Expected JSON payload:
    {
        "feature_path": "/path/to/features.json",
        "hyperparameters": {
            "batch_size": 32,         // Optional, default 32
            "epochs": 20,             // Optional, default 20
            "learning_rate": 0.001    // Optional, default 0.001
        }
    }
    
    Returns:
        JSON with job_id and status_url
    """
    logger.info("Train with features endpoint called")
    
    try:
        # Parse request data
        data = request.get_json()
        logger.debug(f"Request data: {data}")
        
        # Check if feature_path is provided
        if 'feature_path' not in data:
            logger.error("Missing required parameter: feature_path")
            return jsonify({'error': 'feature_path is required'}), 400
            
        feature_path = data['feature_path']
        logger.info(f"Feature path: {feature_path}")
        
        # Check if feature file exists
        feature_file = os.path.join(feature_path, 'features.json')
        if not os.path.isfile(feature_file):
            logger.error(f"Feature file not found: {feature_file}")
            return jsonify({'error': f'Feature file not found: {feature_file}'}), 404
        
        # Extract hyperparameters
        hyperparams = data.get('hyperparameters', {})
        logger.info(f"Hyperparameters: {hyperparams}")
        
        # Create a unique output directory
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")
        
        output_dir = os.path.join(MODELS_DIR, f'efficientnet_b0_features_{job_id}')
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")
        
        # Create model instance
        logger.info("Initializing EfficientNetB0Classifier for feature-based training")
        model = EfficientNetB0Classifier(
            dataset_path=feature_path,  # We'll use this for path reference only
            output_dir=output_dir,
            batch_size=hyperparams.get('batch_size', 32),
            epochs=hyperparams.get('epochs', 20),
            learning_rate=hyperparams.get('learning_rate', 0.001)
        )
        logger.debug("Model initialized successfully")
        
        # Store job info
        active_jobs[job_id] = {
            'id': job_id,
            'status': 'initializing',
            'model': model,
            'start_time': time.time(),
            'feature_path': feature_file,
            'output_dir': output_dir,
            'thread': None,
            'is_feature_based': True
        }
        logger.debug(f"Job info stored with ID: {job_id}")
        
        # Start training in a separate thread
        logger.info(f"Starting feature-based training thread for job {job_id}")
        thread = threading.Thread(
            target=_run_feature_training_job,
            args=(job_id,),
            daemon=True
        )
        thread.start()
        
        active_jobs[job_id]['thread'] = thread
        active_jobs[job_id]['status'] = 'processing'
        logger.info(f"Feature-based training thread started for job {job_id}")
        
        # Return job ID and status URL
        response = {
            'job_id': job_id,
            'status': 'processing',
            'status_url': f'/models/efficientnet_b0/jobs/{job_id}/status'
        }
        logger.info(f"Returning response for job {job_id}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error starting feature-based training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

def _run_training_job(job_id):
    """Run the training job in a background thread."""
    logger.info(f"Training job {job_id} started")
    
    job = active_jobs[job_id]
    model = job['model']
    
    try:
        # Run the full training pipeline
        logger.info(f"Starting full training pipeline for job {job_id}")
        results = model.run_full_training_pipeline()
        
        # Update job status
        job['status'] = 'completed'
        job['results'] = results
        
        logger.info(f"Training job {job_id} completed successfully")
        logger.debug(f"Training results: {results}")
        
    except Exception as e:
        # Update job status on error
        error_msg = str(e)
        job['status'] = 'failed'
        job['error'] = error_msg
        
        logger.error(f"Training job {job_id} failed: {error_msg}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")

def _run_feature_training_job(job_id):
    """Run the feature-based training job in a background thread."""
    logger.info(f"Feature-based training job {job_id} started")
    
    job = active_jobs[job_id]
    model = job['model']
    feature_path = job['feature_path']
    
    try:
        # Run the feature-based training pipeline
        logger.info(f"Starting feature-based training pipeline for job {job_id}")
        logger.debug(f"Feature path: {feature_path}")
        
        results = model.run_feature_training_pipeline(feature_path)
        
        # Update job status
        job['status'] = 'completed'
        job['results'] = results
        
        logger.info(f"Feature-based training job {job_id} completed successfully")
        logger.debug(f"Training results: {results}")
        
    except Exception as e:
        # Update job status on error
        error_msg = str(e)
        job['status'] = 'failed'
        job['error'] = error_msg
        
        logger.error(f"Feature-based training job {job_id} failed: {error_msg}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")

@efficientnet_bp.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """
    Get the status of a training job.
    
    Args:
        job_id (str): The job ID
        
    Returns:
        JSON with job status information
    """
    logger.info(f"Status requested for job {job_id}")
    
    if job_id not in active_jobs:
        logger.error(f"Job {job_id} not found")
        return jsonify({'error': f'Job {job_id} not found'}), 404
    
    job = active_jobs[job_id]
    logger.debug(f"Found job: {job_id}, status: {job['status']}")
    
    # Get progress info from the model's progress.json file
    progress_file = os.path.join(job['output_dir'], 'progress.json')
    progress_data = {}
    
    if os.path.exists(progress_file):
        try:
            logger.debug(f"Reading progress file: {progress_file}")
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            logger.debug(f"Progress data: {progress_data}")
        except Exception as e:
            logger.error(f"Error reading progress file: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Progress file not found: {progress_file}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - job['start_time']
    logger.debug(f"Job elapsed time: {elapsed_time:.2f} seconds")
    
    # Prepare response
    response = {
        'job_id': job_id,
        'status': job['status'],
        'start_time': job['start_time'],
        'elapsed_time': elapsed_time,
        'progress': progress_data
    }
    
    # Add error info if job failed
    if job['status'] == 'failed' and 'error' in job:
        logger.debug(f"Job failed with error: {job['error']}")
        response['error'] = job['error']
    
    # Add download URLs if job completed
    if job['status'] == 'completed':
        logger.debug(f"Job completed, adding download URLs")
        response['download_urls'] = {
            'keras': f'/models/efficientnet_b0/jobs/{job_id}/model?format=keras',
            'tflite': f'/models/efficientnet_b0/jobs/{job_id}/model?format=tflite',
            'saved_model': f'/models/efficientnet_b0/jobs/{job_id}/model?format=saved_model'
        }
        
        # Add evaluation metrics if available
        eval_file = os.path.join(job['output_dir'], 'evaluation_results.json')
        if os.path.exists(eval_file):
            try:
                logger.debug(f"Reading evaluation file: {eval_file}")
                with open(eval_file, 'r') as f:
                    response['evaluation'] = json.load(f)
                logger.debug("Evaluation data added to response")
            except Exception as e:
                logger.error(f"Error reading evaluation file: {str(e)}", exc_info=True)
    
    logger.info(f"Returning status response for job {job_id}")
    return jsonify(response)

@efficientnet_bp.route('/jobs/<job_id>/model', methods=['GET'])
def download_model(job_id):
    """
    Download a trained model.
    
    Args:
        job_id (str): The job ID
        
    Query Parameters:
        format (str): Model format (keras, tflite, or saved_model)
        
    Returns:
        The model file for download
    """
    logger.info(f"Model download requested for job {job_id}")
    
    if job_id not in active_jobs:
        logger.error(f"Job {job_id} not found")
        return jsonify({'error': f'Job {job_id} not found'}), 404
    
    job = active_jobs[job_id]
    
    if job['status'] != 'completed':
        logger.error(f"Job {job_id} training not completed yet, status: {job['status']}")
        return jsonify({'error': 'Model training has not completed yet'}), 400
    
    # Get requested format
    model_format = request.args.get('format', 'keras')
    logger.info(f"Requested model format: {model_format}")
    
    # Define file paths for different formats
    if model_format == 'keras':
        model_path = os.path.join(job['output_dir'], 'final_model.h5')
        if not os.path.exists(model_path):
            logger.debug(f"Final model not found, trying best model at: {model_path}")
            model_path = os.path.join(job['output_dir'], 'best_model.h5')
        
        if os.path.exists(model_path):
            logger.info(f"Sending keras model file: {model_path}")
            return send_file(
                model_path,
                as_attachment=True,
                download_name=f'efficientnet_b0_{job_id}.h5',
                mimetype='application/octet-stream'
            )
    
    elif model_format == 'tflite':
        model_path = os.path.join(job['output_dir'], 'model.tflite')
        
        if os.path.exists(model_path):
            logger.info(f"Sending TFLite model file: {model_path}")
            return send_file(
                model_path,
                as_attachment=True,
                download_name=f'efficientnet_b0_{job_id}.tflite',
                mimetype='application/octet-stream'
            )
    
    elif model_format == 'saved_model':
        model_dir = os.path.join(job['output_dir'], 'saved_model')
        
        if os.path.exists(model_dir):
            # Create a zip file of the SavedModel directory
            zip_path = os.path.join(job['output_dir'], 'saved_model.zip')
            
            if not os.path.exists(zip_path):
                logger.info(f"Creating ZIP archive of SavedModel directory: {model_dir}")
                # Create the zip file
                import zipfile
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, os.path.dirname(model_dir))
                            logger.debug(f"Adding to ZIP: {file_path} -> {rel_path}")
                            zipf.write(file_path, rel_path)
            
            logger.info(f"Sending SavedModel ZIP file: {zip_path}")
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'efficientnet_b0_{job_id}_saved_model.zip',
                mimetype='application/zip'
            )
    
    # If we get here, the model file wasn't found
    logger.error(f"Model file not found for job {job_id} in format {model_format}")
    return jsonify({'error': f'Model file in {model_format} format not found'}), 404

def init_app(app):
    """Register the blueprint with the Flask application."""
    logger.info("Registering EfficientNet-B0 blueprint with Flask app")
    app.register_blueprint(efficientnet_bp)
    logger.info("EfficientNet-B0 blueprint registered successfully")

# Add this at the end of the file
logger.info("EfficientNet-B0 API module loaded successfully") 