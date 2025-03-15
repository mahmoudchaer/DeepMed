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
from flask import Blueprint, request, jsonify, send_file, current_app
import shutil
import traceback
from .model import EfficientNetB0Classifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Blueprint for the EfficientNet-B0 API routes
efficientnet_bp = Blueprint('efficientnet', __name__, url_prefix='/models/efficientnet_b0')

# Dictionary to store active training jobs
active_jobs = {}

# Directory for storing trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'trained_models')
os.makedirs(MODELS_DIR, exist_ok=True)

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
    try:
        # Parse request data
        data = request.get_json()
        
        # Check if dataset_path is provided
        if 'dataset_path' not in data:
            return jsonify({'error': 'dataset_path is required'}), 400
            
        dataset_path = data['dataset_path']
        
        # Check if dataset path exists
        if not os.path.isdir(dataset_path):
            return jsonify({'error': f'Dataset path not found: {dataset_path}'}), 404
        
        # Extract hyperparameters
        hyperparams = data.get('hyperparameters', {})
        
        # Create a unique output directory
        job_id = str(uuid.uuid4())
        output_dir = os.path.join(MODELS_DIR, f'efficientnet_b0_{job_id}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create model instance
        model = EfficientNetB0Classifier(
            dataset_path=dataset_path,
            output_dir=output_dir,
            img_size=hyperparams.get('img_size', 224),
            batch_size=hyperparams.get('batch_size', 32),
            epochs=hyperparams.get('epochs', 20),
            learning_rate=hyperparams.get('learning_rate', 0.001),
            augmentation_intensity=hyperparams.get('augmentation_intensity', 'medium')
        )
        
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
        
        # Start training in a separate thread
        thread = threading.Thread(
            target=_run_training_job,
            args=(job_id,),
            daemon=True
        )
        thread.start()
        
        active_jobs[job_id]['thread'] = thread
        active_jobs[job_id]['status'] = 'processing'
        
        # Return job ID and status URL
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'status_url': f'/models/efficientnet_b0/jobs/{job_id}/status'
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
    try:
        # Parse request data
        data = request.get_json()
        
        # Check if feature_path is provided
        if 'feature_path' not in data:
            return jsonify({'error': 'feature_path is required'}), 400
            
        feature_path = data['feature_path']
        
        # Check if feature file exists
        feature_file = os.path.join(feature_path, 'features.json')
        if not os.path.isfile(feature_file):
            return jsonify({'error': f'Feature file not found: {feature_file}'}), 404
        
        # Extract hyperparameters
        hyperparams = data.get('hyperparameters', {})
        
        # Create a unique output directory
        job_id = str(uuid.uuid4())
        output_dir = os.path.join(MODELS_DIR, f'efficientnet_b0_features_{job_id}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create model instance
        model = EfficientNetB0Classifier(
            dataset_path=feature_path,  # We'll use this for path reference only
            output_dir=output_dir,
            batch_size=hyperparams.get('batch_size', 32),
            epochs=hyperparams.get('epochs', 20),
            learning_rate=hyperparams.get('learning_rate', 0.001)
        )
        
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
        
        # Start training in a separate thread
        thread = threading.Thread(
            target=_run_feature_training_job,
            args=(job_id,),
            daemon=True
        )
        thread.start()
        
        active_jobs[job_id]['thread'] = thread
        active_jobs[job_id]['status'] = 'processing'
        
        # Return job ID and status URL
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'status_url': f'/models/efficientnet_b0/jobs/{job_id}/status'
        })
        
    except Exception as e:
        logger.error(f"Error starting feature-based training: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def _run_training_job(job_id):
    """Run the training job in a background thread."""
    job = active_jobs[job_id]
    model = job['model']
    
    try:
        # Run the full training pipeline
        results = model.run_full_training_pipeline()
        
        # Update job status
        job['status'] = 'completed'
        job['results'] = results
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        # Update job status on error
        job['status'] = 'failed'
        job['error'] = str(e)
        
        logger.error(f"Training job {job_id} failed: {str(e)}")
        traceback.print_exc()

def _run_feature_training_job(job_id):
    """Run the feature-based training job in a background thread."""
    job = active_jobs[job_id]
    model = job['model']
    feature_path = job['feature_path']
    
    try:
        # Run the feature-based training pipeline
        results = model.run_feature_training_pipeline(feature_path)
        
        # Update job status
        job['status'] = 'completed'
        job['results'] = results
        
        logger.info(f"Feature-based training job {job_id} completed successfully")
        
    except Exception as e:
        # Update job status on error
        job['status'] = 'failed'
        job['error'] = str(e)
        
        logger.error(f"Feature-based training job {job_id} failed: {str(e)}")
        traceback.print_exc()

@efficientnet_bp.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """
    Get the status of a training job.
    
    Args:
        job_id (str): The job ID
        
    Returns:
        JSON with job status information
    """
    if job_id not in active_jobs:
        return jsonify({'error': f'Job {job_id} not found'}), 404
    
    job = active_jobs[job_id]
    
    # Get progress info from the model's progress.json file
    progress_file = os.path.join(job['output_dir'], 'progress.json')
    progress_data = {}
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading progress file: {str(e)}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - job['start_time']
    
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
        response['error'] = job['error']
    
    # Add download URLs if job completed
    if job['status'] == 'completed':
        response['download_urls'] = {
            'keras': f'/models/efficientnet_b0/jobs/{job_id}/model?format=keras',
            'tflite': f'/models/efficientnet_b0/jobs/{job_id}/model?format=tflite',
            'saved_model': f'/models/efficientnet_b0/jobs/{job_id}/model?format=saved_model'
        }
        
        # Add evaluation metrics if available
        eval_file = os.path.join(job['output_dir'], 'evaluation_results.json')
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    response['evaluation'] = json.load(f)
            except Exception as e:
                logger.error(f"Error reading evaluation file: {str(e)}")
    
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
    if job_id not in active_jobs:
        return jsonify({'error': f'Job {job_id} not found'}), 404
    
    job = active_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Model training has not completed yet'}), 400
    
    # Get requested format
    model_format = request.args.get('format', 'keras')
    
    # Define file paths for different formats
    if model_format == 'keras':
        model_path = os.path.join(job['output_dir'], 'final_model.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(job['output_dir'], 'best_model.h5')
        
        if os.path.exists(model_path):
            return send_file(
                model_path,
                as_attachment=True,
                download_name=f'efficientnet_b0_{job_id}.h5',
                mimetype='application/octet-stream'
            )
    
    elif model_format == 'tflite':
        model_path = os.path.join(job['output_dir'], 'model.tflite')
        
        if os.path.exists(model_path):
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
                # Create the zip file
                import zipfile
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, os.path.dirname(model_dir))
                            zipf.write(file_path, rel_path)
            
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'efficientnet_b0_{job_id}_saved_model.zip',
                mimetype='application/zip'
            )
    
    # If we get here, the model file wasn't found
    return jsonify({'error': f'Model file in {model_format} format not found'}), 404

def init_app(app):
    """Register the blueprint with the Flask application."""
    app.register_blueprint(efficientnet_bp)
    logger.info("Registered EfficientNet-B0 blueprint") 