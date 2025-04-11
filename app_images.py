from flask import render_template, request, redirect, url_for, session, flash, send_file, jsonify, Response
from flask_login import login_required, current_user
import os
import json
import tempfile
import pandas as pd
import numpy as np
import requests
import logging
import io
from datetime import datetime
from werkzeug.utils import secure_filename
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid
import traceback
import atexit
import time
import threading

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, MODEL_COORDINATOR_URL, AUGMENTATION_SERVICE_URL, MODEL_TRAINING_SERVICE_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, cleanup_session_files, SafeJSONEncoder, logger
from app_api import check_services, save_to_temp_file, clean_data_for_json

# Import data augmentation module - update path to use the Docker module
from docker_for_images.data_augmentation import ClassificationDatasetAugmentor

# Import database models
from db.users import db, TrainingRun, TrainingModel, PreprocessingData

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
        use_docker_service = is_service_available(AUGMENTATION_SERVICE_URL)
        
        if use_docker_service:
            logger.info(f"Starting augmentation process with Docker service for file: {zip_file.filename}")
            
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
                    headers=headers,
                    data=m,
                    stream=True
                )
            
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
            
            # Check the response status
            if response.status_code != 200:
                error_message = "Error in augmentation service"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                except:
                    error_message = f"Error in augmentation service (HTTP {response.status_code})"
                
                return jsonify({"error": error_message}), response.status_code
            
            # Create a Flask response with the file data
            flask_response = Response(response.content)
            flask_response.headers["Content-Type"] = "application/zip"
            flask_response.headers["Content-Disposition"] = f"attachment; filename=augmented_{zip_file.filename}"
            
            return flask_response
        else:
            # Fallback to direct implementation if Docker service is unavailable
            logger.info(f"Docker augmentation service unavailable, using direct implementation for file: {zip_file.filename}")
            
            # Save the uploaded file to a temporary location
            temp_input_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            zip_file.save(temp_input_zip.name)
            temp_input_zip.close()
            temp_files.append(temp_input_zip.name)
            
            # Create a temporary output zip file
            temp_output_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            temp_output_zip.close()
            temp_files.append(temp_output_zip.name)
            
            try:
                # Initialize the augmentor with worker processes
                num_workers = min(os.cpu_count(), 8)  # Use at most 8 workers
                augmentor = ClassificationDatasetAugmentor(
                    image_size=(224, 224),
                    num_workers=num_workers,
                    batch_size=32,
                    use_gpu=False
                )
                
                start_time = time.time()
                
                # Process the dataset directly
                augmentor.process_dataset(
                    zip_path=temp_input_zip.name,
                    output_zip=temp_output_zip.name,
                    level=int(level),
                    num_augmentations=int(num_augmentations)
                )
                
                processing_time = time.time() - start_time
                logger.info(f"Completed augmentation in {processing_time:.2f} seconds")
                
                # Read the file into memory
                with open(temp_output_zip.name, 'rb') as f:
                    file_data = io.BytesIO(f.read())
                
                # Clean up the temporary files
                cleanup_temp_files([temp_input_zip.name, temp_output_zip.name])
                
                # Return the file
                return send_file(
                    file_data,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name=f'augmented_{zip_file.filename}'
                )
            except Exception as e:
                # Clean up temporary files in case of error
                cleanup_temp_files([temp_input_zip.name, temp_output_zip.name])
                raise e
    except Exception as e:
        logger.error(f"Error in image augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Track temp files for cleanup
temp_files = []

# Track processing status
processing_jobs = {}

@app.route('/augmentation/health', methods=['GET'])
def augmentation_health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "augmentation-service"})

# Direct implementation of the augmentation function (for reference)
def direct_augment_dataset(zip_file=None, level=3, num_augmentations=2):
    """
    Direct implementation to augment a dataset (without going through the Docker service)
    
    Args:
        zip_file: The uploaded ZIP file
        level: Augmentation level (1-5)
        num_augmentations: Number of augmentations per image
    
    Returns:
        Flask response with the augmented dataset
    """
    try:
        if zip_file is None:
            return jsonify({"error": "No ZIP file provided"}), 400
        
        # Validate parameters
        if level < 1 or level > 5:
            return jsonify({"error": "Augmentation level must be between 1 and 5"}), 400
        
        if num_augmentations < 1 or num_augmentations > 10:
            return jsonify({"error": "Number of augmentations must be between 1 and 10"}), 400
        
        # Log the start of processing
        file_size = 0
        zip_file.seek(0, os.SEEK_END)
        file_size = zip_file.tell()
        zip_file.seek(0)
        logger.info(f"Starting augmentation of file {zip_file.filename} ({file_size/1024/1024:.2f} MB)")
        
        # Save the uploaded zip file temporarily
        temp_input_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        zip_file.save(temp_input_zip.name)
        temp_input_zip.close()
        
        # Add to the list of files to clean up
        temp_files.append(temp_input_zip.name)
        
        # Create a temporary output zip file
        temp_output_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        temp_output_zip.close()
        
        # Add to the list of files to clean up
        temp_files.append(temp_output_zip.name)
        
        # Initialize the augmentor with more worker processes for large datasets
        num_workers = min(os.cpu_count(), 8)  # Use at most 8 workers
        augmentor = ClassificationDatasetAugmentor(
            image_size=(224, 224),
            num_workers=num_workers,
            batch_size=32,  # Larger batch size for efficiency
            use_gpu=False  # Set to True if GPU is available
        )
        
        start_time = time.time()
        
        # Process the dataset
        augmentor.process_dataset(
            zip_path=temp_input_zip.name,
            output_zip=temp_output_zip.name,
            level=level,
            num_augmentations=num_augmentations
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Completed augmentation in {processing_time:.2f} seconds")
        
        # Read the file into memory
        with open(temp_output_zip.name, 'rb') as f:
            file_data = io.BytesIO(f.read())
        
        # Clean up the temporary files immediately
        cleanup_temp_files([temp_input_zip.name, temp_output_zip.name])
        
        # Return the file from memory
        return send_file(
            file_data,
            mimetype='application/zip',
            as_attachment=True,
            download_name='augmented_dataset.zip'
        )
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Error: {error_msg}")
        return jsonify({"error": str(e), "details": error_msg}), 500

def cleanup_temp_files(file_list=None):
    """Clean up temporary files"""
    global temp_files
    
    if file_list is None:
        file_list = temp_files
    
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                if file_path in temp_files:
                    temp_files.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {file_path}: {e}")

# Register cleanup function to run at exit
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5023))
    logger.info(f"Starting augmentation service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)


