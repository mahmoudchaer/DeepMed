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

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, MODEL_COORDINATOR_URL, AUGMENTATION_SERVICE_URL, MODEL_TRAINING_SERVICE_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, cleanup_session_files, SafeJSONEncoder, logger
from app_api import check_services, save_to_temp_file, clean_data_for_json

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
        
    except Exception as e:
        logger.error(f"Error in image augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


