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
import secrets
import zipfile
# Import the key vault module
import keyvault

# Import common components from app_api.py
from app_api import app, DATA_CLEANER_URL, FEATURE_SELECTOR_URL, MODEL_COORDINATOR_URL, AUGMENTATION_SERVICE_URL, MODEL_TRAINING_SERVICE_URL
from app_api import is_service_available, get_temp_filepath, safe_requests_post, cleanup_session_files, SafeJSONEncoder, logger
from app_api import check_services, save_to_temp_file, clean_data_for_json

# Define new URL for pipeline service
PIPELINE_SERVICE_URL = keyvault.getenv('PIPELINE_SERVICE_URL', 'http://localhost:5025')

# Define URL for anomaly detection service
ANOMALY_DETECTION_SERVICE_URL = keyvault.getenv('ANOMALY_DETECTION_SERVICE_URL', 'http://localhost:5030')

# Import database models
from db.users import db, TrainingRun, TrainingModel, PreprocessingData

@app.route('/images')
@login_required
def images():
    """This endpoint is deprecated and redirects to pipeline"""
    # Redirect to the pipeline page
    flash('The standalone training page has been removed. Please use the Pipeline functionality instead.', 'info')
    return redirect(url_for('pipeline'), code=302)

@app.route('/anomaly_detection')
@login_required
def anomaly_detection():
    """Route for anomaly detection page (PyTorch Autoencoder)"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the anomaly detection page.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    
    # Add anomaly detection service to services status
    services_status['anomaly_detection_service'] = is_service_available(ANOMALY_DETECTION_SERVICE_URL)
    
    return render_template('anomaly_detection.html', services_status=services_status, logout_token=session['logout_token'])

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
    
    try:
        # Check if the augmentation service is available
        if not is_service_available(AUGMENTATION_SERVICE_URL):
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
                    'level': level
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

@app.route('/pipeline')
@login_required
def pipeline():
    """Route for combined augmentation and training pipeline page"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the pipeline page.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    
    # Add pipeline service to services status
    services_status['pipeline_service'] = is_service_available(PIPELINE_SERVICE_URL)
    
    return render_template('pipeline.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/api/pipeline', methods=['POST'])
@login_required
def api_pipeline():
    """API endpoint for the image processing pipeline"""
    # Check for file upload
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    if not zip_file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not zip_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "File must be a ZIP archive"}), 400
    
    try:
        # Check if the pipeline service is available
        if not is_service_available(PIPELINE_SERVICE_URL):
            return jsonify({"error": "Pipeline service is not available. Please try again later."}), 503
        
        logger.info(f"Starting pipeline process for file: {zip_file.filename}")
        
        # Get form parameters
        perform_augmentation = request.form.get('performAugmentation', 'false')
        augmentation_level = request.form.get('augmentationLevel', '3')
        num_classes = request.form.get('numClasses', '5')
        training_level = request.form.get('trainingLevel', '3')
        
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_file.save(temp_file.name)
        temp_file.close()
        
        # Create form data to send to the pipeline service
        with open(temp_file.name, 'rb') as f:
            form_data = MultipartEncoder(
                fields={
                    'zipFile': (zip_file.filename, f, 'application/zip'),
                    'performAugmentation': perform_augmentation,
                    'augmentationLevel': augmentation_level,
                    'numClasses': num_classes,
                    'trainingLevel': training_level
                }
            )
            
            # Forward the request to the pipeline service
            headers = {'Content-Type': form_data.content_type}
            
            # Stream the request to the service
            response = requests.post(
                f"{PIPELINE_SERVICE_URL}/pipeline",
                headers=headers,
                data=form_data,
                stream=True
            )
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")
        
        # Check the response status
        if response.status_code != 200:
            error_message = "Error in pipeline service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in pipeline service (HTTP {response.status_code})"
            
            return jsonify({"error": error_message}), response.status_code
        
        # Create a Flask response with the model file
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=model_package.zip"
        
        # Forward any training metrics in headers
        if 'X-Training-Metrics' in response.headers:
            flask_response.headers["X-Training-Metrics"] = response.headers['X-Training-Metrics']
            flask_response.headers["Access-Control-Expose-Headers"] = "X-Training-Metrics"
        
        return flask_response
        
    except Exception as e:
        logger.error(f"Error in pipeline process: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/train_anomaly', methods=['POST'])
@login_required
def api_train_anomaly():
    """API endpoint for anomaly detection training"""
    # Check for file upload
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    if not zip_file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not zip_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "File must be a ZIP archive"}), 400
    
    try:
        # Check if the anomaly detection service is available
        if not is_service_available(ANOMALY_DETECTION_SERVICE_URL):
            return jsonify({"error": "Anomaly detection service is not available. Please try again later."}), 503
        
        logger.info(f"Starting anomaly detection training for file: {zip_file.filename}")
        
        # Get parameters
        level = request.form.get('level', '3')
        image_size = request.form.get('image_size', '256')
        
        # Save the uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_file.save(temp_file.name)
        temp_file.close()
        
        # Create form data to send to the anomaly detection service
        with open(temp_file.name, 'rb') as f:
            # Create fields dictionary
            fields = {
                'zipFile': (zip_file.filename, f, 'application/zip'),
                'level': level,
                'image_size': image_size
            }
            
            form_data = MultipartEncoder(fields=fields)
            
            # Forward the request to the anomaly detection service
            headers = {'Content-Type': form_data.content_type}
            
            # Stream the request to the service
            response = requests.post(
                f"{ANOMALY_DETECTION_SERVICE_URL}/train",
                headers=headers,
                data=form_data,
                stream=True,
                timeout=600  # Increase timeout to 10 minutes
            )
        
        # Clean up the temporary file
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")
        
        # Check the response status
        if response.status_code != 200:
            error_message = "Error in anomaly detection service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in anomaly detection service (HTTP {response.status_code})"
            
            return jsonify({"error": error_message}), response.status_code
        
        # Create a Flask response with the model zip file
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=anomaly_detection_model.zip"
        
        return flask_response
        
    except Exception as e:
        logger.error(f"Error in anomaly detection training: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/images_prediction')
@login_required
def images_prediction():
    """Route for image prediction page that uses custom models"""
    # Check if the user is logged in
    if not current_user.is_authenticated:
        flash('Please log in to access the image prediction page.', 'info')
        return redirect('/login', code=302)
    
    # Generate a CSRF token for logout form if needed
    if 'logout_token' not in session:
        session['logout_token'] = secrets.token_hex(16)
    
    # Check services health for status display
    services_status = check_services()
    
    # Add images prediction service to services status
    predictor_service_url = keyvault.getenv('PREDICTOR_SERVICE_URL', 'http://localhost:5100')
    services_status['predictor_service'] = is_service_available(predictor_service_url)
    
    return render_template('images_prediction.html', services_status=services_status, logout_token=session['logout_token'])

@app.route('/api/predict_image', methods=['POST'])
@login_required
def api_predict_image():
    """API endpoint for image prediction using custom models"""
    # Check for file uploads
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both model package and input file are required"}), 400
    
    model_package = request.files['model_package']
    input_file = request.files['input_file']
    
    if not model_package.filename or not input_file.filename:
        return jsonify({"error": "Both model package and input file must be selected"}), 400
    
    # Validate file extensions
    if not model_package.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    try:
        # Define the predictor service URL
        predictor_service_url = keyvault.getenv('PREDICTOR_SERVICE_URL', 'http://localhost:5100')
        
        # Check if the predictor service is available
        if not is_service_available(predictor_service_url):
            return jsonify({"error": "Prediction service is not available. Please try again later."}), 503
        
        logger.info(f"Starting image prediction for model: {model_package.filename} and input: {input_file.filename}")
        
        # Forward the files to the prediction service
        files = {
            'model_package': (model_package.filename, model_package.stream, 'application/zip'),
            'input_file': (input_file.filename, input_file.stream, 'application/octet-stream')
        }
        
        # Send request to the predictor service
        response = requests.post(
            f"{predictor_service_url}/predict",
            files=files,
            timeout=600  # 10 minute timeout
        )
        
        # Check response status
        if response.status_code != 200:
            error_message = "Error in prediction service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in prediction service (HTTP {response.status_code})"
            
            return jsonify({"error": error_message}), response.status_code
        
        # Return prediction results
        return jsonify(response.json())
        
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/prediction_details', methods=['POST'])
@login_required
def api_prediction_details():
    """Get prediction details"""
    # Check for file upload
    if 'model_package' not in request.files or 'input_file' not in request.files:
        return jsonify({"error": "Both model package and input file are required"}), 400
    
    model_package = request.files['model_package']
    input_file = request.files['input_file']
    
    if not model_package.filename or not input_file.filename:
        return jsonify({"error": "Both model package and input file must be selected"}), 400
    
    # Validate file extensions
    if not model_package.filename.lower().endswith('.zip'):
        return jsonify({"error": "Model package must be a ZIP archive"}), 400
    
    try:
        # Define the predictor service URL
        predictor_service_url = keyvault.getenv('PREDICTOR_SERVICE_URL', 'http://localhost:5100')
        
        # Check if the predictor service is available
        if not is_service_available(predictor_service_url):
            return jsonify({"error": "Prediction service is not available. Please try again later."}), 503
        
        logger.info(f"Starting prediction details for model: {model_package.filename} and input: {input_file.filename}")
        
        # Forward the files to the prediction service
        files = {
            'model_package': (model_package.filename, model_package.stream, 'application/zip'),
            'input_file': (input_file.filename, input_file.stream, 'application/octet-stream')
        }
        
        # Send request to the predictor service
        response = requests.post(
            f"{predictor_service_url}/predict",
            files=files,
            timeout=600  # 10 minute timeout
        )
        
        # Check response status
        if response.status_code != 200:
            error_message = "Error in prediction service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in prediction service (HTTP {response.status_code})"
            
            return jsonify({"error": error_message}), response.status_code
        
        # Return prediction details
        return jsonify(response.json())
        
    except Exception as e:
        logger.error(f"Error in prediction details: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(keyvault.getenv("PORT", 5023))
    logger.info(f"Starting augmentation service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)


