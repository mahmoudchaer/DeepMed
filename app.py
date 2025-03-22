import os
import io
import zipfile
import tempfile
import json
import requests
import logging
from requests_toolbelt.multipart.encoder import MultipartEncoder

from flask import Flask, request, jsonify, send_file, render_template, Response
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define URL for the EEP service
EEP_SERVICE_URL = "http://localhost:5100"  # This will be our new EEP service

# Alternative URLs to try if localhost doesn't work
ALTERNATIVE_EEP_URLS = [
    "http://127.0.0.1:5100",
    "http://0.0.0.0:5100",
    "http://image-eep-service:5100",  # Docker service name
    "http://host.docker.internal:5100"  # Special Docker DNS name for host
]

def check_model_service_health():
    """Check if the model service is running and healthy"""
    model_service_url = "http://localhost:5110/health"
    logger.info(f"Checking model service health at {model_service_url}")
    
    try:
        response = requests.get(model_service_url, timeout=5)
        healthy = response.status_code == 200
        logger.info(f"Model service health check result: {healthy}")
        return healthy
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to model service: {str(e)}")
        return False

def check_eep_service_health():
    """Check if the EEP service is running and healthy using multiple URLs"""
    global EEP_SERVICE_URL  # Move global declaration to the beginning of the function
    
    # Try primary URL first
    health_url = f"{EEP_SERVICE_URL}/health"
    logger.info(f"Checking EEP service health at {health_url}")
    
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"EEP service health check successful at {health_url}")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to primary EEP service: {str(e)}")
    
    # Try alternative URLs
    for url in ALTERNATIVE_EEP_URLS:
        alt_health_url = f"{url}/health"
        logger.info(f"Trying alternative EEP service URL: {alt_health_url}")
        try:
            response = requests.get(alt_health_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"EEP service health check successful at {alt_health_url}")
                # Update the main URL to the working one
                EEP_SERVICE_URL = url
                logger.info(f"Updated primary EEP service URL to: {EEP_SERVICE_URL}")
                return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to alternative EEP service at {url}: {str(e)}")
    
    logger.error("All EEP service URLs failed health checks")
    return False

def train_model(zip_file, num_classes=5, training_level=3):
    """
    Instead of training locally, send the data to the Docker container
    for processing and return the results.
    """
    # Check if the model service is healthy
    if not check_model_service_health():
        raise Exception("Model service is not available. Please ensure the Docker container is running.")
    
    # The URL of the model training service (Docker container)
    model_service_url = "http://localhost:5110/train"
    
    # Create form data to send to the service using MultipartEncoder
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
        raise Exception(error_message)
    
    # Get metrics from response header
    metrics = {}
    if 'X-Training-Metrics' in response.headers:
        try:
            metrics = json.loads(response.headers['X-Training-Metrics'])
        except:
            logger.error("Failed to parse metrics from model service response")
    
    # Convert the response content (model file) to BytesIO
    model_bytes = io.BytesIO(response.content)
    model_bytes.seek(0)
    
    return model_bytes, metrics

def augment_data(zip_file, augmentation_level=3):
    """
    Send data to the EEP for data augmentation with retry logic
    """
    # First check if the primary EEP service is healthy
    if not check_eep_service_health():
        # This will already have attempted alternative URLs
        raise Exception("EEP service is not available. Please ensure the Docker containers are running.")
    
    # The URL for data augmentation through the EEP
    augmentation_url = f"{EEP_SERVICE_URL}/data_augmentation"
    logger.info(f"Sending augmentation request to {augmentation_url}")
    
    # Create form data to send to the service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'augmentationLevel': str(augmentation_level)
        }
    )
    
    # Make the request to the EEP service
    try:
        logger.info(f"Attempting to send request to {augmentation_url}")
        response = requests.post(
            augmentation_url,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True,
            timeout=60  # Increased timeout for large datasets
        )
        
        # Log the response details
        logger.info(f"Received response from {augmentation_url}: Status {response.status_code}")
        logger.info(f"Response Content-Type: {response.headers.get('Content-Type', 'None')}")
        
        if response.status_code != 200:
            error_message = "Error from data augmentation service"
            try:
                # Check content type before trying to parse as JSON
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                    logger.error(f"Error details: {error_data}")
                else:
                    # If not JSON, log the response text
                    error_text = response.text[:200] if response.text else "No response text"
                    logger.error(f"Non-JSON error response: {error_text}")
                    error_message = f"Error {response.status_code} from augmentation service: {error_text}"
            except Exception as json_error:
                logger.error(f"Could not parse error response: {str(json_error)}")
                # Try to get some of the raw response as a string
                try:
                    error_text = response.text[:200] if response.text else "No response text"
                    error_message = f"Error {response.status_code}: {error_text}"
                except:
                    error_message = f"Error status {response.status_code} from augmentation service"
            
            raise Exception(error_message)
        
        # Verify the content type is what we expect
        content_type = response.headers.get('Content-Type', '')
        if 'application/zip' not in content_type:
            logger.warning(f"Unexpected content type received: {content_type}, expected application/zip")
        
        # Get metrics from response header
        metrics = {}
        if 'X-Augmentation-Metrics' in response.headers:
            try:
                metrics_json = response.headers['X-Augmentation-Metrics']
                logger.info(f"Raw metrics header: {metrics_json[:100]}")
                metrics = json.loads(metrics_json)
                logger.info(f"Received metrics: {metrics}")
            except Exception as metrics_error:
                logger.error(f"Failed to parse metrics from augmentation service response: {str(metrics_error)}")
        
        # Convert the response content (ZIP file) to BytesIO
        logger.info("Successfully received augmented data")
        augmented_zip = io.BytesIO(response.content)
        augmented_zip.seek(0)
        
        return augmented_zip, metrics
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when connecting to {augmentation_url}: {str(e)}")
        raise Exception(f"Failed to connect to augmentation service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during augmentation: {str(e)}", exc_info=True)
        raise

def process_data(zip_file, test_size=0.2, val_size=0.2):
    """
    Send data to the EEP for data processing with retry logic
    """
    # First check if the primary EEP service is healthy
    if not check_eep_service_health():
        # This will already have attempted alternative URLs
        raise Exception("EEP service is not available. Please ensure the Docker containers are running.")
    
    # The URL for data processing through the EEP
    processing_url = f"{EEP_SERVICE_URL}/data_processing"
    logger.info(f"Sending processing request to {processing_url}")
    
    # Create form data to send to the service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'testSize': str(test_size),
            'valSize': str(val_size)
        }
    )
    
    # Make the request to the EEP service
    try:
        logger.info(f"Attempting to send request to {processing_url}")
        response = requests.post(
            processing_url,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True,
            timeout=30  # Increased timeout
        )
        
        # Log the response details
        logger.info(f"Received response from {processing_url}: Status {response.status_code}")
        
        if response.status_code != 200:
            error_message = "Error from data processing service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
                logger.error(f"Error details: {error_data}")
            except:
                logger.error(f"Could not parse error response: {response.text[:200]}")
            raise Exception(error_message)
        
        # Get metrics from response header
        metrics = {}
        if 'X-Processing-Metrics' in response.headers:
            try:
                metrics = json.loads(response.headers['X-Processing-Metrics'])
                logger.info(f"Received metrics: {metrics}")
            except:
                logger.error("Failed to parse metrics from processing service response")
        
        # Convert the response content (ZIP file) to BytesIO
        logger.info("Successfully received processed data")
        processed_zip = io.BytesIO(response.content)
        processed_zip.seek(0)
        
        return processed_zip, metrics
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when connecting to {processing_url}: {str(e)}")
        raise Exception(f"Failed to connect to processing service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    # Render the training page template
    return render_template('train_model.html')

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        num_classes = int(request.form.get('numClasses', 5))
        training_level = int(request.form.get('trainingLevel', 3))
        
        # Train the model on the provided data via the Docker container
        model_bytes, metrics = train_model(zip_file, num_classes=num_classes, training_level=training_level)
        
        # Create a response with both the model file and metrics
        flask_response = Response(model_bytes.getvalue())
        flask_response.headers["Content-Type"] = "application/octet-stream"
        flask_response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics header if available
        if metrics:
            flask_response.headers["X-Training-Metrics"] = json.dumps(metrics)
        
        return flask_response
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/augment_data', methods=['POST'])
def api_augment_data():
    logger.info("Received request to /api/augment_data")
    
    if 'zipFile' not in request.files:
        logger.error("No ZIP file in request")
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        augmentation_level = int(request.form.get('augmentationLevel', 3))
        logger.info(f"Augmentation level: {augmentation_level}")
        
        # Log EEP service status
        eep_healthy = check_eep_service_health()
        logger.info(f"EEP service health check: {'Healthy' if eep_healthy else 'Unhealthy'}")
        
        # Augment the data via the EEP
        augmented_zip, metrics = augment_data(zip_file, augmentation_level=augmentation_level)
        
        # Create a response with the augmented data ZIP and metrics
        flask_response = Response(augmented_zip.getvalue())
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Add metrics header if available
        if metrics:
            flask_response.headers["X-Augmentation-Metrics"] = json.dumps(metrics)
        
        logger.info("Successfully processed augmentation request")
        return flask_response
    except Exception as e:
        logger.error(f"Error in data augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_data', methods=['POST'])
def api_process_data():
    logger.info("Received request to /api/process_data")
    
    if 'zipFile' not in request.files:
        logger.error("No ZIP file in request")
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        test_size = float(request.form.get('testSize', 0.2))
        val_size = float(request.form.get('valSize', 0.2))
        logger.info(f"Test size: {test_size}, Validation size: {val_size}")
        
        # Log EEP service status
        eep_healthy = check_eep_service_health()
        logger.info(f"EEP service health check: {'Healthy' if eep_healthy else 'Unhealthy'}")
        
        # Process the data via the EEP
        processed_zip, metrics = process_data(zip_file, test_size=test_size, val_size=val_size)
        
        # Create a response with the processed data ZIP and metrics
        flask_response = Response(processed_zip.getvalue())
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
        
        # Add metrics header if available
        if metrics:
            flask_response.headers["X-Processing-Metrics"] = json.dumps(metrics)
        
        logger.info("Successfully processed data processing request")
        return flask_response
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def check_health():
    """Health check endpoint that also verifies service connections"""
    model_service_healthy = check_model_service_health()
    eep_service_healthy = check_eep_service_health()
    
    return jsonify({
        "status": "healthy",
        "model_service_status": "connected" if model_service_healthy else "disconnected",
        "eep_service_status": "connected" if eep_service_healthy else "disconnected"
    })

if __name__ == '__main__':
    # Make sure all routes are registered
    print("Starting application with registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"Route: {rule.endpoint} - {rule.rule}")
    app.run(debug=True, host='0.0.0.0')
