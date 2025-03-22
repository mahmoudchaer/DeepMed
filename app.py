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

def check_model_service_health():
    """Check if the model service is running and healthy"""
    try:
        response = requests.get("http://localhost:5110/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def check_eep_service_health():
    """Check if the EEP service is running and healthy"""
    try:
        response = requests.get(f"{EEP_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
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
    Send data to the EEP for data augmentation
    """
    # Check if the EEP service is healthy
    if not check_eep_service_health():
        raise Exception("EEP service is not available. Please ensure the Docker containers are running.")
    
    # The URL for data augmentation through the EEP
    augmentation_url = f"{EEP_SERVICE_URL}/data_augmentation"
    
    # Create form data to send to the service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'augmentationLevel': str(augmentation_level)
        }
    )
    
    # Make the request to the EEP service
    response = requests.post(
        augmentation_url,
        data=form_data,
        headers={'Content-Type': form_data.content_type},
        stream=True
    )
    
    if response.status_code != 200:
        error_message = "Error from data augmentation service"
        try:
            error_data = response.json()
            if 'error' in error_data:
                error_message = error_data['error']
        except:
            pass
        raise Exception(error_message)
    
    # Get metrics from response header
    metrics = {}
    if 'X-Augmentation-Metrics' in response.headers:
        try:
            metrics = json.loads(response.headers['X-Augmentation-Metrics'])
        except:
            logger.error("Failed to parse metrics from augmentation service response")
    
    # Convert the response content (ZIP file) to BytesIO
    augmented_zip = io.BytesIO(response.content)
    augmented_zip.seek(0)
    
    return augmented_zip, metrics

def process_data(zip_file, test_size=0.2, val_size=0.2):
    """
    Send data to the EEP for data processing
    """
    # Check if the EEP service is healthy
    if not check_eep_service_health():
        raise Exception("EEP service is not available. Please ensure the Docker containers are running.")
    
    # The URL for data processing through the EEP
    processing_url = f"{EEP_SERVICE_URL}/data_processing"
    
    # Create form data to send to the service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'testSize': str(test_size),
            'valSize': str(val_size)
        }
    )
    
    # Make the request to the EEP service
    response = requests.post(
        processing_url,
        data=form_data,
        headers={'Content-Type': form_data.content_type},
        stream=True
    )
    
    if response.status_code != 200:
        error_message = "Error from data processing service"
        try:
            error_data = response.json()
            if 'error' in error_data:
                error_message = error_data['error']
        except:
            pass
        raise Exception(error_message)
    
    # Get metrics from response header
    metrics = {}
    if 'X-Processing-Metrics' in response.headers:
        try:
            metrics = json.loads(response.headers['X-Processing-Metrics'])
        except:
            logger.error("Failed to parse metrics from processing service response")
    
    # Convert the response content (ZIP file) to BytesIO
    processed_zip = io.BytesIO(response.content)
    processed_zip.seek(0)
    
    return processed_zip, metrics

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
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        augmentation_level = int(request.form.get('augmentationLevel', 3))
        
        # Augment the data via the EEP
        augmented_zip, metrics = augment_data(zip_file, augmentation_level=augmentation_level)
        
        # Create a response with the augmented data ZIP and metrics
        flask_response = Response(augmented_zip.getvalue())
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Add metrics header if available
        if metrics:
            flask_response.headers["X-Augmentation-Metrics"] = json.dumps(metrics)
        
        return flask_response
    except Exception as e:
        logger.error(f"Error in data augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_data', methods=['POST'])
def api_process_data():
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        test_size = float(request.form.get('testSize', 0.2))
        val_size = float(request.form.get('valSize', 0.2))
        
        # Process the data via the EEP
        processed_zip, metrics = process_data(zip_file, test_size=test_size, val_size=val_size)
        
        # Create a response with the processed data ZIP and metrics
        flask_response = Response(processed_zip.getvalue())
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
        
        # Add metrics header if available
        if metrics:
            flask_response.headers["X-Processing-Metrics"] = json.dumps(metrics)
        
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
    app.run(debug=True)
