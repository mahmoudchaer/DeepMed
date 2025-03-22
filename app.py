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

# Define the service URLs
EEP_URL = "http://localhost:5020"
MODEL_TRAINING_URL = "http://localhost:5010"
DATA_PROCESSING_URL = "http://localhost:5011"
DATA_AUGMENTATION_URL = "http://localhost:5012"

def check_service_health(service_url):
    """Check if a service is running and healthy"""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def check_eep_health():
    """Check if the EEP service is running and healthy"""
    return check_service_health(EEP_URL)

def check_model_service_health():
    """Check if the model service is running and healthy"""
    return check_service_health(MODEL_TRAINING_URL)

def check_data_processing_health():
    """Check if the data processing service is running and healthy"""
    return check_service_health(DATA_PROCESSING_URL)

def check_data_augmentation_health():
    """Check if the data augmentation service is running and healthy"""
    return check_service_health(DATA_AUGMENTATION_URL)

def train_model(zip_file, num_classes=5, training_level=3, use_augmentation=False, 
               validation_split=0.2, test_split=0.1):
    """
    Send the data to the EEP service for processing and model training.
    
    Parameters:
    - zip_file: The uploaded ZIP file containing image data
    - num_classes: Number of classes in the dataset
    - training_level: Level of training (1-5, higher = more thorough)
    - use_augmentation: Whether to use data augmentation
    - validation_split: Percentage of data to use for validation
    - test_split: Percentage of data to use for testing
    
    Returns:
    - model_bytes: Bytes of the trained model
    - metrics: Training metrics
    """
    # Check if the EEP service is healthy
    if not check_eep_health():
        raise Exception("EEP service is not available. Please ensure the Docker container is running.")
    
    # Create form data to send to the EEP service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'numClasses': str(num_classes),
            'trainingLevel': str(training_level),
            'useAugmentation': str(use_augmentation).lower(),
            'validationSplit': str(validation_split),
            'testSplit': str(test_split)
        }
    )
    
    # Make the request to the EEP service
    response = requests.post(
        f"{EEP_URL}/train",
        data=form_data,
        headers={'Content-Type': form_data.content_type},
        stream=True
    )
    
    if response.status_code != 200:
        error_message = "Error from EEP service"
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
            logger.error("Failed to parse metrics from EEP service response")
    
    # Convert the response content (model file) to BytesIO
    model_bytes = io.BytesIO(response.content)
    model_bytes.seek(0)
    
    return model_bytes, metrics

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
        
        # Get optional parameters for data processing and augmentation
        use_augmentation = request.form.get('useAugmentation', 'false').lower() == 'true'
        validation_split = float(request.form.get('validationSplit', 0.2))
        test_split = float(request.form.get('testSplit', 0.1))
        
        # Train the model on the provided data via the EEP
        model_bytes, metrics = train_model(
            zip_file, 
            num_classes=num_classes, 
            training_level=training_level,
            use_augmentation=use_augmentation,
            validation_split=validation_split,
            test_split=test_split
        )
        
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

@app.route('/health', methods=['GET'])
def check_health():
    """Health check endpoint that also verifies service connections"""
    services_status = {
        "eep_service": "connected" if check_eep_health() else "disconnected",
        "model_service": "connected" if check_model_service_health() else "disconnected",
        "data_processing": "connected" if check_data_processing_health() else "disconnected",
        "data_augmentation": "connected" if check_data_augmentation_health() else "disconnected"
    }
    
    return jsonify({
        "status": "healthy",
        "services": services_status
    })

if __name__ == '__main__':
    app.run(debug=True)
