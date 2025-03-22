import os
import io
import json
import zipfile
import tempfile
import logging
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS  # Import CORS for cross-origin support

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define service URLs for IEPs
MODEL_TRAINING_URL = "http://localhost:5110"
DATA_AUGMENTATION_URL = "http://localhost:5111"
DATA_PROCESSING_URL = "http://localhost:5112"

def check_service_health(service_url):
    """
    Check if a service is healthy
    """
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also checks IEP services"""
    logger.info("Health check requested")
    
    # Check all IEP services
    model_training_health = check_service_health(MODEL_TRAINING_URL)
    data_augmentation_health = check_service_health(DATA_AUGMENTATION_URL)
    data_processing_health = check_service_health(DATA_PROCESSING_URL)
    
    logger.info(f"Model training service health: {model_training_health}")
    logger.info(f"Data augmentation service health: {data_augmentation_health}")
    logger.info(f"Data processing service health: {data_processing_health}")
    
    return jsonify({
        "status": "healthy",
        "service": "image-eep-service",
        "iep_services": {
            "model_training": "healthy" if model_training_health else "unhealthy",
            "data_augmentation": "healthy" if data_augmentation_health else "unhealthy",
            "data_processing": "healthy" if data_processing_health else "unhealthy"
        }
    })

@app.route('/model_training', methods=['POST'])
def forward_to_model_training():
    """Forward request to model training service"""
    logger.info("Received model training request")
    
    # Check if model training service is healthy
    if not check_service_health(MODEL_TRAINING_URL):
        logger.error("Model training service is not available")
        return jsonify({"error": "Model training service is not available"}), 503
    
    try:
        # Extract the file from the request
        if 'zipFile' not in request.files:
            logger.error("No ZIP file in request")
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        zip_file = request.files['zipFile']
        num_classes = request.form.get('numClasses', 5)
        training_level = request.form.get('trainingLevel', 3)
        
        logger.info(f"Forwarding model training request with numClasses={num_classes}, trainingLevel={training_level}")
        
        # Create form data to forward to the model training service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'numClasses': str(num_classes),
                'trainingLevel': str(training_level)
            }
        )
        
        # Forward the request to the model training service
        response = requests.post(
            f"{MODEL_TRAINING_URL}/train",
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True
        )
        
        # Log the response status code
        logger.info(f"Model training service response status: {response.status_code}")
        
        # Check if the response is successful
        if response.status_code != 200:
            # Try to extract error message
            try:
                error_data = response.json()
                error_message = error_data.get('error', 'Unknown error from model training service')
            except:
                error_message = f"Error from model training service: {response.status_code}"
            
            logger.error(f"Model training failed: {error_message}")
            return jsonify({"error": error_message}), response.status_code
        
        # Create a response with the model file from the service
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/octet-stream"
        flask_response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Copy metrics header if available
        if 'X-Training-Metrics' in response.headers:
            flask_response.headers["X-Training-Metrics"] = response.headers['X-Training-Metrics']
            logger.info("Training metrics included in response")
        
        logger.info("Successfully forwarded model training response")
        return flask_response
    
    except Exception as e:
        logger.error(f"Error forwarding to model training service: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error forwarding to model training service: {str(e)}"}), 500

@app.route('/data_augmentation', methods=['POST'])
def forward_to_data_augmentation():
    """Forward request to data augmentation service"""
    logger.info("Received data augmentation request")
    
    # Check if data augmentation service is healthy
    if not check_service_health(DATA_AUGMENTATION_URL):
        logger.error("Data augmentation service is not available")
        return jsonify({"error": "Data augmentation service is not available"}), 503
    
    try:
        # Extract the file from the request
        if 'zipFile' not in request.files:
            logger.error("No ZIP file in request")
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        zip_file = request.files['zipFile']
        augmentation_level = request.form.get('augmentationLevel', 3)
        
        logger.info(f"Forwarding data augmentation request with augmentationLevel={augmentation_level}")
        
        # Create form data to forward to the data augmentation service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'augmentationLevel': str(augmentation_level)
            }
        )
        
        # Forward the request to the data augmentation service
        response = requests.post(
            f"{DATA_AUGMENTATION_URL}/augment",
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True,
            timeout=120  # Extended timeout for large datasets
        )
        
        # Log the response status code
        logger.info(f"Data augmentation service response status: {response.status_code}")
        
        # Check if the response is successful
        if response.status_code != 200:
            # Try to extract error message
            try:
                error_data = response.json()
                error_message = error_data.get('error', 'Unknown error from data augmentation service')
            except:
                error_message = f"Error from data augmentation service: {response.status_code}"
            
            logger.error(f"Data augmentation failed: {error_message}")
            return jsonify({"error": error_message}), response.status_code
        
        # Create a response with the augmented data from the service
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Copy metrics header if available
        if 'X-Augmentation-Metrics' in response.headers:
            flask_response.headers["X-Augmentation-Metrics"] = response.headers['X-Augmentation-Metrics']
            logger.info("Augmentation metrics included in response")
        
        logger.info("Successfully forwarded data augmentation response")
        return flask_response
    
    except Exception as e:
        logger.error(f"Error forwarding to data augmentation service: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error forwarding to data augmentation service: {str(e)}"}), 500

@app.route('/data_processing', methods=['POST'])
def forward_to_data_processing():
    """Forward request to data processing service"""
    logger.info("Received data processing request")
    
    # Check if data processing service is healthy
    if not check_service_health(DATA_PROCESSING_URL):
        logger.error("Data processing service is not available")
        return jsonify({"error": "Data processing service is not available"}), 503
    
    try:
        # Extract the file from the request
        if 'zipFile' not in request.files:
            logger.error("No ZIP file in request")
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        zip_file = request.files['zipFile']
        test_size = request.form.get('testSize', 0.2)
        val_size = request.form.get('valSize', 0.2)
        
        logger.info(f"Forwarding data processing request with testSize={test_size}, valSize={val_size}")
        
        # Create form data to forward to the data processing service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'testSize': str(test_size),
                'valSize': str(val_size)
            }
        )
        
        # Forward the request to the data processing service
        response = requests.post(
            f"{DATA_PROCESSING_URL}/process",
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True,
            timeout=120  # Extended timeout for large datasets
        )
        
        # Log the response status code
        logger.info(f"Data processing service response status: {response.status_code}")
        
        # Check if the response is successful
        if response.status_code != 200:
            # Try to extract error message
            try:
                error_data = response.json()
                error_message = error_data.get('error', 'Unknown error from data processing service')
            except:
                error_message = f"Error from data processing service: {response.status_code}"
            
            logger.error(f"Data processing failed: {error_message}")
            return jsonify({"error": error_message}), response.status_code
        
        # Create a response with the processed data from the service
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
        
        # Copy metrics header if available
        if 'X-Processing-Metrics' in response.headers:
            flask_response.headers["X-Processing-Metrics"] = response.headers['X-Processing-Metrics']
            logger.info("Processing metrics included in response")
        
        logger.info("Successfully forwarded data processing response")
        return flask_response
    
    except Exception as e:
        logger.error(f"Error forwarding to data processing service: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error forwarding to data processing service: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5100))
    logger.info(f"Starting image EEP service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 