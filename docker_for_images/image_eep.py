import os
import io
import json
import zipfile
import tempfile
import logging
from flask import Flask, request, jsonify, Response, send_file

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define service URLs for IEPs
MODEL_TRAINING_URL = "http://model-training-service:5010"
DATA_AUGMENTATION_URL = "http://data-augmentation-service:5011"
DATA_PROCESSING_URL = "http://data-processing-service:5012"

def check_service_health(service_url):
    """Check if a service is running and healthy"""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also checks IEPs"""
    services_status = {
        "eep_status": "healthy",
        "services": {
            "model_training": "healthy" if check_service_health(MODEL_TRAINING_URL) else "unhealthy",
            "data_augmentation": "healthy" if check_service_health(DATA_AUGMENTATION_URL) else "unhealthy",
            "data_processing": "healthy" if check_service_health(DATA_PROCESSING_URL) else "unhealthy"
        }
    }
    return jsonify(services_status)

@app.route('/model_training', methods=['POST'])
def model_training():
    """Forward model training request to the model training IEP"""
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        num_classes = request.form.get('numClasses', '5')
        training_level = request.form.get('trainingLevel', '3')
        
        # Check if model training service is healthy
        if not check_service_health(MODEL_TRAINING_URL):
            return jsonify({"error": "Model training service is not available"}), 503
        
        # Forward the request to the model training service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'numClasses': num_classes,
                'trainingLevel': training_level
            }
        )
        
        response = requests.post(
            f"{MODEL_TRAINING_URL}/train",
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
        
        # Create a response with both the model file and metrics
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/octet-stream"
        flask_response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Copy training metrics header if present
        if 'X-Training-Metrics' in response.headers:
            flask_response.headers["X-Training-Metrics"] = response.headers['X-Training-Metrics']
        
        return flask_response
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/data_augmentation', methods=['POST'])
def data_augmentation():
    """Forward data augmentation request to the data augmentation IEP"""
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        augmentation_level = request.form.get('augmentationLevel', '3')
        
        # Check if data augmentation service is healthy
        if not check_service_health(DATA_AUGMENTATION_URL):
            return jsonify({"error": "Data augmentation service is not available"}), 503
        
        # Forward the request to the data augmentation service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'augmentationLevel': augmentation_level
            }
        )
        
        response = requests.post(
            f"{DATA_AUGMENTATION_URL}/augment",
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
            return jsonify({"error": error_message}), response.status_code
        
        # Create a response with the augmented data ZIP file
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Copy augmentation metrics header if present
        if 'X-Augmentation-Metrics' in response.headers:
            flask_response.headers["X-Augmentation-Metrics"] = response.headers['X-Augmentation-Metrics']
        
        return flask_response
    except Exception as e:
        logger.error(f"Error in data augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/data_processing', methods=['POST'])
def data_processing():
    """Forward data processing request to the data processing IEP"""
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        test_size = request.form.get('testSize', '0.2')
        val_size = request.form.get('valSize', '0.2')
        
        # Check if data processing service is healthy
        if not check_service_health(DATA_PROCESSING_URL):
            return jsonify({"error": "Data processing service is not available"}), 503
        
        # Forward the request to the data processing service
        form_data = MultipartEncoder(
            fields={
                'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
                'testSize': test_size,
                'valSize': val_size
            }
        )
        
        response = requests.post(
            f"{DATA_PROCESSING_URL}/process",
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
            return jsonify({"error": error_message}), response.status_code
        
        # Create a response with the processed data ZIP file
        flask_response = Response(response.content)
        flask_response.headers["Content-Type"] = "application/zip"
        flask_response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
        
        # Copy processing metrics header if present
        if 'X-Processing-Metrics' in response.headers:
            flask_response.headers["X-Processing-Metrics"] = response.headers['X-Processing-Metrics']
        
        return flask_response
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 