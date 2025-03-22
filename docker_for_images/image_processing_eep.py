import os
import io
import zipfile
import tempfile
import json
import logging
from flask import Flask, request, jsonify, Response
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define IEP service URLs
DATA_PROCESSING_URL = "http://localhost:5011/process"
DATA_AUGMENTATION_URL = "http://localhost:5012/augment"
MODEL_TRAINING_URL = "http://localhost:5010/train"  # Original model service

def is_service_available(service_url):
    """Check if a service is available by calling its health endpoint"""
    try:
        # Extract base URL without endpoint
        base_url = "/".join(service_url.split("/")[:-1])
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also verifies IEP service connections"""
    services_status = {
        "data_processing": "connected" if is_service_available(DATA_PROCESSING_URL) else "disconnected",
        "data_augmentation": "connected" if is_service_available(DATA_AUGMENTATION_URL) else "disconnected",
        "model_training": "connected" if is_service_available(MODEL_TRAINING_URL) else "disconnected"
    }
    
    return jsonify({
        "status": "healthy",
        "iep_services": services_status
    })

@app.route('/process_data', methods=['POST'])
def process_data():
    """Process data by sending to the data processing IEP"""
    if not is_service_available(DATA_PROCESSING_URL):
        return jsonify({"error": "Data processing service is not available"}), 503
    
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    test_size = float(request.form.get('testSize', 0.2))
    val_size = float(request.form.get('valSize', 0.1))
    
    # Forward the request to the data processing IEP
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'testSize': str(test_size),
            'valSize': str(val_size)
        }
    )
    
    try:
        response = requests.post(
            DATA_PROCESSING_URL,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True
        )
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                if 'error' in error_data:
                    return jsonify({"error": error_data['error']}), response.status_code
            except:
                pass
            return jsonify({"error": "Error from data processing service"}), response.status_code
        
        # Return the processed data (zip file)
        processed_response = Response(response.content)
        processed_response.headers["Content-Type"] = "application/octet-stream"
        processed_response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
        
        # Add metrics header if available
        if 'X-Processing-Metrics' in response.headers:
            processed_response.headers["X-Processing-Metrics"] = response.headers['X-Processing-Metrics']
        
        return processed_response
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/augment_data', methods=['POST'])
def augment_data():
    """Augment data by sending to the data augmentation IEP"""
    if not is_service_available(DATA_AUGMENTATION_URL):
        return jsonify({"error": "Data augmentation service is not available"}), 503
    
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    augmentation_level = int(request.form.get('augmentationLevel', 3))
    
    # Forward the request to the data augmentation IEP
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'augmentationLevel': str(augmentation_level)
        }
    )
    
    try:
        response = requests.post(
            DATA_AUGMENTATION_URL,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True
        )
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                if 'error' in error_data:
                    return jsonify({"error": error_data['error']}), response.status_code
            except:
                pass
            return jsonify({"error": "Error from data augmentation service"}), response.status_code
        
        # Return the augmented data (zip file)
        augmented_response = Response(response.content)
        augmented_response.headers["Content-Type"] = "application/octet-stream"
        augmented_response.headers["Content-Disposition"] = "attachment; filename=augmented_data.zip"
        
        # Add metrics header if available
        if 'X-Augmentation-Metrics' in response.headers:
            augmented_response.headers["X-Augmentation-Metrics"] = response.headers['X-Augmentation-Metrics']
        
        return augmented_response
        
    except Exception as e:
        logger.error(f"Error in data augmentation: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train model by sending to the model training IEP (existing service)"""
    if not is_service_available(MODEL_TRAINING_URL):
        return jsonify({"error": "Model training service is not available"}), 503
    
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    num_classes = int(request.form.get('numClasses', 5))
    training_level = int(request.form.get('trainingLevel', 3))
    
    # Forward the request to the model training IEP
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'numClasses': str(num_classes),
            'trainingLevel': str(training_level)
        }
    )
    
    try:
        response = requests.post(
            MODEL_TRAINING_URL,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True
        )
        
        if response.status_code != 200:
            try:
                error_data = response.json()
                if 'error' in error_data:
                    return jsonify({"error": error_data['error']}), response.status_code
            except:
                pass
            return jsonify({"error": "Error from model training service"}), response.status_code
        
        # Return the trained model
        model_response = Response(response.content)
        model_response.headers["Content-Type"] = "application/octet-stream"
        model_response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics header if available
        if 'X-Training-Metrics' in response.headers:
            model_response.headers["X-Training-Metrics"] = response.headers['X-Training-Metrics']
        
        return model_response
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5020))
    app.run(host='0.0.0.0', port=port, debug=False) 