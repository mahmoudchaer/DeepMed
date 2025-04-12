import os
import io
import json
import tempfile
import zipfile
import logging
import traceback
import requests
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Service URLs (will be configurable via environment variables)
AUGMENTATION_SERVICE_URL = os.environ.get('AUGMENTATION_SERVICE_URL', 'http://augmentation-service:5023')
MODEL_TRAINING_SERVICE_URL = os.environ.get('MODEL_TRAINING_SERVICE_URL', 'http://model-training-service:5021')

# Track temp files for cleanup
temp_files = []

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check if dependent services are available
    services_status = {
        "pipeline_service": "healthy",
        "augmentation_service": "unknown",
        "model_training_service": "unknown"
    }
    
    # Check augmentation service
    try:
        aug_response = requests.get(f"{AUGMENTATION_SERVICE_URL}/health", timeout=2)
        if aug_response.status_code == 200:
            services_status["augmentation_service"] = "healthy"
        else:
            services_status["augmentation_service"] = "unhealthy"
    except Exception:
        services_status["augmentation_service"] = "unavailable"
    
    # Check model training service
    try:
        model_response = requests.get(f"{MODEL_TRAINING_SERVICE_URL}/health", timeout=2)
        if model_response.status_code == 200:
            services_status["model_training_service"] = "healthy"
        else:
            services_status["model_training_service"] = "unhealthy"
    except Exception:
        services_status["model_training_service"] = "unavailable"
    
    overall_status = "healthy" if all(status == "healthy" for service, status in services_status.items() 
                                 if service != "pipeline_service") else "degraded"
    
    return jsonify({
        "status": overall_status,
        "service": "pipeline-service",
        "dependencies": services_status
    })

@app.route('/pipeline', methods=['POST'])
def process_pipeline():
    """
    Pipeline endpoint that optionally performs augmentation and then training
    
    Expects a multipart/form-data POST with:
    - zipFile: A zip file with folders organized by class
    - performAugmentation: "true" or "false" string to indicate whether to perform augmentation
    - augmentationLevel: Augmentation level (1-5)
    - numAugmentations: Number of augmentations per image (if augmenting)
    - numClasses: Number of classes in the dataset
    - trainingLevel: Training level (1-5)
    
    Returns the trained model file
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        if not zip_file.filename:
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension
        if not zip_file.filename.lower().endswith('.zip'):
            return jsonify({"error": "File must be a ZIP archive"}), 400
        
        # Get parameters with defaults
        perform_augmentation = request.form.get('performAugmentation', 'false').lower() == 'true'
        aug_level = int(request.form.get('augmentationLevel', 3))
        num_augmentations = int(request.form.get('numAugmentations', 2))
        num_classes = int(request.form.get('numClasses', 5))
        training_level = int(request.form.get('trainingLevel', 3))
        
        # Save the uploaded zip file temporarily
        temp_input_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        zip_file.save(temp_input_zip.name)
        temp_input_zip.close()
        temp_files.append(temp_input_zip.name)
        
        logger.info(f"Starting pipeline process for file: {zip_file.filename}")
        
        # Step 1: Perform augmentation if requested
        if perform_augmentation:
            logger.info(f"Performing augmentation with level {aug_level} and {num_augmentations} augmentations per image")
            
            # Create a temp file for the augmented data
            temp_augmented_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            temp_augmented_zip.close()
            temp_files.append(temp_augmented_zip.name)
            
            # Send file to augmentation service
            with open(temp_input_zip.name, 'rb') as f:
                m = MultipartEncoder(
                    fields={
                        'zipFile': (zip_file.filename, f, 'application/zip'),
                        'level': str(aug_level),
                        'numAugmentations': str(num_augmentations)
                    }
                )
                
                # Forward the request to the augmentation service
                headers = {'Content-Type': m.content_type}
                
                aug_response = requests.post(
                    f"{AUGMENTATION_SERVICE_URL}/augment",
                    headers=headers,
                    data=m,
                    stream=True
                )
            
            # Check the response status
            if aug_response.status_code != 200:
                error_message = "Error in augmentation service"
                try:
                    error_data = aug_response.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                except:
                    error_message = f"Error in augmentation service (HTTP {aug_response.status_code})"
                
                return jsonify({"error": error_message}), aug_response.status_code
            
            # Save the augmented dataset to the temporary file
            with open(temp_augmented_zip.name, 'wb') as f:
                f.write(aug_response.content)
            
            # Use the augmented dataset for training
            training_zip_path = temp_augmented_zip.name
            logger.info("Augmentation completed successfully")
        else:
            # Use the original dataset for training
            training_zip_path = temp_input_zip.name
            logger.info("Skipping augmentation, proceeding directly to training")
        
        # Step 2: Train the model with the dataset
        logger.info(f"Training model with {num_classes} classes at level {training_level}")
        
        # Send the (potentially augmented) dataset to the model training service
        with open(training_zip_path, 'rb') as f:
            m = MultipartEncoder(
                fields={
                    'zipFile': (zip_file.filename, f, 'application/zip'),
                    'numClasses': str(num_classes),
                    'trainingLevel': str(training_level)
                }
            )
            
            # Forward the request to the model training service
            headers = {'Content-Type': m.content_type}
            
            train_response = requests.post(
                f"{MODEL_TRAINING_SERVICE_URL}/train",
                headers=headers,
                data=m,
                stream=True
            )
        
        # Clean up temporary files
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
                    logger.info(f"Removed temporary file: {tmp_file}")
            except Exception as e:
                logger.error(f"Error removing temporary file {tmp_file}: {str(e)}")
        temp_files.clear()
        
        # Check the response from the training service
        if train_response.status_code != 200:
            error_message = "Error in model training service"
            try:
                error_data = train_response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in model training service (HTTP {train_response.status_code})"
            
            return jsonify({"error": error_message}), train_response.status_code
        
        # Create a Flask response with the model file
        flask_response = Response(train_response.content)
        flask_response.headers["Content-Type"] = "application/octet-stream"
        flask_response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Forward any training metrics headers
        if 'X-Training-Metrics' in train_response.headers:
            flask_response.headers["X-Training-Metrics"] = train_response.headers['X-Training-Metrics']
            flask_response.headers["Access-Control-Expose-Headers"] = "X-Training-Metrics"
        
        logger.info("Pipeline completed successfully")
        return flask_response
        
    except Exception as e:
        logger.error(f"Error in pipeline process: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5025))
    logger.info(f"Starting pipeline service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 