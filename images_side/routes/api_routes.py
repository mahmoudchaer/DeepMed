"""
API routes for the image processing module.
"""
import logging
import json
from flask import request, jsonify, render_template, Response
from images_side import app
from images_side.services.health_service import check_model_service_health, check_eep_service_health
from images_side.services.model_service import train_model
from images_side.services.eep_service import augment_data, process_data

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the training page template"""
    return render_template('train_model.html')

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """
    API endpoint for model training.
    Accepts ZIP file with training data and returns trained model.
    This function is exported for backward compatibility with app_api.py.
    """
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
    """
    API endpoint for data augmentation.
    Accepts ZIP file with image data and returns augmented dataset.
    """
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
    """
    API endpoint for data processing.
    Accepts ZIP file with image data and returns processed dataset.
    """
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