"""
Standalone Service for EfficientNet-B0 Model

This script starts a Flask server that provides the EfficientNet-B0 model API
for training on medical images and downloading trained models.
"""

import os
import logging
import sys
from flask import Flask, jsonify, request
import json
import time
from waitress import serve

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('efficientnet_service.log')
    ]
)
logger = logging.getLogger('efficientnet_service')
logger.info("EfficientNet-B0 service starting up")

# Create and configure Flask app
logger.info("Initializing Flask application")
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload
app.config['JSON_SORT_KEYS'] = False
logger.debug("Flask app configured with MAX_CONTENT_LENGTH=1GB")

# Import EfficientNet-B0 model API
logger.info("Importing EfficientNet-B0 API module")
try:
    # Change the import statement to handle both direct and package imports
    try:
        # First try as a package
        from efficientnet_b0.api import efficientnet_bp, init_app
        logger.info("API module imported as package")
    except ImportError:
        # Then try direct import
        from api import efficientnet_bp, init_app
        logger.info("API module imported directly")
    
    logger.info("API module imported successfully")
except Exception as e:
    logger.error(f"Error importing API module: {str(e)}", exc_info=True)
    raise

# Register the blueprint with the app
logger.info("Registering EfficientNet-B0 blueprint with Flask app")
init_app(app)

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for the service"""
    logger.info("Health check endpoint called")
    
    # Check if TensorFlow and GPU are available
    import tensorflow as tf
    logger.debug(f"TensorFlow version: {tf.__version__}")
    
    gpu_info = "Not available"
    try:
        logger.info("Checking GPU availability")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = f"Available ({len(gpus)} devices)"
            logger.info(f"Found {len(gpus)} GPU devices")
            # Add details about each GPU
            gpu_details = []
            for i, gpu in enumerate(gpus):
                try:
                    logger.debug(f"GPU {i}: {gpu.name}")
                    gpu_details.append({
                        'index': i,
                        'name': gpu.name,
                    })
                except Exception as e:
                    logger.warning(f"Error getting details for GPU {i}: {str(e)}")
                    gpu_details.append({'index': i, 'name': 'Unknown'})
            gpu_info = {'status': gpu_info, 'devices': gpu_details}
        else:
            logger.warning("No GPUs detected")
    except Exception as e:
        error_msg = f"Error checking GPUs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        gpu_info = error_msg
    
    # Get TensorFlow version
    tf_version = tf.__version__
    
    # Memory info
    try:
        import psutil
        memory_info = {
            'total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'percent_used': psutil.virtual_memory().percent
        }
        logger.info(f"Memory info: {memory_info}")
    except Exception as e:
        logger.warning(f"Error getting memory info: {str(e)}")
        memory_info = {"error": str(e)}
    
    response = {
        'status': 'healthy',
        'timestamp': time.time(),
        'tensorflow_version': tf_version,
        'gpu': gpu_info,
        'memory': memory_info,
        'model': 'EfficientNet-B0',
        'service': 'Model Training and Inference Service'
    }
    
    logger.info(f"Health check response: {response}")
    return jsonify(response)

# Start the server
if __name__ == '__main__':
    # Get host and port from environment variables or use defaults
    host = os.environ.get('MODEL_HOST', '0.0.0.0')
    port = int(os.environ.get('MODEL_PORT', 5100))
    
    logger.info(f"Starting EfficientNet-B0 model service on {host}:{port}")
    
    # Log environment variables for debugging
    logger.debug("Environment variables:")
    for key, value in os.environ.items():
        if 'MODEL_' in key or 'TF_' in key or 'CUDA' in key or 'PYTHON' in key:
            logger.debug(f"  {key}={value}")
    
    # Use waitress for production-ready server
    logger.info(f"Starting waitress server with 4 threads on {host}:{port}")
    try:
        serve(app, host=host, port=port, threads=4)
    except Exception as e:
        logger.critical(f"Error starting server: {str(e)}", exc_info=True)
        sys.exit(1)
    
    # For development, you can use:
    # app.run(host=host, port=port, debug=True) 