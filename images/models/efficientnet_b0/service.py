"""
Standalone Service for EfficientNet-B0 Model

This script starts a Flask server that provides the EfficientNet-B0 model API
for training on medical images and downloading trained models.
"""

import os
import logging
from flask import Flask, jsonify, request
import json
import time
from waitress import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create and configure Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload
app.config['JSON_SORT_KEYS'] = False

# Import EfficientNet-B0 model API
from api import efficientnet_bp, init_app

# Register the blueprint with the app
init_app(app)

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for the service"""
    # Check if TensorFlow and GPU are available
    import tensorflow as tf
    
    gpu_info = "Not available"
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = f"Available ({len(gpus)} devices)"
            # Add details about each GPU
            gpu_details = []
            for i, gpu in enumerate(gpus):
                try:
                    gpu_details.append({
                        'index': i,
                        'name': gpu.name,
                    })
                except:
                    gpu_details.append({'index': i, 'name': 'Unknown'})
            gpu_info = {'status': gpu_info, 'devices': gpu_details}
    except Exception as e:
        gpu_info = f"Error checking GPUs: {str(e)}"
    
    # Get TensorFlow version
    tf_version = tf.__version__
    
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'tensorflow_version': tf_version,
        'gpu': gpu_info,
        'model': 'EfficientNet-B0',
        'service': 'Model Training and Inference Service'
    })

# Start the server
if __name__ == '__main__':
    # Get host and port from environment variables or use defaults
    host = os.environ.get('MODEL_HOST', '0.0.0.0')
    port = int(os.environ.get('MODEL_PORT', 5100))
    
    logger.info(f"Starting EfficientNet-B0 model service on {host}:{port}")
    
    # Use waitress for production-ready server
    serve(app, host=host, port=port, threads=4)
    
    # For development, you can use:
    # app.run(host=host, port=port, debug=True) 