"""
Model service functions for training models.
"""
import requests
import logging
import json
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder
from images_side.config import MODEL_SERVICE_TRAIN_URL
from images_side.services.health_service import check_model_service_health

logger = logging.getLogger(__name__)

def train_model(zip_file, num_classes=5, training_level=3):
    """
    Instead of training locally, send the data to the Docker container
    for processing and return the results.
    """
    # Check if the model service is healthy
    if not check_model_service_health():
        raise Exception("Model service is not available. Please ensure the Docker container is running.")
    
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
        MODEL_SERVICE_TRAIN_URL,
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