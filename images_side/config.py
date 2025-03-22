"""
Configuration for the images_side module.
Contains all URLs and settings for connecting to services.
"""
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model service URL
MODEL_SERVICE_URL = "http://localhost:5110"
MODEL_SERVICE_HEALTH_URL = f"{MODEL_SERVICE_URL}/health"
MODEL_SERVICE_TRAIN_URL = f"{MODEL_SERVICE_URL}/train"

# EEP service URLs
EEP_SERVICE_URL = "http://localhost:5100"

# Alternative URLs to try if localhost doesn't work
ALTERNATIVE_EEP_URLS = [
    "http://127.0.0.1:5100",
    "http://0.0.0.0:5100",
    "http://image-eep-service:5100",  # Docker service name
    "http://host.docker.internal:5100"  # Special Docker DNS name for host
]

# EEP service endpoints
EEP_HEALTH_ENDPOINT = "/health"
EEP_AUGMENTATION_ENDPOINT = "/data_augmentation"
EEP_PROCESSING_ENDPOINT = "/data_processing" 