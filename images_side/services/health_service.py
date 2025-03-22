"""
Health check services for EEP and model services.
"""
import requests
import logging
from images_side.config import (
    MODEL_SERVICE_HEALTH_URL, 
    EEP_SERVICE_URL, 
    ALTERNATIVE_EEP_URLS,
    EEP_HEALTH_ENDPOINT
)

logger = logging.getLogger(__name__)

def check_model_service_health():
    """Check if the model service is running and healthy"""
    logger.info(f"Checking model service health at {MODEL_SERVICE_HEALTH_URL}")
    
    try:
        response = requests.get(MODEL_SERVICE_HEALTH_URL, timeout=5)
        healthy = response.status_code == 200
        logger.info(f"Model service health check result: {healthy}")
        return healthy
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to model service: {str(e)}")
        return False

def check_eep_service_health():
    """Check if the EEP service is running and healthy using multiple URLs"""
    global EEP_SERVICE_URL  # We'll modify the global URL if needed
    
    # Try primary URL first
    health_url = f"{EEP_SERVICE_URL}{EEP_HEALTH_ENDPOINT}"
    logger.info(f"Checking EEP service health at {health_url}")
    
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"EEP service health check successful at {health_url}")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to primary EEP service: {str(e)}")
    
    # Try alternative URLs
    for url in ALTERNATIVE_EEP_URLS:
        alt_health_url = f"{url}{EEP_HEALTH_ENDPOINT}"
        logger.info(f"Trying alternative EEP service URL: {alt_health_url}")
        try:
            response = requests.get(alt_health_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"EEP service health check successful at {alt_health_url}")
                # Update the main URL to the working one
                from images_side import config
                config.EEP_SERVICE_URL = url
                logger.info(f"Updated primary EEP service URL to: {url}")
                return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to alternative EEP service at {url}: {str(e)}")
    
    logger.error("All EEP service URLs failed health checks")
    return False 