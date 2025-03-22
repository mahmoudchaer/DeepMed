"""
EEP (External Endpoint Processing) service functions.
Handles data augmentation and processing.
"""
import requests
import logging
import json
import io
from requests_toolbelt.multipart.encoder import MultipartEncoder
from images_side.config import (
    EEP_SERVICE_URL, 
    EEP_AUGMENTATION_ENDPOINT, 
    EEP_PROCESSING_ENDPOINT
)
from images_side.services.health_service import check_eep_service_health

logger = logging.getLogger(__name__)

def augment_data(zip_file, augmentation_level=3):
    """
    Send data to the EEP for data augmentation with retry logic
    """
    # First check if the primary EEP service is healthy
    if not check_eep_service_health():
        # This will already have attempted alternative URLs
        raise Exception("EEP service is not available. Please ensure the Docker containers are running.")
    
    # The URL for data augmentation through the EEP
    from images_side.config import EEP_SERVICE_URL  # Get fresh URL in case it was updated
    augmentation_url = f"{EEP_SERVICE_URL}{EEP_AUGMENTATION_ENDPOINT}"
    logger.info(f"Sending augmentation request to {augmentation_url}")
    
    # Create form data to send to the service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'augmentationLevel': str(augmentation_level)
        }
    )
    
    # Make the request to the EEP service
    try:
        logger.info(f"Attempting to send request to {augmentation_url}")
        response = requests.post(
            augmentation_url,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True,
            timeout=60  # Increased timeout for large datasets
        )
        
        # Log the response details
        logger.info(f"Received response from {augmentation_url}: Status {response.status_code}")
        logger.info(f"Response Content-Type: {response.headers.get('Content-Type', 'None')}")
        
        if response.status_code != 200:
            error_message = "Error from data augmentation service"
            try:
                # Check content type before trying to parse as JSON
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                    logger.error(f"Error details: {error_data}")
                else:
                    # If not JSON, log the response text
                    error_text = response.text[:200] if response.text else "No response text"
                    logger.error(f"Non-JSON error response: {error_text}")
                    error_message = f"Error {response.status_code} from augmentation service: {error_text}"
            except Exception as json_error:
                logger.error(f"Could not parse error response: {str(json_error)}")
                # Try to get some of the raw response as a string
                try:
                    error_text = response.text[:200] if response.text else "No response text"
                    error_message = f"Error {response.status_code}: {error_text}"
                except:
                    error_message = f"Error status {response.status_code} from augmentation service"
            
            raise Exception(error_message)
        
        # Verify the content type is what we expect
        content_type = response.headers.get('Content-Type', '')
        if 'application/zip' not in content_type:
            logger.warning(f"Unexpected content type received: {content_type}, expected application/zip")
        
        # Get metrics from response header
        metrics = {}
        if 'X-Augmentation-Metrics' in response.headers:
            try:
                metrics_json = response.headers['X-Augmentation-Metrics']
                logger.info(f"Raw metrics header: {metrics_json[:100]}")
                metrics = json.loads(metrics_json)
                logger.info(f"Received metrics: {metrics}")
            except Exception as metrics_error:
                logger.error(f"Failed to parse metrics from augmentation service response: {str(metrics_error)}")
        
        # Convert the response content (ZIP file) to BytesIO
        logger.info("Successfully received augmented data")
        augmented_zip = io.BytesIO(response.content)
        augmented_zip.seek(0)
        
        return augmented_zip, metrics
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when connecting to {augmentation_url}: {str(e)}")
        raise Exception(f"Failed to connect to augmentation service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during augmentation: {str(e)}", exc_info=True)
        raise

def process_data(zip_file, test_size=0.2, val_size=0.2):
    """
    Send data to the EEP for data processing with retry logic
    """
    # First check if the primary EEP service is healthy
    if not check_eep_service_health():
        # This will already have attempted alternative URLs
        raise Exception("EEP service is not available. Please ensure the Docker containers are running.")
    
    # The URL for data processing through the EEP
    from images_side.config import EEP_SERVICE_URL  # Get fresh URL in case it was updated
    processing_url = f"{EEP_SERVICE_URL}{EEP_PROCESSING_ENDPOINT}"
    logger.info(f"Sending processing request to {processing_url}")
    
    # Create form data to send to the service using MultipartEncoder
    form_data = MultipartEncoder(
        fields={
            'zipFile': (zip_file.filename, zip_file.stream, zip_file.content_type),
            'testSize': str(test_size),
            'valSize': str(val_size)
        }
    )
    
    # Make the request to the EEP service
    try:
        logger.info(f"Attempting to send request to {processing_url}")
        response = requests.post(
            processing_url,
            data=form_data,
            headers={'Content-Type': form_data.content_type},
            stream=True,
            timeout=30  # Increased timeout
        )
        
        # Log the response details
        logger.info(f"Received response from {processing_url}: Status {response.status_code}")
        
        if response.status_code != 200:
            error_message = "Error from data processing service"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
                logger.error(f"Error details: {error_data}")
            except:
                logger.error(f"Could not parse error response: {response.text[:200]}")
            raise Exception(error_message)
        
        # Get metrics from response header
        metrics = {}
        if 'X-Processing-Metrics' in response.headers:
            try:
                metrics = json.loads(response.headers['X-Processing-Metrics'])
                logger.info(f"Received metrics: {metrics}")
            except:
                logger.error("Failed to parse metrics from processing service response")
        
        # Convert the response content (ZIP file) to BytesIO
        logger.info("Successfully received processed data")
        processed_zip = io.BytesIO(response.content)
        processed_zip.seek(0)
        
        return processed_zip, metrics
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when connecting to {processing_url}: {str(e)}")
        raise Exception(f"Failed to connect to processing service: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {str(e)}", exc_info=True)
        raise 