"""
Utility functions for the image analysis module.
"""

import os
import uuid
from werkzeug.utils import secure_filename
import tempfile
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm', 'dicom', 'tif', 'tiff', 'webp'}

# Define temporary folder for image uploads (created in setup.py)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'medicai_images_temp')

def allowed_image_file(filename):
    """
    Check if the file has an allowed image extension.
    
    Args:
        filename: The name of the file to check
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def get_temp_filepath(original_filename=None, extension=None):
    """
    Generate a unique temporary filepath for image uploads.
    
    Args:
        original_filename: Original filename to preserve (optional)
        extension: File extension to use if original_filename is not provided (optional)
        
    Returns:
        str: Path where the temporary file should be stored
    """
    if extension is None and original_filename:
        extension = os.path.splitext(original_filename)[1]
    elif extension is None:
        extension = '.tmp'
    
    unique_id = str(uuid.uuid4())
    if original_filename:
        safe_name = secure_filename(original_filename)
        filename = f"{unique_id}_{safe_name}"
    else:
        filename = f"{unique_id}{extension}"
    
    return os.path.join(UPLOAD_FOLDER, filename)

def process_image(image_path, analysis_type):
    """
    Process an image file based on the requested analysis type.
    This is a placeholder for future implementation.
    
    Args:
        image_path: Path to the image file
        analysis_type: Type of analysis to perform ('classification', 'segmentation', 'detection')
        
    Returns:
        dict: Results of the image analysis
    """
    # This is just a placeholder for now
    logger.info(f"Processing image {image_path} with {analysis_type}")
    
    results = {
        'status': 'success',
        'message': f'Image processed with {analysis_type}',
        'file': os.path.basename(image_path),
        'analysis_type': analysis_type,
        # In the future, this would contain actual results
        'results': {
            'confidence': 0.95,
            'predictions': ['normal', 'no abnormalities detected'],
        }
    }
    
    return results 