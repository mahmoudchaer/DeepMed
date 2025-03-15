"""
Setup and initialization for the image analysis module.
"""

import os
import tempfile
import atexit
import shutil
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define temporary folder for image uploads
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'medicai_images_temp')
PROCESSED_FOLDER = os.path.join('static', 'processed_images')

def initialize(app):
    """
    Initialize the image analysis module with necessary directories and resources.
    
    Args:
        app: The Flask application instance
    """
    # Create temporary folder for image uploads if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Created temporary upload folder: {UPLOAD_FOLDER}")
    
    # Create folder for processed images if it doesn't exist
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    logger.info(f"Created processed images folder: {PROCESSED_FOLDER}")
    
    # Register cleanup function to remove temporary files when the application exits
    atexit.register(cleanup_temp_files)
    
    # Configure app with image-specific settings
    app.config['IMAGE_UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['IMAGE_PROCESSED_FOLDER'] = PROCESSED_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    logger.info("Image analysis module initialized successfully")

def cleanup_temp_files():
    """
    Remove all temporary files when the application exits.
    """
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info(f"Cleaned up temporary directory: {UPLOAD_FOLDER}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")

def cleanup_session_image_files(session):
    """
    Remove image files associated with the current session.
    
    Args:
        session: The Flask session object
    """
    files_to_cleanup = []
    
    # Check for file paths in session
    image_keys = [
        'uploaded_image', 'processed_image', 'results_file'
    ]
    
    for key in image_keys:
        if key in session:
            filepath = session.get(key)
            if filepath and os.path.exists(filepath):
                files_to_cleanup.append(filepath)
                
    # Delete the files
    for filepath in files_to_cleanup:
        try:
            os.remove(filepath)
            logger.info(f"Deleted temporary file: {filepath}")
        except Exception as e:
            logger.error(f"Error deleting temporary file {filepath}: {str(e)}")
            
    # Clear session references to files
    for key in image_keys:
        if key in session:
            session.pop(key) 