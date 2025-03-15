"""
Image Model Collection for Medical Image Analysis

This package contains various deep learning models for medical image analysis.
"""

import logging

# Set up logging
logger = logging.getLogger(__name__)

def init_app(app):
    """
    Initialize all model modules and register their blueprints with the Flask app.
    
    Args:
        app: The Flask application instance
    """
    # Import and initialize the EfficientNet-B0 model
    try:
        from .efficientnet_b0 import init_app as init_efficientnet
        init_efficientnet(app)
        logger.info("Initialized EfficientNet-B0 model")
    except ImportError as e:
        logger.warning(f"Could not initialize EfficientNet-B0 model: {str(e)}")
        
    # Add more models here as they are implemented 