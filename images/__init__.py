"""
Medical Image Analysis Module for MedicAI

This package contains the functionality for analyzing medical images.
"""

from flask import Blueprint, current_app
import logging

# Set up logging
logger = logging.getLogger('images')
logger.info("Initializing Images Module")

# Create Blueprint for image-related routes
images_bp = Blueprint('images', __name__, url_prefix='/images')
logger.info("Created images blueprint with url_prefix='/images'")

# Import routes after creating the blueprint to avoid circular imports
from . import routes
logger.info("Imported routes module")

def init_app(app):
    """
    Initialize the images module and register it with the Flask app.
    
    Args:
        app: The Flask application instance
    """
    logger.info("Initializing images module for Flask app")
    
    # Register the blueprint with the app
    app.register_blueprint(images_bp)
    logger.info("Registered images blueprint with Flask app")
    
    # Log the available routes
    logger.debug("Available routes in the images blueprint:")
    for rule in app.url_map.iter_rules():
        if rule.endpoint.startswith('images.'):
            logger.debug(f"  {rule.rule} -> {rule.endpoint}")
    
    # Set up necessary directories and other initialization
    logger.info("Setting up directories and initialization")
    from . import setup
    setup.initialize(app)
    
    # Initialize the models module if available
    try:
        logger.info("Attempting to initialize models module")
        from .models import init_app as init_models
        init_models(app)
        logger.info("Models module initialized successfully")
    except ImportError as e:
        # Log the error but don't crash - models might not be available yet
        logger.warning(f"Could not initialize image models: {str(e)}")
    
    # Add a direct route at the app level that redirects to the blueprint
    @app.route('/images')
    def images_redirect():
        from flask import redirect, url_for
        logger.debug("Root /images route called, redirecting to images.index")
        return redirect(url_for('images.index'))
    
    logger.info("Images module initialization complete") 