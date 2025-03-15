"""
Medical Image Analysis Module for MedicAI

This package contains the functionality for analyzing medical images.
"""

from flask import Blueprint

# Create Blueprint for image-related routes
images_bp = Blueprint('images', __name__, url_prefix='/images')

# Import routes after creating the blueprint to avoid circular imports
from . import routes

def init_app(app):
    """
    Initialize the images module and register it with the Flask app.
    
    Args:
        app: The Flask application instance
    """
    app.register_blueprint(images_bp)
    
    # Set up necessary directories and other initialization
    from . import setup
    setup.initialize(app)
    
    # Initialize the models module if available
    try:
        from .models import init_app as init_models
        init_models(app)
    except ImportError as e:
        # Log the error but don't crash - models might not be available yet
        app.logger.warning(f"Could not initialize image models: {str(e)}")
    
    # Add a direct route at the app level that redirects to the blueprint
    @app.route('/images')
    def images_redirect():
        from flask import redirect, url_for
        return redirect(url_for('images.index')) 