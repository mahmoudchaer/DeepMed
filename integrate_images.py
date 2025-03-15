"""
Integration for MedicAI Image Analysis Module

This file provides the integration point between the main application and 
the image analysis module.

Usage:
1. Import this file in your main application
2. Call integrate_images(app) where app is your Flask application instance
"""

from images import init_app as init_images_app
import os

def integrate_images(app):
    """
    Integrate the image analysis module with the main Flask app.
    
    Args:
        app: The Flask application instance from app_api.py
    """
    # Register the images module with the app
    init_images_app(app)
    
    print("✅ Successfully integrated image analysis functionality!")
    print("   Images dashboard is available at: /images")
    
    # Optional: You can add any additional setup code here, such as:
    # - Creating directories for image storage
    # - Setting up additional configuration
    # - Registering cleanup handlers
    
    # Create permanent storage for processed images if needed
    image_storage_dir = os.path.join('static', 'processed_images')
    os.makedirs(image_storage_dir, exist_ok=True)

# Example usage in app_api.py:
"""
if __name__ == '__main__':
    print("Starting MedicAI with API services integration")
    
    # Initialize image analysis functionality
    from integrate_images import integrate_images
    integrate_images(app)
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        print("Database tables created/verified")
    
    try:
        app.run(debug=True, port=5000)
    finally:
        # Clean up all temporary files when the application stops
        cleanup_temp_files() 
""" 