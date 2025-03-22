"""
Runner script for the images_side module.
This is a standalone way to run the image processing module directly.
"""
from images_side import app

if __name__ == '__main__':
    # Make sure all routes are registered
    print("Starting Images Side Application with registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"Route: {rule.endpoint} - {rule.rule}")
        
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5200)  # Running on a different port by default 