from flask import Blueprint, render_template, jsonify, request
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Blueprint for the chatbot routes
chatbot_blueprint = Blueprint('chatbot', __name__)

@chatbot_blueprint.route('/chatbot-js')
def chatbot_js():
    """Serve the chatbot JavaScript file"""
    return render_template('chatbot/chatbot.js'), 200, {'Content-Type': 'application/javascript'}

@chatbot_blueprint.route('/chatbot-css')
def chatbot_css():
    """Serve the chatbot CSS file"""
    return render_template('chatbot/chatbot.css'), 200, {'Content-Type': 'text/css'}

@chatbot_blueprint.route('/inject-chatbot')
def inject_chatbot():
    """Inject the chatbot HTML into the page"""
    return render_template('chatbot/chatbot_inject.html')

@chatbot_blueprint.route('/test-api', methods=['POST'])
def test_api():
    """Test endpoint for the chatbot API"""
    try:
        data = request.get_json()
        logger.info(f"Received test request: {data}")
        
        # Echo back the request with a sample response
        return jsonify({
            "status": "success",
            "received": data,
            "response": "This is a test response from the chatbot API."
        })
    except Exception as e:
        logger.error(f"Error in test API: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def register_chatbot_blueprint(app):
    """Register the chatbot blueprint with the Flask app"""
    app.register_blueprint(chatbot_blueprint, url_prefix='/chatbot')
    
    # Create the template directories if they don't exist
    template_dirs = [
        os.path.join(app.template_folder, 'chatbot')
    ]
    
    for directory in template_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    # Modify the base template to include the chatbot
    @app.context_processor
    def inject_chatbot_assets():
        """Inject chatbot assets into all templates"""
        return {
            'chatbot_enabled': True
        }
    
    return app 