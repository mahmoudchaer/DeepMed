from flask import Flask, request, jsonify, Response
import requests
import os
import logging
from requests_toolbelt.multipart.encoder import MultipartEncoder

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define URL for the actual anomaly detection service
ANOMALY_DETECTION_SERVICE_URL = os.environ.get('ANOMALY_DETECTION_SERVICE_URL', 'http://localhost:5029')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if the actual anomaly detection service is available
        response = requests.get(f"{ANOMALY_DETECTION_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            return jsonify({"status": "ok", "message": "Anomaly detection EEP service is running"}), 200
        else:
            return jsonify({
                "status": "degraded",
                "message": "Anomaly detection EEP service is running but the underlying service is not responding"
            }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "degraded", 
            "message": f"Underlying service is not available: {str(e)}"
        }), 200

@app.route('/train', methods=['POST'])
def train():
    """Proxy endpoint for anomaly detection training"""
    try:
        logger.info("Received training request")
        
        # Get all the files from the request
        files = {}
        for file_key in request.files:
            files[file_key] = (
                request.files[file_key].filename,
                request.files[file_key].stream,
                request.files[file_key].content_type
            )
        
        # Get all form data
        form_data = {}
        for key in request.form:
            form_data[key] = request.form[key]
        
        # Create MultipartEncoder for forwarding the request
        fields = {}
        fields.update(form_data)
        fields.update(files)
        
        multipart_data = MultipartEncoder(fields=fields)
        
        # Forward the request to the actual anomaly detection service
        logger.info(f"Forwarding request to {ANOMALY_DETECTION_SERVICE_URL}/train")
        response = requests.post(
            f"{ANOMALY_DETECTION_SERVICE_URL}/train",
            headers={'Content-Type': multipart_data.content_type},
            data=multipart_data,
            stream=True,
            timeout=600  # 10 minute timeout
        )
        
        # Forward the response back
        logger.info(f"Received response from anomaly detection service with status code {response.status_code}")
        
        # Create a Flask response with the same content and headers
        flask_response = Response(response.content)
        flask_response.status_code = response.status_code
        
        # Copy relevant headers
        for header_key, header_value in response.headers.items():
            if header_key.lower() != 'transfer-encoding':  # Skip transfer-encoding header
                flask_response.headers[header_key] = header_value
        
        return flask_response
        
    except Exception as e:
        logger.error(f"Error in train proxy: {str(e)}", exc_info=True)
        return jsonify({"error": f"Middleware error: {str(e)}"}), 500

@app.route('/detect', methods=['POST'])
def detect():
    """Proxy endpoint for anomaly detection"""
    try:
        logger.info("Received detection request")
        
        # Similar to train endpoint, forward the request and response
        files = {}
        for file_key in request.files:
            files[file_key] = (
                request.files[file_key].filename,
                request.files[file_key].stream,
                request.files[file_key].content_type
            )
        
        form_data = {}
        for key in request.form:
            form_data[key] = request.form[key]
        
        fields = {}
        fields.update(form_data)
        fields.update(files)
        
        multipart_data = MultipartEncoder(fields=fields)
        
        logger.info(f"Forwarding request to {ANOMALY_DETECTION_SERVICE_URL}/detect")
        response = requests.post(
            f"{ANOMALY_DETECTION_SERVICE_URL}/detect",
            headers={'Content-Type': multipart_data.content_type},
            data=multipart_data,
            stream=True,
            timeout=600
        )
        
        logger.info(f"Received response from anomaly detection service with status code {response.status_code}")
        
        flask_response = Response(response.content)
        flask_response.status_code = response.status_code
        
        for header_key, header_value in response.headers.items():
            if header_key.lower() != 'transfer-encoding':
                flask_response.headers[header_key] = header_value
        
        return flask_response
        
    except Exception as e:
        logger.error(f"Error in detect proxy: {str(e)}", exc_info=True)
        return jsonify({"error": f"Middleware error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5030))
    logger.info(f"Starting anomaly detection EEP service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 