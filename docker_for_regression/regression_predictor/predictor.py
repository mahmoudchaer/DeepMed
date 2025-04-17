from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
import time
import requests
from waitress import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

class RegressionPredictor:
    def __init__(self):
        """Initialize the regression predictor"""
        self.model_services = {
            'linear_regression': os.environ.get('LINEAR_REGRESSION_URL', 'http://localhost:5041'),
            'lasso_regression': os.environ.get('LASSO_REGRESSION_URL', 'http://localhost:5042'),
            'ridge_regression': os.environ.get('RIDGE_REGRESSION_URL', 'http://localhost:5043'),
            'random_forest_regression': os.environ.get('RANDOM_FOREST_REGRESSION_URL', 'http://localhost:5044'),
            'knn_regression': os.environ.get('KNN_REGRESSION_URL', 'http://localhost:5045'),
            'xgboost_regression': os.environ.get('XGBOOST_REGRESSION_URL', 'http://localhost:5046')
        }
    
    def _get_model_service(self, model_url):
        """Determine which model service should handle the prediction based on the model URL"""
        if not model_url:
            return None, "No model URL provided"
        
        # Extract model type from URL
        for model_type, service_url in self.model_services.items():
            if model_type in model_url.lower():
                return service_url, model_type
                
        return None, f"Unknown model type in URL: {model_url}"
    
    def predict(self, data, model_url):
        """Make predictions using the specified model"""
        logger.info(f"Making predictions with model: {model_url}")
        
        # Get the appropriate model service
        service_url, model_type = self._get_model_service(model_url)
        if not service_url:
            logger.error(f"Error determining model service: {model_type}")
            return None, model_type
        
        try:
            # Make the request to the appropriate model service
            service_base_url = service_url.split('/saved_models')[0] if '/saved_models' in service_url else service_url
            
            # Convert data to appropriate format if needed
            if isinstance(data, pd.DataFrame):
                data = data.to_dict(orient='list')
            
            logger.info(f"Sending prediction request to {service_base_url}/predict")
            
            response = requests.post(
                f"{service_base_url}/predict",
                json={
                    'X': data,
                    'model_url': model_url
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Error from model service: {response.text}")
                return None, f"Error from model service: {response.text}"
            
            # Extract predictions
            result = response.json()
            predictions = result.get('predictions', [])
            
            logger.info(f"Received {len(predictions)} predictions")
            
            return predictions, None
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, str(e)

# Create a predictor instance
predictor = RegressionPredictor()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "regression_predictor"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with a specified model"""
    try:
        # Get request data
        request_data = request.json
        
        if not request_data or 'data' not in request_data or 'model_url' not in request_data:
            return jsonify({"error": "Missing required data fields"}), 400
        
        # Extract data and model URL
        data = request_data['data']
        model_url = request_data['model_url']
        
        # Convert data to DataFrame if it's not already
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Make predictions
        predictions, error = predictor.predict(data, model_url)
        
        if error:
            return jsonify({"error": error}), 400
        
        return jsonify({
            "predictions": predictions
        })
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    logger.info(f"Starting Regression Predictor Service on port {port}")
    serve(app, host='0.0.0.0', port=port) 