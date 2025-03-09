from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/model/model.pkl')
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'unknown')
MODEL_FEATURES = []

# Load the model
def load_model():
    """Load the trained model from disk"""
    try:
        logger.info(f"Loading {MODEL_TYPE} model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Try to get model features if available
        features_path = os.path.join(os.path.dirname(MODEL_PATH), 'features.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                global MODEL_FEATURES
                MODEL_FEATURES = json.load(f)
                logger.info(f"Loaded {len(MODEL_FEATURES)} features")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

MODEL = load_model()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if MODEL is not None:
        return jsonify({"status": "healthy", "model_type": MODEL_TYPE})
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions endpoint"""
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get data from request
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({"error": "No data provided"}), 400
        
        # Handle different input formats
        input_data = data['data']
        if isinstance(input_data, dict):
            # Input is a dictionary of feature arrays
            # Convert to DataFrame
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, list):
            # Input is a list of dictionaries (records)
            df = pd.DataFrame(input_data)
        else:
            return jsonify({"error": "Unsupported data format"}), 400
        
        # Ensure we have all required features
        if MODEL_FEATURES and not all(feature in df.columns for feature in MODEL_FEATURES):
            missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
            return jsonify({
                "error": f"Missing features: {missing_features}"
            }), 400
        
        # If we have feature list, select only those features
        if MODEL_FEATURES:
            df = df[MODEL_FEATURES]
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Make prediction
        predictions = MODEL.predict(df).tolist()
        
        # If model has predict_proba method, include probabilities
        probabilities = None
        if hasattr(MODEL, 'predict_proba'):
            try:
                probabilities = MODEL.predict_proba(df).tolist()
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        # Prepare response
        response = {
            "predictions": predictions
        }
        
        if probabilities:
            response["probabilities"] = probabilities
            
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": f"Error making prediction: {str(e)}"}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get model information
    info = {
        "model_type": MODEL_TYPE,
        "features": MODEL_FEATURES,
        "parameters": {}
    }
    
    # Try to get model parameters if available
    if hasattr(MODEL, 'get_params'):
        try:
            info["parameters"] = MODEL.get_params()
        except:
            pass
    
    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 