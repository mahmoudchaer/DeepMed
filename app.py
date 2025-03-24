import os
import io
import zipfile
import tempfile
import json
import requests

from flask import Flask, request, jsonify, send_file, render_template, Response
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

app = Flask(__name__)

def check_model_service_health():
    """Check if the model service is running and healthy"""
    try:
        response = requests.get("http://localhost:5020/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def train_model(zip_file, num_classes=5, training_level=3):
    """
    Instead of training locally, send the data to the Docker container
    for processing and return the results.
    """
    # Check if the model service is healthy
    if not check_model_service_health():
        raise Exception("Model service is not available. Please ensure the Docker container is running.")
    
    # The URL of the model training service (Docker container)
    model_service_url = "http://localhost:5020/train"
    
    # Create a new form with the file and parameters
    files = {'zipFile': (zip_file.filename, zip_file, 'application/zip')}
    data = {
        'numClasses': num_classes,
        'trainingLevel': training_level
    }
    
    # Make the request to the model service
    response = requests.post(model_service_url, files=files, data=data)
    
    if response.status_code != 200:
        error_message = f"Model service error: {response.text}"
        raise Exception(error_message)
    
    # Get metrics from response header
    metrics = json.loads(response.headers.get('X-Training-Metrics', '{}'))
    
    # Convert the response content (model file) to BytesIO
    model_bytes = io.BytesIO(response.content)
    model_bytes.seek(0)
    
    return model_bytes, metrics

@app.route('/')
def index():
    # Render the training page template
    return render_template('train_model.html')

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    zip_file = request.files['zipFile']
    try:
        # Get parameters from the form
        num_classes = int(request.form.get('numClasses', 5))
        training_level = int(request.form.get('trainingLevel', 3))
        
        # Train the model on the provided data via the Docker container
        model_bytes, metrics = train_model(zip_file, num_classes=num_classes, training_level=training_level)
        
        # Set metrics as a header in the response
        response = Response(model_bytes.getvalue())
        response.headers["Content-Type"] = "application/octet-stream"
        response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics as a JSON string in a custom header
        response.headers["X-Training-Metrics"] = json.dumps(metrics)
        
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def check_health():
    """Health check endpoint that also verifies model service connection"""
    model_service_healthy = check_model_service_health()
    return jsonify({
        "status": "healthy",
        "model_service_status": "connected" if model_service_healthy else "disconnected"
    })

if __name__ == '__main__':
    app.run(debug=True)
