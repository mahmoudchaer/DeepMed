import os
import io
import zipfile
import tempfile
import json
import requests
from flask import Flask, request, jsonify, Response
from requests_toolbelt.multipart.encoder import MultipartEncoder

app = Flask(__name__)

# Define service URLs for IEPs
DATA_PROCESSING_URL = "http://data-processing-service:5011"
DATA_AUGMENTATION_URL = "http://data-augmentation-service:5012"
MODEL_TRAINING_URL = "http://model-training-service:5010"

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check all IEP services are available
    services_status = {
        "data_processing": check_service_health(DATA_PROCESSING_URL),
        "data_augmentation": check_service_health(DATA_AUGMENTATION_URL),
        "model_training": check_service_health(MODEL_TRAINING_URL)
    }
    
    # Service is healthy if the EEP itself is running, even if some IEPs are down
    return jsonify({
        "status": "healthy", 
        "service": "image-classification-eep",
        "iep_services": services_status
    })

def check_service_health(service_url):
    """Check if a service is running and healthy"""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        return "healthy" if response.status_code == 200 else "unhealthy"
    except requests.exceptions.RequestException:
        return "unreachable"

@app.route('/train', methods=['POST'])
def train_model():
    """
    End-to-End Process for model training. 
    This coordinates the data processing, data augmentation (if requested), 
    and model training steps.
    """
    # Check if all required parameters are present
    if 'zipFile' not in request.files:
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    zip_file = request.files['zipFile']
    
    # Get parameters from the form
    num_classes = int(request.form.get('numClasses', 5))
    training_level = int(request.form.get('trainingLevel', 3))
    
    # Check if user wants data augmentation
    use_augmentation = request.form.get('useAugmentation', 'false').lower() == 'true'
    
    # Get validation split percentage
    validation_split = float(request.form.get('validationSplit', 0.2))
    test_split = float(request.form.get('testSplit', 0.1))
    
    try:
        # 1. First, process the data using the data_processing service
        # Save the ZIP file to a temporary location to send to the data processing service
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            zip_file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        # Send the data to the data processing service
        with open(tmp_file_path, 'rb') as f:
            data_processing_form = MultipartEncoder(
                fields={
                    'zipFile': (zip_file.filename, f, 'application/zip'),
                    'validationSplit': str(validation_split),
                    'testSplit': str(test_split)
                }
            )
            
            headers = {'Content-Type': data_processing_form.content_type}
            resp = requests.post(
                f"{DATA_PROCESSING_URL}/process", 
                data=data_processing_form,
                headers=headers,
                stream=True
            )
            
            if resp.status_code != 200:
                error_message = "Error from data processing service"
                try:
                    error_data = resp.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                except:
                    pass
                raise Exception(error_message)
            
            # Get the processed data as ZIP
            processed_data = io.BytesIO(resp.content)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        # 2. If augmentation is requested, send the processed data to the augmentation service
        if use_augmentation:
            augmentation_form = MultipartEncoder(
                fields={
                    'zipFile': ('processed_data.zip', processed_data, 'application/zip')
                }
            )
            
            headers = {'Content-Type': augmentation_form.content_type}
            resp = requests.post(
                f"{DATA_AUGMENTATION_URL}/augment", 
                data=augmentation_form,
                headers=headers,
                stream=True
            )
            
            if resp.status_code != 200:
                error_message = "Error from data augmentation service"
                try:
                    error_data = resp.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                except:
                    pass
                raise Exception(error_message)
            
            # Get the augmented data
            training_data = io.BytesIO(resp.content)
        else:
            # If no augmentation, use the processed data directly
            training_data = processed_data
        
        # Reset buffer position before sending
        training_data.seek(0)
        
        # 3. Send the final data to the model training service
        training_form = MultipartEncoder(
            fields={
                'zipFile': ('training_data.zip', training_data, 'application/zip'),
                'numClasses': str(num_classes),
                'trainingLevel': str(training_level)
            }
        )
        
        headers = {'Content-Type': training_form.content_type}
        resp = requests.post(
            f"{MODEL_TRAINING_URL}/train", 
            data=training_form,
            headers=headers,
            stream=True
        )
        
        if resp.status_code != 200:
            error_message = "Error from model training service"
            try:
                error_data = resp.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                pass
            raise Exception(error_message)
        
        # Get the trained model and metrics
        model_bytes = io.BytesIO(resp.content)
        model_bytes.seek(0)
        
        # Extract metrics if available
        metrics = {}
        if 'X-Training-Metrics' in resp.headers:
            try:
                metrics = json.loads(resp.headers['X-Training-Metrics'])
                
                # Add EEP processing info to metrics
                metrics["eep_processing"] = {
                    "data_processing": True,
                    "data_augmentation": use_augmentation,
                    "validation_split": validation_split,
                    "test_split": test_split
                }
            except:
                pass
        
        # Create a response with both the model file and metrics
        response = Response(model_bytes.getvalue())
        response.headers["Content-Type"] = "application/octet-stream"
        response.headers["Content-Disposition"] = "attachment; filename=trained_model.pt"
        
        # Add metrics as a JSON string in a custom header
        response.headers["X-Training-Metrics"] = json.dumps(metrics)
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5020))
    app.run(host='0.0.0.0', port=port, debug=False) 