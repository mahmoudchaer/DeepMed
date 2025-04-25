import os
import io
import json
import tempfile
import zipfile
import logging
import traceback
import requests
from flask import Flask, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from requests_toolbelt.multipart.encoder import MultipartEncoder
from torchvision import transforms
from PIL import Image
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Service URLs (will be configurable via environment variables)
AUGMENTATION_SERVICE_URL = os.environ.get('AUGMENTATION_SERVICE_URL', 'http://augmentation-service:5023')
MODEL_TRAINING_SERVICE_URL = os.environ.get('MODEL_TRAINING_SERVICE_URL', 'http://model-training-service:5021')

# Track temp files for cleanup
temp_files = []

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check if dependent services are available
    services_status = {
        "pipeline_service": "healthy",
        "augmentation_service": "unknown",
        "model_training_service": "unknown"
    }
    
    # Check augmentation service
    try:
        aug_response = requests.get(f"{AUGMENTATION_SERVICE_URL}/health", timeout=602)
        if aug_response.status_code == 200:
            services_status["augmentation_service"] = "healthy"
        else:
            services_status["augmentation_service"] = "unhealthy"
    except Exception:
        services_status["augmentation_service"] = "unavailable"
    
    # Check model training service
    try:
        model_response = requests.get(f"{MODEL_TRAINING_SERVICE_URL}/health", timeout=602)
        if model_response.status_code == 200:
            services_status["model_training_service"] = "healthy"
        else:
            services_status["model_training_service"] = "unhealthy"
    except Exception:
        services_status["model_training_service"] = "unavailable"
    
    overall_status = "healthy" if all(status == "healthy" for service, status in services_status.items() 
                                 if service != "pipeline_service") else "degraded"
    
    return jsonify({
        "status": overall_status,
        "service": "pipeline-service",
        "dependencies": services_status
    })

@app.route('/pipeline', methods=['POST'])
def process_pipeline():
    """
    Pipeline endpoint that optionally performs augmentation and then training
    
    Expects a multipart/form-data POST with:
    - zipFile: A zip file with folders organized by class
    - performAugmentation: "true" or "false" string to indicate whether to perform augmentation
    - augmentationLevel: Augmentation level (1-5) which determines both strength and number of augmentations
    - numClasses: Number of classes in the dataset
    - trainingLevel: Training level (1-5)
    
    Returns a zip file containing:
    - The trained model
    - An inference script for using the model
    - Requirements file
    - README with instructions
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        if not zip_file.filename:
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension
        if not zip_file.filename.lower().endswith('.zip'):
            return jsonify({"error": "File must be a ZIP archive"}), 400
        
        # Get parameters with defaults
        perform_augmentation = request.form.get('performAugmentation', 'false').lower() == 'true'
        aug_level = int(request.form.get('augmentationLevel', 3))
        num_classes = int(request.form.get('numClasses', 5))
        training_level = int(request.form.get('trainingLevel', 3))
        
        # Save the uploaded zip file temporarily
        temp_input_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        zip_file.save(temp_input_zip.name)
        temp_input_zip.close()
        temp_files.append(temp_input_zip.name)
        
        logger.info(f"Starting pipeline process for file: {zip_file.filename}")
        
        # Step 1: Perform augmentation if requested
        if perform_augmentation:
            logger.info(f"Performing augmentation with level {aug_level}")
            
            # Create a temp file for the augmented data
            temp_augmented_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
            temp_augmented_zip.close()
            temp_files.append(temp_augmented_zip.name)
            
            # Send file to augmentation service
            with open(temp_input_zip.name, 'rb') as f:
                m = MultipartEncoder(
                    fields={
                        'zipFile': (zip_file.filename, f, 'application/zip'),
                        'level': str(aug_level)
                    }
                )
                
                # Forward the request to the augmentation service
                headers = {'Content-Type': m.content_type}
                
                aug_response = requests.post(
                    f"{AUGMENTATION_SERVICE_URL}/augment",
                    headers=headers,
                    data=m,
                    stream=True
                )
            
            # Check the response status
            if aug_response.status_code != 200:
                error_message = "Error in augmentation service"
                try:
                    error_data = aug_response.json()
                    if 'error' in error_data:
                        error_message = error_data['error']
                except:
                    error_message = f"Error in augmentation service (HTTP {aug_response.status_code})"
                
                return jsonify({"error": error_message}), aug_response.status_code
            
            # Save the augmented dataset to the temporary file
            with open(temp_augmented_zip.name, 'wb') as f:
                f.write(aug_response.content)
            
            # Use the augmented dataset for training
            training_zip_path = temp_augmented_zip.name
            logger.info("Augmentation completed successfully")
        else:
            # Use the original dataset for training
            training_zip_path = temp_input_zip.name
            logger.info("Skipping augmentation, proceeding directly to training")
        
        # Step 2: Train the model with the dataset
        logger.info(f"Training model with {num_classes} classes at level {training_level}")
        
        # Send the (potentially augmented) dataset to the model training service
        with open(training_zip_path, 'rb') as f:
            m = MultipartEncoder(
                fields={
                    'zipFile': (zip_file.filename, f, 'application/zip'),
                    'numClasses': str(num_classes),
                    'trainingLevel': str(training_level)
                }
            )
            
            # Forward the request to the model training service
            headers = {'Content-Type': m.content_type}
            
            train_response = requests.post(
                f"{MODEL_TRAINING_SERVICE_URL}/train",
                headers=headers,
                data=m,
                stream=True
            )
        
        # Clean up temporary files
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
                    logger.info(f"Removed temporary file: {tmp_file}")
            except Exception as e:
                logger.error(f"Error removing temporary file {tmp_file}: {str(e)}")
        temp_files.clear()
        
        # Check the response from the training service
        if train_response.status_code != 200:
            error_message = "Error in model training service"
            try:
                error_data = train_response.json()
                if 'error' in error_data:
                    error_message = error_data['error']
            except:
                error_message = f"Error in model training service (HTTP {train_response.status_code})"
            
            return jsonify({"error": error_message}), train_response.status_code
        
        # Get training metrics if available
        training_metrics = {}
        if 'X-Training-Metrics' in train_response.headers:
            try:
                training_metrics = json.loads(train_response.headers['X-Training-Metrics'])
            except:
                logger.warning("Failed to parse training metrics")
        
        # Create the model package directly in memory
        model_content = train_response.content
        
        # Create a zip file in memory
        memory_zip = io.BytesIO()
        
        with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add the model file
            zf.writestr('trained_model.pt', model_content)
            
            # Add class mapping if available
            if training_metrics and 'class_to_idx' in training_metrics:
                # Create a class mapping JSON file
                class_mapping = json.dumps(training_metrics.get('class_to_idx'), indent=2)
                zf.writestr('class_mapping.json', class_mapping)
                
                # Also create a simple text file with class names for easy reference
                if 'classes' in training_metrics:
                    # Log the class mapping for debugging
                    logger.info(f"Class mapping: {training_metrics.get('class_to_idx')}")
                    logger.info(f"Class names: {training_metrics.get('classes')}")
                    
                    # Create a better formatted class_names.txt file
                    class_names_text = ""
                    for i, name in enumerate(training_metrics.get('classes')):
                        class_names_text += f"{i}: {name}\n"
                    zf.writestr('class_names.txt', class_names_text)
            
            # Add inference script
            inference_code = """import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import glob

def load_model(model_path, num_classes):
    # Load EfficientNet-B0
    model = models.efficientnet_b0(pretrained=False)
    
    # Replace the classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def predict_image(model, image_path, class_names=None):
    # Preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    # Get the predicted class name or index
    predicted_class = predicted_idx.item()
    if class_names and predicted_class < len(class_names):
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = f"Class {predicted_class}"
    
    # Calculate the confidence score
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class].item() * 100
    
    return predicted_label, confidence, predicted_class

def main():
    parser = argparse.ArgumentParser(description='Predict using a trained medical image model')
    # Use an optional positional argument for the image
    parser.add_argument('image', nargs='?', default=None, help='Path to the image file (optional)')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes in the model')
    parser.add_argument('--class_names', type=str, help='Optional comma-separated list of class names')
    parser.add_argument('--all', action='store_true', help='Process all images in the current folder')
    
    args = parser.parse_args()
    
    # Load the model file (assumed to be named "trained_model.pt" in the package)
    model_path = 'trained_model.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    # Try to automatically load class names from class_names.txt if available
    class_names = None
    if os.path.exists('class_names.txt') and not args.class_names:
        try:
            with open('class_names.txt', 'r') as f:
                class_names_content = f.read().splitlines()
            class_names = []
            for line in class_names_content:
                if ':' in line:
                    # Extract the class name after the colon
                    class_names.append(line.split(':', 1)[1].strip())
            print(f"Loaded {len(class_names)} class names from class_names.txt")
        except Exception as e:
            print(f"Error loading class names from file: {str(e)}")
    
    # Overwrite with command-line provided class names if given
    if args.class_names:
        class_names = args.class_names.split(',')
        print(f"Using {len(class_names)} class names from command line argument")
    
    # Determine number of classes from class names or parameter
    if class_names:
        num_classes = len(class_names)
    else:
        num_classes = args.num_classes
    print(f"Using model with {num_classes} classes")
    
    model = load_model(model_path, num_classes)
    
    # Processing images
    if args.all:
        # Process all images in the current directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(ext))
        
        if not image_files:
            print("No image files found in the current directory.")
            return
        
        print(f"Processing {len(image_files)} images...")
        for img_path in image_files:
            try:
                predicted_label, confidence, _ = predict_image(model, img_path, class_names)
                print(f"{img_path}: {predicted_label} (Confidence: {confidence:.2f}%)")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    else:
        # If an image is provided as an argument, use that.
        # Otherwise, search for the only image in the current directory.
        img_path = args.image
        if img_path is None:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(ext))
            
            if len(image_files) == 0:
                print("No image file found in the current directory.")
                return
            elif len(image_files) > 1:
                print("Multiple image files found. Please specify one as an argument or use --all to process all images.")
                return
            else:
                img_path = image_files[0]
                print(f"Found one image file: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"Error: Image file '{img_path}' not found!")
            return
        
        predicted_label, confidence, _ = predict_image(model, img_path, class_names)
        print(f"Prediction: {predicted_label}")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == '__main__':
    main()

"""
            zf.writestr('predict.py', inference_code)
            
            # Add requirements.txt
            requirements = """torch>=1.8.0
torchvision>=0.9.0
Pillow>=8.0.0
numpy>=1.19.0
"""
            zf.writestr('requirements.txt', requirements)
            
            # Add README
            readme = """# Medical Image Model Inference Package

This package contains a trained medical image model and code to use it for making predictions.

## Contents

- `trained_model.pt`: The trained PyTorch model
- `predict.py`: Python script for making predictions with the model
- `requirements.txt`: Required Python packages

## Setup Instructions

1. Install Python 3.8 or newer if not already installed
2. Place your image files in the same folder as these files
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
To predict a single image:
```
python predict.py --image your_image.jpg
```

### Process All Images
To process all images in the current directory:
```
python predict.py --all
```

### Specify Class Names
If you know the names of the classes:
```
python predict.py --image your_image.jpg --class_names "class1,class2,class3,class4,class5"
```

### Specify Number of Classes
If your model was trained with a different number of classes:
```
python predict.py --image your_image.jpg --num_classes 3
```

## Model Information
"""
            # Add model information to README from training metrics
            if training_metrics:
                readme += "\nModel training details:\n"
                if 'num_classes' in training_metrics:
                    readme += f"- Number of classes: {training_metrics.get('num_classes')}\n"
                if 'train_accuracy' in training_metrics:
                    readme += f"- Training accuracy: {training_metrics.get('train_accuracy'):.2f}%\n"
                if 'test_accuracy' in training_metrics:
                    readme += f"- Test accuracy: {training_metrics.get('test_accuracy'):.2f}%\n"
                if 'training_level' in training_metrics:
                    readme += f"- Training level used: {training_metrics.get('training_level')}\n"
                if 'classes' in training_metrics:
                    readme += "\nClasses in dataset:\n"
                    for i, class_name in enumerate(training_metrics.get('classes')):
                        readme += f"- Class {i}: {class_name}\n"
                
            zf.writestr('README.md', readme)
        
        # Rewind the file-like object
        memory_zip.seek(0)
        
        # Return the zipped package
        return send_file(
            memory_zip,
            mimetype='application/zip',
            as_attachment=True,
            download_name='model_package.zip'
        )
        
    except Exception as e:
        logger.error(f"Error in pipeline process: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5025))
    logger.info(f"Starting pipeline service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True) 