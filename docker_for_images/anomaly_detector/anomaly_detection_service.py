import os
import io
import json
import zipfile
import tempfile
import logging
import traceback
import shutil
import yaml
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, Response, send_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Track temp files for cleanup
temp_dirs = []

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Custom dataset for anomaly detection
class AnomalyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                          if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "anomaly-detection-service"})

@app.route('/train', methods=['POST'])
def train_model():
    """
    Endpoint to train an anomaly detection model using a PyTorch Autoencoder
    
    Expects a multipart/form-data POST with:
    - zipFile: A zip file with normal (non-anomalous) images for training
    - level: Level of training (1-5) which determines the complexity and training time
    - image_size: (Optional) Size to resize images to (default: 256)
    
    Returns a zip file of the trained model
    """
    try:
        if 'zipFile' not in request.files:
            return jsonify({"error": "No ZIP file uploaded"}), 400
        
        # Get the zip file from the request
        zip_file = request.files['zipFile']
        
        # Get level parameter with default
        level = int(request.form.get('level', 3))
        
        # Get image size with default
        image_size = int(request.form.get('image_size', 256))
        
        # Define preset configurations for each level
        level_configs = {
            1: {
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-3
            },
            2: {
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 1e-3
            },
            3: {
                'epochs': 30,
                'batch_size': 16,
                'learning_rate': 1e-3
            },
            4: {
                'epochs': 50,
                'batch_size': 8,
                'learning_rate': 5e-4
            },
            5: {
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 1e-4
            }
        }
        
        # Validate level
        if level < 1 or level > 5:
            return jsonify({"error": "Level must be between 1 and 5"}), 400
        
        # Get configuration for the selected level
        config = level_configs[level]
        epochs = config['epochs']
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        
        # Log the start of processing
        logger.info(f"Starting Anomaly Detection training with level {level} (epochs: {epochs})")
        
        # Create a temporary working directory
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        
        # Create dataset directory
        dataset_dir = os.path.join(temp_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save the uploaded zip file
        zip_path = os.path.join(temp_dir, "dataset.zip")
        zip_file.save(zip_path)
        
        # Extract the dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Find image directory
        image_dir = None
        image_files = []
        
        # Find all image files in the extracted zip folder (including subfolders)
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            return jsonify({"error": "No image files found in the uploaded dataset"}), 400
        
        # If all images are in a single directory, use that directory
        parent_dirs = set(os.path.dirname(f) for f in image_files)
        if len(parent_dirs) == 1:
            image_dir = parent_dirs.pop()
        else:
            # If images are in multiple directories, create a flat directory with all images
            image_dir = os.path.join(temp_dir, "all_images")
            os.makedirs(image_dir, exist_ok=True)
            for i, img_path in enumerate(image_files):
                img_ext = os.path.splitext(img_path)[1]
                shutil.copy(img_path, os.path.join(image_dir, f"image_{i}{img_ext}"))
        
        logger.info(f"Found {len(image_files)} images for training")
        
        # Create results directory
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # Create dataset and dataloader
        dataset = AnomalyDataset(image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        if len(dataset) == 0:
            return jsonify({"error": "No valid images found in the dataset"}), 400
        
        logger.info(f"Created dataset with {len(dataset)} images")
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model = Autoencoder(input_channels=3)
        model = model.to(device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        logger.info("Starting training...")
        losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_idx, data in enumerate(dataloader):
                data = data.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.6f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_epoch_loss:.6f}")
        
        # Save the trained model
        model_save_path = os.path.join(results_dir, "autoencoder.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'image_size': image_size,
                'input_channels': 3
            }
        }, model_save_path)
        
        # Calculate reconstruction loss threshold based on training data
        logger.info("Calculating threshold for anomaly detection...")
        model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                outputs = model(data)
                
                # Calculate MSE for each image in the batch
                for i in range(data.shape[0]):
                    mse = ((data[i] - outputs[i]) ** 2).mean().item()
                    reconstruction_errors.append(mse)
        
        # Set threshold as mean + 2*std of reconstruction errors (95% confidence)
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        threshold = mean_error + 2 * std_error
        
        # Create a metadata file with model info
        metadata = {
            'model_type': 'autoencoder',
            'image_size': image_size,
            'threshold': threshold,
            'mean_error': mean_error,
            'std_error': std_error,
            'training_losses': losses
        }
        
        metadata_path = os.path.join(results_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Create a zip file to return the model and related files
        temp_output_zip = os.path.join(temp_dir, "model_output.zip")
        
        # Create predict.py file
        detect_script_content = '''import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import os
import glob
import argparse

# Define the Autoencoder architecture (must match the one used for training)
class Autoencoder(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU()
        )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def detect_anomaly(image_path, model_path='autoencoder.pt', metadata_path='metadata.json'):
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata.get('threshold')
    image_size = metadata.get('image_size', 256)
    
    # Load model
    model = Autoencoder(input_channels=3)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Detect anomaly
    with torch.no_grad():
        output = model(image_tensor)
        mse = ((image_tensor - output) ** 2).mean().item()
    
    is_anomaly = mse > threshold
    
    return {
        'reconstruction_error': mse,
        'threshold': threshold,
        'is_anomaly': is_anomaly
    }

def find_unique_image():
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(ext))
        
    if len(image_files) == 0:
        return None, "No image file found in the current directory."
    elif len(image_files) > 1:
        return None, "Multiple image files found. Please specify one as an argument."
    else:
        return image_files[0], f"Found one image file: {image_files[0]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect anomaly using a trained autoencoder model.')
    # Define an optional positional argument for the image path.
    parser.add_argument('image', nargs='?', default=None, help='Path to the image file (optional; auto-detect if not provided)')
    parser.add_argument('--model_path', type=str, default='autoencoder.pt', help='Path to the autoencoder model file')
    parser.add_argument('--metadata_path', type=str, default='metadata.json', help='Path to the metadata JSON file')
    
    args = parser.parse_args()
    
    image_path = args.image
    if image_path is None:
        image_path, message = find_unique_image()
        if image_path is None:
            print(message)
            exit(1)
        else:
            print(message)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found")
        exit(1)
    
    if not os.path.exists(args.metadata_path):
        print(f"Error: Metadata file '{args.metadata_path}' not found")
        exit(1)
    
    result = detect_anomaly(image_path, args.model_path, args.metadata_path)
    print("Anomaly detection result:")
    print(f"  Reconstruction error: {result['reconstruction_error']:.6f}")
    print(f"  Threshold: {result['threshold']:.6f}")
    print(f"  Is anomaly: {result['is_anomaly']}")

'''
        
        # Create requirements.txt content
        requirements_content = '''torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.2.0
numpy>=1.19.5
'''
        
        # Create README.md content
        readme_content = '''# Anomaly Detection Model

This package contains a trained autoencoder model for image anomaly detection. The model was trained on normal (non-anomalous) images and can be used to detect anomalies in new images.

## Contents

- `autoencoder.pt`: The trained PyTorch model
- `metadata.json`: Model metadata including the detection threshold
- `predict.py`: Python script to run inference on new images
- `requirements.txt`: Required Python packages

## Setup Instructions

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv anomaly_env

# Activate the virtual environment
# On Windows:
anomaly_env\\Scripts\\activate
# On macOS/Linux:
source anomaly_env/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run Anomaly Detection

Place the image(s) you want to analyze in the same directory as the model files, or provide the full path to the images.

```bash
# Basic usage (uses default model and metadata filenames)
python predict.py your_image.jpg

# Specify custom paths for model and metadata if needed
python predict.py your_image.jpg custom_model_name.pt custom_metadata.json
```

### Example Usage

```bash
# Detect anomalies in a single image
python predict.py test_image.jpg

# Expected output:
# Anomaly detection result:
#   Reconstruction error: 0.023456
#   Threshold: 0.018765
#   Is anomaly: True
```

### Understanding Results

- **Reconstruction error**: The mean squared error between the original image and its reconstruction by the autoencoder
- **Threshold**: The value above which an image is considered anomalous
- **Is anomaly**: Boolean result indicating whether the image contains an anomaly

## Batch Processing

To process multiple images, you can create a simple script that iterates through files in a directory:

```python
import os
import glob
from detect_anomaly import detect_anomaly

# Process all jpg images in a directory
image_dir = "your_images_directory"
for image_path in glob.glob(os.path.join(image_dir, "*.jpg")):
    print(f"Processing {image_path}...")
    result = detect_anomaly(image_path)
    print(f"  Is anomaly: {result['is_anomaly']}")
    print(f"  Error: {result['reconstruction_error']:.6f}")
```

## Troubleshooting

If you encounter any issues:

1. Make sure you've activated the virtual environment
2. Verify that all requirements are installed correctly
3. Check that the model and metadata files are in the correct location
4. Ensure your images are valid and can be opened by PIL/Pillow
5. For CUDA errors, try running with CPU only by modifying the `map_location` parameter in the script
'''
        
        # Save files in the results directory
        detect_script_path = os.path.join(results_dir, "predict.py")
        requirements_path = os.path.join(results_dir, "requirements.txt")
        readme_path = os.path.join(results_dir, "README.md")
        
        try:
            with open(detect_script_path, 'w') as f:
                f.write(detect_script_content)
            logger.info(f"Successfully created predict.py at {detect_script_path}")
            logger.info(f"File exists: {os.path.exists(detect_script_path)}, Size: {os.path.getsize(detect_script_path)} bytes")
        except Exception as e:
            logger.error(f"Error creating predict.py: {str(e)}")
        
        try:
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            logger.info(f"Successfully created requirements.txt at {requirements_path}")
            logger.info(f"File exists: {os.path.exists(requirements_path)}, Size: {os.path.getsize(requirements_path)} bytes")
        except Exception as e:
            logger.error(f"Error creating requirements.txt: {str(e)}")
            
        try:
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            logger.info(f"Successfully created README.md at {readme_path}")
            logger.info(f"File exists: {os.path.exists(readme_path)}, Size: {os.path.getsize(readme_path)} bytes")
        except Exception as e:
            logger.error(f"Error creating README.md: {str(e)}")
        
        logger.info("Created all necessary files for inclusion in output")
        
        # NEW APPROACH: Create the zip file in memory directly
        memory_file = io.BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w') as zipf:
            try:
                # Add the model by reading and writing the content
                with open(model_save_path, 'rb') as f:
                    model_data = f.read()
                zipf.writestr("autoencoder.pt", model_data)
                logger.info(f"Added autoencoder.pt to zip")
                
                # Add metadata
                with open(metadata_path, 'r') as f:
                    metadata_data = f.read()
                zipf.writestr("metadata.json", metadata_data)
                logger.info(f"Added metadata.json to zip")
                
                # Add detection script directly
                zipf.writestr("predict.py", detect_script_content)
                logger.info(f"Added predict.py to zip")
                
                # Add requirements directly
                zipf.writestr("requirements.txt", requirements_content)
                logger.info(f"Added requirements.txt to zip")
                
                # Add README directly
                zipf.writestr("README.md", readme_content)
                logger.info(f"Added README.md to zip")
            except Exception as e:
                logger.error(f"Error adding files to zip: {str(e)}")
            
            logger.info(f"Added all files to output zip: {zipf.namelist()}")
        
        # Verify zip contents
        memory_file.seek(0)
        with zipfile.ZipFile(memory_file, 'r') as zipf:
            contents = zipf.namelist()
            logger.info(f"Zip file contains: {contents}")
            
            # Ensure all required files are in the zip
            required_files = ["autoencoder.pt", "metadata.json", "predict.py", "requirements.txt", "README.md"]
            for file in required_files:
                if file not in contents:
                    logger.error(f"Required file {file} is missing from the zip!")
                else:
                    logger.info(f"Verified {file} is in the zip")
                    
                    # Debug: Print file size
                    file_info = zipf.getinfo(file)
                    logger.info(f"  - Size: {file_info.file_size} bytes")
        
        # Reset the file pointer
        memory_file.seek(0)
        
        # Clean up
        cleanup_temp_dirs()
        
        # Return the zip file
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='anomaly_detection_model.zip'
        )
        
    except Exception as e:
        logger.error(f"Error during anomaly detection training: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500
    finally:
        # Make sure to clean up temp files even if there's an error
        cleanup_temp_dirs()

@app.route('/detect', methods=['POST'])
def detect_anomalies():
    """
    Endpoint to detect anomalies in images using a trained autoencoder
    
    Expects a multipart/form-data POST with:
    - modelFile: A trained autoencoder model
    - imageFile: An image to check for anomalies
    
    Returns JSON with anomaly score and whether it's anomalous
    """
    try:
        if 'modelFile' not in request.files or 'imageFile' not in request.files:
            return jsonify({"error": "Both model file and image file are required"}), 400
        
        # Get the files from the request
        model_file = request.files['modelFile']
        image_file = request.files['imageFile']
        
        # Create a temporary working directory
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        
        # Save the uploaded files
        model_path = os.path.join(temp_dir, "autoencoder.pt")
        image_path = os.path.join(temp_dir, "test_image.jpg")
        
        model_file.save(model_path)
        image_file.save(image_path)
        
        # Load the model
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_config = checkpoint['config']
        image_size = model_config.get('image_size', 256)
        
        # Load metadata if it exists in the same zip
        metadata = None
        if 'metadataFile' in request.files:
            metadata_file = request.files['metadataFile']
            metadata_path = os.path.join(temp_dir, "metadata.json")
            metadata_file.save(metadata_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Initialize model
        model = Autoencoder(input_channels=model_config.get('input_channels', 3))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load and preprocess the image
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Detect anomaly
        with torch.no_grad():
            output = model(image_tensor)
            
            # Calculate reconstruction error
            mse = ((image_tensor - output) ** 2).mean().item()
        
        # Determine if it's an anomaly based on threshold
        threshold = metadata.get('threshold', 0.02) if metadata else 0.02
        is_anomaly = mse > threshold
        
        # Prepare the response
        result = {
            'reconstruction_error': float(mse),
            'threshold': float(threshold) if threshold is not None else None,
            'is_anomaly': bool(is_anomaly)
        }
        
        # Clean up
        cleanup_temp_dirs()
        
        # Return result
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during anomaly detection: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500
    finally:
        # Make sure to clean up temp files even if there's an error
        cleanup_temp_dirs()

def cleanup_temp_dirs():
    """Clean up temporary directories"""
    global temp_dirs
    for dir_path in temp_dirs:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.info(f"Cleaned up temporary directory: {dir_path}")
        except Exception as e:
            logger.error(f"Error cleaning up {dir_path}: {e}")
    temp_dirs = []

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5029))
    logger.info(f"Starting anomaly detection service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False) 