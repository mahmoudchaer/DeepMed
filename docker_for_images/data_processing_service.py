import os
import io
import zipfile
import tempfile
import json
import logging
import random
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, jsonify, Response
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from flask_cors import CORS  # Import CORS for cross-origin support

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def process_data(zip_file, test_size=0.2, val_size=0.2):
    """
    Process image data by normalizing and splitting into train/validation/test sets
    
    Args:
        zip_file: The ZIP file containing image folders
        test_size: Proportion of data to use for testing (default 0.2)
        val_size: Proportion of data to use for validation (default 0.2)
        
    Returns:
        A BytesIO object containing the processed data as a ZIP file and metrics
    """
    # Validate the split parameters
    test_size = float(test_size)
    val_size = float(val_size)
    
    # Ensure test_size and val_size are valid
    if test_size < 0 or test_size > 0.5:
        test_size = 0.2  # Reset to default if invalid
    if val_size < 0 or val_size > 0.5:
        val_size = 0.2  # Reset to default if invalid
    
    # Calculate train size based on test and validation sizes
    train_size = 1.0 - test_size - val_size
    
    # Adjust if train_size is too small
    if train_size < 0.3:
        # Recalculate to ensure at least 30% for training
        train_size = 0.3
        # Distribute the rest between val and test proportionally
        total_val_test = test_size + val_size
        test_size = 0.7 * (test_size / total_val_test)
        val_size = 0.7 * (val_size / total_val_test)
    
    # Create temporary directories for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the uploaded ZIP file
        zip_path = os.path.join(tmpdir, "data.zip")
        zip_file.save(zip_path)
        
        # Extract the ZIP file
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Create directories for processed data
        processed_dir = os.path.join(tmpdir, "processed")
        train_dir = os.path.join(processed_dir, "train")
        val_dir = os.path.join(processed_dir, "val")
        test_dir = os.path.join(processed_dir, "test")
        
        # Create the directories
        for directory in [train_dir, val_dir, test_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Find all class folders
        class_folders = [f for f in os.listdir(extract_dir) 
                        if os.path.isdir(os.path.join(extract_dir, f))]
        
        # Initialize metrics
        metrics = {
            "classes": len(class_folders),
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "splits": {},
            "class_counts": {}
        }
        
        # Process each class folder
        total_files = 0
        train_files = 0
        val_files = 0
        test_files = 0
        
        for class_folder in class_folders:
            # Create corresponding folders in train, val, and test directories
            train_class_dir = os.path.join(train_dir, class_folder)
            val_class_dir = os.path.join(val_dir, class_folder)
            test_class_dir = os.path.join(test_dir, class_folder)
            
            for directory in [train_class_dir, val_class_dir, test_class_dir]:
                os.makedirs(directory, exist_ok=True)
            
            # Get all image files in the class folder
            src_dir = os.path.join(extract_dir, class_folder)
            image_files = [f for f in os.listdir(src_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # Skip if no images found
            if not image_files:
                print(f"No images found in class folder: {class_folder}")
                continue
                
            # Shuffle the images to ensure random splits
            random.shuffle(image_files)
            
            # Count images
            class_total = len(image_files)
            total_files += class_total
            
            # Split the image files into train, val, and test sets using stratified sampling
            # Since we're working with file names, create dummy labels (all 0s)
            dummy_labels = [0] * class_total
            
            # First split off the test set
            train_val_files, test_files_list = train_test_split(
                image_files, 
                test_size=test_size, 
                random_state=42
            )
            
            # Then split the train_val set into train and val sets
            # Recalculate val proportion relative to the remaining data
            val_proportion = val_size / (train_size + val_size)
            train_files_list, val_files_list = train_test_split(
                train_val_files, 
                test_size=val_proportion, 
                random_state=42
            )
            
            # Update metrics for this class
            class_train = len(train_files_list)
            class_val = len(val_files_list)
            class_test = len(test_files_list)
            
            train_files += class_train
            val_files += class_val
            test_files += class_test
            
            metrics["class_counts"][class_folder] = {
                "total": class_total,
                "train": class_train,
                "val": class_val,
                "test": class_test
            }
            
            # Define transformation for preprocessing images
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to standard size
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]     # ImageNet stds
                )
            ])
            
            # Helper function to process and save an image
            def process_and_save_image(img_file, output_dir):
                try:
                    # Load the image
                    img_path = os.path.join(src_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    
                    # No need to apply full preprocessing here - just resize for consistency
                    # We'll save in the original format because we're creating a reusable dataset
                    img_resized = transforms.Resize((224, 224))(img)
                    
                    # Save the processed image
                    output_path = os.path.join(output_dir, img_file)
                    img_resized.save(output_path)
                    
                    return True
                except Exception as e:
                    print(f"Error processing image {img_file}: {str(e)}")
                    return False
            
            # Process and save train images
            for img_file in train_files_list:
                process_and_save_image(img_file, train_class_dir)
            
            # Process and save validation images
            for img_file in val_files_list:
                process_and_save_image(img_file, val_class_dir)
            
            # Process and save test images
            for img_file in test_files_list:
                process_and_save_image(img_file, test_class_dir)
        
        # Update overall metrics
        metrics["total_images"] = total_files
        metrics["splits"] = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        # Create a ZIP file with the processed data
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for root, _, files in os.walk(processed_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, processed_dir)
                    zip_out.write(file_path, arcname)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        return zip_buffer, metrics

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    logger.info("Health check request received")
    return jsonify({"status": "healthy", "service": "data-processing-service"})

@app.route('/process', methods=['POST'])
def process():
    """
    Process a ZIP file containing image folders and return processed dataset
    """
    logger.info("Received /process request")
    
    if 'zipFile' not in request.files:
        logger.error("No zipFile in request")
        return jsonify({"error": "No ZIP file uploaded"}), 400
    
    try:
        zip_file = request.files['zipFile']
        test_size = float(request.form.get('testSize', 0.2))
        val_size = float(request.form.get('valSize', 0.2))
        
        logger.info(f"Processing data with test_size={test_size}, val_size={val_size}")
        
        # Process the data
        processed_zip, metrics = process_data(zip_file, test_size=test_size, val_size=val_size)
        
        # Create response with the processed data
        response = Response(processed_zip.getvalue())
        response.headers["Content-Type"] = "application/zip"
        response.headers["Content-Disposition"] = "attachment; filename=processed_data.zip"
        
        # Add metrics as a custom header
        response.headers["X-Processing-Metrics"] = json.dumps(metrics)
        
        logger.info(f"Processing complete: {metrics}")
        return response
    
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5112))
    logger.info(f"Starting data processing service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 